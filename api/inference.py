from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TYPE_CHECKING:
    import dgl

import config
from config import CHECKPOINT_PATH

# NOTE: we deliberately do NOT do `from config import ACTIVE_CLASS_MAP` here.
# That would bind a local name to whatever dict object config.ACTIVE_CLASS_MAP
# pointed to at import time. `_load_model()` below sometimes reassigns
# `config.ACTIVE_CLASS_MAP = {...}` at runtime (e.g. to auto-correct a
# num_classes mismatch) — a locally-imported name would never see that
# update, since `config.ACTIVE_CLASS_MAP = ...` rebinds the module
# attribute, it does not mutate the dict a prior `from x import y` copied a
# reference to. Always go through `config.ACTIVE_CLASS_MAP` at call time.

logger = logging.getLogger(__name__)

_model: object | None = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None





def _load_model(checkpoint_path: Path | None = None) -> object:
    global _model

    ckpt = checkpoint_path or CHECKPOINT_PATH

    if not ckpt.exists():
        logger.error("Checkpoint not found at '%s'. Inference cannot proceed.", ckpt)
        raise FileNotFoundError(f"Checkpoint not found at '{ckpt}'")

    try:
        # Attempt to import the real UV-Net segmentation model
        from original_uvnet import Segmentation

        # --- Try Lightning-style loading first ---
        try:
            _model = Segmentation.load_from_checkpoint(str(ckpt), map_location=_device)
            logger.info("✅ Loaded UV-Net checkpoint (Lightning) from '%s' on %s", ckpt, _device)
        except Exception as lightning_exc:
            logger.warning(
                "Lightning checkpoint loading failed (%s). Trying direct torch.load…",
                lightning_exc,
            )
            # --- Fallback: manual torch.load ---
            checkpoint = torch.load(str(ckpt), map_location=_device, weights_only=False)

            # Auto-detect num_classes from checkpoint
            hparams = checkpoint.get("hyper_parameters", checkpoint.get("hparams", {}))
            state_dict = checkpoint.get("state_dict", checkpoint)

            # Try to infer num_classes from the classifier's final linear layer
            num_classes = hparams.get("num_classes", None)
            if num_classes is None:
                for key in state_dict:
                    if "classifier.linear3.weight" in key or "linear3.weight" in key:
                        num_classes = state_dict[key].shape[0]
                        break
            if num_classes is None:
                num_classes = len(config.ACTIVE_CLASS_MAP)
                logger.warning(
                    "Could not detect num_classes from checkpoint; defaulting to %d",
                    num_classes,
                )

            # Dynamically update ACTIVE_CLASS_MAP if num_classes differs.
            # This now actually takes effect everywhere, since run_inference()
            # and map_predictions() below read config.ACTIVE_CLASS_MAP at call
            # time instead of holding a stale imported reference.
            if num_classes != len(config.ACTIVE_CLASS_MAP):
                logger.warning(
                    "Checkpoint num_classes=%d differs from ACTIVE_CLASS_MAP (%d entries). "
                    "Generating generic class labels.",
                    num_classes,
                    len(config.ACTIVE_CLASS_MAP),
                )
                config.ACTIVE_CLASS_MAP = {
                    i: config.ACTIVE_CLASS_MAP.get(i, f"class_{i}")
                    for i in range(num_classes)
                }

            # Build model and load weights
            model_kwargs = {k: v for k, v in hparams.items() if k not in ("num_classes", "lr")}
            _model = Segmentation(num_classes=num_classes, **model_kwargs)

            # Clean up state dict keys (strip Lightning prefixes)
            cleaned = {}
            for k, v in state_dict.items():
                new_key = k.replace("model.model.", "model.").replace("module.", "")
                cleaned[new_key] = v
            _model.load_state_dict(cleaned, strict=False)
            logger.info("✅ Loaded UV-Net checkpoint (torch.load fallback, num_classes=%d)", num_classes)

        _model.eval()
        if hasattr(_model, "to"):
            _model.to(_device)

    except ImportError:
        logger.error("Could not import 'original_uvnet.Segmentation'.")
        raise
    except Exception as exc:
        logger.error("Failed to load checkpoint: %s", exc, exc_info=True)
        _model = None

    return _model



def get_model():
    global _model
    if _model is None:
        _load_model()
    return _model



def run_inference(graph: "dgl.DGLGraph") -> list[int]:
    model = get_model()

    if model is None:
        raise RuntimeError("Model is not loaded for inference.")

    # ── Real inference ────────────────────────────────────────────────────
    with torch.no_grad():
        # Permute to channel-first format expected by the convolutional encoders
        # ndata['x']: [N, H, W, C] → [N, C, H, W]
        # edata['x']: [E, L, C]    → [E, C, L]
        graph = graph.to(_device)
        graph.ndata["x"] = graph.ndata["x"].permute(0, 3, 1, 2).float()
        graph.edata["x"] = graph.edata["x"].permute(0, 2, 1).float()

        logits = model(graph)  # [total_nodes, num_classes]
        logger.debug("Raw logits per face:\n%s", logits.cpu().numpy())

        if not torch.isfinite(logits).all():
            # If this ever fires, a face produced NaN/Inf features upstream
            # (degenerate uvgrid sample, e.g. a surface pole) and it has
            # propagated through message passing into other faces' logits.
            # pipeline.py now sanitizes ndata/edata for this, but this check
            # is left in place so a regression shows up in the logs instead
            # of as a silent all-one-class collapse.
            logger.error(
                "Non-finite values detected in model logits — predictions "
                "may be unreliable. Check pipeline.py's NaN sanitization."
            )

        preds = torch.argmax(logits, dim=-1)  # softmax doesn't change argmax
        preds_list = preds.cpu().tolist()
        logger.debug("Final predictions: %s", preds_list)
        return preds_list



def map_predictions(class_indices: list[int]) -> list[dict]:
    return [
        {
            "face_id": idx,
            "type": config.ACTIVE_CLASS_MAP.get(cls, f"unknown_{cls}"),
        }
        for idx, cls in enumerate(class_indices)
    ]