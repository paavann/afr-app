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

from config import MODEL_REGISTRY, DEFAULT_MODEL

logger = logging.getLogger(__name__)

# ── Model cache ───────────────────────────────────────────────────────────
# Maps model_type → loaded nn.Module so we only load each checkpoint once.
_models: dict[str, object] = {}
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None



def _load_model(model_type: str) -> object:
    """Load a model checkpoint by its registry key and cache it."""
    if model_type in _models:
        return _models[model_type]

    entry = MODEL_REGISTRY.get(model_type)
    if entry is None:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    ckpt_path = entry["checkpoint"]
    class_map = entry["class_map"]

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at '{ckpt_path}'")

    try:
        from original_uvnet import Segmentation

        # --- Try Lightning-style loading first ---
        try:
            model = Segmentation.load_from_checkpoint(str(ckpt_path), map_location=_device)
            num_c = getattr(model.model.classifier.linear3, "out_features", "unknown")
            print(f"[INFO] Loaded '{model_type}' checkpoint (Lightning) from '{ckpt_path}' on {_device} (num_classes={num_c})")
        except Exception as lightning_exc:
            print(f"[WARN] Lightning loading failed for '{model_type}' ({lightning_exc}). Trying torch.load…")
            checkpoint = torch.load(str(ckpt_path), map_location=_device, weights_only=False)

            hparams = checkpoint.get("hyper_parameters", checkpoint.get("hparams", {}))
            state_dict = checkpoint.get("state_dict", checkpoint)

            # Auto-detect num_classes from checkpoint
            num_classes = hparams.get("num_classes", None)
            if num_classes is None:
                for key in state_dict:
                    if "classifier.linear3.weight" in key or "linear3.weight" in key:
                        num_classes = state_dict[key].shape[0]
                        break
            if num_classes is None:
                num_classes = len(class_map)
                print(f"[WARN] Could not detect num_classes from '{model_type}' checkpoint; defaulting to {num_classes} from class map")

            model_kwargs = {k: v for k, v in hparams.items() if k not in ("num_classes", "lr")}
            model = Segmentation(num_classes=num_classes, **model_kwargs)

            cleaned = {}
            for k, v in state_dict.items():
                new_key = k.replace("model.model.", "model.").replace("module.", "")
                cleaned[new_key] = v
            model.load_state_dict(cleaned, strict=False)
            num_c = getattr(model.model.classifier.linear3, "out_features", num_classes)
            print(f"[INFO] Loaded '{model_type}' checkpoint (torch.load fallback, num_classes={num_c})")

        model.eval()
        if hasattr(model, "to"):
            model.to(_device)

        _models[model_type] = model
        return model

    except ImportError:
        print("[ERROR] Could not import 'original_uvnet.Segmentation'.")
        raise
    except Exception as exc:
        print(f"[ERROR] Failed to load '{model_type}' checkpoint: {exc}")
        raise



def get_model(model_type: str | None = None) -> object:
    """Return a cached model, loading it on first access."""
    mt = model_type or DEFAULT_MODEL
    return _load_model(mt)



def run_inference(graph: "dgl.DGLGraph", model_type: str | None = None) -> list[int]:
    """Run UV-Net segmentation inference on a DGL graph."""
    mt = model_type or DEFAULT_MODEL
    model = get_model(mt)

    if model is None:
        raise RuntimeError(f"Model '{mt}' is not loaded for inference.")

    with torch.no_grad():
        graph = graph.to(_device)
        graph.ndata["x"] = graph.ndata["x"].permute(0, 3, 1, 2).float()
        graph.edata["x"] = graph.edata["x"].permute(0, 2, 1).float()

        logits = model(graph)
        # print(f"[DEBUG] Raw logits per face:\n{logits.cpu().numpy()}")

        if not torch.isfinite(logits).all():
            print("[ERROR] Non-finite values detected in model logits — predictions may be unreliable. Check pipeline.py's NaN sanitization.")

        preds = torch.argmax(logits, dim=-1)
        preds_list = preds.cpu().tolist()
        print(f"[INFO] Final predictions for {mt}: {preds_list}")
        return preds_list



def map_predictions(class_indices: list[int], model_type: str | None = None) -> list[dict]:
    """Map raw class indices to human-readable feature names using the model's class map."""
    mt = model_type or DEFAULT_MODEL
    entry = MODEL_REGISTRY.get(mt)
    if entry is None:
        raise ValueError(f"Unknown model_type '{mt}'.")

    class_map = entry["class_map"]
    return [
        {
            "face_id": idx,
            "type": class_map.get(cls, f"unknown_{cls}"),
        }
        for idx, cls in enumerate(class_indices)
    ]