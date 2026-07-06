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

from config import ACTIVE_CLASS_MAP, CHECKPOINT_PATH

logger = logging.getLogger(__name__)

_model: object | None = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None





def _load_model(checkpoint_path: Path | None = None) -> object:
    global _model

    ckpt = checkpoint_path or CHECKPOINT_PATH

    if not ckpt.exists():
        logger.warning(
            "Checkpoint not found at '%s'. Inference will use STUB predictions. "
            "Place your trained best.ckpt in api/checkpoints/ to enable real inference.",
            ckpt,
        )
        _model = None
        return _model

    try:
        # Attempt to import the real UV-Net segmentation model
        from uvnet.models import Segmentation

        _model = Segmentation.load_from_checkpoint(str(ckpt), map_location=_device)
        _model.eval()
        _model.to(_device)
        logger.info("✅ Loaded UV-Net checkpoint from '%s' on %s", ckpt, _device)
    except ImportError:
        logger.warning(
            "Could not import 'uvnet.models.Segmentation'. "
            "Ensure the UV-Net repo is on PYTHONPATH. Using stub predictions."
        )
        _model = None
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
        # ── Stub mode: return mock predictions so the API is testable
        # without a real checkpoint.
        num_faces = graph.num_nodes()
        logger.info(
            "🔶 STUB inference: generating mock predictions for %d faces", num_faces
        )
        num_classes = len(ACTIVE_CLASS_MAP)
        return [i % num_classes for i in range(num_faces)]

    # ── Real inference ────────────────────────────────────────────────────
    with torch.no_grad():
        # Permute to channel-first format expected by the convolutional encoders
        # ndata['x']: [N, H, W, C] → [N, C, H, W]
        # edata['x']: [E, L, C]    → [E, C, L]
        graph = graph.to(_device)
        graph.ndata["x"] = graph.ndata["x"].permute(0, 3, 1, 2).float()
        graph.edata["x"] = graph.edata["x"].permute(0, 2, 1).float()

        logits = model(graph)  # [total_nodes, num_classes]
        preds = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
        preds_list = preds.cpu().tolist()
        print(f"[INFO] Final Predictions: {preds_list}")
        return preds_list



def map_predictions(class_indices: list[int]) -> list[dict]:
    return [
        {
            "face_id": idx,
            "type": ACTIVE_CLASS_MAP.get(cls, f"unknown_{cls}"),
        }
        for idx, cls in enumerate(class_indices)
    ]
