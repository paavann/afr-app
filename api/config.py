import os
from pathlib import Path



BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/tmp/uploads"))
GRAPH_DIR = Path(os.getenv("GRAPH_DIR", "/tmp/graphs"))
MESH_DIR = Path(os.getenv("MESH_DIR", "/tmp/meshes"))
STATIC_DIR = Path(os.getenv("STATIC_DIR", "/tmp/static"))

for _d in (UPLOAD_DIR, GRAPH_DIR, MESH_DIR, STATIC_DIR):
    _d.mkdir(parents=True, exist_ok=True)



CURV_NUM_U_SAMPLES: int = 10
SURF_NUM_U_SAMPLES: int = 10
SURF_NUM_V_SAMPLES: int = 10


# Mesh triangulation tolerances
TRIANGLE_FACE_TOL: float = 0.01
ANGLE_TOL_RADS: float = 0.1


CORS_ORIGINS: list[str] = ["*"]


# ── Model registry ────────────────────────────────────────────────────────
# Each entry maps a model key (sent by the frontend) to its checkpoint path
# and class map. The frontend sends `model_type` as a string; the backend
# looks it up here.

FUSIONGALLERY_CLASS_MAP: dict[int, str] = {
    0: "extrude_side",
    1: "extrude_end",
    2: "cut_side",
    3: "cut_end",
    4: "fillet",
    5: "chamfer",
    6: "revolve_side",
    7: "revolve_end",
}

MFCAD_CLASS_MAP: dict[int, str] = {
    0: "rectangular_through_slot",
    1: "triangular_through_slot",
    2: "rectangular_passage",
    3: "triangular_passage",
    4: "6sides_passage",
    5: "rectangular_through_step",
    6: "2sides_through_step",
    7: "slanted_through_step",
    8: "rectangular_blind_step",
    9: "triangular_blind_step",
    10: "rectangular_blind_slot",
    11: "rectangular_pocket",
    12: "triangular_pocket",
    13: "6sides_pocket",
    14: "chamfer",
    15: "stock",
}


MODEL_REGISTRY: dict[str, dict] = {
    "fusiongallery": {
        "checkpoint": BASE_DIR / "checkpoints" / "uvnet-fusiongallery.ckpt",
        "class_map": FUSIONGALLERY_CLASS_MAP,
        "display_name": "Fusion 360 Gallery",
    },
    "mfcad": {
        "checkpoint": BASE_DIR / "checkpoints" / "uvnet-mfcad.ckpt",
        "class_map": MFCAD_CLASS_MAP,
        "display_name": "MFCAD",
    },
}

DEFAULT_MODEL: str = "fusiongallery"


# Server
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))