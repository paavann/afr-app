import os
from pathlib import Path



BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/tmp/uploads"))
GRAPH_DIR = Path(os.getenv("GRAPH_DIR", "/tmp/graphs"))
MESH_DIR = Path(os.getenv("MESH_DIR", "/tmp/meshes"))
STATIC_DIR = Path(os.getenv("STATIC_DIR", "/tmp/static"))

for _d in (UPLOAD_DIR, GRAPH_DIR, MESH_DIR, STATIC_DIR):
    _d.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = Path(
    os.getenv("CHECKPOINT_PATH", str(BASE_DIR / "checkpoints" / "best.ckpt"))
)



CURV_NUM_U_SAMPLES: int = 10
SURF_NUM_U_SAMPLES: int = 10
SURF_NUM_V_SAMPLES: int = 10


# Mesh triangulation tolerances
TRIANGLE_FACE_TOL: float = 0.01
ANGLE_TOL_RADS: float = 0.1


CORS_ORIGINS: list[str] = [ "*" ]

# MFCAD class map — 16 machining feature classes
# Maps predicted class index → human-readable feature name.
# Source: MFCAD dataset (Cao et al., 2020)
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

# Simplified / generic class map (for broader B-rep surface type labelling)
SURFACE_TYPE_CLASS_MAP: dict[int, str] = {
    0: "plane",
    1: "cylinder",
    2: "cone",
    3: "sphere",
    4: "torus",
    5: "fillet",
    6: "chamfer",
}

# Active class map — swap to SURFACE_TYPE_CLASS_MAP if using a different model
ACTIVE_CLASS_MAP: dict[int, str] = MFCAD_CLASS_MAP

# Server
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))