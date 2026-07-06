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
    os.getenv("CHECKPOINT_PATH", str(BASE_DIR / "checkpoints" / "uvnet-fusiongallery.ckpt"))
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
# NOTE: kept for reference only. This taxonomy is NOT well-posed for
# fillet/chamfer recognition: a fillet IS a cylindrical/toroidal surface,
# and a chamfer IS a planar surface, so asking a classifier to separate
# "cylinder" from "fillet" using that face's own geometry alone conflates
# surface *type* with feature *semantics*. Prefer FEATURE_OPERATION_CLASS_MAP
# below for real fillet/chamfer work.
SURFACE_TYPE_CLASS_MAP: dict[int, str] = {
    0: "plane",
    1: "cylinder",
    2: "cone",
    3: "sphere",
    4: "torus",
    5: "fillet",
    6: "chamfer",
}

# CAD-operation-based labels, mirroring the ASM-derived scheme the UV-Net
# paper uses on the ABC dataset (Appendix A.3.4). This encodes *why* a face
# exists (which modeling operation produced it) rather than its raw
# primitive type, which is what actually lets "fillet" be distinguished
# from a generic cylinder, and "chamfer" from a generic plane.
#
# IMPORTANT: swapping this dict alone does NOT fix model accuracy — it only
# changes how predicted class indices are *displayed*. Your checkpoint must
# actually be trained (or fine-tuned) against ground-truth labels drawn
# from this same taxonomy, generated e.g. via a CAD kernel's rule-based
# operation classifier (see paper Appendix A.3.4 for the exact recipe).
FEATURE_OPERATION_CLASS_MAP: dict[int, str] = {
    0: "extrude_side",
    1: "extrude_end",
    2: "cut_side",
    3: "cut_end",
    4: "fillet",
    5: "chamfer",
    6: "revolve",
    7: "stock",
}

# Active class map. MUST match, in order, the exact label space your
# checkpoint's classifier head was trained against — index i here must mean
# the same thing as class index i in the model's output logits.
ACTIVE_CLASS_MAP: dict[int, str] = FEATURE_OPERATION_CLASS_MAP

# Server
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))