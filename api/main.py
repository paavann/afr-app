from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from config import (
    API_HOST,
    API_PORT,
    CORS_ORIGINS,
    GRAPH_DIR,
    MESH_DIR,
    STATIC_DIR,
    UPLOAD_DIR,
)
from inference import map_predictions, run_inference
from pipeline import build_face_adjacency_graph, build_render_mesh, save_graph
from schemas import ErrorResponse, FacePrediction, PredictionResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("cad_api")

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="CAD Feature Recognition API",
    description=(
        "Inference gateway for the UV-Net face segmentation model. "
        "Upload a .step CAD file and receive per-face feature predictions "
        "plus a downloadable .stl mesh for 3D visualization."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS — allow the React frontend to talk to us
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Static file serving — generated .stl meshes
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/api/health", tags=["System"])
async def health_check():
    """Lightweight liveness probe."""
    return {"status": "healthy", "service": "cad-feature-recognition-api"}


# ---------------------------------------------------------------------------
# Core prediction endpoint
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS = {".step", ".stp", ".STEP", ".STP"}


@app.post(
    "/api/predict-cad",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input file"},
        422: {"model": ErrorResponse, "description": "CAD kernel processing failure"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    tags=["Inference"],
    summary="Run CAD face segmentation on an uploaded STEP file",
)
async def predict_cad(request: Request, file: UploadFile = File(...)):
    """
    Accept a `.step` CAD file, process it through the full UV-Net pipeline,
    and return per-face feature type predictions alongside a mesh URL.

    **Pipeline steps:**

    1. Validate and save the uploaded file.
    2. Convert STEP → DGL face-adjacency graph (``solid_to_graph``).
    3. Convert STEP → STL render mesh (``solid_to_rendermesh``).
    4. Run UV-Net segmentation inference on the graph.
    5. Map predicted class indices to human-readable labels.
    6. Return JSON response with predictions and mesh download URL.
    """

    # ── 0. Validate file extension ────────────────────────────────────────
    if file.filename is None:
        raise HTTPException(status_code=400, detail="No filename provided.")

    file_ext = Path(file.filename).suffix
    if file_ext.lower() not in {e.lower() for e in ALLOWED_EXTENSIONS}:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{file_ext}'. "
                f"Please upload a STEP file ({', '.join(ALLOWED_EXTENSIONS)})."
            ),
        )

    # Unique job ID to isolate concurrent requests
    job_id = uuid.uuid4().hex[:12]
    original_filename = file.filename
    stem = Path(original_filename).stem

    # Paths for this job
    upload_path = UPLOAD_DIR / f"{stem}_{job_id}{file_ext}"
    graph_path = GRAPH_DIR / f"{stem}_{job_id}.bin"
    stl_filename = f"{stem}_{job_id}.stl"
    stl_path = STATIC_DIR / stl_filename

    try:
        # ── 1. Save uploaded file ─────────────────────────────────────────
        logger.info("📥 Received '%s' (job=%s)", original_filename, job_id)
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        upload_path.write_bytes(contents)
        logger.info("💾 Saved upload → %s (%d bytes)", upload_path, len(contents))

        # ── 2. STEP → DGL graph ──────────────────────────────────────────
        logger.info("🔧 Building face-adjacency graph…")
        try:
            dgl_graph = build_face_adjacency_graph(upload_path)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error("Graph construction failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=422,
                detail=(
                    "CAD kernel failed to parse the boundary representation. "
                    f"Error: {exc}"
                ),
            )

        save_graph(dgl_graph, graph_path)

        # ── 3. STEP → STL mesh ───────────────────────────────────────────
        logger.info("🔧 Building render mesh…")
        try:
            build_render_mesh(upload_path, stl_path)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error("Mesh generation failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=422,
                detail=f"Mesh triangulation failed: {exc}",
            )

        # ── 4. Model inference ────────────────────────────────────────────
        logger.info("🧠 Running UV-Net segmentation inference…")
        class_indices = run_inference(dgl_graph)

        # ── 5. Build response ─────────────────────────────────────────────
        predictions = map_predictions(class_indices)
        base_url = str(request.base_url).rstrip("/")
        mesh_url = f"{base_url}/static/{stl_filename}"

        logger.info(
            "✅ Prediction complete: %d faces, mesh → %s",
            len(predictions),
            mesh_url,
        )

        return PredictionResponse(
            status="success",
            filename=original_filename,
            mesh_url=mesh_url,
            predictions=[FacePrediction(**p) for p in predictions],
            num_faces=len(predictions),
        )

    except HTTPException:
        # Re-raise FastAPI HTTP exceptions as-is
        raise

    except Exception as exc:
        logger.exception("Unexpected error during prediction pipeline")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {exc}",
        )

    finally:
        # ── 6. Cleanup raw upload & intermediate graph ────────────────────
        # The STL in /static is intentionally kept so the frontend can fetch it.
        for cleanup_path in (upload_path, graph_path):
            try:
                if cleanup_path.exists():
                    cleanup_path.unlink()
                    logger.debug("🗑️  Cleaned up %s", cleanup_path)
            except OSError as e:
                logger.warning("Cleanup failed for %s: %s", cleanup_path, e)


# ---------------------------------------------------------------------------
# Global exception handler — catch-all for unhandled errors
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            status="error",
            detail=f"An unexpected error occurred: {exc}",
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info",
    )
