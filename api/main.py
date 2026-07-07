from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from config import (
    API_HOST, API_PORT,
    CORS_ORIGINS,
    GRAPH_DIR, MESH_DIR, STATIC_DIR, UPLOAD_DIR,
    MODEL_REGISTRY, DEFAULT_MODEL,
)
from inference import map_predictions, run_inference
from pipeline import build_face_adjacency_graph, build_render_mesh, save_graph
from schemas import ErrorResponse, FacePrediction, PredictionResponse




logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("cad_api")


app = FastAPI(
    title="Automated Feature Recognition - CAD",
    description=(
        "Inference gateway for the UV-Net face segmentation model. "
        "Upload a .step CAD file and receive per-face feature predictions "
        "plus a downloadable .stl mesh for 3D visualization."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")




@app.get("/api/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "service": "cad-feature-recognition-api"
    }


@app.get("/api/models", tags=["System"])
async def list_models():
    """Return available model types and their display names."""
    return {
        "default": DEFAULT_MODEL,
        "models": {
            key: {
                "display_name": entry["display_name"],
                "num_classes": len(entry["class_map"]),
                "class_map": entry["class_map"],
            }
            for key, entry in MODEL_REGISTRY.items()
        },
    }


ALLOWED_EXTENSIONS = { ".step", ".stp", ".STEP", ".STP" }
@app.post(
    "/api/afr",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input file"},
        422: {"model": ErrorResponse, "description": "CAD kernel processing failure"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    tags=["Inference"],
    summary="Run CAD face segmentation on an uploaded STEP file",
)
async def predict_cad(
    request: Request,
    file: UploadFile = File(...),
    model_type: str = Form(DEFAULT_MODEL, description="Model to use: 'fusiongallery' or 'mfcad'"),
):
    # Validate model_type
    if model_type not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown model_type '{model_type}'. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            ),
        )

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

    job_id = uuid.uuid4().hex[:12]
    original_filename = file.filename
    stem = Path(original_filename).stem

    upload_path = UPLOAD_DIR / f"{stem}_{job_id}{file_ext}"
    graph_path = GRAPH_DIR / f"{stem}_{job_id}.bin"
    stl_filename = f"{stem}_{job_id}.stl"
    stl_path = STATIC_DIR / stl_filename

    try:
        logger.info("Received '%s' (job=%s, model=%s)", original_filename, job_id, model_type)
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        upload_path.write_bytes(contents)
        logger.info("Saved upload -> %s (%d bytes)", upload_path, len(contents))

        logger.info("Building face-adjacency graph...")
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

        logger.info("Building render mesh...")
        try:
            stl_path_out, tri_mapping = build_render_mesh(upload_path, stl_path)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error("Mesh generation failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=422,
                detail=f"Mesh triangulation failed: {exc}",
            )

        logger.info("Running UV-Net segmentation inference (model=%s)...", model_type)
        class_indices = run_inference(dgl_graph, model_type=model_type)

        predictions = map_predictions(class_indices, model_type=model_type)

        # --- Print human-readable results to console ---
        print("=" * 60)
        print(f"  RESULTS FOR: {original_filename} (model={model_type})")
        print(f"  TOTAL FACES: {len(predictions)}")
        print("-" * 60)
        print("  Face ID  |  Feature Type")
        print("-" * 60)
        for p in predictions:
            print(f"    {p['face_id']:4d}   |  {p['type']}")
        print("=" * 60)

        base_url = str(request.base_url).rstrip("/")
        mesh_url = f"{base_url}/static/{stl_filename}"

        logger.info(
            "Prediction complete: %d faces, mesh -> %s",
            len(predictions),
            mesh_url,
        )

        return PredictionResponse(
            status="success",
            filename=original_filename,
            mesh_url=mesh_url,
            predictions=[FacePrediction(**p) for p in predictions],
            num_faces=len(predictions),
            face_mapping=tri_mapping.tolist() if tri_mapping is not None else None,
            model_type=model_type,
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Unexpected error during prediction pipeline")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {exc}",
        )

    finally:
        for cleanup_path in (upload_path, graph_path):
            try:
                if cleanup_path.exists():
                    cleanup_path.unlink()
                    logger.debug("Cleaned up %s", cleanup_path)
            except OSError as e:
                logger.warning("Cleanup failed for %s: %s", cleanup_path, e)



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



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info",
    )
