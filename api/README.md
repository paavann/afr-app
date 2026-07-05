# CAD Feature Recognition — Inference API

Production-ready FastAPI application that serves as the inference gateway for the UV-Net CAD face segmentation model.

## Architecture

```
api/
├── main.py           # FastAPI app — routes, CORS, static file mount
├── config.py         # Centralized configuration (paths, CORS, class maps)
├── schemas.py        # Pydantic response models
├── pipeline.py       # CAD processing: STEP → DGL graph + STL mesh
├── inference.py      # Model loading + inference (with stub fallback)
├── checkpoints/      # Place best.ckpt here
│   └── README.md
└── requirements.txt  # Python dependencies
```

## Quick Start

### 1. Install dependencies

```bash
# Create conda env with OCC + DGL
conda create -n afr_api python=3.10
conda activate afr_api
conda install -c conda-forge pythonocc-core
pip install -r requirements.txt
```

### 2. (Optional) Add model checkpoint

Place your trained `best.ckpt` in `api/checkpoints/`. Without it, the API runs in **stub mode** with mock predictions.

### 3. Run the server

```bash
cd afr-app/api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Test it

```bash
curl -X POST http://localhost:8000/api/predict-cad \
  -F "file=@/path/to/your/model.step"
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Liveness probe |
| `POST` | `/api/predict-cad` | Upload STEP file → get predictions + mesh URL |
| `GET` | `/static/{filename}` | Download generated STL meshes |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/redoc` | ReDoc API documentation |

## Response Format

```json
{
    "status": "success",
    "filename": "bracket.step",
    "mesh_url": "http://localhost:8000/static/bracket_a1b2c3d4e5f6.stl",
    "predictions": [
        {"face_id": 0, "type": "rectangular_through_slot"},
        {"face_id": 1, "type": "chamfer"},
        {"face_id": 2, "type": "stock"}
    ],
    "num_faces": 3
}
```

## Class Labels (MFCAD Dataset)

| Index | Feature Type |
|-------|-------------|
| 0 | rectangular_through_slot |
| 1 | triangular_through_slot |
| 2 | rectangular_passage |
| 3 | triangular_passage |
| 4 | 6sides_passage |
| 5 | rectangular_through_step |
| 6 | 2sides_through_step |
| 7 | slanted_through_step |
| 8 | rectangular_blind_step |
| 9 | triangular_blind_step |
| 10 | rectangular_blind_slot |
| 11 | rectangular_pocket |
| 12 | triangular_pocket |
| 13 | 6sides_pocket |
| 14 | chamfer |
| 15 | stock |
