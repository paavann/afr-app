# AFR Backend

This directory contains the production-ready FastAPI backend for the Automated Feature Recognition application. It acts as the inference gateway for the UV-Net CAD face segmentation model.

## How It Works

The backend processes 3D CAD files through a specialized pipeline:
1. **Upload & Storage**: Receives `.step` or `.stp` files via a REST endpoint.
2. **CAD Parsing**: Uses `pythonocc-core` (OpenCASCADE) to parse the boundary representation (B-Rep) of the CAD file.
3. **Graph Construction**: Builds a face-adjacency graph representing the CAD geometry.
4. **Mesh Generation**: Generates an `.stl` mesh for frontend visualization.
5. **Inference**: Feeds the graph into a neural network (UV-Net) to classify each face into specific machining features (e.g., chamfer, pocket, slot).
6. **Response**: Returns a JSON object containing per-face predictions and a URL to the generated 3D mesh.

## Setup Instructions

### Why Conda?

This backend relies heavily on `pythonocc-core`, which is a Python wrapper for the OpenCASCADE C++ library. Because it requires complex system-level C++ binaries and libraries to handle CAD kernels, installing it via standard `pip` is extremely difficult and prone to errors. **Conda** manages these heavy C++ dependencies seamlessly, ensuring that the environment is isolated, reproducible, and has all necessary system libraries pre-compiled.

### 1. Create the Conda Environment

You will need [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda installed on your system.

```bash
# Create a dedicated environment with Python 3.10
conda create -n afr_api python=3.10 -y

# Activate the environment
conda activate afr_api
```

### 2. Install Dependencies

First, install the CAD kernel dependencies via `conda-forge`, and then install the rest of the Python packages via `pip`.

```bash
# Install pythonocc-core for CAD processing
conda install -c conda-forge pythonocc-core -y

# Install FastAPI, Uvicorn, and Machine Learning dependencies
pip install -r requirements.txt
```

### 3. (Optional) Add Model Checkpoint

To enable actual AI predictions, place your trained PyTorch checkpoint (e.g., `best.ckpt`) inside the `checkpoints/` directory. If a checkpoint is not found, the API will automatically fall back to generating mock predictions for testing purposes.

### 4. Run the Server

Start the development server using Uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`. You can explore the interactive API documentation at `http://localhost:8000/docs`.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Liveness probe |
| `POST` | `/api/afr` | Upload STEP file → get predictions + mesh URL |
| `GET` | `/static/{filename}` | Download generated STL meshes |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/redoc` | ReDoc API documentation |
