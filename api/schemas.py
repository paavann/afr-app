"""
Pydantic response schemas for the prediction API.
"""

from pydantic import BaseModel, Field


class FacePrediction(BaseModel):
    """A single face's predicted feature type."""

    face_id: int = Field(..., description="Zero-indexed B-rep face identifier")
    type: str = Field(..., description="Human-readable feature type name")


class PredictionResponse(BaseModel):
    """Successful prediction result returned by POST /api/predict-cad."""

    status: str = Field(default="success", description="Request outcome")
    filename: str = Field(..., description="Original uploaded filename")
    mesh_url: str = Field(
        ..., description="URL to download the generated .stl render mesh"
    )
    predictions: list[FacePrediction] = Field(
        ..., description="Per-face feature predictions"
    )
    num_faces: int = Field(..., description="Total number of B-rep faces detected")


class ErrorResponse(BaseModel):
    """Error payload returned on processing failures."""

    status: str = Field(default="error")
    detail: str = Field(..., description="Human-readable error description")
    filename: str | None = Field(
        default=None, description="Original filename, if available"
    )
