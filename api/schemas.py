from pydantic import BaseModel, Field



class FacePrediction(BaseModel):
    face_id: int = Field(..., description="Zero-indexed B-rep face identifier")
    type: str = Field(..., description="Human-readable feature type name")


class PredictionResponse(BaseModel):
    status: str = Field(default="success", description="Request outcome")
    filename: str = Field(..., description="Original uploaded filename")
    mesh_url: str = Field(
        ..., description="URL to download the generated .stl render mesh"
    )
    predictions: list[FacePrediction] = Field(
        ..., description="Per-face feature predictions"
    )
    num_faces: int = Field(..., description="Total number of B-rep faces detected")
    face_mapping: list[int] | None = Field(
        default=None, description="Triangle-to-face mapping: face_mapping[tri_idx] = face_id"
    )
    model_type: str = Field(
        default="fusiongallery",
        description="Model type used for inference: 'fusiongallery' or 'mfcad'",
    )


class ErrorResponse(BaseModel):
    status: str = Field(default="error")
    detail: str = Field(..., description="Human-readable error description")
    filename: str | None = Field(
        default=None, description="Original filename, if available"
    )
