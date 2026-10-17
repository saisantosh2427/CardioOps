"""Request / response models for the prediction API (pydantic v2)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EXAMPLE_PATIENT = {
    "age": 63,
    "sex": 1,
    "cp": 4,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 2,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 3,
    "ca": 0,
    "thal": 6,
}


class PatientFeatures(BaseModel):
    """Clinical measurements for one patient (UCI Heart Disease coding).

    ``age``, ``sex`` and ``cp`` are required; other measurements may be null
    and are imputed by the model pipeline exactly as during training.
    """

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_PATIENT]})

    age: int = Field(..., ge=18, le=100, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(
        ..., description="Chest pain: 1 typical, 2 atypical, 3 non-anginal, 4 asymptomatic"
    )
    trestbps: float | None = Field(None, ge=60, le=250, description="Resting BP (mm Hg)")
    chol: float | None = Field(None, ge=80, le=700, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] | None = Field(None, description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] | None = Field(None, description="Resting ECG result")
    thalach: float | None = Field(None, ge=50, le=250, description="Max heart rate achieved")
    exang: Literal[0, 1] | None = Field(None, description="Exercise induced angina")
    oldpeak: float | None = Field(None, ge=-5, le=10, description="ST depression (exercise)")
    slope: Literal[1, 2, 3] | None = Field(None, description="Slope of peak exercise ST")
    ca: Literal[0, 1, 2, 3] | None = Field(None, description="Major vessels by fluoroscopy")
    thal: Literal[3, 6, 7] | None = Field(None, description="3 normal, 6 fixed, 7 reversible")


class PredictionResponse(BaseModel):
    request_id: str = Field(..., description="Use this id to send ground truth to /feedback")
    prediction: Literal[0, 1]
    probability: float = Field(..., ge=0, le=1, description="P(heart disease)")
    risk_label: Literal["low", "high"]
    model_name: str
    model_version: str


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instances: list[PatientFeatures] = Field(..., min_length=1)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    model_name: str
    model_version: str


class FeedbackItem(BaseModel):
    request_id: str
    actual: Literal[0, 1] = Field(..., description="Confirmed diagnosis (1 = heart disease)")


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[FeedbackItem] = Field(..., min_length=1)


class FeedbackResponse(BaseModel):
    updated: int
    unknown_request_ids: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    model_uri: str
    source: Literal["registry", "local"]
    alias: str | None
    run_id: str | None
    model_type: str | None
    loaded_at: datetime
    decision_threshold: float
    features: list[str]
    tags: dict[str, str]
