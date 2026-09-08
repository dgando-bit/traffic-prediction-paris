from datetime import datetime

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class PredictionResponse(BaseModel):
    iu_ac: str
    prediction_timestamp_utc: datetime
    target_timestamp_utc: datetime
    predicted_k: float
    model_version: str


class RoadResponse(BaseModel):
    iu_ac: str