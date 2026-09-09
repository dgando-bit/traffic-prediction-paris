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
    libelle: str | None
    latitude: float
    longitude: float
    road_length_m: float | None
    geo_shape: str | None

class TrafficObservationResponse(BaseModel):
    timestamp_utc: datetime
    q: float | None
    k: float | None


class RoadHistoryResponse(BaseModel):
    iu_ac: str
    observations: list[
        TrafficObservationResponse
    ]
    prediction: PredictionResponse | None