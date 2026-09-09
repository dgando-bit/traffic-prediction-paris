import uvicorn
from fastapi import FastAPI, HTTPException
from sqlalchemy import text

from api_service.schemas import (
    HealthResponse,
    PredictionResponse,
    RoadResponse,
)
from traffic_prediction.storage.database import (
    get_db_session,
)
from traffic_prediction.storage.repositories import (
    get_latest_prediction_for_road,
    get_latest_predictions,
    get_road_segments,
)


app = FastAPI(
    title="Paris Traffic Prediction API",
    version="0.2.0",
    description=(
        "API exposing one-hour-ahead traffic "
        "predictions for Paris road segments."
    ),
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
)
def health() -> HealthResponse:
    try:
        with get_db_session() as session:
            session.execute(
                text("SELECT 1")
            )

        return HealthResponse(
            status="ok"
        )

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable",
        ) from exc


@app.get(
    "/roads",
    response_model=list[RoadResponse],
    tags=["roads"],
)
def roads() -> list[RoadResponse]:
    """
    Return monitored road segments and
    their geographic metadata.
    """
    with get_db_session() as session:
        road_segments = get_road_segments(
            session
        )

        return [
            RoadResponse(
                iu_ac=road.iu_ac,
                libelle=road.libelle,
                latitude=road.latitude,
                longitude=road.longitude,
                road_length_m=road.road_length_m,
                geo_shape=road.geo_shape,
            )
            for road in road_segments
        ]

@app.get(
    "/predictions/latest",
    response_model=list[PredictionResponse],
    tags=["predictions"],
)
def latest_predictions() -> list[PredictionResponse]:
    """
    Return the most recent prediction available
    for every road segment.
    """
    with get_db_session() as session:
        predictions = get_latest_predictions(
            session
        )

        return [
            PredictionResponse(
                iu_ac=row.iu_ac,
                prediction_timestamp_utc=(
                    row.prediction_timestamp_utc
                ),
                target_timestamp_utc=(
                    row.target_timestamp_utc
                ),
                predicted_k=row.predicted_k,
                model_version=row.model_version,
            )
            for row in predictions
        ]


@app.get(
    "/predictions/{iu_ac}",
    response_model=PredictionResponse,
    tags=["predictions"],
)
def latest_prediction(
    iu_ac: str,
) -> PredictionResponse:
    """
    Return the latest prediction for one road.
    """
    with get_db_session() as session:
        prediction = (
            get_latest_prediction_for_road(
                session,
                iu_ac,
            )
        )

        if prediction is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No prediction found "
                    f"for road {iu_ac}"
                ),
            )

        return PredictionResponse(
            iu_ac=prediction.iu_ac,
            prediction_timestamp_utc=(
                prediction.prediction_timestamp_utc
            ),
            target_timestamp_utc=(
                prediction.target_timestamp_utc
            ),
            predicted_k=prediction.predicted_k,
            model_version=prediction.model_version,
        )


def run() -> None:
    uvicorn.run(
        "api_service.main:app",
        host="0.0.0.0",
        port=8000,
    )


if __name__ == "__main__":
    run()