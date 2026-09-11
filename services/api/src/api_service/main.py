import time

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import text
from traffic_prediction.features.schema import (
    PREDICTION_HORIZONS,
)
from traffic_prediction.monitoring.metrics import (
    HTTP_REQUEST_DURATION_SECONDS,
    HTTP_REQUESTS_TOTAL,
    refresh_business_metrics,
)
from traffic_prediction.storage.database import (
    get_db_session,
)
from traffic_prediction.storage.repositories import (
    get_latest_prediction_for_road,
    get_latest_predictions,
    get_road_segments,
    get_traffic_history_for_road,
)

from api_service.schemas import (
    HealthResponse,
    PredictionResponse,
    RoadHistoryResponse,
    RoadResponse,
    TrafficObservationResponse,
)

app = FastAPI(
    title="Paris Traffic Prediction API",
    version="0.3.0",
    description=(
        "API exposing multi-horizon traffic "
        "predictions for Paris road segments."
    ),
)

@app.middleware("http")
async def prometheus_middleware(
    request: Request,
    call_next,
) -> Response:
    start_time = time.perf_counter()

    response = await call_next(request)

    duration = time.perf_counter() - start_time

    route = request.scope.get("route")

    path = (
        getattr(route, "path", request.url.path)
        if route is not None
        else request.url.path
    )

    HTTP_REQUESTS_TOTAL.labels(
        method=request.method,
        path=path,
        status_code=str(response.status_code),
    ).inc()

    HTTP_REQUEST_DURATION_SECONDS.labels(
        method=request.method,
        path=path,
    ).observe(duration)

    return response

def validate_horizon(
    horizon_hours: int,
) -> int:
    """
    Validate that a prediction horizon is supported.
    """
    if horizon_hours not in PREDICTION_HORIZONS:
        raise HTTPException(
            status_code=422,
            detail=(
                "Unsupported prediction horizon. "
                f"Allowed values: "
                f"{PREDICTION_HORIZONS}"
            ),
        )

    return horizon_hours


def prediction_to_response(
    prediction,
) -> PredictionResponse:
    """
    Convert a Prediction ORM object into
    the API response schema.
    """
    return PredictionResponse(
        iu_ac=prediction.iu_ac,
        prediction_timestamp_utc=(
            prediction.prediction_timestamp_utc
        ),
        target_timestamp_utc=(
            prediction.target_timestamp_utc
        ),
        horizon_hours=(
            prediction.horizon_hours
        ),
        predicted_k=prediction.predicted_k,
        model_version=prediction.model_version,
    )

@app.get(
    "/metrics",
    include_in_schema=False,
)
def metrics() -> Response:
    with get_db_session() as session:
        refresh_business_metrics(session)

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
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
def latest_predictions(
    horizon_hours: int = Query(
        default=1,
        description=(
            "Prediction horizon in hours."
        ),
    ),
) -> list[PredictionResponse]:
    """
    Return the latest prediction batch for
    the requested prediction horizon.
    """
    horizon_hours = validate_horizon(
        horizon_hours
    )

    with get_db_session() as session:
        predictions = get_latest_predictions(
            session,
            horizon_hours=horizon_hours,
        )

        return [
            prediction_to_response(
                prediction
            )
            for prediction in predictions
        ]


@app.get(
    "/roads/{iu_ac}/history",
    response_model=RoadHistoryResponse,
    tags=["roads"],
)
def road_history(
    iu_ac: str,
    hours: int = Query(
        default=24,
        ge=1,
        le=168,
    ),
    horizon_hours: int = Query(
        default=1,
        description=(
            "Prediction horizon in hours."
        ),
    ),
) -> RoadHistoryResponse:
    """
    Return recent traffic observations for one
    road and its latest prediction for the
    requested horizon.
    """
    horizon_hours = validate_horizon(
        horizon_hours
    )

    with get_db_session() as session:
        observations = (
            get_traffic_history_for_road(
                session,
                iu_ac=iu_ac,
                hours=hours,
            )
        )

        if not observations:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No traffic history found "
                    f"for road {iu_ac}"
                ),
            )

        prediction = (
            get_latest_prediction_for_road(
                session,
                iu_ac,
                horizon_hours=horizon_hours,
            )
        )

        prediction_response = (
            prediction_to_response(
                prediction
            )
            if prediction is not None
            else None
        )

        return RoadHistoryResponse(
            iu_ac=iu_ac,
            observations=[
                TrafficObservationResponse(
                    timestamp_utc=(
                        observation.timestamp_utc
                    ),
                    q=observation.q,
                    k=observation.k,
                )
                for observation in observations
            ],
            prediction=prediction_response,
        )


@app.get(
    "/predictions/{iu_ac}",
    response_model=PredictionResponse,
    tags=["predictions"],
)
def latest_prediction(
    iu_ac: str,
    horizon_hours: int = Query(
        default=1,
        description=(
            "Prediction horizon in hours."
        ),
    ),
) -> PredictionResponse:
    """
    Return the latest prediction for one road
    and one prediction horizon.
    """
    horizon_hours = validate_horizon(
        horizon_hours
    )

    with get_db_session() as session:
        prediction = (
            get_latest_prediction_for_road(
                session,
                iu_ac,
                horizon_hours=horizon_hours,
            )
        )

        if prediction is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No +{horizon_hours}h "
                    f"prediction found "
                    f"for road {iu_ac}"
                ),
            )

        return prediction_to_response(
            prediction
        )


def run() -> None:
    uvicorn.run(
        "api_service.main:app",
        host="0.0.0.0",
        port=8000,
    )


if __name__ == "__main__":
    run()