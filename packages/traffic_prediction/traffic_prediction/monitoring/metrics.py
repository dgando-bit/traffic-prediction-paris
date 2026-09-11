from __future__ import annotations

from datetime import datetime, timezone

from prometheus_client import Counter, Gauge, Histogram
from sqlalchemy.orm import Session

from traffic_prediction.storage.repositories import (
    get_all_road_ids,
    get_latest_predictions,
    get_latest_traffic_timestamp,
)


# ---------------------------------------------------------------------------
# HTTP API metrics
# ---------------------------------------------------------------------------

HTTP_REQUESTS_TOTAL = Counter(
    "traffic_api_http_requests_total",
    "Total number of HTTP requests received by the API.",
    ["method", "path", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "traffic_api_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ["method", "path"],
)


# ---------------------------------------------------------------------------
# Inference metrics
# ---------------------------------------------------------------------------

INFERENCE_RUNS_TOTAL = Counter(
    "traffic_inference_runs_total",
    "Total number of inference runs.",
    ["status"],
)

INFERENCE_DURATION_SECONDS = Histogram(
    "traffic_inference_duration_seconds",
    "Inference pipeline duration in seconds.",
)

PREDICTIONS_GENERATED_TOTAL = Counter(
    "traffic_predictions_generated_total",
    "Total number of generated traffic predictions.",
    ["horizon_hours"],
)


# ---------------------------------------------------------------------------
# Data freshness metrics
# ---------------------------------------------------------------------------

LATEST_OBSERVATION_TIMESTAMP = Gauge(
    "traffic_latest_observation_timestamp_seconds",
    "Unix timestamp of the latest traffic observation.",
)

OBSERVATION_AGE_SECONDS = Gauge(
    "traffic_observation_age_seconds",
    "Age in seconds of the latest traffic observation.",
)

LATEST_PREDICTION_TIMESTAMP = Gauge(
    "traffic_latest_prediction_timestamp_seconds",
    "Unix timestamp of the latest traffic prediction.",
    ["horizon_hours"],
)

PREDICTION_AGE_SECONDS = Gauge(
    "traffic_prediction_age_seconds",
    "Age in seconds of the latest traffic prediction.",
    ["horizon_hours"],
)


# ---------------------------------------------------------------------------
# Prediction coverage metrics
# ---------------------------------------------------------------------------

PREDICTED_ROADS = Gauge(
    "traffic_predicted_roads",
    "Number of roads with a latest prediction.",
    ["horizon_hours"],
)

TOTAL_ROADS = Gauge(
    "traffic_total_roads",
    "Total number of monitored road segments.",
)

PREDICTION_COVERAGE_RATIO = Gauge(
    "traffic_prediction_coverage_ratio",
    "Ratio of monitored roads having a latest prediction.",
    ["horizon_hours"],
)


# ---------------------------------------------------------------------------
# Model metrics
# ---------------------------------------------------------------------------

CHAMPION_MODEL_VERSION = Gauge(
    "traffic_champion_model_version",
    "Active model version stored in latest predictions.",
    ["horizon_hours"],
)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


def refresh_business_metrics(
    session: Session,
    horizons: tuple[int, ...] = (1, 2, 3),
) -> None:
    """
    Refresh Prometheus business metrics from PostgreSQL.

    This function is intended to be called when Prometheus scrapes
    the API /metrics endpoint.
    """
    now = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Traffic observations
    # ------------------------------------------------------------------

    latest_observation = get_latest_traffic_timestamp(session)

    if latest_observation is not None:
        latest_observation = _ensure_utc(latest_observation)

        LATEST_OBSERVATION_TIMESTAMP.set(
            latest_observation.timestamp()
        )

        observation_age = max(
            0.0,
            (now - latest_observation).total_seconds(),
        )

        OBSERVATION_AGE_SECONDS.set(observation_age)

    # ------------------------------------------------------------------
    # Roads
    # ------------------------------------------------------------------

    road_ids = get_all_road_ids(session)
    total_roads = len(road_ids)

    TOTAL_ROADS.set(total_roads)

    # ------------------------------------------------------------------
    # Predictions by horizon
    # ------------------------------------------------------------------

    for horizon_hours in horizons:
        predictions = get_latest_predictions(
            session,
            horizon_hours=horizon_hours,
        )

        predicted_roads = len(predictions)

        PREDICTED_ROADS.labels(
            horizon_hours=str(horizon_hours),
        ).set(predicted_roads)

        coverage_ratio = (
            predicted_roads / total_roads
            if total_roads > 0
            else 0.0
        )

        PREDICTION_COVERAGE_RATIO.labels(
            horizon_hours=str(horizon_hours),
        ).set(coverage_ratio)

        if not predictions:
            continue

        latest_prediction = max(
            predictions,
            key=lambda prediction: (
                prediction.prediction_timestamp_utc
            ),
        )

        prediction_timestamp = _ensure_utc(
            latest_prediction.prediction_timestamp_utc
        )

        LATEST_PREDICTION_TIMESTAMP.labels(
            horizon_hours=str(horizon_hours),
        ).set(
            prediction_timestamp.timestamp()
        )

        prediction_age = max(
            0.0,
            (now - prediction_timestamp).total_seconds(),
        )

        PREDICTION_AGE_SECONDS.labels(
            horizon_hours=str(horizon_hours),
        ).set(prediction_age)

        try:
            model_version = float(
                latest_prediction.model_version
            )
        except (TypeError, ValueError):
            continue

        CHAMPION_MODEL_VERSION.labels(
            horizon_hours=str(horizon_hours),
        ).set(model_version)