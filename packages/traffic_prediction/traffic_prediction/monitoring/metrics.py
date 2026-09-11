from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram


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

LATEST_OBSERVATION_TIMESTAMP = Gauge(
    "traffic_latest_observation_timestamp_seconds",
    "Unix timestamp of the latest traffic observation.",
)

LATEST_PREDICTION_TIMESTAMP = Gauge(
    "traffic_latest_prediction_timestamp_seconds",
    "Unix timestamp of the latest traffic prediction.",
    ["horizon_hours"],
)

PREDICTED_ROADS = Gauge(
    "traffic_predicted_roads",
    "Number of roads with predictions.",
    ["horizon_hours"],
)

CHAMPION_MODEL_VERSION = Gauge(
    "traffic_champion_model_version",
    "Active MLflow champion model version.",
    ["horizon_hours"],
)