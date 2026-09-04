from datetime import datetime

from airflow.sdk import dag, task

from traffic_prediction.inference.predictor import run_predictions
from traffic_prediction.pipelines.traffic_ingestion import (
    run_incremental_ingestion,
)


@dag(
    dag_id="traffic_prediction_pipeline",
    description="Ingestion and prediction pipeline for Paris traffic",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["traffic", "prediction", "paris"],
)
def traffic_prediction_pipeline():

    @task
    def ingest_traffic() -> int:
        return run_incremental_ingestion()

    @task
    def predict_traffic() -> int:
        predictions = run_predictions()
        return len(predictions)

    ingestion_result = ingest_traffic()
    prediction_result = predict_traffic()

    ingestion_result >> prediction_result


traffic_prediction_pipeline()