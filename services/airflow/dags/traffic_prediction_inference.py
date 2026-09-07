from __future__ import annotations

from datetime import datetime

from airflow.sdk import dag, task

from traffic_prediction.inference.predictor import run_predictions
from traffic_prediction.pipelines.traffic_ingestion import (
    run_incremental_ingestion,
)


@dag(
    dag_id="traffic_prediction_inference",
    description="Hourly inference pipeline for Paris traffic prediction",
    schedule="0 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=[
        "traffic",
        "inference",
        "mlflow",
        "paris",
    ],
)
def traffic_prediction_inference():

    @task
    def ingestion() -> int:
        inserted_rows = run_incremental_ingestion()

        print(
            f"Ingestion completed: "
            f"{inserted_rows} observations processed"
        )

        return inserted_rows

    @task
    def predict(ingestion_result: int) -> int:
        print(
            f"Ingestion result received: "
            f"{ingestion_result}"
        )

        prediction_count = run_predictions()

        print(
            f"Prediction completed: "
            f"{prediction_count} predictions stored"
        )

        return prediction_count

    @task
    def notify(prediction_count: int) -> None:
        print()
        print("================================")
        print(" Traffic prediction completed")
        print("================================")
        print(
            f"Predictions stored: "
            f"{prediction_count}"
        )
        print("================================")

    ingestion_result = ingestion()
    prediction_count = predict(
        ingestion_result
    )
    notify(prediction_count)


traffic_prediction_inference()