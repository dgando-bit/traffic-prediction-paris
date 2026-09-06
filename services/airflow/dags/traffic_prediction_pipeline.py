from __future__ import annotations

from datetime import datetime
from typing import Any

from airflow.sdk import dag, task

from traffic_prediction.pipelines.dataset import (
    make_training_dataset,
)
from traffic_prediction.pipelines.features import (
    build_training_features,
)
from traffic_prediction.pipelines.traffic_ingestion import (
    run_incremental_ingestion,
)
from traffic_prediction.pipelines.training import (
    train_and_register_model,
)


@dag(
    dag_id="traffic_training_pipeline",
    description="Training pipeline for Paris traffic prediction",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=[
        "traffic",
        "training",
        "mlflow",
        "paris",
    ],
)
def traffic_training_pipeline():

    @task
    def ingestion() -> int:
        """
        Fetch new traffic observations and persist them
        into PostgreSQL.
        """
        inserted_rows = run_incremental_ingestion()

        print(
            f"Ingestion completed: "
            f"{inserted_rows} observations processed"
        )

        return inserted_rows

    @task
    def make_dataset(
        ingestion_result: int,
    ) -> str:
        """
        Extract the training history from PostgreSQL and
        create the interim training dataset.

        ingestion_result is intentionally received so that
        Airflow creates an explicit dependency on ingestion.
        """
        print(
            f"Ingestion result received: "
            f"{ingestion_result}"
        )

        path = make_training_dataset()

        print(
            f"Training dataset created: {path}"
        )

        return str(path)

    @task
    def build_features(
        dataset_path: str,
    ) -> str:
        """
        Build the 14 features used by the final model.
        """
        path = build_training_features(
            input_path=dataset_path,
        )

        print(
            f"Training features created: {path}"
        )

        return str(path)

    @task
    def train_model(
        features_path: str,
    ) -> dict[str, Any]:
        """
        Train LightGBM, log the run to MLflow and register
        the new model version as @candidate.
        """
        result = train_and_register_model(
            dataset_path=features_path,
        )

        return result

    @task
    def notify(
        training_result: dict[str, Any],
    ) -> None:
        """
        Initial notification task.

        For now the training summary is written to the Airflow
        logs. A real notification channel can be added later.
        """
        print()
        print("================================")
        print(" Traffic training completed")
        print("================================")

        print(
            "Model:",
            training_result[
                "registered_model_name"
            ],
        )

        print(
            "Version:",
            training_result[
                "model_version"
            ],
        )

        print(
            "Alias:",
            training_result["alias"],
        )

        print(
            "Run ID:",
            training_result["run_id"],
        )

        print(
            "MAE:",
            f"{training_result['mae']:.4f}",
        )

        print(
            "RMSE:",
            f"{training_result['rmse']:.4f}",
        )

        print("================================")

    ingestion_result = ingestion()

    dataset_path = make_dataset(
        ingestion_result
    )

    features_path = build_features(
        dataset_path
    )

    training_result = train_model(
        features_path
    )

    notify(
        training_result
    )


traffic_training_pipeline()