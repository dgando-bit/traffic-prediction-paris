from datetime import datetime

from airflow.sdk import dag, task

from traffic_prediction.pipelines.traffic_ingestion import (
    run_incremental_ingestion,
)
from traffic_prediction.pipelines.dataset import (
    make_training_dataset,
)
from traffic_prediction.pipelines.features import (
    build_training_features,
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
    tags=["traffic", "training", "mlflow", "paris"],
)
def traffic_training_pipeline():

    @task
    def ingestion() -> int:
        return run_incremental_ingestion()

    @task
    def make_dataset() -> str:
        path = make_training_dataset()
        return str(path)

    @task
    def build_features(dataset_path: str) -> str:
        path = build_training_features(
            input_path=dataset_path,
        )
        return str(path)

    @task
    def train_model(features_path: str) -> dict:
        return train_and_register_model(
            dataset_path=features_path,
        )

    @task
    def notify(training_result: dict) -> None:
        print("Training pipeline completed")
        print(
            f"Model: "
            f"{training_result['registered_model_name']}"
        )
        print(
            f"Version: "
            f"{training_result['model_version']}"
        )
        print(
            f"Alias: "
            f"{training_result['alias']}"
        )
        print(
            f"MAE: "
            f"{training_result['mae']:.4f}"
        )
        print(
            f"RMSE: "
            f"{training_result['rmse']:.4f}"
        )

    ingestion_result = ingestion()

    dataset_path = make_dataset()

    features_path = build_features(
        dataset_path
    )

    training_result = train_model(
        features_path
    )

    notification = notify(
        training_result
    )

    ingestion_result >> dataset_path
    notification


traffic_training_pipeline()