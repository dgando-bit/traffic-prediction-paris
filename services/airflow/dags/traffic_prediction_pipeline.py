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
from traffic_prediction.pipelines.promotion import (
    promote_candidate,
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
    max_active_runs=1,
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
        result = train_and_register_model(
            dataset_path=features_path,
        )

        return result

    @task
    def quality_gate(
        features_path: str,
        training_result: dict[str, Any],
    ) -> dict[str, Any]:
        print(
            "Evaluating newly trained candidate..."
        )

        print(
            "Candidate version:",
            training_result["model_version"],
        )

        result = promote_candidate(
            dataset_path=features_path,
            metric="mae",
            min_improvement_pct=1.0,
        )

        return result

    @task
    def notify(
        training_result: dict[str, Any],
        promotion_result: dict[str, Any],
    ) -> None:
        print()
        print(
            "================================"
        )
        print(
            " Traffic training completed"
        )
        print(
            "================================"
        )

        print(
            "Model:",
            training_result[
                "registered_model_name"
            ],
        )

        print(
            "Candidate version:",
            promotion_result[
                "candidate_version"
            ],
        )

        print(
            "Champion version:",
            promotion_result[
                "champion_version"
            ],
        )

        print(
            "Training MAE:",
            f"{training_result['mae']:.4f}",
        )

        print(
            "Training RMSE:",
            f"{training_result['rmse']:.4f}",
        )

        print(
            "Quality gate:",
            (
                "PROMOTED"
                if promotion_result["promoted"]
                else "REJECTED"
            ),
        )

        if (
            promotion_result.get(
                "improvement_pct"
            )
            is not None
        ):
            print(
                "Improvement:",
                f"{promotion_result['improvement_pct']:+.2f}%",
            )

        print(
            "Reason:",
            promotion_result["reason"],
        )

        print(
            "Run ID:",
            training_result["run_id"],
        )

        print(
            "================================"
        )

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

    promotion_result = quality_gate(
        features_path,
        training_result,
    )

    notify(
        training_result,
        promotion_result,
    )


traffic_training_pipeline()