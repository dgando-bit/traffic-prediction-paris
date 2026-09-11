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
    promote_all_candidates,
)
from traffic_prediction.pipelines.traffic_ingestion import (
    run_incremental_ingestion,
)
from traffic_prediction.pipelines.training import (
    train_and_register_all_horizons,
)


TRAINING_HISTORY_HOURS = 24 * 90


@dag(
    dag_id="traffic_training_pipeline",
    description=(
        "Multi-horizon training pipeline "
        "for Paris traffic prediction"
    ),
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=[
        "traffic",
        "training",
        "mlflow",
        "multi-horizon",
        "paris",
    ],
)
def traffic_training_pipeline():

    @task
    def ingestion() -> int:
        inserted_rows = (
            run_incremental_ingestion()
        )

        print(
            "Ingestion completed: "
            f"{inserted_rows} observations processed"
        )

        return inserted_rows

    @task
    def make_dataset(
        ingestion_result: int,
    ) -> str:
        print(
            "Ingestion result received: "
            f"{ingestion_result}"
        )

        print(
            "Building training dataset "
            f"from the last "
            f"{TRAINING_HISTORY_HOURS // 24} days"
        )

        path = make_training_dataset(
            history_hours=(
                TRAINING_HISTORY_HOURS
            ),
        )

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
            "Multi-horizon training "
            f"features created: {path}"
        )

        return str(path)

    @task(multiple_outputs=False)
    def train_models(
        features_path: str,
    ) -> dict[int, dict[str, Any]]:
        print()
        print(
            "================================"
        )
        print(
            " Multi-horizon model training"
        )
        print(
            "================================"
        )

        results = (
            train_and_register_all_horizons(
                dataset_path=features_path,
            )
        )

        print()
        print(
            f"{len(results)} horizon models "
            "trained and registered."
        )

        return results

    @task(multiple_outputs=False)
    def quality_gate(
        features_path: str,
        training_results: dict[
            int,
            dict[str, Any],
        ],
    ) -> dict[int, dict[str, Any]]:
        print()
        print(
            "================================"
        )
        print(
            " Multi-horizon quality gate"
        )
        print(
            "================================"
        )

        print(
            f"Training results received "
            f"for {len(training_results)} "
            "horizons."
        )

        results = promote_all_candidates(
            dataset_path=features_path,
            metric="mae",
            min_improvement_pct=0.5,
            min_baseline_improvement_pct=0.0,
        )

        return results

    @task
    def notify(
        training_results: dict[
            int,
            dict[str, Any],
        ],
        promotion_results: dict[
            int,
            dict[str, Any],
        ],
    ) -> None:
        print()
        print(
            "========================================"
        )
        print(
            " Traffic multi-horizon training completed"
        )
        print(
            "========================================"
        )

        for raw_horizon, result in sorted(
            promotion_results.items(),
            key=lambda item: int(item[0]),
        ):
            horizon = int(
                raw_horizon
            )

            status = (
                "PROMOTED"
                if result.get(
                    "promoted",
                    False,
                )
                else "REJECTED"
            )

            candidate_metrics = (
                result.get(
                    "candidate_metrics",
                    {},
                )
            )

            mae = candidate_metrics.get(
                "mae"
            )

            rmse = candidate_metrics.get(
                "rmse"
            )

            candidate_version = (
                result.get(
                    "candidate_version"
                )
            )

            champion_version = (
                result.get(
                    "champion_version"
                )
            )

            reason = result.get(
                "reason"
            )

            baseline_improvement = (
                result.get(
                    "baseline_improvement_pct"
                )
            )

            champion_improvement = (
                result.get(
                    "champion_improvement_pct"
                )
            )

            print()
            print(
                f"Horizon: +{horizon}h"
            )
            print(
                f"Status: {status}"
            )
            print(
                "Candidate version: "
                f"v{candidate_version}"
            )

            if champion_version is not None:
                print(
                    "Champion version: "
                    f"v{champion_version}"
                )

            if mae is not None:
                print(
                    f"MAE: {mae:.4f}"
                )

            if rmse is not None:
                print(
                    f"RMSE: {rmse:.4f}"
                )

            if (
                baseline_improvement
                is not None
            ):
                print(
                    "Improvement vs "
                    "persistence: "
                    f"{baseline_improvement:+.2f}%"
                )

            if (
                champion_improvement
                is not None
            ):
                print(
                    "Improvement vs "
                    "champion: "
                    f"{champion_improvement:+.2f}%"
                )

            print(
                f"Reason: {reason}"
            )

        print()
        print(
            "========================================"
        )
        print(
            "All configured horizons processed."
        )
        print(
            "========================================"
        )

    ingestion_result = ingestion()

    dataset_path = make_dataset(
        ingestion_result
    )

    features_path = build_features(
        dataset_path
    )

    training_results = train_models(
        features_path
    )

    promotion_results = quality_gate(
        features_path,
        training_results,
    )

    notify(
        training_results,
        promotion_results,
    )


traffic_training_pipeline()