from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import pendulum
from airflow.exceptions import (
    AirflowSkipException,
)
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
    get_last_training_dataset_max_timestamp,
    train_and_register_all_horizons,
)
from traffic_prediction.storage.database import (
    get_db_session,
)
from traffic_prediction.storage.repositories import (
    get_latest_traffic_timestamp,
)


TRAINING_HISTORY_HOURS = 24 * 90

#
# Training is aborted when the newest traffic
# observation is older than this threshold.
#
MAX_DATA_AGE_HOURS = 24


@dag(
    dag_id="traffic_training_pipeline",
    description=(
        "Weekly multi-horizon training pipeline "
        "for Paris traffic prediction"
    ),

    #
    # Every Sunday at 05:00 Europe/Paris.
    #
    schedule="0 5 * * 0",

    start_date=pendulum.datetime(
        2026,
        1,
        1,
        tz="Europe/Paris",
    ),

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
        """
        Retrieve any observations that have appeared
        since the last incremental ingestion.
        """
        inserted_rows = (
            run_incremental_ingestion()
        )

        print(
            "Ingestion completed: "
            f"{inserted_rows} observations processed"
        )

        return inserted_rows

    @task
    def validate_data(
        ingestion_result: int,
    ) -> str:
        """
        Validate data freshness and determine whether
        the training dataset has advanced since the
        previous successful MLflow training.

        If no new observation exists, downstream
        training tasks are skipped.
        """
        print()
        print(
            "================================"
        )
        print(
            " Training data validation"
        )
        print(
            "================================"
        )

        print(
            "Rows processed during ingestion: "
            f"{ingestion_result}"
        )

        with get_db_session() as session:
            latest_timestamp = (
                get_latest_traffic_timestamp(
                    session
                )
            )

        if latest_timestamp is None:
            raise RuntimeError(
                "No traffic observations found "
                "in PostgreSQL."
            )

        #
        # Normalize PostgreSQL timestamp to UTC.
        #
        if latest_timestamp.tzinfo is None:
            latest_timestamp = (
                latest_timestamp.replace(
                    tzinfo=timezone.utc
                )
            )
        else:
            latest_timestamp = (
                latest_timestamp.astimezone(
                    timezone.utc
                )
            )

        now = datetime.now(
            timezone.utc
        )

        age_hours = (
            now - latest_timestamp
        ).total_seconds() / 3600

        print(
            "Latest traffic observation: "
            f"{latest_timestamp.isoformat()}"
        )

        print(
            "Traffic data age: "
            f"{age_hours:.2f} hours"
        )

        #
        # Do not train on a stale source dataset.
        #
        if age_hours > MAX_DATA_AGE_HOURS:
            raise RuntimeError(
                "Training aborted: latest traffic "
                f"observation is {age_hours:.2f} "
                "hours old. "
                "Maximum allowed age is "
                f"{MAX_DATA_AGE_HOURS} hours."
            )

        print(
            "Training data freshness: OK"
        )

        #
        # Determine the maximum timestamp used during
        # the previous training.
        #
        last_training_timestamp = (
            get_last_training_dataset_max_timestamp()
        )

        if last_training_timestamp is None:
            print()
            print(
                "No previous training dataset "
                "timestamp found in MLflow."
            )
            print(
                "Training will continue."
            )

            return (
                latest_timestamp.isoformat()
            )

        current_timestamp = (
            pd.Timestamp(
                latest_timestamp
            )
        )

        if current_timestamp.tzinfo is None:
            current_timestamp = (
                current_timestamp.tz_localize(
                    "UTC"
                )
            )
        else:
            current_timestamp = (
                current_timestamp.tz_convert(
                    "UTC"
                )
            )

        print(
            "Last trained dataset timestamp: "
            f"{last_training_timestamp.isoformat()}"
        )

        #
        # No observation newer than the data used by
        # the previous training.
        #
        if (
            current_timestamp
            <= last_training_timestamp
        ):
            print()
            print(
                "================================"
            )
            print(
                " TRAINING SKIPPED"
            )
            print(
                "================================"
            )
            print(
                "No new traffic data since "
                "the previous training."
            )
            print(
                "Current latest timestamp: "
                f"{current_timestamp.isoformat()}"
            )
            print(
                "Previous training timestamp: "
                f"{last_training_timestamp.isoformat()}"
            )
            print(
                "================================"
            )

            raise AirflowSkipException(
                "Training skipped because the "
                "traffic dataset has not advanced "
                "since the previous training."
            )

        new_data_hours = (
            current_timestamp
            - last_training_timestamp
        ).total_seconds() / 3600

        print(
            "New data since previous training: "
            f"{new_data_hours:.2f} hours"
        )

        print(
            "Training required: YES"
        )

        return (
            latest_timestamp.isoformat()
        )

    @task
    def make_dataset(
        validation_result: str,
    ) -> str:
        """
        Build the rolling 90-day training dataset
        from PostgreSQL.
        """
        print(
            "Data validation received: "
            f"{validation_result}"
        )

        print(
            "Building training dataset "
            "from the last "
            f"{TRAINING_HISTORY_HOURS // 24} days"
        )

        path = make_training_dataset(
            history_hours=(
                TRAINING_HISTORY_HOURS
            ),
        )

        print(
            "Training dataset created: "
            f"{path}"
        )

        return str(
            path
        )

    @task
    def build_features(
        dataset_path: str,
    ) -> str:
        """
        Generate the shared multi-horizon feature
        dataset.
        """
        path = (
            build_training_features(
                input_path=dataset_path,
            )
        )

        print(
            "Multi-horizon training "
            f"features created: {path}"
        )

        return str(
            path
        )

    @task(
        multiple_outputs=False
    )
    def train_models(
        features_path: str,
    ) -> dict[int, dict[str, Any]]:
        """
        Train and register one candidate model for
        every configured prediction horizon.
        """
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

    @task(
        multiple_outputs=False
    )
    def quality_gate(
        features_path: str,
        training_results: dict[
            int,
            dict[str, Any],
        ],
    ) -> dict[int, dict[str, Any]]:
        """
        Compare each candidate against persistence
        and the existing champion.

        A candidate is promoted only when it satisfies
        the configured quality rules.
        """
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
            "Training results received "
            f"for {len(training_results)} "
            "horizons."
        )

        results = (
            promote_all_candidates(
                dataset_path=features_path,
                metric="mae",

                #
                # Candidate must improve the existing
                # champion by at least 0.5%.
                #
                min_improvement_pct=0.5,

                #
                # Candidate must at least beat the
                # persistence baseline.
                #
                min_baseline_improvement_pct=0.0,
            )
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
        """
        Write a final training/promotion summary into
        the Airflow task logs.
        """
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

        print(
            "Models trained: "
            f"{len(training_results)}"
        )

        for raw_horizon, result in sorted(
            promotion_results.items(),
            key=lambda item: int(
                item[0]
            ),
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

            mae = (
                candidate_metrics.get(
                    "mae"
                )
            )

            rmse = (
                candidate_metrics.get(
                    "rmse"
                )
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

            reason = (
                result.get(
                    "reason"
                )
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

    #
    # DAG dependency graph
    #
    ingestion_result = (
        ingestion()
    )

    validation_result = (
        validate_data(
            ingestion_result
        )
    )

    dataset_path = (
        make_dataset(
            validation_result
        )
    )

    features_path = (
        build_features(
            dataset_path
        )
    )

    training_results = (
        train_models(
            features_path
        )
    )

    promotion_results = (
        quality_gate(
            features_path,
            training_results,
        )
    )

    notify(
        training_results,
        promotion_results,
    )


traffic_training_pipeline()