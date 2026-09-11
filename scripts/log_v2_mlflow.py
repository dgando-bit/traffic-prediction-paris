from pathlib import Path

import mlflow
import mlflow.lightgbm
import pandas as pd
from traffic_prediction.models.baseline import (
    evaluate_persistence_baseline,
    temporal_train_test_split,
)
from traffic_prediction.models.evaluate import evaluate_regression
from traffic_prediction.models.lightgbm import train_lightgbm

INPUT_PATH = Path(
    "data/processed/ml_dataset_multi_20_long.parquet"
)

TARGET_COLUMN = "target_k_1h"

FEATURE_COLUMNS = [
    "q",
    "k",
    "hour",
    "day_of_week",
    "is_weekend",
    "q_lag_1h",
    "k_lag_1h",
    "q_lag_2h",
    "k_lag_2h",
    "q_lag_24h",
    "k_lag_24h",
    "latitude",
    "longitude",
    "road_length_m",
]

EXPERIMENT_NAME = "traffic-prediction-v2"


def main() -> None:
    df = pd.read_parquet(INPUT_PATH)

    train_df, test_df = temporal_train_test_split(
        df,
        test_size=0.2,
    )

    evaluation_df = test_df[
        test_df[TARGET_COLUMN].notna()
        & test_df["k"].notna()
    ].copy()

    baseline_metrics = evaluate_persistence_baseline(
        evaluation_df
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    with mlflow.start_run(
        run_name="lightgbm-road-context"
    ):
        model = train_lightgbm(
            train_df=train_df,
            feature_columns=FEATURE_COLUMNS,
            target_column=TARGET_COLUMN,
        )

        predictions = model.predict(
            evaluation_df[FEATURE_COLUMNS]
        )

        metrics = evaluate_regression(
            evaluation_df[TARGET_COLUMN],
            predictions,
        )

        # -----------------------------
        # Dataset information
        # -----------------------------

        mlflow.log_param(
            "dataset",
            INPUT_PATH.name,
        )

        mlflow.log_param(
            "target",
            TARGET_COLUMN,
        )

        mlflow.log_param(
            "prediction_horizon_hours",
            1,
        )

        mlflow.log_param(
            "n_features",
            len(FEATURE_COLUMNS),
        )

        mlflow.log_param(
            "train_rows",
            len(train_df),
        )

        mlflow.log_param(
            "test_rows",
            len(evaluation_df),
        )

        # -----------------------------
        # Model parameters
        # -----------------------------

        mlflow.log_params(
            {
                "model_type": "LightGBM",
                "objective": "regression",
                "n_estimators": 300,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "random_state": 42,
            }
        )

        # -----------------------------
        # Metrics
        # -----------------------------

        mlflow.log_metric(
            "mae",
            metrics.mae,
        )

        mlflow.log_metric(
            "rmse",
            metrics.rmse,
        )

        mlflow.log_metric(
            "baseline_mae",
            baseline_metrics.mae,
        )

        mlflow.log_metric(
            "baseline_rmse",
            baseline_metrics.rmse,
        )

        mae_improvement = (
            (
                baseline_metrics.mae
                - metrics.mae
            )
            / baseline_metrics.mae
            * 100
        )

        rmse_improvement = (
            (
                baseline_metrics.rmse
                - metrics.rmse
            )
            / baseline_metrics.rmse
            * 100
        )

        mlflow.log_metric(
            "mae_improvement_vs_persistence_pct",
            mae_improvement,
        )

        mlflow.log_metric(
            "rmse_improvement_vs_persistence_pct",
            rmse_improvement,
        )

        # -----------------------------
        # Feature metadata
        # -----------------------------

        feature_file = Path(
            "data/processed/v2_features.txt"
        )

        feature_file.write_text(
            "\n".join(FEATURE_COLUMNS),
            encoding="utf-8",
        )

        mlflow.log_artifact(
            feature_file,
            artifact_path="metadata",
        )

        # -----------------------------
        # Model artifact
        # -----------------------------

        mlflow.lightgbm.log_model(
            model,
            name="model",
        )

        # -----------------------------
        # Output
        # -----------------------------

        print()
        print("=== MLflow V2 run ===")

        print(
            f"MAE  : {metrics.mae:.4f}"
        )

        print(
            f"RMSE : {metrics.rmse:.4f}"
        )

        print(
            f"Baseline MAE  : "
            f"{baseline_metrics.mae:.4f}"
        )

        print(
            f"Baseline RMSE : "
            f"{baseline_metrics.rmse:.4f}"
        )

        print(
            f"MAE improvement  : "
            f"{mae_improvement:+.2f}%"
        )

        print(
            f"RMSE improvement : "
            f"{rmse_improvement:+.2f}%"
        )

        print()

        print(
            f"Run ID: "
            f"{mlflow.active_run().info.run_id}"
        )


if __name__ == "__main__":
    main()