from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import mlflow.lightgbm
import pandas as pd
from mlflow import MlflowClient

from shared.config import get_settings
from traffic_prediction.models.baseline import temporal_train_test_split
from traffic_prediction.models.evaluate import evaluate_regression
from traffic_prediction.models.lightgbm import train_lightgbm


DATASET_PATH = Path("data/processed/ml_dataset_multi_20_long.parquet")

EXPERIMENT_NAME = "traffic-prediction-v3"

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


def _load_dataset(
    dataset_path: str | Path = DATASET_PATH,
) -> pd.DataFrame:
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {dataset_path}"
        )

    df = pd.read_parquet(dataset_path)

    required_columns = {
        "timestamp_utc",
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "Training dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return df


def _get_registered_model_version(
    *,
    client: MlflowClient,
    registered_model_name: str,
    run_id: str,
) -> str:
    versions = client.search_model_versions(
        f"name='{registered_model_name}'"
    )

    matching_versions = [
        version
        for version in versions
        if version.run_id == run_id
    ]

    if not matching_versions:
        raise RuntimeError(
            "Unable to find a registered model version "
            f"for run {run_id}"
        )

    latest_version = max(
        matching_versions,
        key=lambda version: int(version.version),
    )

    return str(latest_version.version)


def train_and_register_model(
    *,
    dataset_path: str | Path = DATASET_PATH,
) -> dict[str, Any]:
    settings = get_settings()

    tracking_uri = settings.mlflow_tracking_uri
    registered_model_name = settings.registered_model_name
    trained_model_alias = settings.trained_model_alias

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("MLflow tracking URI:", mlflow.get_tracking_uri())
    print("Experiment:", EXPERIMENT_NAME)
    print("Registered model:", registered_model_name)
    print("Target alias:", trained_model_alias)

    df = _load_dataset(dataset_path)

    print("Dataset rows:", len(df))

    train_df, test_df = temporal_train_test_split(
	    df,
	    test_size=0.2,
    )

    train_df = train_df.dropna(
        subset=[TARGET_COLUMN]
    ).copy()

    test_df = test_df.dropna(
        subset=[TARGET_COLUMN]
    ).copy()

    if train_df.empty:
        raise ValueError(
            "Training dataset is empty after removing missing targets."
        )

    if test_df.empty:
        raise ValueError(
            "Test dataset is empty after removing missing targets."
        )

    print("Train rows:", len(train_df))
    print("Test rows:", len(test_df))

    with mlflow.start_run(
        run_name="lightgbm-road-context"
    ) as run:
        run_id = run.info.run_id

        model = train_lightgbm(
            train_df=train_df,
            feature_columns=FEATURE_COLUMNS,
            target_column=TARGET_COLUMN,
        )

        predictions = model.predict(
            test_df[FEATURE_COLUMNS]
        )

        metrics = evaluate_regression(
            y_true=test_df[TARGET_COLUMN],
            y_pred=predictions,
        )

        mlflow.log_params(
            {
                "model_type": "LightGBM",
                "target_column": TARGET_COLUMN,
                "n_features": len(FEATURE_COLUMNS),
                "train_rows": len(train_df),
                "test_rows": len(test_df),
            }
        )

        mlflow.log_metric(
            "mae",
            float(metrics.mae),
        )

        mlflow.log_metric(
            "rmse",
            float(metrics.rmse),
        )

        mlflow.log_dict(
            {
                "features": FEATURE_COLUMNS,
            },
            "features.json",
        )

        model_info = mlflow.lightgbm.log_model(
	        lgb_model=model,
	        name="model",
	        registered_model_name=registered_model_name,
        )

        print("Run ID:", run_id)
        print("Model URI:", model_info.model_uri)
        print("MAE:", metrics.mae)
        print("RMSE:", metrics.rmse)

    client = MlflowClient()

    model_version = _get_registered_model_version(
        client=client,
        registered_model_name=registered_model_name,
        run_id=run_id,
    )

    client.set_registered_model_alias(
        name=registered_model_name,
        alias=trained_model_alias,
        version=model_version,
    )

    print(
        f"Alias @{trained_model_alias} -> "
        f"{registered_model_name} version {model_version}"
    )

    return {
        "run_id": run_id,
        "experiment_name": EXPERIMENT_NAME,
        "registered_model_name": registered_model_name,
        "model_version": model_version,
        "alias": trained_model_alias,
        "mae": float(metrics.mae),
        "rmse": float(metrics.rmse),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
    }