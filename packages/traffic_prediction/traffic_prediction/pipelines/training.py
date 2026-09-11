from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import mlflow
import mlflow.lightgbm
import pandas as pd
from mlflow import MlflowClient

from shared.config import get_settings
from traffic_prediction.features.schema import (
    FEATURE_COLUMNS,
    PREDICTION_HORIZONS,
    TARGET_COLUMNS,
    get_target_column,
)
from traffic_prediction.models.baseline import (
    temporal_train_test_split,
)
from traffic_prediction.models.evaluate import (
    evaluate_regression,
)
from traffic_prediction.models.lightgbm import (
    train_lightgbm,
)


DATASET_PATH = Path(
    "data/processed/training_features.parquet"
)

EXPERIMENT_NAME = "traffic-prediction-multihorizon"


def _load_dataset(
    dataset_path: str | Path = DATASET_PATH,
    *,
    target_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {dataset_path}"
        )

    df = pd.read_parquet(
        dataset_path
    )

    if target_columns is None:
        target_columns = TARGET_COLUMNS.values()

    required_columns = {
        "timestamp_utc",
        *FEATURE_COLUMNS,
        *target_columns,
    }

    missing_columns = (
        required_columns.difference(
            df.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Training dataset is missing required "
            "columns: "
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
            "Unable to find a registered model "
            "version "
            f"for run {run_id}"
        )

    latest_version = max(
        matching_versions,
        key=lambda version: int(
            version.version
        ),
    )

    return str(
        latest_version.version
    )


def _get_registered_model_name(
    *,
    base_name: str,
    horizon_hours: int,
) -> str:
    return (
        f"{base_name}-{horizon_hours}h"
    )


def _train_horizon(
    *,
    train_base_df: pd.DataFrame,
    test_base_df: pd.DataFrame,
    horizon_hours: int,
    base_registered_model_name: str,
    trained_model_alias: str,
) -> dict[str, Any]:
    target_column = get_target_column(
        horizon_hours
    )

    registered_model_name = (
        _get_registered_model_name(
            base_name=(
                base_registered_model_name
            ),
            horizon_hours=horizon_hours,
        )
    )

    train_df = (
        train_base_df.dropna(
            subset=[target_column]
        )
        .copy()
    )

    test_df = (
        test_base_df.dropna(
            subset=[target_column]
        )
        .copy()
    )

    if train_df.empty:
        raise ValueError(
            "Training dataset is empty after "
            "removing missing targets for "
            f"horizon +{horizon_hours}h."
        )

    if test_df.empty:
        raise ValueError(
            "Test dataset is empty after "
            "removing missing targets for "
            f"horizon +{horizon_hours}h."
        )

    print()
    print(
        "=" * 60
    )
    print(
        f"Training horizon: "
        f"+{horizon_hours}h"
    )
    print(
        f"Target column: "
        f"{target_column}"
    )
    print(
        f"Registered model: "
        f"{registered_model_name}"
    )
    print(
        f"Train rows: "
        f"{len(train_df):,}"
    )
    print(
        f"Test rows: "
        f"{len(test_df):,}"
    )
    print(
        "=" * 60
    )

    with mlflow.start_run(
        run_name=(
            "lightgbm-road-context-"
            f"{horizon_hours}h"
        )
    ) as run:
        run_id = (
            run.info.run_id
        )

        model = train_lightgbm(
            train_df=train_df,
            feature_columns=(
                FEATURE_COLUMNS
            ),
            target_column=target_column,
        )

        predictions = model.predict(
            test_df[
                FEATURE_COLUMNS
            ]
        )

        metrics = (
            evaluate_regression(
                y_true=(
                    test_df[
                        target_column
                    ]
                ),
                y_pred=predictions,
            )
        )

        mlflow.log_params(
            {
                "model_type": (
                    "LightGBM"
                ),
                "horizon_hours": (
                    horizon_hours
                ),
                "target_column": (
                    target_column
                ),
                "n_features": len(
                    FEATURE_COLUMNS
                ),
                "train_rows": len(
                    train_df
                ),
                "test_rows": len(
                    test_df
                ),
            }
        )

        mlflow.log_metric(
            "mae",
            float(
                metrics.mae
            ),
        )

        mlflow.log_metric(
            "rmse",
            float(
                metrics.rmse
            ),
        )

        mlflow.log_dict(
            {
                "features": (
                    FEATURE_COLUMNS
                ),
                "target": (
                    target_column
                ),
                "horizon_hours": (
                    horizon_hours
                ),
            },
            "features.json",
        )

        model_info = (
            mlflow.lightgbm.log_model(
                lgb_model=model,
                name="model",
                registered_model_name=(
                    registered_model_name
                ),
            )
        )

        print(
            "Run ID:",
            run_id,
        )
        print(
            "Model URI:",
            model_info.model_uri,
        )
        print(
            "MAE:",
            metrics.mae,
        )
        print(
            "RMSE:",
            metrics.rmse,
        )

    client = MlflowClient()

    model_version = (
        _get_registered_model_version(
            client=client,
            registered_model_name=(
                registered_model_name
            ),
            run_id=run_id,
        )
    )

    client.set_registered_model_alias(
        name=registered_model_name,
        alias=trained_model_alias,
        version=model_version,
    )

    print(
        f"Alias "
        f"@{trained_model_alias} -> "
        f"{registered_model_name} "
        f"version {model_version}"
    )

    return {
        "horizon_hours": (
            horizon_hours
        ),
        "target_column": (
            target_column
        ),
        "run_id": (
            run_id
        ),
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "registered_model_name": (
            registered_model_name
        ),
        "model_version": (
            model_version
        ),
        "alias": (
            trained_model_alias
        ),
        "mae": float(
            metrics.mae
        ),
        "rmse": float(
            metrics.rmse
        ),
        "train_rows": len(
            train_df
        ),
        "test_rows": len(
            test_df
        ),
    }


def train_and_register_model(
    *,
    horizon_hours: int = 1,
    dataset_path: str | Path = DATASET_PATH,
) -> dict[str, Any]:
    """
    Train and register one prediction horizon.

    Kept for backward compatibility with the
    existing training service.
    """
    if (
        horizon_hours
        not in PREDICTION_HORIZONS
    ):
        raise ValueError(
            "Unsupported prediction horizon: "
            f"{horizon_hours}. "
            "Supported horizons: "
            f"{PREDICTION_HORIZONS}"
        )

    settings = get_settings()

    tracking_uri = (
        settings.mlflow_tracking_uri
    )

    base_registered_model_name = (
        settings.registered_model_name
    )

    trained_model_alias = (
        settings.trained_model_alias
    )

    target_column = (
        get_target_column(
            horizon_hours
        )
    )

    mlflow.set_tracking_uri(
        tracking_uri
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    print(
        "MLflow tracking URI:",
        mlflow.get_tracking_uri(),
    )
    print(
        "Experiment:",
        EXPERIMENT_NAME,
    )

    df = _load_dataset(
        dataset_path,
        target_columns=[
            target_column
        ],
    )

    print(
        "Dataset rows:",
        f"{len(df):,}",
    )

    train_base_df, test_base_df = (
        temporal_train_test_split(
            df,
            test_size=0.2,
        )
    )

    print(
        "Train period:",
        train_base_df[
            "timestamp_utc"
        ].min(),
        "->",
        train_base_df[
            "timestamp_utc"
        ].max(),
    )

    print(
        "Test period:",
        test_base_df[
            "timestamp_utc"
        ].min(),
        "->",
        test_base_df[
            "timestamp_utc"
        ].max(),
    )

    return _train_horizon(
        train_base_df=(
            train_base_df
        ),
        test_base_df=(
            test_base_df
        ),
        horizon_hours=(
            horizon_hours
        ),
        base_registered_model_name=(
            base_registered_model_name
        ),
        trained_model_alias=(
            trained_model_alias
        ),
    )


def train_and_register_all_horizons(
    *,
    dataset_path: str | Path = DATASET_PATH,
) -> dict[int, dict[str, Any]]:
    """
    Train and register all supported prediction
    horizons using exactly the same temporal
    train/test split.
    """
    settings = get_settings()

    tracking_uri = (
        settings.mlflow_tracking_uri
    )

    base_registered_model_name = (
        settings.registered_model_name
    )

    trained_model_alias = (
        settings.trained_model_alias
    )

    mlflow.set_tracking_uri(
        tracking_uri
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    print(
        "MLflow tracking URI:",
        mlflow.get_tracking_uri(),
    )

    print(
        "Experiment:",
        EXPERIMENT_NAME,
    )

    print(
        "Horizons:",
        PREDICTION_HORIZONS,
    )

    df = _load_dataset(
        dataset_path,
    )

    print(
        "Dataset rows:",
        f"{len(df):,}",
    )

    #
    # IMPORTANT:
    # The split is performed only once.
    # Every horizon therefore uses exactly
    # the same temporal train/test boundary.
    #
    train_base_df, test_base_df = (
        temporal_train_test_split(
            df,
            test_size=0.2,
        )
    )

    train_start = (
        train_base_df[
            "timestamp_utc"
        ].min()
    )

    train_end = (
        train_base_df[
            "timestamp_utc"
        ].max()
    )

    test_start = (
        test_base_df[
            "timestamp_utc"
        ].min()
    )

    test_end = (
        test_base_df[
            "timestamp_utc"
        ].max()
    )

    print()
    print(
        "Shared temporal split"
    )
    print(
        "Train:",
        train_start,
        "->",
        train_end,
    )
    print(
        "Test :",
        test_start,
        "->",
        test_end,
    )

    results: dict[
        int,
        dict[str, Any],
    ] = {}

    for horizon_hours in (
        PREDICTION_HORIZONS
    ):
        result = _train_horizon(
            train_base_df=(
                train_base_df
            ),
            test_base_df=(
                test_base_df
            ),
            horizon_hours=(
                horizon_hours
            ),
            base_registered_model_name=(
                base_registered_model_name
            ),
            trained_model_alias=(
                trained_model_alias
            ),
        )

        results[
            horizon_hours
        ] = result

    print()
    print(
        "=" * 60
    )
    print(
        "MULTI-HORIZON TRAINING SUMMARY"
    )
    print(
        "=" * 60
    )

    for horizon_hours, result in (
        results.items()
    ):
        print(
            f"+{horizon_hours}h | "
            f"MAE={result['mae']:.4f} | "
            f"RMSE={result['rmse']:.4f} | "
            f"{result['registered_model_name']} "
            f"v{result['model_version']}"
        )

    return results