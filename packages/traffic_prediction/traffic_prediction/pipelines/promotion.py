from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.lightgbm
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from shared.config import get_settings
from traffic_prediction.models.baseline import temporal_train_test_split
from traffic_prediction.models.evaluate import evaluate_regression


DEFAULT_DATASET_PATH = Path(
    "data/processed/training_features.parquet"
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


def _load_evaluation_dataset(
    dataset_path: str | Path,
    *,
    test_size: float = 0.2,
) -> pd.DataFrame:
    """
    Load the current feature dataset and return a common
    temporal evaluation window for all compared models.
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {dataset_path}"
        )

    df = pd.read_parquet(dataset_path)

    required_columns = {
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
        "timestamp_utc",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing evaluation columns: "
            f"{sorted(missing_columns)}"
        )

    _, test_df = temporal_train_test_split(
        df,
        test_size=test_size,
    )

    if test_df.empty:
        raise ValueError(
            "The evaluation dataset is empty."
        )

    return test_df


def _evaluate_model_version(
    *,
    model_name: str,
    version: str,
    test_df: pd.DataFrame,
) -> dict[str, float]:
    """
    Evaluate one concrete MLflow model version on the
    supplied common evaluation dataset.
    """
    model_uri = (
        f"models:/{model_name}/{version}"
    )

    print(
        f"Loading {model_name} "
        f"version {version}"
    )

    model = mlflow.lightgbm.load_model(
        model_uri
    )

    predictions = model.predict(
        test_df[FEATURE_COLUMNS]
    )

    metrics = evaluate_regression(
        test_df[TARGET_COLUMN],
        predictions,
    )

    return {
        "mae": float(metrics.mae),
        "rmse": float(metrics.rmse),
    }


def promote_candidate(
    *,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    test_size: float = 0.2,
    metric: str = "mae",
    min_improvement_pct: float = 0.0,
) -> dict:
    """
    Compare candidate and champion on exactly the same
    temporal evaluation window.

    The candidate is promoted only when its selected metric
    improves on the champion by at least
    ``min_improvement_pct``.

    Lower metric values are considered better.
    """
    if metric not in {"mae", "rmse"}:
        raise ValueError(
            "metric must be either 'mae' or 'rmse'"
        )

    settings = get_settings()

    mlflow.set_tracking_uri(
        settings.mlflow_tracking_uri
    )

    client = MlflowClient()

    model_name = (
        settings.registered_model_name
    )

    candidate_alias = (
        settings.trained_model_alias
    )

    champion_alias = (
        settings.api_model_alias
    )

    candidate = (
        client.get_model_version_by_alias(
            model_name,
            candidate_alias,
        )
    )

    print(
        f"Candidate: v{candidate.version}"
    )

    try:
        champion = (
            client.get_model_version_by_alias(
                model_name,
                champion_alias,
            )
        )

    except MlflowException:
        client.set_registered_model_alias(
            name=model_name,
            alias=champion_alias,
            version=candidate.version,
        )

        print(
            "No champion exists. "
            f"Candidate v{candidate.version} "
            "promoted as initial champion."
        )

        return {
            "promoted": True,
            "reason": "no_existing_champion",
            "candidate_version": str(
                candidate.version
            ),
            "champion_version": str(
                candidate.version
            ),
        }

    print(
        f"Champion: v{champion.version}"
    )

    test_df = _load_evaluation_dataset(
        dataset_path,
        test_size=test_size,
    )

    print()
    print(
        "Common evaluation window:"
    )

    print(
        f"Rows: {len(test_df)}"
    )

    print(
        "From:",
        test_df["timestamp_utc"].min(),
    )

    print(
        "To:",
        test_df["timestamp_utc"].max(),
    )

    print()

    candidate_metrics = (
        _evaluate_model_version(
            model_name=model_name,
            version=str(
                candidate.version
            ),
            test_df=test_df,
        )
    )

    champion_metrics = (
        _evaluate_model_version(
            model_name=model_name,
            version=str(
                champion.version
            ),
            test_df=test_df,
        )
    )

    candidate_metric = (
        candidate_metrics[metric]
    )

    champion_metric = (
        champion_metrics[metric]
    )

    improvement_pct = (
        (
            champion_metric
            - candidate_metric
        )
        / champion_metric
        * 100
    )

    print()
    print(
        f"Candidate v{candidate.version}: "
        f"MAE={candidate_metrics['mae']:.4f} | "
        f"RMSE={candidate_metrics['rmse']:.4f}"
    )

    print(
        f"Champion v{champion.version}: "
        f"MAE={champion_metrics['mae']:.4f} | "
        f"RMSE={champion_metrics['rmse']:.4f}"
    )

    print(
        f"{metric.upper()} improvement: "
        f"{improvement_pct:+.2f}%"
    )

    promoted = (
        improvement_pct
        >= min_improvement_pct
    )

    if promoted:
        client.set_registered_model_alias(
            name=model_name,
            alias=champion_alias,
            version=candidate.version,
        )

        print(
            f"Candidate v{candidate.version} "
            f"promoted to @{champion_alias}."
        )

        final_champion_version = str(
            candidate.version
        )

        reason = "quality_gate_passed"

    else:
        print(
            f"Candidate v{candidate.version} "
            "rejected. "
            f"Champion remains "
            f"v{champion.version}."
        )

        final_champion_version = str(
            champion.version
        )

        reason = "quality_gate_failed"

    return {
        "promoted": promoted,
        "reason": reason,
        "candidate_version": str(
            candidate.version
        ),
        "champion_version": (
            final_champion_version
        ),
        "candidate_mae": (
            candidate_metrics["mae"]
        ),
        "candidate_rmse": (
            candidate_metrics["rmse"]
        ),
        "champion_mae": (
            champion_metrics["mae"]
        ),
        "champion_rmse": (
            champion_metrics["rmse"]
        ),
        "metric": metric,
        "improvement_pct": float(
            improvement_pct
        ),
        "evaluation_rows": len(
            test_df
        ),
        "evaluation_start": str(
            test_df[
                "timestamp_utc"
            ].min()
        ),
        "evaluation_end": str(
            test_df[
                "timestamp_utc"
            ].max()
        ),
    }