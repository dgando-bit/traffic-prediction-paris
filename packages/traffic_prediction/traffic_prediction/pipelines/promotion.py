from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import mlflow.lightgbm
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from shared.config import get_settings

from traffic_prediction.features.schema import (
    FEATURE_COLUMNS,
    PREDICTION_HORIZONS,
    get_target_column,
)
from traffic_prediction.models.baseline import (
    temporal_train_test_split,
)
from traffic_prediction.models.evaluate import (
    evaluate_regression,
)

DEFAULT_DATASET_PATH = Path(
    "data/processed/training_features.parquet"
)


def _get_registered_model_name(
    *,
    base_name: str,
    horizon_hours: int,
) -> str:
    return (
        f"{base_name}-{horizon_hours}h"
    )


def _load_evaluation_dataset(
    dataset_path: str | Path,
    *,
    horizon_hours: int,
    test_size: float = 0.2,
) -> pd.DataFrame:
    """
    Load the current feature dataset and return
    the temporal test window for one horizon.

    Rows are filtered so that both the target and
    the persistence baseline value `k` are present.

    This guarantees that persistence, candidate and
    champion are evaluated on exactly the same rows.
    """
    dataset_path = Path(
        dataset_path
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            "Evaluation dataset not found: "
            f"{dataset_path}"
        )

    target_column = (
        get_target_column(
            horizon_hours
        )
    )

    df = pd.read_parquet(
        dataset_path
    )

    required_columns = {
        "timestamp_utc",
        "k",
        target_column,
        *FEATURE_COLUMNS,
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing evaluation columns: "
            f"{sorted(missing_columns)}"
        )

    _, test_df = (
        temporal_train_test_split(
            df,
            test_size=test_size,
        )
    )

    test_df = (
        test_df
        .dropna(
            subset=[
                target_column,
                "k",
            ]
        )
        .copy()
    )

    if test_df.empty:
        raise ValueError(
            "The evaluation dataset is empty "
            f"for horizon +{horizon_hours}h."
        )

    return test_df


def _evaluate_persistence(
    *,
    test_df: pd.DataFrame,
    target_column: str,
) -> dict[str, float]:
    """
    Persistence baseline:

        predicted k(t+h) = current k(t)
    """
    metrics = (
        evaluate_regression(
            y_true=(
                test_df[
                    target_column
                ]
            ),
            y_pred=(
                test_df["k"]
            ),
        )
    )

    return {
        "mae": float(
            metrics.mae
        ),
        "rmse": float(
            metrics.rmse
        ),
    }


def _evaluate_model_version(
    *,
    model_name: str,
    version: str,
    test_df: pd.DataFrame,
    target_column: str,
) -> dict[str, float]:
    """
    Evaluate one concrete MLflow model version
    on the supplied common evaluation dataset.
    """
    model_uri = (
        f"models:/{model_name}/{version}"
    )

    print(
        f"Loading {model_name} "
        f"version {version}"
    )

    model = (
        mlflow.lightgbm.load_model(
            model_uri
        )
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

    return {
        "mae": float(
            metrics.mae
        ),
        "rmse": float(
            metrics.rmse
        ),
    }


def _compute_improvement_pct(
    *,
    reference_value: float,
    candidate_value: float,
) -> float:
    """
    Compute percentage improvement for a metric
    where lower values are better.
    """
    if reference_value == 0:
        if candidate_value == 0:
            return 0.0

        return float("-inf")

    return (
        (
            reference_value
            - candidate_value
        )
        / reference_value
        * 100
    )


def promote_candidate(
    *,
    horizon_hours: int = 1,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    test_size: float = 0.2,
    metric: str = "mae",
    min_improvement_pct: float = 0.0,
    min_baseline_improvement_pct: float = 0.0,
) -> dict[str, Any]:
    """
    Apply the quality gate for one prediction
    horizon.

    Rules:

    1. Candidate must beat persistence by at least
       `min_baseline_improvement_pct`.

    2. If no champion exists and rule 1 passes,
       candidate becomes the initial champion.

    3. If a champion exists, candidate must also
       improve on it by at least
       `min_improvement_pct`.

    Lower MAE/RMSE values are better.
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

    if metric not in {
        "mae",
        "rmse",
    }:
        raise ValueError(
            "metric must be either "
            "'mae' or 'rmse'"
        )

    settings = get_settings()

    mlflow.set_tracking_uri(
        settings.mlflow_tracking_uri
    )

    client = MlflowClient()

    model_name = (
        _get_registered_model_name(
            base_name=(
                settings.registered_model_name
            ),
            horizon_hours=(
                horizon_hours
            ),
        )
    )

    candidate_alias = (
        settings.trained_model_alias
    )

    champion_alias = (
        settings.api_model_alias
    )

    target_column = (
        get_target_column(
            horizon_hours
        )
    )

    print()
    print(
        "=" * 64
    )
    print(
        f"QUALITY GATE +{horizon_hours}h"
    )
    print(
        "=" * 64
    )
    print(
        "Model:",
        model_name,
    )
    print(
        "Target:",
        target_column,
    )
    print(
        "Metric:",
        metric,
    )

    candidate = (
        client.get_model_version_by_alias(
            model_name,
            candidate_alias,
        )
    )

    print(
        f"Candidate: "
        f"v{candidate.version}"
    )

    test_df = (
        _load_evaluation_dataset(
            dataset_path,
            horizon_hours=(
                horizon_hours
            ),
            test_size=test_size,
        )
    )

    print()
    print(
        "Common evaluation window:"
    )
    print(
        f"Rows: {len(test_df):,}"
    )
    print(
        "From:",
        test_df[
            "timestamp_utc"
        ].min(),
    )
    print(
        "To:",
        test_df[
            "timestamp_utc"
        ].max(),
    )

    print()
    print(
        "Evaluating persistence baseline..."
    )

    baseline_metrics = (
        _evaluate_persistence(
            test_df=test_df,
            target_column=(
                target_column
            ),
        )
    )

    print(
        "Evaluating candidate..."
    )

    candidate_metrics = (
        _evaluate_model_version(
            model_name=model_name,
            version=str(
                candidate.version
            ),
            test_df=test_df,
            target_column=(
                target_column
            ),
        )
    )

    baseline_metric = (
        baseline_metrics[
            metric
        ]
    )

    candidate_metric = (
        candidate_metrics[
            metric
        ]
    )

    baseline_improvement_pct = (
        _compute_improvement_pct(
            reference_value=(
                baseline_metric
            ),
            candidate_value=(
                candidate_metric
            ),
        )
    )

    print()
    print(
        "Persistence: "
        f"MAE={baseline_metrics['mae']:.4f} | "
        f"RMSE={baseline_metrics['rmse']:.4f}"
    )

    print(
        f"Candidate v{candidate.version}: "
        f"MAE={candidate_metrics['mae']:.4f} | "
        f"RMSE={candidate_metrics['rmse']:.4f}"
    )

    print(
        f"Improvement vs persistence "
        f"({metric.upper()}): "
        f"{baseline_improvement_pct:+.2f}%"
    )

    baseline_gate_passed = (
        baseline_improvement_pct
        >= min_baseline_improvement_pct
    )

    if not baseline_gate_passed:
        print()
        print(
            "Quality gate FAILED: "
            "candidate does not beat persistence "
            "by the required margin."
        )

        return {
            "horizon_hours": (
                horizon_hours
            ),
            "model_name": (
                model_name
            ),
            "promoted": False,
            "reason": (
                "baseline_gate_failed"
            ),
            "candidate_version": str(
                candidate.version
            ),
            "champion_version": None,
            "metric": metric,
            "baseline_metrics": (
                baseline_metrics
            ),
            "candidate_metrics": (
                candidate_metrics
            ),
            "baseline_improvement_pct": float(
                baseline_improvement_pct
            ),
            "champion_improvement_pct": None,
            "evaluation_rows": len(
                test_df
            ),
        }

    #
    # Candidate beats persistence.
    # Now check whether a champion exists.
    #
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

        print()
        print(
            "No existing champion."
        )
        print(
            f"Candidate v{candidate.version} "
            f"promoted as initial "
            f"@{champion_alias}."
        )

        return {
            "horizon_hours": (
                horizon_hours
            ),
            "model_name": (
                model_name
            ),
            "promoted": True,
            "reason": (
                "initial_champion"
            ),
            "candidate_version": str(
                candidate.version
            ),
            "champion_version": str(
                candidate.version
            ),
            "metric": metric,
            "baseline_metrics": (
                baseline_metrics
            ),
            "candidate_metrics": (
                candidate_metrics
            ),
            "champion_metrics": None,
            "baseline_improvement_pct": float(
                baseline_improvement_pct
            ),
            "champion_improvement_pct": None,
            "evaluation_rows": len(
                test_df
            ),
        }

    print()
    print(
        f"Current champion: "
        f"v{champion.version}"
    )

    #
    # If candidate already is champion there is
    # nothing useful to compare/promote.
    #
    if (
        str(candidate.version)
        == str(champion.version)
    ):
        print(
            "Candidate and champion already "
            "reference the same model version."
        )

        return {
            "horizon_hours": (
                horizon_hours
            ),
            "model_name": (
                model_name
            ),
            "promoted": False,
            "reason": (
                "candidate_already_champion"
            ),
            "candidate_version": str(
                candidate.version
            ),
            "champion_version": str(
                champion.version
            ),
            "metric": metric,
            "baseline_metrics": (
                baseline_metrics
            ),
            "candidate_metrics": (
                candidate_metrics
            ),
            "champion_metrics": (
                candidate_metrics
            ),
            "baseline_improvement_pct": float(
                baseline_improvement_pct
            ),
            "champion_improvement_pct": 0.0,
            "evaluation_rows": len(
                test_df
            ),
        }

    print(
        "Evaluating champion..."
    )

    champion_metrics = (
        _evaluate_model_version(
            model_name=model_name,
            version=str(
                champion.version
            ),
            test_df=test_df,
            target_column=(
                target_column
            ),
        )
    )

    champion_metric = (
        champion_metrics[
            metric
        ]
    )

    champion_improvement_pct = (
        _compute_improvement_pct(
            reference_value=(
                champion_metric
            ),
            candidate_value=(
                candidate_metric
            ),
        )
    )

    print()
    print(
        f"Champion v{champion.version}: "
        f"MAE={champion_metrics['mae']:.4f} | "
        f"RMSE={champion_metrics['rmse']:.4f}"
    )

    print(
        f"Improvement vs champion "
        f"({metric.upper()}): "
        f"{champion_improvement_pct:+.2f}%"
    )

    promoted = (
        champion_improvement_pct
        >= min_improvement_pct
    )

    if promoted:
        client.set_registered_model_alias(
            name=model_name,
            alias=champion_alias,
            version=candidate.version,
        )

        print()
        print(
            f"Candidate v{candidate.version} "
            f"promoted to @{champion_alias}."
        )

        final_champion_version = str(
            candidate.version
        )

        reason = (
            "quality_gate_passed"
        )

    else:
        print()
        print(
            f"Candidate v{candidate.version} "
            "NOT promoted."
        )

        print(
            f"Required improvement: "
            f"{min_improvement_pct:.2f}%"
        )

        final_champion_version = str(
            champion.version
        )

        reason = (
            "champion_gate_failed"
        )

    return {
        "horizon_hours": (
            horizon_hours
        ),
        "model_name": (
            model_name
        ),
        "promoted": (
            promoted
        ),
        "reason": (
            reason
        ),
        "candidate_version": str(
            candidate.version
        ),
        "champion_version": (
            final_champion_version
        ),
        "metric": (
            metric
        ),
        "baseline_metrics": (
            baseline_metrics
        ),
        "candidate_metrics": (
            candidate_metrics
        ),
        "champion_metrics": (
            champion_metrics
        ),
        "baseline_improvement_pct": float(
            baseline_improvement_pct
        ),
        "champion_improvement_pct": float(
            champion_improvement_pct
        ),
        "evaluation_rows": len(
            test_df
        ),
    }


def promote_all_candidates(
    *,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    test_size: float = 0.2,
    metric: str = "mae",
    min_improvement_pct: float = 0.0,
    min_baseline_improvement_pct: float = 0.0,
) -> dict[int, dict[str, Any]]:
    """
    Run the quality gate independently for every
    configured prediction horizon.
    """
    results: dict[
        int,
        dict[str, Any],
    ] = {}

    for horizon_hours in (
        PREDICTION_HORIZONS
    ):
        result = promote_candidate(
            horizon_hours=(
                horizon_hours
            ),
            dataset_path=(
                dataset_path
            ),
            test_size=(
                test_size
            ),
            metric=metric,
            min_improvement_pct=(
                min_improvement_pct
            ),
            min_baseline_improvement_pct=(
                min_baseline_improvement_pct
            ),
        )

        results[
            horizon_hours
        ] = result

    print()
    print(
        "=" * 64
    )
    print(
        "MULTI-HORIZON PROMOTION SUMMARY"
    )
    print(
        "=" * 64
    )

    for horizon_hours, result in (
        results.items()
    ):
        status = (
            "PROMOTED"
            if result["promoted"]
            else "NOT PROMOTED"
        )

        candidate_metrics = (
            result[
                "candidate_metrics"
            ]
        )

        print(
            f"+{horizon_hours}h | "
            f"{status:<12} | "
            f"MAE="
            f"{candidate_metrics['mae']:.4f} | "
            f"RMSE="
            f"{candidate_metrics['rmse']:.4f} | "
            f"reason="
            f"{result['reason']}"
        )

    return results