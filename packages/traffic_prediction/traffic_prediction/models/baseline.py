from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

from shared.logging import get_logger


logger = get_logger(__name__)


@dataclass
class BaselineMetrics:
    mae: float
    rmse: float
    n_samples: int


def temporal_train_test_split(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a time-series dataset chronologically.

    The oldest observations are used for training and the
    most recent observations for testing.
    """
    if not 0 < test_size < 1:
        raise ValueError(
            "test_size must be between 0 and 1."
        )

    if "timestamp_utc" not in df.columns:
        raise ValueError(
            "Column 'timestamp_utc' is required."
        )

    if len(df) < 2:
        raise ValueError(
            "At least 2 observations are required."
        )

    df = df.sort_values(
        "timestamp_utc"
    ).reset_index(drop=True)

    split_index = int(
        len(df) * (1 - test_size)
    )

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    logger.info(
        "Temporal split: %s train rows / %s test rows",
        len(train_df),
        len(test_df),
    )

    return train_df, test_df


def evaluate_persistence_baseline(
    test_df: pd.DataFrame,
) -> BaselineMetrics:
    """
    Evaluate the persistence baseline:

        prediction k(t + 1h) = k(t)

    Only rows with both current k and future target available
    are evaluated.
    """
    required_columns = {
        "k",
        "target_k_1h",
    }

    missing_columns = (
        required_columns - set(test_df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    evaluation_df = test_df[
        test_df["k"].notna()
        & test_df["target_k_1h"].notna()
    ].copy()

    if evaluation_df.empty:
        raise ValueError(
            "No valid observations available "
            "for baseline evaluation."
        )

    y_true = evaluation_df["target_k_1h"]
    y_pred = evaluation_df["k"]

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = mean_squared_error(
        y_true,
        y_pred,
    ) ** 0.5

    metrics = BaselineMetrics(
        mae=float(mae),
        rmse=float(rmse),
        n_samples=len(evaluation_df),
    )

    logger.info(
        "Persistence baseline - MAE: %.4f | RMSE: %.4f | n=%s",
        metrics.mae,
        metrics.rmse,
        metrics.n_samples,
    )

    return metrics