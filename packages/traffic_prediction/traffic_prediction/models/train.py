from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from shared.logging import get_logger
from traffic_prediction.features.ml_dataset import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


logger = get_logger(__name__)


@dataclass
class TrainingResult:
    """Result of a model training operation."""

    model: HistGradientBoostingRegressor
    feature_columns: list[str]


def train_hist_gradient_boosting(
    train_df: pd.DataFrame,
) -> TrainingResult:
    """
    Train the V0 HistGradientBoosting regression model.

    The model predicts traffic occupancy k one hour ahead.
    """
    required_columns = {
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
    }

    missing_columns = (
        required_columns - set(train_df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    training_df = train_df[
        train_df[TARGET_COLUMN].notna()
    ].copy()

    if training_df.empty:
        raise ValueError(
            "No valid observations available for training."
        )

    X_train = training_df[FEATURE_COLUMNS].copy()
    y_train = training_df[TARGET_COLUMN].copy()

    logger.info(
        "Training HistGradientBoostingRegressor: "
        "%s rows / %s features",
        len(X_train),
        len(FEATURE_COLUMNS),
    )

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
    )

    logger.info(
        "HistGradientBoostingRegressor training completed"
    )

    return TrainingResult(
        model=model,
        feature_columns=list(FEATURE_COLUMNS),
    )