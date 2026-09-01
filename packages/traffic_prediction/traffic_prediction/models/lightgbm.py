from __future__ import annotations

import lightgbm as lgb
import pandas as pd

from shared.logging import get_logger


logger = get_logger(__name__)


def train_lightgbm(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "target_k_1h",
) -> lgb.LGBMRegressor:
    logger.info(
        "Training LightGBM: %s rows / %s features",
        len(train_df),
        len(feature_columns),
    )

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbosity=-1,
    )

    model.fit(
        train_df[feature_columns],
        train_df[target_column],
    )

    logger.info(
        "LightGBM training completed"
    )

    return model