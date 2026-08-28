from __future__ import annotations

import pandas as pd

from shared.logging import get_logger


logger = get_logger(__name__)


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
]

TARGET_COLUMN = "target_k_1h"


def build_ml_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the V0 supervised ML dataset.

    Rows without a target are removed because they cannot
    participate in supervised training.

    Missing lag values are kept for now so that the model
    preparation step can decide how to handle them.
    """
    required_columns = {
        "iu_ac",
        "timestamp_utc",
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    dataset = df[
        [
            "iu_ac",
            "timestamp_utc",
            *FEATURE_COLUMNS,
            TARGET_COLUMN,
        ]
    ].copy()

    # A supervised training row requires a known future target.
    dataset = dataset[
        dataset[TARGET_COLUMN].notna()
    ].copy()

    dataset = dataset.sort_values(
        ["timestamp_utc", "iu_ac"]
    ).reset_index(drop=True)

    logger.info(
        "ML dataset created: %s rows, %s features",
        len(dataset),
        len(FEATURE_COLUMNS),
    )

    return dataset