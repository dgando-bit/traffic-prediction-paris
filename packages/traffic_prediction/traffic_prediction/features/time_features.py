from __future__ import annotations

import pandas as pd

from shared.logging import get_logger


logger = get_logger(__name__)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create calendar/time features from the Paris-local timestamp.

    The timestamp must be timezone-aware because traffic patterns
    should be interpreted using Paris local time.
    """
    if "timestamp_paris" not in df.columns:
        raise ValueError(
            "Column 'timestamp_paris' is required "
            "to create time features."
        )

    df = df.copy()

    timestamp = df["timestamp_paris"]

    df["hour"] = timestamp.dt.hour
    df["day_of_week"] = timestamp.dt.dayofweek
    df["day_of_month"] = timestamp.dt.day
    df["month"] = timestamp.dt.month

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    )

    logger.info(
        "Time features created for %s records",
        len(df),
    )

    return df