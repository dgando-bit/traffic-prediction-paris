from __future__ import annotations

import pandas as pd

from shared.logging import get_logger
from traffic_prediction.features.time_features import (
    add_time_features,
)
from traffic_prediction.features.traffic_features import (
    add_traffic_lag_features,
    add_traffic_target,
)
from traffic_prediction.features.calendar_features import (
    add_public_holiday_features,
    add_school_holiday_features,
)

logger = get_logger(__name__)


def build_traffic_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the V0 traffic feature set.

    Features:
    - calendar/time features
    - traffic lags at 1h, 2h and 24h
    - prediction target k(t + 1h)
    """
    logger.info(
        "Building traffic features for %s records",
        len(df),
    )

    # ------------------------------------------------------------------
    # Time features
    # ------------------------------------------------------------------

    df = add_time_features(df)

    # ------------------------------------------------------------------
    # Holidays features
    # ------------------------------------------------------------------

    df = add_public_holiday_features(df)

    df = add_school_holiday_features(df)

    # ------------------------------------------------------------------
    # Historical traffic features
    # ------------------------------------------------------------------

    df = add_traffic_lag_features(
        df,
        lags=(1, 2, 24),
    )

    # ------------------------------------------------------------------
    # ML target
    # ------------------------------------------------------------------

    df = add_traffic_target(
        df,
        horizon_hours=1,
    )

    logger.info(
        "Feature engineering completed: %s records",
        len(df),
    )

    return df