from __future__ import annotations

import pandas as pd

from shared.logging import get_logger


logger = get_logger(__name__)


def add_traffic_lag_features(
    df: pd.DataFrame,
    lags: tuple[int, ...] = (1, 2, 24),
) -> pd.DataFrame:
    """
    Create traffic lag features using exact time differences.

    A lag of 1h means exactly one hour earlier, not simply
    the previous row.
    """
    required_columns = {
        "iu_ac",
        "timestamp_utc",
        "q",
        "k",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    df = df.copy()

    df = df.sort_values(
        ["iu_ac", "timestamp_utc"]
    ).reset_index(drop=True)

    for lag in lags:
        lookup = df[
            [
                "iu_ac",
                "timestamp_utc",
                "q",
                "k",
            ]
        ].copy()

        # Shift historical timestamps forward so they match
        # the current observation.
        #
        # Example for lag=1:
        # historical row at 17:00 becomes 18:00 in lookup,
        # allowing it to match the current row at 18:00.
        lookup["timestamp_utc"] = (
            lookup["timestamp_utc"]
            + pd.Timedelta(hours=lag)
        )

        lookup = lookup.rename(
            columns={
                "q": f"q_lag_{lag}h",
                "k": f"k_lag_{lag}h",
            }
        )

        df = df.merge(
            lookup,
            on=["iu_ac", "timestamp_utc"],
            how="left",
            validate="one_to_one",
        )

    logger.info(
        "Traffic lag features created with exact time matching: %s",
        lags,
    )

    return df


def add_traffic_target(
    df: pd.DataFrame,
    horizon_hours: int = 1,
) -> pd.DataFrame:
    """
    Create the future occupancy target.

    For horizon_hours=1:
        target_k_1h = occupancy exactly one hour in the future.

    The target is matched using timestamps rather than row position.
    """
    required_columns = {
        "iu_ac",
        "timestamp_utc",
        "k",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if horizon_hours <= 0:
        raise ValueError(
            "horizon_hours must be greater than 0."
        )

    df = df.copy()

    target_column = f"target_k_{horizon_hours}h"

    lookup = df[
        [
            "iu_ac",
            "timestamp_utc",
            "k",
        ]
    ].copy()

    # Example:
    #
    # k measured at 18:00 must become the target
    # of the observation at 17:00.
    #
    # Therefore:
    #
    # 18:00 -> 17:00 in the lookup table.
    lookup["timestamp_utc"] = (
        lookup["timestamp_utc"]
        - pd.Timedelta(hours=horizon_hours)
    )

    lookup = lookup.rename(
        columns={
            "k": target_column,
        }
    )

    df = df.merge(
        lookup,
        on=["iu_ac", "timestamp_utc"],
        how="left",
        validate="one_to_one",
    )

    logger.info(
        "Traffic target created: %s",
        target_column,
    )

    return df