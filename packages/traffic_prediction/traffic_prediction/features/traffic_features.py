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
        if lag <= 0:
            raise ValueError(
                "All lags must be greater than 0."
            )

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
    Create one future occupancy target.

    Example:
        horizon_hours=1
        -> target_k_1h

    The target is matched using exact timestamps rather than
    row position.
    """
    return add_traffic_targets(
        df,
        horizons=(horizon_hours,),
    )


def add_traffic_targets(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 2, 3),
) -> pd.DataFrame:
    """
    Create future occupancy targets for multiple horizons.

    Examples:
        target_k_1h = occupancy exactly 1 hour in the future
        target_k_2h = occupancy exactly 2 hours in the future
        target_k_3h = occupancy exactly 3 hours in the future

    Targets are matched using exact timestamps rather than
    row position.

    This avoids incorrect targets when observations are missing.
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

    if not horizons:
        raise ValueError(
            "At least one prediction horizon is required."
        )

    if any(horizon <= 0 for horizon in horizons):
        raise ValueError(
            "All horizons must be greater than 0."
        )

    if len(set(horizons)) != len(horizons):
        raise ValueError(
            "Prediction horizons must be unique."
        )

    df = df.copy()

    df = df.sort_values(
        ["iu_ac", "timestamp_utc"]
    ).reset_index(drop=True)

    for horizon_hours in horizons:
        target_column = (
            f"target_k_{horizon_hours}h"
        )

        lookup = df[
            [
                "iu_ac",
                "timestamp_utc",
                "k",
            ]
        ].copy()

        # Example for horizon=2:
        #
        # k measured at 19:00 must become the target
        # of the observation at 17:00.
        #
        # Therefore:
        #
        # 19:00 -> 17:00 in the lookup table.
        lookup["timestamp_utc"] = (
            lookup["timestamp_utc"]
            - pd.Timedelta(
                hours=horizon_hours
            )
        )

        lookup = lookup.rename(
            columns={
                "k": target_column,
            }
        )

        df = df.merge(
            lookup,
            on=[
                "iu_ac",
                "timestamp_utc",
            ],
            how="left",
            validate="one_to_one",
        )

        logger.info(
            "Traffic target created: %s",
            target_column,
        )

    logger.info(
        "Traffic targets created for horizons: %s",
        horizons,
    )

    return df