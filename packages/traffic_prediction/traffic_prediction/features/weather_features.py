from __future__ import annotations

import pandas as pd

from shared.logging import get_logger


logger = get_logger(__name__)


WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "wind_speed_10m",
    "weather_code",
]


def add_weather_features(
    traffic_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    horizon_hours: int = 1,
) -> pd.DataFrame:
    if "timestamp_utc" not in traffic_df.columns:
        raise ValueError(
            "Column 'timestamp_utc' is required "
            "in traffic dataframe."
        )

    if "timestamp_utc" not in weather_df.columns:
        raise ValueError(
            "Column 'timestamp_utc' is required "
            "in weather dataframe."
        )

    traffic = traffic_df.copy()
    weather = weather_df.copy()

    # Timestamp corresponding to the prediction horizon.
    traffic["weather_target_timestamp"] = (
        traffic["timestamp_utc"]
        + pd.Timedelta(hours=horizon_hours)
    )

    rename_map = {
        column: f"{column}_target_1h"
        for column in WEATHER_COLUMNS
    }

    weather = weather[
        ["timestamp_utc", *WEATHER_COLUMNS]
    ].rename(
        columns={
            "timestamp_utc": "weather_target_timestamp",
            **rename_map,
        }
    )

    weather = weather.drop_duplicates(
        subset=["weather_target_timestamp"]
    )

    result = traffic.merge(
        weather,
        on="weather_target_timestamp",
        how="left",
        validate="many_to_one",
    )

    result = result.drop(
        columns=["weather_target_timestamp"]
    )

    target_weather_columns = list(
        rename_map.values()
    )

    missing_weather = (
        result[target_weather_columns]
        .isna()
        .all(axis=1)
        .sum()
    )

    logger.info(
        "Target weather features merged: "
        "%s records / %s without weather match",
        len(result),
        missing_weather,
    )

    return result