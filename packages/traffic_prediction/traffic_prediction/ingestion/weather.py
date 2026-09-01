from __future__ import annotations

from pathlib import Path

import httpx
import pandas as pd

from shared.logging import get_logger


logger = get_logger(__name__)


HISTORICAL_WEATHER_URL = (
    "https://historical-forecast-api.open-meteo.com/v1/forecast"
)

# Paris
PARIS_LATITUDE = 48.8566
PARIS_LONGITUDE = 2.3522

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "wind_speed_10m",
    "weather_code",
]


def fetch_historical_weather(
    start_date: str,
    end_date: str,
) -> dict:
    params = {
        "latitude": PARIS_LATITUDE,
        "longitude": PARIS_LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARIABLES),

        # Important:
        # canonical timestamps remain UTC.
        "timezone": "UTC",
    }

    logger.info(
        "Fetching historical weather: %s -> %s",
        start_date,
        end_date,
    )

    response = httpx.get(
        HISTORICAL_WEATHER_URL,
        params=params,
        timeout=60.0,
    )

    response.raise_for_status()

    return response.json()


def normalize_historical_weather(
    payload: dict,
) -> pd.DataFrame:
    hourly = payload["hourly"]

    df = pd.DataFrame(hourly)

    df["timestamp_utc"] = pd.to_datetime(
        df["time"],
        utc=True,
    )

    df = df.drop(
        columns=["time"],
    )

    df = df.sort_values(
        "timestamp_utc"
    ).reset_index(drop=True)

    logger.info(
        "Historical weather normalized: %s records",
        len(df),
    )

    return df


def save_weather_parquet(
    df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        output_path,
        index=False,
    )

    logger.info(
        "Weather data saved: %s",
        output_path,
    )

    return output_path


def ingest_historical_weather(
    start_date: str,
    end_date: str,
    output_path: str | Path,
) -> pd.DataFrame:
    payload = fetch_historical_weather(
        start_date=start_date,
        end_date=end_date,
    )

    df = normalize_historical_weather(
        payload
    )

    save_weather_parquet(
        df,
        output_path,
    )

    return df