from __future__ import annotations

import json
from pathlib import Path

import httpx
import pandas as pd

from shared.logging import get_logger


logger = get_logger(__name__)


EVENTS_API_URL = (
    "https://opendata.paris.fr/api/explore/v2.1/catalog/"
    "datasets/que-faire-a-paris-/records"
)

DEFAULT_PAGE_SIZE = 100


def fetch_events_page(
    *,
    offset: int = 0,
    limit: int = DEFAULT_PAGE_SIZE,
) -> dict:
    """
    Fetch one page of Paris events.
    """

    response = httpx.get(
        EVENTS_API_URL,
        params={
            "limit": limit,
            "offset": offset,
        },
        timeout=30.0,
    )

    response.raise_for_status()

    return response.json()


def fetch_events() -> pd.DataFrame:
    """
    Fetch all currently available events from the Paris Open Data API.
    """

    records: list[dict] = []
    offset = 0

    while True:
        payload = fetch_events_page(
            offset=offset,
        )

        results = payload.get(
            "results",
            [],
        )

        if not results:
            break

        records.extend(results)

        logger.info(
            "Fetched %s event records",
            len(records),
        )

        offset += len(results)

        total_count = payload.get(
            "total_count"
        )

        if (
            total_count is not None
            and offset >= total_count
        ):
            break

    return pd.DataFrame(records)


def normalize_events(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize event dates and coordinates.
    """

    result = df.copy()

    result["date_start"] = pd.to_datetime(
        result["date_start"],
        utc=True,
        errors="coerce",
    )

    result["date_end"] = pd.to_datetime(
        result["date_end"],
        utc=True,
        errors="coerce",
    )

    # Extract coordinates from:
    # {"lon": ..., "lat": ...}
    result["latitude"] = result[
        "lat_lon"
    ].apply(
        lambda value: (
            value.get("lat")
            if isinstance(value, dict)
            else None
        )
    )

    result["longitude"] = result[
        "lat_lon"
    ].apply(
        lambda value: (
            value.get("lon")
            if isinstance(value, dict)
            else None
        )
    )

    result["latitude"] = pd.to_numeric(
        result["latitude"],
        errors="coerce",
    )

    result["longitude"] = pd.to_numeric(
        result["longitude"],
        errors="coerce",
    )

    columns = [
        "event_id",
        "title",
        "date_start",
        "date_end",
        "occurrences",
        "latitude",
        "longitude",
    ]

    result = result[
        columns
    ].copy()

    # An event without dates cannot be aligned with traffic.
    result = result[
        result["date_start"].notna()
        & result["date_end"].notna()
    ].copy()

    result = result.drop_duplicates(
        subset=[
            "event_id",
            "date_start",
            "date_end",
        ]
    )

    result = result.sort_values(
        [
            "date_start",
            "event_id",
        ]
    ).reset_index(drop=True)

    logger.info(
        "Normalized events: %s records",
        len(result),
    )

    return result


def save_events(
    df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save normalized events to Parquet.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        output_path,
        index=False,
    )

    logger.info(
        "Events saved to %s",
        output_path,
    )


def ingest_events(
    output_path: str | Path,
) -> pd.DataFrame:
    """
    Fetch, normalize and save Paris events.
    """

    raw_df = fetch_events()

    normalized_df = normalize_events(
        raw_df
    )

    save_events(
        normalized_df,
        output_path,
    )

    return normalized_df


def build_event_occurrences(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Expand Paris events into precise occurrence intervals.

    When the API provides the `occurrences` field, each occurrence
    is used independently.

    When occurrences are missing, only events lasting at most
    24 hours are kept to avoid treating long-running events as
    continuously active.
    """

    records = []

    for _, row in df.iterrows():
        event_id = row.get("event_id")
        title = row.get("title")
        latitude = row.get("latitude")
        longitude = row.get("longitude")

        occurrences = row.get("occurrences")

        # -----------------------------------------------------------
        # Precise occurrences provided by the API
        # -----------------------------------------------------------

        if (
            isinstance(occurrences, str)
            and occurrences.strip()
        ):
            for occurrence in occurrences.split(";"):
                occurrence = occurrence.strip()

                if not occurrence:
                    continue

                try:
                    start_raw, end_raw = occurrence.split(
                        "_",
                        maxsplit=1,
                    )
                except ValueError:
                    continue

                occurrence_start = pd.to_datetime(
                    start_raw,
                    utc=True,
                    errors="coerce",
                )

                occurrence_end = pd.to_datetime(
                    end_raw,
                    utc=True,
                    errors="coerce",
                )

                if (
                    pd.isna(occurrence_start)
                    or pd.isna(occurrence_end)
                ):
                    continue

                records.append(
                    {
                        "event_id": event_id,
                        "title": title,
                        "occurrence_start": occurrence_start,
                        "occurrence_end": occurrence_end,
                        "latitude": latitude,
                        "longitude": longitude,
                    }
                )

            continue

        # -----------------------------------------------------------
        # Fallback for simple short events
        # -----------------------------------------------------------

        date_start = row.get("date_start")
        date_end = row.get("date_end")

        if (
            pd.isna(date_start)
            or pd.isna(date_end)
        ):
            continue

        duration = date_end - date_start

        if duration > pd.Timedelta(hours=24):
            continue

        records.append(
            {
                "event_id": event_id,
                "title": title,
                "occurrence_start": date_start,
                "occurrence_end": date_end,
                "latitude": latitude,
                "longitude": longitude,
            }
        )

    result = pd.DataFrame(records)

    if result.empty:
        return result

    result = result[
        result["latitude"].notna()
        & result["longitude"].notna()
    ].copy()

    result = result.drop_duplicates(
        subset=[
            "event_id",
            "occurrence_start",
            "occurrence_end",
        ]
    )

    result = result.sort_values(
        [
            "occurrence_start",
            "event_id",
        ]
    ).reset_index(drop=True)

    logger.info(
        "Event occurrences created: %s",
        len(result),
    )

    return result