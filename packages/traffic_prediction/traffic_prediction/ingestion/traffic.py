from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from shared.config import get_settings
from shared.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)

def fetch_traffic_page(
    *,
    limit: int | None = None,
    offset: int = 0,
    where: str | None = None,
    order_by: str | None = None,
    select: str | None = None,
    group_by: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Fetch one page of traffic data from Paris Open Data."""

    limit = limit or settings.api_page_size
    timeout = timeout or settings.http_timeout

    params: dict[str, Any] = {
        "limit": limit,
        "offset": offset,
    }

    if where:
        params["where"] = where

    if order_by:
        params["order_by"] = order_by

    if select:
        params["select"] = select

    if group_by:
        params["group_by"] = group_by

    logger.debug(
        "Fetching traffic data: offset=%s limit=%s",
        offset,
        limit,
    )

    response = httpx.get(
        settings.traffic_api_url,
        params=params,
        timeout=timeout,
    )

    response.raise_for_status()

    return response.json()

def fetch_traffic_data(
    *,
    where: str | None = None,
    order_by: str | None = None,
    page_size: int | None = None,
    max_records: int | None = None,
    timeout: float | None = None,
) -> pd.DataFrame:
    """
    Fetch traffic observations with pagination.

    Raw observations are preserved as returned by Paris Open Data.
    Missing q/k values are intentionally not removed.

    Parameters
    ----------
    where:
        Optional Opendatasoft filter expression.
    order_by:
        Optional sort expression.
    page_size:
        Number of records requested per API call.
    max_records:
        Optional maximum number of records to retrieve.
        Useful during development and testing.
    timeout:
        HTTP request timeout in seconds.

    Returns
    -------
    pandas.DataFrame
        Traffic observations.
    """
    page_size = page_size or settings.api_page_size
    timeout = timeout or settings.http_timeout

    logger.info(
        "Starting traffic ingestion (max_records=%s)",
        max_records or "unlimited",
    )

    rows: list[dict[str, Any]] = []
    offset = 0

    while True:
        current_limit = page_size

        if max_records is not None:
            remaining = max_records - len(rows)

            if remaining <= 0:
                break

            current_limit = min(page_size, remaining)

        data = fetch_traffic_page(
            limit=current_limit,
            offset=offset,
            where=where,
            order_by=order_by,
            timeout=timeout,
        )

        page = data.get("results", [])

        if not page:
            break

        rows.extend(page)

        if len(page) < current_limit:
            break

        offset += len(page)

    df = pd.DataFrame(rows)

    logger.info(
        "Traffic download completed: %s records",
        len(df),
    )

    return df

def normalize_traffic_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply minimal structural normalization to raw traffic data.

    Missing q/k observations are intentionally preserved.
    Complex JSON fields are serialized to JSON strings so they
    can be stored safely in Parquet.
    """
    df = df.copy()

    if "t_1h" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(
            df["t_1h"],
            utc=True,
            errors="coerce",
        )

    if "iu_ac" in df.columns:
        df["iu_ac"] = df["iu_ac"].astype("string")

    for column in ("q", "k"):
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    for column in ("geo_point_2d", "geo_shape"):
        if column in df.columns:
            df[column] = df[column].apply(
                lambda value: (
                    json.dumps(value)
                    if isinstance(value, dict)
                    else None
                )
            )

    return df

def save_traffic_parquet(
    df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Save traffic data as a Parquet file."""

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        output_path,
        index=False,
        engine="pyarrow",
    )

    logger.info(
        "Traffic data saved to %s (%s records)",
        output_path,
        len(df),
    )

    return output_path

def ingest_traffic_data(
    *,
    output_path: str | Path,
    where: str | None = None,
    order_by: str | None = None,
    page_size: int | None = None,
    max_records: int | None = None,
) -> Path:
    """
    Execute the V0 traffic ingestion pipeline.

    Paris Open Data
        -> fetch
        -> minimal normalization
        -> Parquet
    """
    df = fetch_traffic_data(
        where=where,
        order_by=order_by,
        page_size=page_size,
        max_records=max_records,
    )

    df = normalize_traffic_data(df)

    return save_traffic_parquet(
        df,
        output_path,
    )