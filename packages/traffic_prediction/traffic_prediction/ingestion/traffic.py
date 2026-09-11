from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from shared.config import get_settings
from shared.logging import get_logger


settings = get_settings()
logger = get_logger(__name__)


DEFAULT_MAX_RETRIES = 6
DEFAULT_PAGE_DELAY_SECONDS = 1.0
DEFAULT_INITIAL_BACKOFF_SECONDS = 30.0
DEFAULT_MAX_BACKOFF_SECONDS = 300.0

def fetch_traffic_page(
    *,
    limit: int | None = None,
    offset: int = 0,
    where: str | None = None,
    order_by: str | None = None,
    select: str | None = None,
    group_by: str | None = None,
    timeout: float | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict[str, Any]:
    """
    Fetch one page of traffic data from Paris Open Data.

    HTTP 429 responses are retried automatically using either
    the Retry-After header or exponential backoff.
    """
    limit = limit or settings.api_page_size
    timeout = timeout or settings.http_timeout

    if max_retries <= 0:
        raise ValueError(
            "max_retries must be greater than 0"
        )

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

    for attempt in range(
        1,
        max_retries + 1,
    ):
        try:
            response = httpx.get(
                settings.traffic_api_url,
                params=params,
                timeout=timeout,
            )

        except httpx.RequestError as exc:
            if attempt >= max_retries:
                logger.error(
                    "Traffic API request failed after "
                    "%s attempts: %s",
                    max_retries,
                    exc,
                )
                raise

            wait_seconds = min(
                DEFAULT_INITIAL_BACKOFF_SECONDS
                * (2 ** (attempt - 1)),
                DEFAULT_MAX_BACKOFF_SECONDS,
            )

            logger.warning(
                "Traffic API request error: %s. "
                "Retrying in %.1f seconds "
                "(attempt %s/%s)",
                exc,
                wait_seconds,
                attempt,
                max_retries,
            )

            time.sleep(
                wait_seconds
            )

            continue

        # -----------------------------------------------------
        # Rate limit
        # -----------------------------------------------------

        if response.status_code == 429:
            retry_after = response.headers.get(
                "Retry-After"
            )

            if retry_after:
                try:
                    wait_seconds = float(
                        retry_after
                    )
                except ValueError:
                    wait_seconds = min(
                        DEFAULT_INITIAL_BACKOFF_SECONDS
                        * (2 ** (attempt - 1)),
                        DEFAULT_MAX_BACKOFF_SECONDS,
                    )
            else:
                wait_seconds = min(
                    DEFAULT_INITIAL_BACKOFF_SECONDS
                    * (2 ** (attempt - 1)),
                    DEFAULT_MAX_BACKOFF_SECONDS,
                )

            if attempt >= max_retries:
                logger.error(
                    "Paris Open Data rate limit persisted "
                    "after %s attempts.",
                    max_retries,
                )

                response.raise_for_status()

            logger.warning(
                "Rate limited by Paris Open Data "
                "(HTTP 429). "
                "Retrying in %.1f seconds "
                "(attempt %s/%s)",
                wait_seconds,
                attempt,
                max_retries,
            )

            time.sleep(
                wait_seconds
            )

            continue

        # -----------------------------------------------------
        # Other HTTP errors
        # -----------------------------------------------------

        response.raise_for_status()

        return response.json()

    raise RuntimeError(
        "Unable to fetch traffic data after "
        f"{max_retries} attempts."
    )


def fetch_traffic_data(
    *,
    where: str | None = None,
    order_by: str | None = None,
    page_size: int | None = None,
    max_records: int | None = None,
    timeout: float | None = None,
    page_delay_seconds: float = (
        DEFAULT_PAGE_DELAY_SECONDS
    ),
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

    page_delay_seconds:
        Delay between paginated API calls. This helps avoid
        HTTP 429 rate-limit responses.

    Returns
    -------
    pandas.DataFrame
        Traffic observations.
    """
    page_size = (
        page_size
        or settings.api_page_size
    )

    timeout = (
        timeout
        or settings.http_timeout
    )

    if page_delay_seconds < 0:
        raise ValueError(
            "page_delay_seconds must be greater "
            "than or equal to 0"
        )

    logger.info(
        "Starting traffic ingestion "
        "(max_records=%s)",
        max_records
        if max_records is not None
        else "unlimited",
    )

    rows: list[dict[str, Any]] = []
    offset = 0

    while True:
        current_limit = page_size

        if max_records is not None:
            remaining = (
                max_records
                - len(rows)
            )

            if remaining <= 0:
                break

            current_limit = min(
                page_size,
                remaining,
            )

        data = fetch_traffic_page(
            limit=current_limit,
            offset=offset,
            where=where,
            order_by=order_by,
            timeout=timeout,
        )

        page = data.get(
            "results",
            [],
        )

        if not page:
            break

        rows.extend(
            page
        )

        logger.debug(
            "Fetched page: offset=%s "
            "records=%s total=%s",
            offset,
            len(page),
            len(rows),
        )

        if len(page) < current_limit:
            break

        offset += len(page)

        if page_delay_seconds > 0:
            time.sleep(
                page_delay_seconds
            )

    df = pd.DataFrame(
        rows
    )

    logger.info(
        "Traffic download completed: "
        "%s records",
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
        df["iu_ac"] = (
            df["iu_ac"]
            .astype("string")
        )

    for column in (
        "q",
        "k",
    ):
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    for column in (
        "geo_point_2d",
        "geo_shape",
    ):
        if column in df.columns:
            df[column] = df[column].apply(
                lambda value: (
                    json.dumps(
                        value
                    )
                    if isinstance(
                        value,
                        dict,
                    )
                    else None
                )
            )

    return df


def save_traffic_parquet(
    df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """
    Save traffic data as a Parquet file.
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
        engine="pyarrow",
    )

    logger.info(
        "Traffic data saved to %s "
        "(%s records)",
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
    page_delay_seconds: float = (
        DEFAULT_PAGE_DELAY_SECONDS
    ),
) -> Path:
    """
    Execute the traffic ingestion pipeline.

    Paris Open Data
        -> fetch
        -> pagination
        -> retry / rate-limit handling
        -> minimal normalization
        -> Parquet
    """
    df = fetch_traffic_data(
        where=where,
        order_by=order_by,
        page_size=page_size,
        max_records=max_records,
        page_delay_seconds=(
            page_delay_seconds
        ),
    )

    df = normalize_traffic_data(
        df
    )

    return save_traffic_parquet(
        df,
        output_path,
    )