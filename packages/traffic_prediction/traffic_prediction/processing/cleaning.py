from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from shared.logging import get_logger

logger = get_logger(__name__)


def clean_traffic_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize raw Paris traffic observations.

    This step performs structural cleaning only.
    Rows with missing q/k values are preserved.
    """
    df = df.copy()

    logger.info(
        "Starting traffic cleaning: %s records",
        len(df),
    )

    # ------------------------------------------------------------------
    # Identifiers
    # ------------------------------------------------------------------

    if "iu_ac" in df.columns:
        df["iu_ac"] = df["iu_ac"].astype("string").str.strip()

    for column in ("iu_nd_amont", "iu_nd_aval"):
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip()

    # ------------------------------------------------------------------
    # Numeric measurements
    # ------------------------------------------------------------------

    for column in ("q", "k"):
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # ------------------------------------------------------------------
    # Timestamp
    # ------------------------------------------------------------------

    if "timestamp_utc" not in df.columns and "t_1h" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(
            df["t_1h"],
            utc=True,
            errors="coerce",
        )

    elif "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(
            df["timestamp_utc"],
            utc=True,
            errors="coerce",
        )

    if "timestamp_utc" in df.columns:
        df["timestamp_paris"] = (
            df["timestamp_utc"]
            .dt.tz_convert("Europe/Paris")
        )

    # ------------------------------------------------------------------
    # Dates
    # ------------------------------------------------------------------

    for column in ("date_debut", "date_fin"):
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce",
            )

    # ------------------------------------------------------------------
    # Geographic fields
    # ------------------------------------------------------------------

    for column in ("geo_point_2d", "geo_shape"):
        if column in df.columns:
            df[column] = df[column].apply(
                _normalize_json_value
            )

    logger.info(
        "Traffic cleaning completed: %s records",
        len(df),
    )

    return df


def _normalize_json_value(value: object) -> str | None:
    """
    Normalize geographic JSON fields.

    Raw ingestion may already have serialized these fields.
    """
    if value is None:
        return None

    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        return json.dumps(value)

    return None


def process_traffic_file(
    input_path: str | Path,
    output_path: str | Path,
) -> Path:
    """
    Read raw traffic data, clean it, add quality flags
    and save the resulting interim dataset.
    """
    from traffic_prediction.processing.validation import (
        add_traffic_quality_flags,
    )

    input_path = Path(input_path)
    output_path = Path(output_path)

    logger.info(
        "Processing raw traffic file: %s",
        input_path,
    )

    df = pd.read_parquet(input_path)

    df = clean_traffic_data(df)
    df = add_traffic_quality_flags(df)

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
        "Processed traffic data saved to %s",
        output_path,
    )

    return output_path