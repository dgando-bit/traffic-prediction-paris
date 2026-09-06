from __future__ import annotations

from pathlib import Path

from shared.logging import get_logger
from traffic_prediction.processing.cleaning import clean_traffic_data
from traffic_prediction.storage.database import get_db_session
from traffic_prediction.storage.repositories import (
    get_recent_traffic_dataframe,
)


logger = get_logger(__name__)


DEFAULT_OUTPUT_PATH = Path(
    "data/interim/traffic_training.parquet"
)

# One year of traffic history.
DEFAULT_HISTORY_HOURS = 24 * 365


def make_training_dataset(
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    history_hours: int = DEFAULT_HISTORY_HOURS,
) -> Path:
    """
    Build the interim traffic dataset used for model training.

    Traffic observations are extracted from PostgreSQL, normalized,
    deduplicated and stored as a Parquet file.

    Feature engineering is intentionally performed in a later stage.
    """
    output_path = Path(output_path)

    if history_hours <= 0:
        raise ValueError(
            "history_hours must be greater than 0."
        )

    logger.info(
        "Building training dataset from PostgreSQL "
        "(history_hours=%s)",
        history_hours,
    )

    with get_db_session() as session:
        df = get_recent_traffic_dataframe(
            session,
            hours=history_hours,
        )

    if df.empty:
        raise ValueError(
            "No traffic observations available for training."
        )

    logger.info(
        "Loaded %s observations from PostgreSQL",
        len(df),
    )

    # Normalize identifiers, numeric values and timestamps.
    df = clean_traffic_data(df)

    required_columns = {
        "iu_ac",
        "timestamp_utc",
        "q",
        "k",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Training dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # Rows without a usable identifier or timestamp cannot
    # participate in the time-series feature pipeline.
    df = df.dropna(
        subset=[
            "iu_ac",
            "timestamp_utc",
        ]
    ).copy()

    # PostgreSQL already enforces uniqueness, but keeping this
    # protection makes the pipeline robust if its input changes.
    df = (
        df.sort_values(
            [
                "timestamp_utc",
                "iu_ac",
            ]
        )
        .drop_duplicates(
            subset=[
                "iu_ac",
                "timestamp_utc",
            ],
            keep="last",
        )
        .reset_index(drop=True)
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
        "Training dataset saved to %s: "
        "%s rows / %s roads / %s -> %s",
        output_path,
        len(df),
        df["iu_ac"].nunique(),
        df["timestamp_utc"].min(),
        df["timestamp_utc"].max(),
    )

    return output_path