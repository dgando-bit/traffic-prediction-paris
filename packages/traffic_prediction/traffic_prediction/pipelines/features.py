from __future__ import annotations

from pathlib import Path

import pandas as pd

from shared.logging import get_logger
from traffic_prediction.features.time_features import (
    add_time_features,
)
from traffic_prediction.features.traffic_features import (
    add_traffic_lag_features,
    add_traffic_target,
)
from traffic_prediction.features.road_features import (
    add_road_features,
)
from traffic_prediction.features.schema import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)

logger = get_logger(__name__)


DEFAULT_INPUT_PATH = Path(
    "data/interim/traffic_training.parquet"
)

DEFAULT_ROAD_REFERENCE_PATH = Path(
    "data/raw/reference/road_reference.parquet"
)

DEFAULT_OUTPUT_PATH = Path(
    "data/processed/training_features.parquet"
)


def build_training_features(
    *,
    input_path: str | Path = DEFAULT_INPUT_PATH,
    road_reference_path: str | Path = DEFAULT_ROAD_REFERENCE_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """
    Build the final training feature dataset.

    The resulting dataset contains the 14 features retained
    by the final LightGBM model plus the prediction target.
    """
    input_path = Path(input_path)
    road_reference_path = Path(road_reference_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Training input dataset not found: {input_path}"
        )

    if not road_reference_path.exists():
        raise FileNotFoundError(
            f"Road reference dataset not found: "
            f"{road_reference_path}"
        )

    logger.info(
        "Building training features from %s",
        input_path,
    )

    df = pd.read_parquet(input_path)
    road_reference_df = pd.read_parquet(
        road_reference_path
    )

    required_input_columns = {
        "iu_ac",
        "timestamp_utc",
        "q",
        "k",
    }

    missing_columns = (
        required_input_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Input dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # Time features
    df = add_time_features(df)

    # Static road context
    df = add_road_features(
        df,
        road_reference_df,
    )

    # Historical traffic features
    df = add_traffic_lag_features(
        df,
        lags=(1, 2, 24),
    )

    # Prediction target k(t + 1h)
    df = add_traffic_target(
        df,
        horizon_hours=1,
    )

    required_output_columns = {
        "iu_ac",
        "timestamp_utc",
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
    }

    missing_output_columns = (
        required_output_columns - set(df.columns)
    )

    if missing_output_columns:
        raise ValueError(
            "Feature engineering did not create all "
            "required columns: "
            f"{sorted(missing_output_columns)}"
        )

    dataset = df[
        [
            "iu_ac",
            "timestamp_utc",
            *FEATURE_COLUMNS,
            TARGET_COLUMN,
        ]
    ].copy()

    # Supervised training requires a known future target.
    dataset = dataset[
        dataset[TARGET_COLUMN].notna()
    ].copy()

    dataset = (
        dataset.sort_values(
            [
                "timestamp_utc",
                "iu_ac",
            ]
        )
        .reset_index(drop=True)
    )

    if dataset.empty:
        raise ValueError(
            "Training feature dataset is empty."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_parquet(
        output_path,
        index=False,
        engine="pyarrow",
    )

    logger.info(
        "Training features saved to %s: "
        "%s rows / %s roads / %s features",
        output_path,
        len(dataset),
        dataset["iu_ac"].nunique(),
        len(FEATURE_COLUMNS),
    )

    return output_path