from __future__ import annotations

from pathlib import Path

import pandas as pd

from traffic_prediction.features.road_features import (
    add_road_features,
)
from traffic_prediction.features.schema import (
    FEATURE_COLUMNS,
    PREDICTION_HORIZONS,
    TARGET_COLUMNS,
)
from traffic_prediction.features.time_features import (
    add_time_features,
)
from traffic_prediction.features.traffic_features import (
    add_traffic_lag_features,
    add_traffic_targets,
)
from traffic_prediction.processing.cleaning import (
    clean_traffic_data,
)

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
    Build the machine-learning training dataset.

    The dataset contains:
    - traffic features
    - time features
    - road context
    - lag features
    - future occupancy targets for every configured horizon

    Returns
    -------
    Path
        Path of the generated Parquet file.
    """
    input_path = Path(input_path)
    road_reference_path = Path(
        road_reference_path
    )
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Training input file not found: "
            f"{input_path}"
        )

    if not road_reference_path.exists():
        raise FileNotFoundError(
            f"Road reference file not found: "
            f"{road_reference_path}"
        )

    print(
        f"Loading training data from "
        f"{input_path}"
    )

    traffic_df = pd.read_parquet(
        input_path
    )

    print(
        f"Loaded {len(traffic_df)} "
        "traffic observations"
    )

    road_reference_df = pd.read_parquet(
        road_reference_path
    )

    # ---------------------------------------------------------
    # Cleaning
    # ---------------------------------------------------------

    features_df = clean_traffic_data(
        traffic_df
    )

    # ---------------------------------------------------------
    # Time features
    # ---------------------------------------------------------

    features_df = add_time_features(
        features_df
    )

    # ---------------------------------------------------------
    # Road context
    # ---------------------------------------------------------

    features_df = add_road_features(
        features_df,
        road_reference_df,
    )

    # Keep only currently eligible roads.
    eligible_road_ids = set(
        road_reference_df["iu_ac"]
        .astype(str)
    )

    features_df = features_df[
        features_df["iu_ac"]
        .astype(str)
        .isin(eligible_road_ids)
    ].copy()

    # ---------------------------------------------------------
    # Traffic lag features
    # ---------------------------------------------------------

    features_df = add_traffic_lag_features(
        features_df,
        lags=(1, 2, 24),
    )

    # ---------------------------------------------------------
    # Multi-horizon targets
    # ---------------------------------------------------------

    features_df = add_traffic_targets(
        features_df,
        horizons=PREDICTION_HORIZONS,
    )

    # ---------------------------------------------------------
    # Validate feature schema
    # ---------------------------------------------------------

    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in features_df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing training features: "
            f"{missing_features}"
        )

    missing_targets = [
        target_column
        for target_column
        in TARGET_COLUMNS.values()
        if target_column
        not in features_df.columns
    ]

    if missing_targets:
        raise ValueError(
            "Missing training targets: "
            f"{missing_targets}"
        )

    # ---------------------------------------------------------
    # Sort and save
    # ---------------------------------------------------------

    features_df = (
        features_df
        .sort_values(
            [
                "iu_ac",
                "timestamp_utc",
            ]
        )
        .reset_index(drop=True)
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features_df.to_parquet(
        output_path,
        index=False,
    )

    print(
        f"Training features saved to "
        f"{output_path}"
    )

    print(
        f"Rows: {len(features_df)}"
    )

    print(
        f"Roads: "
        f"{features_df['iu_ac'].nunique()}"
    )

    print(
        f"Features: {len(FEATURE_COLUMNS)}"
    )

    for horizon in PREDICTION_HORIZONS:
        target_column = (
            TARGET_COLUMNS[horizon]
        )

        available_targets = (
            features_df[
                target_column
            ]
            .notna()
            .sum()
        )

        print(
            f"{target_column}: "
            f"{available_targets} "
            "available targets"
        )

    return output_path