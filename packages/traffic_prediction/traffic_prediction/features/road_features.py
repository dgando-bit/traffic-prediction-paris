from __future__ import annotations

import pandas as pd

from shared.logging import get_logger


logger = get_logger(__name__)


ROAD_FEATURE_COLUMNS = [
    "latitude",
    "longitude",
    "road_length_m",
]


def add_road_features(
    traffic_df: pd.DataFrame,
    road_reference_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add road characteristics to traffic observations.

    The join is performed using the road segment identifier iu_ac.
    """

    if "iu_ac" not in traffic_df.columns:
        raise ValueError(
            "Column 'iu_ac' is required in traffic dataframe."
        )

    if "iu_ac" not in road_reference_df.columns:
        raise ValueError(
            "Column 'iu_ac' is required in road reference dataframe."
        )

    missing_columns = [
        column
        for column in ROAD_FEATURE_COLUMNS
        if column not in road_reference_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing road reference columns: "
            f"{missing_columns}"
        )

    traffic = traffic_df.copy()
    road_reference = road_reference_df.copy()

    # Ensure the join key has the same type.
    traffic["iu_ac"] = traffic["iu_ac"].astype("string")
    road_reference["iu_ac"] = road_reference[
        "iu_ac"
    ].astype("string")

    # Only keep columns needed for ML.
    road_reference = road_reference[
        [
            "iu_ac",
            *ROAD_FEATURE_COLUMNS,
        ]
    ].drop_duplicates(
        subset=["iu_ac"]
    )

    result = traffic.merge(
        road_reference,
        on="iu_ac",
        how="left",
        validate="many_to_one",
    )

    missing_road_context = (
        result[ROAD_FEATURE_COLUMNS]
        .isna()
        .all(axis=1)
        .sum()
    )

    logger.info(
        "Road features merged: %s records / "
        "%s without road context",
        len(result),
        missing_road_context,
    )

    return result