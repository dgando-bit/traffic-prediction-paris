from __future__ import annotations

import pandas as pd

from shared.logging import get_logger


logger = get_logger(__name__)


REQUIRED_TRAFFIC_COLUMNS = {
    "iu_ac",
    "timestamp_utc",
    "q",
    "k",
    "etat_trafic",
    "etat_barre",
}


def validate_traffic_schema(df: pd.DataFrame) -> None:
    """
    Validate that mandatory traffic columns are present.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    missing_columns = REQUIRED_TRAFFIC_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required traffic columns: "
            f"{sorted(missing_columns)}"
        )


def add_traffic_quality_flags(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add data-quality indicators without removing observations.
    """
    validate_traffic_schema(df)

    df = df.copy()

    # iu_ac should contain a road arc identifier.
    # We already observed corrupted textual values in the source dataset.
    df["is_valid_iu_ac"] = (
        df["iu_ac"]
        .astype("string")
        .str.fullmatch(r"\d+")
        .fillna(False)
    )

    df["has_q"] = df["q"].notna()
    df["has_k"] = df["k"].notna()

    df["has_complete_measurement"] = (
        df["has_q"]
        & df["has_k"]
    )

    df["is_invalid_road_state"] = (
        df["etat_barre"]
        .astype("string")
        .eq("Invalide")
    )

    df["is_unknown_traffic_state"] = (
        df["etat_trafic"]
        .astype("string")
        .eq("Inconnu")
    )

    logger.info(
        "Traffic validation completed: "
        "%s invalid iu_ac, %s missing k, %s missing q",
        (~df["is_valid_iu_ac"]).sum(),
        (~df["has_k"]).sum(),
        (~df["has_q"]).sum(),
    )

    return df