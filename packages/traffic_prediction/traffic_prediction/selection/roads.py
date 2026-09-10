from __future__ import annotations

from traffic_prediction.ingestion.traffic import fetch_traffic_page
import pandas as pd


DEFAULT_MIN_OBSERVATIONS = 24 * 90
DEFAULT_MIN_VALID_RATIO = 0.95


def select_eligible_roads(
    road_stats: pd.DataFrame,
    road_reference: pd.DataFrame,
    *,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
    min_valid_ratio: float = DEFAULT_MIN_VALID_RATIO,
) -> pd.DataFrame:
    """
    Select road segments suitable for traffic prediction.

    A road is eligible when:
    - it has enough historical observations;
    - enough q/k observations are valid;
    - it exists in the road reference;
    - coordinates and geometry are available.
    """

    required_stats = {
        "iu_ac",
        "n_observations",
        "n_valid_q",
        "n_valid_k",
    }

    missing = required_stats - set(road_stats.columns)

    if missing:
        raise ValueError(
            f"Missing road statistics columns: "
            f"{sorted(missing)}"
        )

    required_reference = {
        "iu_ac",
        "latitude",
        "longitude",
        "road_length_m",
        "geo_shape",
    }

    missing = (
        required_reference
        - set(road_reference.columns)
    )

    if missing:
        raise ValueError(
            f"Missing road reference columns: "
            f"{sorted(missing)}"
        )

    stats = road_stats.copy()
    reference = road_reference.copy()

    stats["iu_ac"] = stats["iu_ac"].astype("string")
    reference["iu_ac"] = reference["iu_ac"].astype(
        "string"
    )

    stats["valid_q_ratio"] = (
        stats["n_valid_q"]
        / stats["n_observations"]
    )

    stats["valid_k_ratio"] = (
        stats["n_valid_k"]
        / stats["n_observations"]
    )

    eligible = stats[
        (
            stats["n_observations"]
            >= min_observations
        )
        & (
            stats["valid_q_ratio"]
            >= min_valid_ratio
        )
        & (
            stats["valid_k_ratio"]
            >= min_valid_ratio
        )
    ].copy()

    reference = reference[
        [
            "iu_ac",
            "latitude",
            "longitude",
            "road_length_m",
            "geo_shape",
        ]
    ].drop_duplicates(
        subset=["iu_ac"]
    )

    eligible = eligible.merge(
        reference,
        on="iu_ac",
        how="inner",
    )

    eligible = eligible.dropna(
        subset=[
            "latitude",
            "longitude",
            "road_length_m",
            "geo_shape",
        ]
    )

    return eligible.sort_values(
        [
            "n_observations",
            "iu_ac",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

def fetch_road_statistics(
    *,
    page_size: int = 100,
) -> pd.DataFrame:
    """
    Fetch aggregated traffic statistics for every road segment.

    Pagination stops when the API returns fewer rows than requested.
    """
    rows: list[dict] = []
    offset = 0

    while True:
        data = fetch_traffic_page(
            limit=page_size,
            offset=offset,
            select=(
                "iu_ac, "
                "count(*) as n_observations, "
                "count(q) as n_valid_q, "
                "count(k) as n_valid_k"
            ),
            group_by="iu_ac",
        )

        page = data.get("results", [])

        if not page:
            break

        rows.extend(page)

        print(
            f"Road statistics fetched: "
            f"{len(rows)}"
        )

        if len(page) < page_size:
            break

        offset += len(page)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["iu_ac"] = df["iu_ac"].astype("string")

    for column in (
        "n_observations",
        "n_valid_q",
        "n_valid_k",
    ):
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        ).fillna(0).astype(int)

    # Road IDs in the traffic reference are numeric.
    # This removes malformed values such as "*"
    # or API/database error strings.
    df = df[
        df["iu_ac"].str.fullmatch(
            r"\d+",
            na=False,
        )
    ].copy()

    return (
        df.drop_duplicates(subset=["iu_ac"])
        .sort_values("iu_ac")
        .reset_index(drop=True)
    )