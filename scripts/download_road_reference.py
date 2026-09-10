from pathlib import Path

from traffic_prediction.ingestion.road_reference import (
    fetch_road_reference,
    normalize_road_reference,
    save_road_reference,
)
from traffic_prediction.selection.roads import (
    fetch_road_statistics,
    select_eligible_roads,
)


OUTPUT_PATH = Path(
    "data/raw/reference/road_reference.parquet"
)

MIN_OBSERVATIONS = 24 * 90
MIN_VALID_RATIO = 0.95


def main() -> None:
    print(
        "=== Fetching traffic statistics ==="
    )

    road_stats = fetch_road_statistics()

    print(
        f"Traffic roads found: "
        f"{road_stats['iu_ac'].nunique()}"
    )

    # ---------------------------------------------------------
    # Pre-select traffic candidates
    # ---------------------------------------------------------

    stats = road_stats.copy()

    stats["valid_q_ratio"] = (
        stats["n_valid_q"]
        / stats["n_observations"]
    )

    stats["valid_k_ratio"] = (
        stats["n_valid_k"]
        / stats["n_observations"]
    )

    candidate_mask = (
        (
            stats["n_observations"]
            >= MIN_OBSERVATIONS
        )
        & (
            stats["valid_q_ratio"]
            >= MIN_VALID_RATIO
        )
        & (
            stats["valid_k_ratio"]
            >= MIN_VALID_RATIO
        )
    )

    candidate_ids = (
        stats.loc[
            candidate_mask,
            "iu_ac",
        ]
        .astype(str)
        .tolist()
    )

    print(
        f"Traffic candidates: "
        f"{len(candidate_ids)}"
    )

    # ---------------------------------------------------------
    # Download road reference only for candidates
    # ---------------------------------------------------------

    print(
        "\n=== Fetching road reference ==="
    )

    raw_reference = fetch_road_reference(
        candidate_ids,
    )

    reference = normalize_road_reference(
        raw_reference
    )

    print(
        f"Normalized reference roads: "
        f"{reference['iu_ac'].nunique()}"
    )

    # ---------------------------------------------------------
    # Final eligibility
    # ---------------------------------------------------------

    eligible = select_eligible_roads(
        road_stats,
        reference,
        min_observations=MIN_OBSERVATIONS,
        min_valid_ratio=MIN_VALID_RATIO,
    )

    eligible_ids = set(
        eligible["iu_ac"]
        .astype(str)
        .tolist()
    )

    # Keep complete road-reference columns.
    final_reference = (
        reference[
            reference["iu_ac"]
            .astype(str)
            .isin(eligible_ids)
        ]
        .copy()
        .sort_values("iu_ac")
        .reset_index(drop=True)
    )

    print(
        f"Final eligible roads: "
        f"{len(final_reference)}"
    )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    required_columns = [
        "latitude",
        "longitude",
        "road_length_m",
        "geo_shape",
    ]

    missing = (
        final_reference[
            required_columns
        ]
        .isna()
        .sum()
    )

    print(
        "\n=== Missing values ==="
    )
    print(missing)

    print(
        "\n=== Road length statistics ==="
    )
    print(
        final_reference[
            "road_length_m"
        ].describe()
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    save_road_reference(
        final_reference,
        OUTPUT_PATH,
    )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()