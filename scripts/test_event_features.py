from pathlib import Path

import pandas as pd

from traffic_prediction.features.event_features import (
    add_event_features,
)


TRAFFIC_PATH = Path(
    "data/processed/ml_dataset_multi_20_long.parquet"
)

EVENTS_PATH = Path(
    "data/processed/event_occurrences.parquet"
)


def main() -> None:
    traffic_df = pd.read_parquet(
        TRAFFIC_PATH
    )

    events_df = pd.read_parquet(
        EVENTS_PATH
    )

    # Sample distributed across the whole period
    step = max(
        len(traffic_df) // 2000,
        1,
    )

    sample_df = (
        traffic_df.iloc[::step]
        .head(2000)
        .copy()
    )

    result = add_event_features(
        sample_df,
        events_df,
    )

    print()
    print("=== Event features ===")
    print()

    print(
        result[
            [
                "has_event_nearby_target_1h",
                "event_count_nearby_target_1h",
            ]
        ].describe()
    )

    print()

    print(
        "Rows with nearby event:",
        result[
            "has_event_nearby_target_1h"
        ].sum(),
    )

    print(
        "Max nearby events:",
        result[
            "event_count_nearby_target_1h"
        ].max(),
    )


if __name__ == "__main__":
    main()