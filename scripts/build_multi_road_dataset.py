from pathlib import Path

import pandas as pd

from traffic_prediction.features.build_features import (
    build_traffic_features,
)
from traffic_prediction.features.ml_dataset import (
    build_ml_dataset,
)


INPUT_PATH = Path(
    "data/interim/traffic_multi_20_clean.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/ml_dataset_multi_20.parquet"
)


def main() -> None:
    # ---------------------------------------------------------------
    # Load processed data
    # ---------------------------------------------------------------

    df = pd.read_parquet(INPUT_PATH)

    print("=== Input dataset ===")
    print(f"Rows  : {len(df)}")
    print(f"Roads : {df['iu_ac'].nunique()}")

    print(
        "Period:",
        df["timestamp_utc"].min(),
        "->",
        df["timestamp_utc"].max(),
    )

    # ---------------------------------------------------------------
    # Feature engineering
    # ---------------------------------------------------------------

    df_features = build_traffic_features(df)

    print()
    print("=== Feature engineering ===")

    print(f"Rows  : {len(df_features)}")
    print(f"Roads : {df_features['iu_ac'].nunique()}")

    lag_columns = [
        "q_lag_1h",
        "k_lag_1h",
        "q_lag_2h",
        "k_lag_2h",
        "q_lag_24h",
        "k_lag_24h",
    ]

    print()
    print("Missing lag values:")

    print(
        df_features[
            lag_columns
        ].isna().sum()
    )

    print()
    print(
        "Available targets:",
        df_features[
            "target_k_1h"
        ].notna().sum(),
    )

    print(
        "Missing targets:",
        df_features[
            "target_k_1h"
        ].isna().sum(),
    )

    # ---------------------------------------------------------------
    # ML dataset
    # ---------------------------------------------------------------

    ml_dataset = build_ml_dataset(
        df_features
    )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ml_dataset.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    print()
    print("=== ML Dataset ===")

    print(
        f"Rows  : {len(ml_dataset)}"
    )

    print(
        f"Roads : "
        f"{ml_dataset['iu_ac'].nunique()}"
    )

    print(
        "Period:",
        ml_dataset["timestamp_utc"].min(),
        "->",
        ml_dataset["timestamp_utc"].max(),
    )

    print(
        "Missing target:",
        ml_dataset[
            "target_k_1h"
        ].isna().sum(),
    )

    print()
    print("Rows per road:")

    print(
        ml_dataset.groupby("iu_ac")
        .size()
        .sort_values(
            ascending=False
        )
    )

    input_roads = set(df["iu_ac"].unique())
    ml_roads = set(ml_dataset["iu_ac"].unique())

    excluded_roads = sorted(
        input_roads - ml_roads
    )

    print()
    print("Roads excluded from ML dataset:")

    for iu_ac in excluded_roads:
        print(f"  - {iu_ac}")

    # ---------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------

    assert len(ml_dataset) > 0

    assert (
            1
            < ml_dataset["iu_ac"].nunique()
            <= df["iu_ac"].nunique()
    )

    assert (
        ml_dataset["target_k_1h"]
        .notna()
        .all()
    )

    assert (
        len(ml_dataset)
        <= len(df_features)
    )

    print()
    print(
        f"Dataset saved: {OUTPUT_PATH}"
    )

    print(
        "✅ Multi-road ML dataset test OK"
    )


if __name__ == "__main__":
    main()