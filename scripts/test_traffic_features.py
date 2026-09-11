from pathlib import Path

import pandas as pd
from traffic_prediction.features.build_features import (
    build_traffic_features,
)
from traffic_prediction.features.ml_dataset import (
    build_ml_dataset,
)

INPUT_PATH = "data/interim/traffic_4836_clean.parquet"


def main() -> None:
    # ------------------------------------------------------------------
    # Load processed traffic data
    # ------------------------------------------------------------------

    df = pd.read_parquet(INPUT_PATH)

    print("=== Input dataset ===")
    print(f"Nombre de lignes : {len(df)}")
    print(f"Nombre de iu_ac : {df['iu_ac'].nunique()}")

    print(
        "Période :",
        df["timestamp_utc"].min(),
        "->",
        df["timestamp_utc"].max(),
    )

    # ------------------------------------------------------------------
    # Build features
    # ------------------------------------------------------------------

    df_features = build_traffic_features(df)

    # ------------------------------------------------------------------
    # Display resulting features
    # ------------------------------------------------------------------

    columns = [
        "iu_ac",
        "timestamp_paris",
        "q",
        "k",
        "hour",
        "day_of_week",
        "is_weekend",
        "q_lag_1h",
        "k_lag_1h",
        "q_lag_2h",
        "k_lag_2h",
        "q_lag_24h",
        "k_lag_24h",
        "target_k_1h",
    ]

    print()
    print("=== Features ===")

    print(
        df_features[columns].tail(30)
    )

    # ------------------------------------------------------------------
    # Missing lag values
    # ------------------------------------------------------------------

    lag_columns = [
        "q_lag_1h",
        "k_lag_1h",
        "q_lag_2h",
        "k_lag_2h",
        "q_lag_24h",
        "k_lag_24h",
    ]

    print()
    print("=== Valeurs manquantes des lags ===")

    print(
        df_features[
            lag_columns
        ]
        .isna()
        .sum()
    )

    # ------------------------------------------------------------------
    # Target statistics
    # ------------------------------------------------------------------

    print()
    print("=== Target ===")

    target_available = (
        df_features["target_k_1h"]
        .notna()
        .sum()
    )

    target_missing = (
        df_features["target_k_1h"]
        .isna()
        .sum()
    )

    print(
        "Target k+1h disponible :",
        target_available,
    )

    print(
        "Target k+1h manquante :",
        target_missing,
    )

    print()
    print("=== ML Dataset ===")

    ml_dataset = build_ml_dataset(df_features)

    output_path = Path(
        "data/processed/ml_dataset_4836.parquet"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ml_dataset.to_parquet(
        output_path,
        index=False,
    )

    print(f"Nombre de lignes : {len(ml_dataset)}")
    print(f"Nombre de colonnes : {len(ml_dataset.columns)}")
    print(f"Target manquante : {ml_dataset['target_k_1h'].isna().sum()}")
    print(f"Dataset sauvegardé : {output_path}")

    # ------------------------------------------------------------------
    # Basic validation
    # ------------------------------------------------------------------

    print()
    print("=== Contrôles ===")

    assert len(df_features) == len(df)

    assert df_features["iu_ac"].nunique() == 1
    assert df_features["iu_ac"].iloc[0] == "4836"

    assert "k_lag_1h" in df_features.columns
    assert "k_lag_24h" in df_features.columns
    assert "target_k_1h" in df_features.columns

    assert df_features["k_lag_1h"].notna().any()
    assert df_features["k_lag_24h"].notna().any()
    assert df_features["target_k_1h"].notna().any()

    assert len(ml_dataset) > 0
    assert ml_dataset["target_k_1h"].notna().all()
    assert len(ml_dataset) <= len(df_features)

    print("✅ Traffic feature engineering test OK")

if __name__ == "__main__":
    main()