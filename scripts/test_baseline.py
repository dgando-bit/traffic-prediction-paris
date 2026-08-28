import pandas as pd

from traffic_prediction.models.baseline import (
    evaluate_persistence_baseline,
    temporal_train_test_split,
)


INPUT_PATH = (
    "data/processed/ml_dataset_4836.parquet"
)


def main() -> None:
    # ---------------------------------------------------------------
    # Load ML dataset
    # ---------------------------------------------------------------

    df = pd.read_parquet(INPUT_PATH)

    print("=== ML Dataset ===")
    print(f"Nombre de lignes : {len(df)}")

    print(
        "Période :",
        df["timestamp_utc"].min(),
        "->",
        df["timestamp_utc"].max(),
    )

    # ---------------------------------------------------------------
    # Temporal split
    # ---------------------------------------------------------------

    train_df, test_df = temporal_train_test_split(
        df,
        test_size=0.2,
    )

    print()
    print("=== Temporal split ===")

    print(
        f"Train : {len(train_df)} lignes"
    )

    print(
        train_df["timestamp_utc"].min(),
        "->",
        train_df["timestamp_utc"].max(),
    )

    print()

    print(
        f"Test : {len(test_df)} lignes"
    )

    print(
        test_df["timestamp_utc"].min(),
        "->",
        test_df["timestamp_utc"].max(),
    )

    # ---------------------------------------------------------------
    # Baseline
    # ---------------------------------------------------------------

    metrics = evaluate_persistence_baseline(
        test_df
    )

    print()
    print("=== Persistence baseline ===")

    print(
        f"MAE     : {metrics.mae:.4f}"
    )

    print(
        f"RMSE    : {metrics.rmse:.4f}"
    )

    print(
        f"Samples : {metrics.n_samples}"
    )

    # ---------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------

    print()
    print("=== Contrôles ===")

    assert len(train_df) > 0
    assert len(test_df) > 0

    assert (
        train_df["timestamp_utc"].max()
        < test_df["timestamp_utc"].min()
    )

    assert metrics.mae >= 0
    assert metrics.rmse >= 0
    assert metrics.n_samples > 0

    print("✅ Persistence baseline test OK")


if __name__ == "__main__":
    main()