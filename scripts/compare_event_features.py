from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from traffic_prediction.features.event_features import (
    add_event_features,
)
from traffic_prediction.models.lightgbm import (
    train_lightgbm,
)

TRAFFIC_PATH = Path(
    "data/processed/ml_dataset_multi_20_long.parquet"
)

EVENTS_PATH = Path(
    "data/processed/event_occurrences.parquet"
)

TARGET_COLUMN = "target_k_1h"

SPLIT_CUTOFF = pd.Timestamp(
    "2026-04-09 16:00:00+00:00"
)

TRAFFIC_FEATURES = [
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
]

EVENT_FEATURES = [
    "event_count_nearby_target_1h",
    "has_event_nearby_target_1h",
]


def evaluate_model(
    name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
) -> dict:
    model = train_lightgbm(
        train_df=train_df,
        feature_columns=features,
        target_column=TARGET_COLUMN,
    )

    predictions = model.predict(
        test_df[features]
    )

    mae = mean_absolute_error(
        test_df[TARGET_COLUMN],
        predictions,
    )

    rmse = (
        mean_squared_error(
            test_df[TARGET_COLUMN],
            predictions,
        )
        ** 0.5
    )

    return {
        "model": name,
        "features": len(features),
        "mae": mae,
        "rmse": rmse,
    }


def main() -> None:
    traffic_df = pd.read_parquet(
        TRAFFIC_PATH
    )

    events_df = pd.read_parquet(
        EVENTS_PATH
    )

    print(
        "Traffic rows:",
        len(traffic_df),
    )

    print(
        "Event occurrences:",
        len(events_df),
    )

    print()
    print("Building event features...")

    df = add_event_features(
        traffic_df=traffic_df,
        event_occurrences_df=events_df,
    )

    df = df[
        df[TARGET_COLUMN].notna()
    ].copy()

    train_df = df[
        df["timestamp_utc"]
        <= SPLIT_CUTOFF
    ].copy()

    test_df = df[
        df["timestamp_utc"]
        > SPLIT_CUTOFF
    ].copy()

    print()
    print("=== Split ===")
    print("Cutoff:", SPLIT_CUTOFF)
    print("Train:", len(train_df))
    print("Test:", len(test_df))

    print()
    print(
        "Test event coverage:",
        f"{test_df['has_event_nearby_target_1h'].mean() * 100:.2f}%",
    )

    traffic_result = evaluate_model(
        name="Traffic only",
        train_df=train_df,
        test_df=test_df,
        features=TRAFFIC_FEATURES,
    )

    event_result = evaluate_model(
        name="Traffic + Events",
        train_df=train_df,
        test_df=test_df,
        features=(
            TRAFFIC_FEATURES
            + EVENT_FEATURES
        ),
    )

    results = pd.DataFrame(
        [
            traffic_result,
            event_result,
        ]
    )

    print()
    print(
        "=== Event feature comparison ==="
    )

    print(
        results.to_string(
            index=False
        )
    )

    baseline_mae = traffic_result["mae"]
    baseline_rmse = traffic_result["rmse"]

    event_mae = event_result["mae"]
    event_rmse = event_result["rmse"]

    mae_gain = (
        (
            baseline_mae
            - event_mae
        )
        / baseline_mae
        * 100
    )

    rmse_gain = (
        (
            baseline_rmse
            - event_rmse
        )
        / baseline_rmse
        * 100
    )

    print()
    print("=== Event contribution ===")

    print(
        f"MAE improvement: {mae_gain:+.2f}%"
    )

    print(
        f"RMSE improvement: {rmse_gain:+.2f}%"
    )


if __name__ == "__main__":
    main()