from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from traffic_prediction.models.baseline import (
    evaluate_persistence_baseline,
    temporal_train_test_split,
)
from traffic_prediction.models.evaluate import (
    evaluate_regression,
)

INPUT_PATH = Path(
    "data/processed/ml_dataset_multi_20_long.parquet"
)

TARGET_COLUMN = "target_k_1h"


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


WEATHER_FEATURES = [
    "temperature_2m_target_1h",
    "relative_humidity_2m_target_1h",
    "precipitation_target_1h",
    "rain_target_1h",
    "wind_speed_10m_target_1h",
    "weather_code_target_1h",
]


FEATURE_SETS = {
    "Traffic only": TRAFFIC_FEATURES,
    "Traffic + Weather": (
        TRAFFIC_FEATURES
        + WEATHER_FEATURES
    ),
}


def main() -> None:
    df = pd.read_parquet(INPUT_PATH)

    print("=== Dataset ===")
    print(f"Rows  : {len(df)}")
    print(f"Roads : {df['iu_ac'].nunique()}")

    print(
        "Period:",
        df["timestamp_utc"].min(),
        "->",
        df["timestamp_utc"].max(),
    )

    train_df, test_df = temporal_train_test_split(
        df,
        test_size=0.2,
    )

    # Same rows for every model.
    evaluation_df = test_df[
        test_df[TARGET_COLUMN].notna()
        & test_df["k"].notna()
    ].copy()

    baseline = evaluate_persistence_baseline(
        evaluation_df
    )

    results = []

    for name, features in FEATURE_SETS.items():
        print()
        print("=" * 60)
        print(name)
        print(f"Features: {len(features)}")
        print("=" * 60)

        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        )

        model.fit(
            train_df[features],
            train_df[TARGET_COLUMN],
        )

        predictions = model.predict(
            evaluation_df[features]
        )

        metrics = evaluate_regression(
            evaluation_df[TARGET_COLUMN],
            predictions,
        )

        results.append(
            {
                "name": name,
                "features": len(features),
                "mae": metrics.mae,
                "rmse": metrics.rmse,
            }
        )

    print()
    print("=== Weather ablation study ===")
    print()

    print(
        f"{'Experiment':<25}"
        f"{'Features':>10}"
        f"{'MAE':>12}"
        f"{'RMSE':>12}"
    )

    print("-" * 59)

    print(
        f"{'Persistence':<25}"
        f"{'-':>10}"
        f"{baseline.mae:>12.4f}"
        f"{baseline.rmse:>12.4f}"
    )

    for result in results:
        print(
            f"{result['name']:<25}"
            f"{result['features']:>10}"
            f"{result['mae']:>12.4f}"
            f"{result['rmse']:>12.4f}"
        )

    traffic = results[0]
    weather = results[1]

    mae_improvement = (
        (traffic["mae"] - weather["mae"])
        / traffic["mae"]
        * 100
    )

    rmse_improvement = (
        (traffic["rmse"] - weather["rmse"])
        / traffic["rmse"]
        * 100
    )

    print()
    print(
        "=== Weather contribution ==="
    )

    print(
        f"MAE  : {mae_improvement:+.2f}%"
    )

    print(
        f"RMSE : {rmse_improvement:+.2f}%"
    )


if __name__ == "__main__":
    main()