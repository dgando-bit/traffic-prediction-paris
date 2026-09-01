from pathlib import Path

import pandas as pd

from traffic_prediction.models.baseline import (
    evaluate_persistence_baseline,
    temporal_train_test_split,
)
from traffic_prediction.models.evaluate import evaluate_regression
from traffic_prediction.models.train import train_hist_gradient_boosting


INPUT_PATH = Path(
    "data/processed/ml_dataset_multi_20_long.parquet"
)

TARGET_COLUMN = "target_k_1h"


FEATURE_SETS = {
    "A - No calendar": [
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
    ],

    "B - Public holidays": [
        "q",
        "k",
        "hour",
        "day_of_week",
        "is_weekend",
        "is_public_holiday",
        "is_day_before_public_holiday",
        "is_day_after_public_holiday",
        "q_lag_1h",
        "k_lag_1h",
        "q_lag_2h",
        "k_lag_2h",
        "q_lag_24h",
        "k_lag_24h",
    ],

    "C - Full calendar": [
        "q",
        "k",
        "hour",
        "day_of_week",
        "is_weekend",
        "is_public_holiday",
        "is_day_before_public_holiday",
        "is_day_after_public_holiday",
        "is_school_holiday",
        "q_lag_1h",
        "k_lag_1h",
        "q_lag_2h",
        "k_lag_2h",
        "q_lag_24h",
        "k_lag_24h",
    ],
}


def main() -> None:
    df = pd.read_parquet(INPUT_PATH)

    print("=== Dataset ===")
    print(f"Rows   : {len(df)}")
    print(f"Roads  : {df['iu_ac'].nunique()}")
    print(
        f"Period : {df['timestamp_utc'].min()}"
        f" -> {df['timestamp_utc'].max()}"
    )

    train_df, test_df = temporal_train_test_split(
        df,
        test_size=0.2,
    )

    # Exact same evaluation rows for every experiment.
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

        # train_hist_gradient_boosting expects FEATURE_COLUMNS
        # from ml_dataset.py, so for the ablation study we
        # temporarily train directly here.
        from sklearn.ensemble import (
            HistGradientBoostingRegressor,
        )

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
                "model": name,
                "features": len(features),
                "mae": metrics.mae,
                "rmse": metrics.rmse,
            }
        )

    print()
    print("=== Ablation study ===")
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
            f"{result['model']:<25}"
            f"{result['features']:>10}"
            f"{result['mae']:>12.4f}"
            f"{result['rmse']:>12.4f}"
        )

    print()
    print("=== Delta vs no-calendar model ===")

    reference = results[0]

    for result in results[1:]:
        mae_improvement = (
            (reference["mae"] - result["mae"])
            / reference["mae"]
            * 100
        )

        rmse_improvement = (
            (reference["rmse"] - result["rmse"])
            / reference["rmse"]
            * 100
        )

        print()
        print(result["model"])
        print(
            f"MAE  : {mae_improvement:+.2f}%"
        )
        print(
            f"RMSE : {rmse_improvement:+.2f}%"
        )


if __name__ == "__main__":
    main()