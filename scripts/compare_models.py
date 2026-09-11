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
from traffic_prediction.models.lightgbm import (
    train_lightgbm,
)

INPUT_PATH = Path(
    "data/processed/ml_dataset_multi_20_long.parquet"
)

TARGET_COLUMN = "target_k_1h"

# V1 reference features:
# no calendar and no weather.
FEATURE_COLUMNS = [
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


def main() -> None:
    df = pd.read_parquet(INPUT_PATH)

    train_df, test_df = temporal_train_test_split(
        df,
        test_size=0.2,
    )

    evaluation_df = test_df[
        test_df[TARGET_COLUMN].notna()
        & test_df["k"].notna()
    ].copy()

    print("=== Dataset ===")
    print(f"Rows  : {len(df)}")
    print(f"Train : {len(train_df)}")
    print(f"Test  : {len(evaluation_df)}")
    print(f"Features: {len(FEATURE_COLUMNS)}")

    # ---------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------

    baseline = evaluate_persistence_baseline(
        evaluation_df
    )

    # ---------------------------------------------------------
    # HistGradientBoosting
    # ---------------------------------------------------------

    hgb = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )

    hgb.fit(
        train_df[FEATURE_COLUMNS],
        train_df[TARGET_COLUMN],
    )

    hgb_predictions = hgb.predict(
        evaluation_df[FEATURE_COLUMNS]
    )

    hgb_metrics = evaluate_regression(
        evaluation_df[TARGET_COLUMN],
        hgb_predictions,
    )

    # ---------------------------------------------------------
    # LightGBM
    # ---------------------------------------------------------

    lightgbm = train_lightgbm(
        train_df=train_df,
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
    )

    lightgbm_predictions = lightgbm.predict(
        evaluation_df[FEATURE_COLUMNS]
    )

    lightgbm_metrics = evaluate_regression(
        evaluation_df[TARGET_COLUMN],
        lightgbm_predictions,
    )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print()
    print("=== Model comparison ===")

    print(
        f"{'Model':<25}"
        f"{'MAE':>12}"
        f"{'RMSE':>12}"
    )

    print("-" * 49)

    print(
        f"{'Persistence':<25}"
        f"{baseline.mae:>12.4f}"
        f"{baseline.rmse:>12.4f}"
    )

    print(
        f"{'HistGradientBoosting':<25}"
        f"{hgb_metrics.mae:>12.4f}"
        f"{hgb_metrics.rmse:>12.4f}"
    )

    print(
        f"{'LightGBM':<25}"
        f"{lightgbm_metrics.mae:>12.4f}"
        f"{lightgbm_metrics.rmse:>12.4f}"
    )


if __name__ == "__main__":
    main()