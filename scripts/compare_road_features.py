from pathlib import Path

import pandas as pd

from traffic_prediction.models.baseline import (
    temporal_train_test_split,
)
from traffic_prediction.models.evaluate import (
    evaluate_regression,
)
from traffic_prediction.models.lightgbm import (
    train_lightgbm,
)


DATASET_PATH = Path(
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


ROAD_CONTEXT_FEATURES = [
    *TRAFFIC_FEATURES,
    "latitude",
    "longitude",
    "road_length_m",
]


def evaluate_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
):
    model = train_lightgbm(
        train_df=train_df,
        feature_columns=feature_columns,
        target_column=TARGET_COLUMN,
    )

    predictions = model.predict(
        test_df[feature_columns]
    )

    return evaluate_regression(
        y_true=test_df[TARGET_COLUMN],
        y_pred=predictions,
    )


def main() -> None:
    df = pd.read_parquet(
        DATASET_PATH
    )

    print("=== Dataset ===")
    print(f"Rows : {len(df)}")
    print(
        f"Roads: {df['iu_ac'].nunique()}"
    )

    print(
        "Period:",
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
        f"Train rows: {len(train_df)}"
    )

    print(
        f"Test rows : {len(test_df)}"
    )

    print(
        "Train period:",
        train_df["timestamp_utc"].min(),
        "->",
        train_df["timestamp_utc"].max(),
    )

    print(
        "Test period:",
        test_df["timestamp_utc"].min(),
        "->",
        test_df["timestamp_utc"].max(),
    )

    # ---------------------------------------------------------------
    # Common evaluation rows
    # ---------------------------------------------------------------

    test_eval = test_df[
        test_df[TARGET_COLUMN].notna()
    ].copy()

    print()
    print(
        "Evaluation rows:",
        len(test_eval),
    )

    # ---------------------------------------------------------------
    # V1 model
    # ---------------------------------------------------------------

    print()
    print(
        "=== LightGBM V1: traffic only ==="
    )

    traffic_metrics = evaluate_model(
        train_df=train_df,
        test_df=test_eval,
        feature_columns=TRAFFIC_FEATURES,
    )

    print(
        f"Features: {len(TRAFFIC_FEATURES)}"
    )
    print(
        f"MAE : {traffic_metrics.mae:.4f}"
    )
    print(
        f"RMSE: {traffic_metrics.rmse:.4f}"
    )

    # ---------------------------------------------------------------
    # Road context model
    # ---------------------------------------------------------------

    print()
    print(
        "=== LightGBM V1 + road context ==="
    )

    road_metrics = evaluate_model(
        train_df=train_df,
        test_df=test_eval,
        feature_columns=ROAD_CONTEXT_FEATURES,
    )

    print(
        f"Features: "
        f"{len(ROAD_CONTEXT_FEATURES)}"
    )
    print(
        f"MAE : {road_metrics.mae:.4f}"
    )
    print(
        f"RMSE: {road_metrics.rmse:.4f}"
    )

    # ---------------------------------------------------------------
    # Improvement
    # ---------------------------------------------------------------

    mae_gain = (
        (
            traffic_metrics.mae
            - road_metrics.mae
        )
        / traffic_metrics.mae
        * 100
    )

    rmse_gain = (
        (
            traffic_metrics.rmse
            - road_metrics.rmse
        )
        / traffic_metrics.rmse
        * 100
    )

    print()
    print("=== Road context contribution ===")

    print(
        f"MAE improvement : "
        f"{mae_gain:+.2f}%"
    )

    print(
        f"RMSE improvement: "
        f"{rmse_gain:+.2f}%"
    )


if __name__ == "__main__":
    main()