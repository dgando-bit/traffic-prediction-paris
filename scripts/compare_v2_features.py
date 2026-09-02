from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

from traffic_prediction.features.calendar_features import (
    add_public_holiday_features,
    add_school_holiday_features,
)
from traffic_prediction.features.event_features import (
    add_event_features,
)
from traffic_prediction.models.lightgbm import (
    train_lightgbm,
)


DATASET_PATH = Path(
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


WEATHER_FEATURES = [
    "temperature_2m_target_1h",
    "relative_humidity_2m_target_1h",
    "precipitation_target_1h",
    "rain_target_1h",
    "wind_speed_10m_target_1h",
    "weather_code_target_1h",
]


ROAD_FEATURES = [
    "latitude",
    "longitude",
    "road_length_m",
]


EVENT_FEATURES = [
    "event_count_nearby_target_1h",
    "has_event_nearby_target_1h",
]


def evaluate_lightgbm(
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


def evaluate_persistence(
    test_df: pd.DataFrame,
) -> dict:
    predictions = test_df["k"]

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
        "model": "Persistence",
        "features": 1,
        "mae": mae,
        "rmse": rmse,
    }


def main() -> None:
    print("Loading dataset...")

    df = pd.read_parquet(
        DATASET_PATH
    )

    events_df = pd.read_parquet(
        EVENTS_PATH
    )

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
    )

    df["timestamp_paris"] = (
	    df["timestamp_utc"]
	    .dt.tz_convert("Europe/Paris")
    )

    print(
        f"Dataset rows: {len(df)}"
    )

    # --------------------------------------------------
    # Calendar features
    # --------------------------------------------------

    columns_before_calendar = set(
        df.columns
    )

    df = add_public_holiday_features(
        df
    )

    df = add_school_holiday_features(
        df
    )

    calendar_features = [
        column
        for column in df.columns
        if column not in columns_before_calendar
    ]

    print(
        "Calendar features:",
        calendar_features,
    )

    # --------------------------------------------------
    # Event features
    # --------------------------------------------------

    df = add_event_features(
        traffic_df=df,
        event_occurrences_df=events_df,
    )

    # --------------------------------------------------
    # Common evaluation dataset
    # --------------------------------------------------

    # Persistence requires current k.
    # We use exactly these same rows for every model.
    df = df[
        df[TARGET_COLUMN].notna()
        & df["k"].notna()
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
    print("=== Common split ===")
    print("Cutoff:", SPLIT_CUTOFF)
    print("Train rows:", len(train_df))
    print("Test rows:", len(test_df))

    print(
        "Event coverage in test:",
        f"{test_df['has_event_nearby_target_1h'].mean() * 100:.2f}%",
    )

    # --------------------------------------------------
    # Feature sets
    # --------------------------------------------------

    feature_sets = {
        "Traffic only":
            TRAFFIC_FEATURES,

        "Traffic + Calendar":
            TRAFFIC_FEATURES
            + calendar_features,

        "Traffic + Weather":
            TRAFFIC_FEATURES
            + WEATHER_FEATURES,

        "Traffic + Road":
            TRAFFIC_FEATURES
            + ROAD_FEATURES,

        "Traffic + Events":
            TRAFFIC_FEATURES
            + EVENT_FEATURES,

        "Traffic + Road + Events":
            TRAFFIC_FEATURES
            + ROAD_FEATURES
            + EVENT_FEATURES,
    }

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    results = [
        evaluate_persistence(
            test_df
        )
    ]

    for name, features in feature_sets.items():
        result = evaluate_lightgbm(
            name=name,
            train_df=train_df,
            test_df=test_df,
            features=features,
        )

        results.append(
            result
        )

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------
    # Improvement vs Traffic-only
    # --------------------------------------------------

    traffic_only = results_df[
        results_df["model"]
        == "Traffic only"
    ].iloc[0]

    results_df[
        "mae_gain_vs_traffic_pct"
    ] = (
        (
            traffic_only["mae"]
            - results_df["mae"]
        )
        / traffic_only["mae"]
        * 100
    )

    results_df[
        "rmse_gain_vs_traffic_pct"
    ] = (
        (
            traffic_only["rmse"]
            - results_df["rmse"]
        )
        / traffic_only["rmse"]
        * 100
    )

    print()
    print(
        "=== Final V2 comparison ==="
    )

    print(
        results_df.to_string(
            index=False,
            formatters={
                "mae": "{:.4f}".format,
                "rmse": "{:.4f}".format,
                "mae_gain_vs_traffic_pct":
                    "{:+.2f}%".format,
                "rmse_gain_vs_traffic_pct":
                    "{:+.2f}%".format,
            },
        )
    )

    # --------------------------------------------------
    # Best LightGBM model
    # --------------------------------------------------

    lightgbm_results = (
        results_df[
            results_df["model"]
            != "Persistence"
        ]
        .sort_values("mae")
    )

    best = lightgbm_results.iloc[0]

    print()
    print("=== Best V2 model ===")
    print("Model:", best["model"])
    print(
        f"MAE: {best['mae']:.4f}"
    )
    print(
        f"RMSE: {best['rmse']:.4f}"
    )


if __name__ == "__main__":
    main()