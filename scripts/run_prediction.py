from datetime import timedelta

import mlflow.lightgbm
import pandas as pd

from traffic_prediction.features.road_features import (
    add_road_features,
)
from traffic_prediction.features.time_features import (
    add_time_features,
)
from traffic_prediction.features.traffic_features import (
    add_traffic_lag_features,
)
from traffic_prediction.storage.database import (
    get_db_session,
)
from traffic_prediction.storage.repositories import (
    get_recent_traffic_dataframe,
    insert_predictions,
)


ROAD_REFERENCE_PATH = (
    "data/raw/reference/road_reference.parquet"
)

MODEL_RUN_ID = (
    "5881705e77a14c2e991655ebe76c5b06"
)

MODEL_URI = f"runs:/{MODEL_RUN_ID}/model"

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
    "latitude",
    "longitude",
    "road_length_m",
]


def build_prediction_features() -> pd.DataFrame:
    with get_db_session() as session:
        df = get_recent_traffic_dataframe(
            session,
            hours=48,
        )

    if df.empty:
        raise RuntimeError(
            "No traffic observations available."
        )

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
    )

    df["timestamp_paris"] = (
        df["timestamp_utc"]
        .dt.tz_convert("Europe/Paris")
    )

    df = add_time_features(df)

    df = add_traffic_lag_features(
        df,
        lags=(1, 2, 24),
    )

    road_reference_df = pd.read_parquet(
        ROAD_REFERENCE_PATH
    )

    df = add_road_features(
        df,
        road_reference_df,
    )

    latest_timestamp = (
        df["timestamp_utc"].max()
    )

    prediction_df = df[
        df["timestamp_utc"]
        == latest_timestamp
    ].copy()

    return prediction_df


def main() -> None:
    print("Building prediction features...")

    prediction_df = (
        build_prediction_features()
    )

    print(
        f"Prediction rows: {len(prediction_df)}"
    )

    source_timestamp = (
        prediction_df[
            "timestamp_utc"
        ].max()
    )

    target_timestamp = (
        source_timestamp
        + timedelta(hours=1)
    )

    print(
        f"Source timestamp : {source_timestamp}"
    )
    print(
        f"Target timestamp : {target_timestamp}"
    )

    print()
    print(
        f"Loading model: {MODEL_URI}"
    )

    model = mlflow.lightgbm.load_model(
        MODEL_URI
    )

    predictions = model.predict(
        prediction_df[FEATURE_COLUMNS]
    )

    prediction_df["predicted_k"] = (
        predictions
    )

    rows = []

    for _, row in prediction_df.iterrows():
        rows.append(
            {
                "iu_ac": str(
                    row["iu_ac"]
                ),
                "prediction_timestamp_utc":
                    source_timestamp.to_pydatetime(),
                "target_timestamp_utc":
                    target_timestamp.to_pydatetime(),
                "predicted_k": float(
                    row["predicted_k"]
                ),
                "model_version":
                    MODEL_RUN_ID,
            }
        )

    with get_db_session() as session:
        insert_predictions(
            session,
            rows,
        )

    print()
    print(
        "=== Predictions completed ==="
    )

    print(
        prediction_df[
            [
                "iu_ac",
                "k",
                "predicted_k",
            ]
        ]
        .sort_values(
            "predicted_k",
            ascending=False,
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved {len(rows)} predictions "
        "to PostgreSQL"
    )


if __name__ == "__main__":
    main()