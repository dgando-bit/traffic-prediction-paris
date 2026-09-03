from traffic_prediction.features.time_features import (
    add_time_features,
)
from traffic_prediction.features.traffic_features import (
    add_traffic_lag_features,
)
from traffic_prediction.features.road_features import (
    add_road_features,
)
from traffic_prediction.storage.database import (
    get_db_session,
)
from traffic_prediction.storage.repositories import (
    get_recent_traffic_dataframe,
)

import pandas as pd


ROAD_REFERENCE_PATH = (
    "data/raw/reference/road_reference.parquet"
)


def main() -> None:
    with get_db_session() as session:
        df = get_recent_traffic_dataframe(
            session,
            hours=48,
        )

    print(
        f"Loaded {len(df)} rows from PostgreSQL"
    )

    if df.empty:
        print("No traffic data available.")
        return

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
    )

    df["timestamp_paris"] = (
        df["timestamp_utc"]
        .dt.tz_convert("Europe/Paris")
    )

    df = add_time_features(
        df
    )

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

    print()
    print(
        "Latest timestamp:",
        latest_timestamp,
    )

    print(
        "Prediction rows:",
        len(prediction_df),
    )

    columns = [
        "iu_ac",
        "timestamp_utc",
        "q",
        "k",
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

    print()
    print(
        prediction_df[
            columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Missing values:"
    )

    print(
        prediction_df[
            columns
        ].isna().sum()
    )


if __name__ == "__main__":
    main()