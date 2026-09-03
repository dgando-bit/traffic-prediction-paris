from pathlib import Path

import pandas as pd

from traffic_prediction.ingestion.traffic import (
    ingest_traffic_data,
)
from traffic_prediction.storage.database import (
    get_db_session,
)
from traffic_prediction.storage.repositories import (
    get_latest_traffic_timestamp,
    upsert_traffic_dataframe,
)


OUTPUT_PATH = Path(
    "data/raw/traffic/traffic_latest.parquet"
)

ROAD_IDS = [
    "4632",
    "4634",
    "1029",
    "1030",
    "1031",
    "1032",
    "1067",
    "1072",
    "4630",
    "4633",
    "4637",
    "985",
    "1033",
    "1034",
    "1111",
    "1112",
    "1222",
    "1043",
    "1044",
    "1038",
]


def build_where_clause(
    latest_timestamp: pd.Timestamp | None,
) -> str:
    road_filter = " OR ".join(
        f'iu_ac="{road_id}"'
        for road_id in ROAD_IDS
    )

    road_filter = f"({road_filter})"

    if latest_timestamp is None:
        return road_filter

    timestamp = (
        latest_timestamp
        .tz_convert("UTC")
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    return (
        f"{road_filter} "
        f'AND t_1h > "{timestamp}"'
    )


def main() -> None:
    print(
        "Starting incremental traffic ingestion..."
    )

    with get_db_session() as session:
        latest_timestamp = (
            get_latest_traffic_timestamp(
                session
            )
        )

    if latest_timestamp is not None:
        latest_timestamp = pd.Timestamp(
            latest_timestamp
        )

        print(
            "Latest database observation: "
            f"{latest_timestamp}"
        )

    where = build_where_clause(
        latest_timestamp
    )

    print(
        f"API filter: {where}"
    )

    parquet_path = ingest_traffic_data(
        output_path=OUTPUT_PATH,
        where=where,
        order_by="t_1h ASC",
        max_records=5000,
    )

    df = pd.read_parquet(
        parquet_path
    )

    if df.empty:
        print(
            "No new traffic observations."
        )
        return

    print(
        f"Fetched {len(df)} rows"
    )

    print(
        "Batch period: "
        f"{df['timestamp_utc'].min()} "
        "→ "
        f"{df['timestamp_utc'].max()}"
    )

    with get_db_session() as session:
        upserted = (
            upsert_traffic_dataframe(
                session,
                df,
            )
        )

    print()
    print(
        "=== Incremental ingestion completed ==="
    )
    print(
        f"Rows fetched  : {len(df)}"
    )
    print(
        f"Rows upserted : {upserted}"
    )
    print(
        "Latest timestamp: "
        f"{df['timestamp_utc'].max()}"
    )


if __name__ == "__main__":
    main()