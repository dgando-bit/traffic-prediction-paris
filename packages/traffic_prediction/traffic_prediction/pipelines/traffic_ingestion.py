from pathlib import Path

import pandas as pd

from traffic_prediction.ingestion.traffic import ingest_traffic_data
from traffic_prediction.storage.database import get_db_session
from traffic_prediction.storage.repositories import (
    get_latest_traffic_timestamp,
    upsert_traffic_dataframe,
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

DEFAULT_OUTPUT_PATH = Path("data/raw/traffic/traffic_latest.parquet")
DEFAULT_MAX_RECORDS = 5000


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

    latest_timestamp = pd.Timestamp(latest_timestamp)

    if latest_timestamp.tzinfo is None:
        latest_timestamp = latest_timestamp.tz_localize("UTC")
    else:
        latest_timestamp = latest_timestamp.tz_convert("UTC")

    timestamp = latest_timestamp.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    return (
        f'{road_filter} '
        f'AND t_1h > "{timestamp}"'
    )


def run_incremental_ingestion(
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    max_records: int = DEFAULT_MAX_RECORDS,
) -> int:
    output_path = Path(output_path)

    with get_db_session() as session:
        latest_timestamp = get_latest_traffic_timestamp(
            session
        )

    print(
        "Latest timestamp in database:",
        latest_timestamp,
    )

    where_clause = build_where_clause(
        latest_timestamp
    )

    print("API where clause:", where_clause)

    parquet_path = ingest_traffic_data(
        output_path=output_path,
        where=where_clause,
        order_by="t_1h ASC",
        max_records=max_records,
    )

    df = pd.read_parquet(parquet_path)

    if df.empty:
        print("No new traffic observations found.")
        return 0

    with get_db_session() as session:
        upsert_traffic_dataframe(
            session,
            df,
        )

    print(
        f"{len(df)} traffic observations "
        "loaded into PostgreSQL."
    )

    return len(df)