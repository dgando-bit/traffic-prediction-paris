from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from traffic_prediction.ingestion.traffic import ingest_traffic_data
from traffic_prediction.storage.database import get_db_session
from traffic_prediction.storage.repositories import (
    get_latest_traffic_timestamps_by_road,
    upsert_traffic_dataframe,
)


DEFAULT_OUTPUT_PATH = Path(
    "data/raw/traffic/traffic_latest.parquet"
)

DEFAULT_MAX_RECORDS = 5000


def build_where_clause(
    latest_timestamp: pd.Timestamp | None,
    road_ids: Sequence[str] | None = None,
) -> str | None:
    filters: list[str] = []

    if road_ids:
        road_filter = " OR ".join(
            f'iu_ac="{road_id}"'
            for road_id in road_ids
        )
        filters.append(f"({road_filter})")

    if latest_timestamp is not None:
        latest_timestamp = pd.Timestamp(
            latest_timestamp
        )

        if latest_timestamp.tzinfo is None:
            latest_timestamp = (
                latest_timestamp.tz_localize("UTC")
            )
        else:
            latest_timestamp = (
                latest_timestamp.tz_convert("UTC")
            )

        timestamp = latest_timestamp.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        filters.append(
            f't_1h > "{timestamp}"'
        )

    if not filters:
        return None

    return " AND ".join(filters)


def group_roads_by_latest_timestamp(
    road_ids: Sequence[str],
    latest_timestamps: dict[str, object],
) -> dict[pd.Timestamp | None, list[str]]:
    grouped: dict[
        pd.Timestamp | None,
        list[str],
    ] = defaultdict(list)

    for road_id in road_ids:
        timestamp = latest_timestamps.get(
            road_id
        )

        if timestamp is None:
            grouped[None].append(road_id)
            continue

        timestamp = pd.Timestamp(timestamp)

        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(
                "UTC"
            )
        else:
            timestamp = timestamp.tz_convert(
                "UTC"
            )

        grouped[timestamp].append(road_id)

    return dict(grouped)


def run_incremental_ingestion(
    *,
    road_ids: Sequence[str],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    max_records: int = DEFAULT_MAX_RECORDS,
) -> int:
    if not road_ids:
        print("No road IDs provided.")
        return 0

    road_ids = list(dict.fromkeys(road_ids))

    with get_db_session() as session:
        latest_timestamps = (
            get_latest_traffic_timestamps_by_road(
                session,
                road_ids,
            )
        )

    grouped_roads = (
        group_roads_by_latest_timestamp(
            road_ids,
            latest_timestamps,
        )
    )

    print(
        f"{len(road_ids)} roads split into "
        f"{len(grouped_roads)} timestamp groups."
    )

    total_loaded = 0

    for group_index, (
        latest_timestamp,
        group_road_ids,
    ) in enumerate(
        grouped_roads.items(),
        start=1,
    ):
        print(
            f"\nGroup {group_index}/"
            f"{len(grouped_roads)}"
        )

        print(
            "Roads:",
            len(group_road_ids),
        )

        print(
            "Latest timestamp:",
            latest_timestamp,
        )

        where_clause = build_where_clause(
            latest_timestamp=latest_timestamp,
            road_ids=group_road_ids,
        )

        print(
            "API where clause:",
            where_clause,
        )

        group_output_path = (
            output_path.parent
            / (
                f"{output_path.stem}"
                f"_group_{group_index}"
                f"{output_path.suffix}"
            )
        )

        parquet_path = ingest_traffic_data(
            output_path=group_output_path,
            where=where_clause,
            order_by="t_1h ASC",
            max_records=max_records,
        )

        df = pd.read_parquet(
            parquet_path
        )

        if df.empty:
            print(
                "No new observations "
                "for this group."
            )
            continue

        with get_db_session() as session:
            loaded_rows = (
                upsert_traffic_dataframe(
                    session,
                    df,
                )
            )

        total_loaded += loaded_rows

        print(
            f"{loaded_rows} observations "
            "loaded for this group."
        )

    print(
        f"\nTotal loaded: "
        f"{total_loaded} observations."
    )

    return total_loaded