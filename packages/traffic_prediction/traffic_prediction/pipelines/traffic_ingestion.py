from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from traffic_prediction.ingestion.traffic import (
    ingest_traffic_data,
)
from traffic_prediction.storage.database import (
    get_db_session,
)
from traffic_prediction.storage.repositories import (
    get_all_road_ids,
    get_latest_traffic_timestamps_by_road,
    upsert_traffic_dataframe,
)

DEFAULT_OUTPUT_PATH = Path(
    "data/raw/traffic/traffic_latest.parquet"
)

DEFAULT_MAX_RECORDS = 5000

DEFAULT_BOOTSTRAP_HOURS = 72

DEFAULT_ROAD_BATCH_SIZE = 25


def build_where_clause(
    latest_timestamp: pd.Timestamp | None,
    road_ids: Sequence[str] | None = None,
) -> str | None:
    """
    Build a Paris Open Data API where clause.

    Parameters
    ----------
    latest_timestamp:
        Only observations newer than this timestamp are fetched.
        If None, no timestamp filter is applied.

    road_ids:
        Optional road segment identifiers.

    Returns
    -------
    str | None
        API where clause, or None when there is no filter.
    """
    filters: list[str] = []

    if road_ids:
        road_filter = " OR ".join(
            f'iu_ac="{road_id}"'
            for road_id in road_ids
        )

        filters.append(
            f"({road_filter})"
        )

    if latest_timestamp is not None:
        timestamp = pd.Timestamp(
            latest_timestamp
        )

        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(
                "UTC"
            )
        else:
            timestamp = timestamp.tz_convert(
                "UTC"
            )

        formatted_timestamp = timestamp.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        filters.append(
            f't_1h > "{formatted_timestamp}"'
        )

    if not filters:
        return None

    return " AND ".join(filters)


def group_roads_by_latest_timestamp(
    road_ids: Sequence[str],
    latest_timestamps: dict[str, object],
) -> dict[pd.Timestamp | None, list[str]]:
    """
    Group roads sharing the same latest observation timestamp.

    This keeps a per-road ingestion cursor while avoiding
    one HTTP request per road.
    """
    grouped: dict[
        pd.Timestamp | None,
        list[str],
    ] = defaultdict(list)

    for road_id in road_ids:
        timestamp = latest_timestamps.get(
            road_id
        )

        if timestamp is None:
            grouped[None].append(
                road_id
            )
            continue

        timestamp = pd.Timestamp(
            timestamp
        )

        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(
                "UTC"
            )
        else:
            timestamp = timestamp.tz_convert(
                "UTC"
            )

        grouped[timestamp].append(
            road_id
        )

    return dict(grouped)


def chunk_road_ids(
    road_ids: Sequence[str],
    batch_size: int,
) -> list[list[str]]:
    """
    Split road identifiers into smaller batches.
    """
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than 0"
        )

    return [
        list(
            road_ids[
                index:index + batch_size
            ]
        )
        for index in range(
            0,
            len(road_ids),
            batch_size,
        )
    ]


def run_incremental_ingestion(
    *,
    road_ids: Sequence[str] | None = None,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    max_records: int = DEFAULT_MAX_RECORDS,
    bootstrap_hours: int = DEFAULT_BOOTSTRAP_HOURS,
    road_batch_size: int = DEFAULT_ROAD_BATCH_SIZE,
) -> int:
    """
    Incrementally ingest Paris traffic observations.

    When road_ids is omitted, the list of eligible road segments
    is automatically loaded from PostgreSQL.

    Existing roads use their latest observation timestamp as
    their incremental ingestion cursor.

    Roads without traffic history receive a bounded bootstrap
    window.

    Roads sharing the same cursor are grouped together and then
    split into smaller API batches.

    Existing-road groups are fully paginated without an artificial
    max_records limit.

    Bootstrap groups keep max_records as a safety limit.
    """
    output_path = Path(
        output_path
    )

    # ---------------------------------------------------------
    # Resolve eligible roads
    # ---------------------------------------------------------

    if road_ids is None:
        with get_db_session() as session:
            road_ids = get_all_road_ids(
                session
            )

    if not road_ids:
        print(
            "No road IDs available."
        )
        return 0

    road_ids = list(
        dict.fromkeys(
            str(road_id)
            for road_id in road_ids
        )
    )

    print(
        f"Roads selected: {len(road_ids)}"
    )

    # ---------------------------------------------------------
    # Read per-road ingestion cursors
    # ---------------------------------------------------------

    with get_db_session() as session:
        latest_timestamps = (
            get_latest_traffic_timestamps_by_road(
                session,
                road_ids,
            )
        )

    known_road_ids = [
        road_id
        for road_id in road_ids
        if road_id in latest_timestamps
    ]

    new_road_ids = [
        road_id
        for road_id in road_ids
        if road_id not in latest_timestamps
    ]

    print(
        f"Known roads: {len(known_road_ids)}"
    )

    print(
        f"New roads: {len(new_road_ids)}"
    )

    # ---------------------------------------------------------
    # Group existing roads by cursor
    # ---------------------------------------------------------

    grouped_roads = (
        group_roads_by_latest_timestamp(
            known_road_ids,
            latest_timestamps,
        )
    )

    # ---------------------------------------------------------
    # Bootstrap new roads
    # ---------------------------------------------------------

    bootstrap_group_timestamp: pd.Timestamp | None = None

    if new_road_ids:
        now_utc = pd.Timestamp.now(
            tz="UTC"
        )

        bootstrap_group_timestamp = (
            now_utc
            - pd.Timedelta(
                hours=bootstrap_hours
            )
        )

        print(
            "Bootstrap timestamp:",
            bootstrap_group_timestamp,
        )

        grouped_roads[
            bootstrap_group_timestamp
        ] = new_road_ids

    print(
        f"Timestamp groups: "
        f"{len(grouped_roads)}"
    )

    # ---------------------------------------------------------
    # Ingest every group
    # ---------------------------------------------------------

    total_loaded = 0

    for group_index, (
        latest_timestamp,
        group_road_ids,
    ) in enumerate(
        grouped_roads.items(),
        start=1,
    ):
        road_batches = chunk_road_ids(
            group_road_ids,
            road_batch_size,
        )

        is_bootstrap_group = (
            bootstrap_group_timestamp is not None
            and latest_timestamp
            == bootstrap_group_timestamp
        )

        for batch_index, batch_road_ids in enumerate(
            road_batches,
            start=1,
        ):
            print(
                f"\nGroup {group_index}/"
                f"{len(grouped_roads)} "
                f"- batch {batch_index}/"
                f"{len(road_batches)}"
            )

            print(
                "Roads:",
                len(batch_road_ids),
            )

            print(
                "Latest timestamp:",
                latest_timestamp,
            )

            print(
                "Bootstrap:",
                is_bootstrap_group,
            )

            where_clause = build_where_clause(
                latest_timestamp=latest_timestamp,
                road_ids=batch_road_ids,
            )

            print(
                "API where clause:",
                where_clause,
            )

            # Existing roads must not be silently truncated.
            # Bootstrap batches are bounded for safety.
            batch_max_records = (
                max_records
                if is_bootstrap_group
                else None
            )

            print(
                "Max records:",
                batch_max_records,
            )

            group_output_path = (
                output_path.parent
                / (
                    f"{output_path.stem}"
                    f"_group_{group_index}"
                    f"_batch_{batch_index}"
                    f"{output_path.suffix}"
                )
            )

            parquet_path = ingest_traffic_data(
                output_path=group_output_path,
                where=where_clause,
                order_by="t_1h ASC",
                max_records=batch_max_records,
            )

            df = pd.read_parquet(
                parquet_path
            )

            if df.empty:
                print(
                    "No new observations "
                    "for this batch."
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
                "loaded for this batch."
            )

    print(
        f"\nTotal loaded: "
        f"{total_loaded} observations."
    )

    return total_loaded