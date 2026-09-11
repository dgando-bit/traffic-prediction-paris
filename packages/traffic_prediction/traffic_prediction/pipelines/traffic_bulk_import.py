from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from traffic_prediction.storage.database import (
    get_db_session,
)
from traffic_prediction.storage.repositories import (
    get_all_road_ids,
    upsert_traffic_dataframe,
)

DEFAULT_BULK_DIR = Path(
    "data/raw/traffic/bulk"
)

DEFAULT_COLUMNS = [
    "iu_ac",
    "t_1h",
    "q",
    "k",
]

DEFAULT_DB_BATCH_SIZE = 5000


def _normalize_bulk_chunk(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize a chunk coming from a Paris Data
    bulk Parquet export.
    """
    required_columns = {
        "iu_ac",
        "t_1h",
        "q",
        "k",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df = df.copy()

    df["iu_ac"] = (
        df["iu_ac"]
        .astype("string")
    )

    df["timestamp_utc"] = (
        pd.to_datetime(
            df["t_1h"],
            utc=True,
            errors="coerce",
        )
    )

    df["q"] = pd.to_numeric(
        df["q"],
        errors="coerce",
    )

    df["k"] = pd.to_numeric(
        df["k"],
        errors="coerce",
    )

    df = df[
        [
            "iu_ac",
            "timestamp_utc",
            "q",
            "k",
        ]
    ]

    df = df.dropna(
        subset=[
            "iu_ac",
            "timestamp_utc",
        ]
    )

    return df


def iter_bulk_parquet_chunks(
    *,
    bulk_dir: str | Path = DEFAULT_BULK_DIR,
    columns: Iterable[str] = DEFAULT_COLUMNS,
):
    """
    Yield one pandas DataFrame per Parquet row group.

    This avoids loading all 6.5 million rows
    into memory at once.
    """
    bulk_dir = Path(
        bulk_dir
    )

    files = sorted(
        bulk_dir.glob("*.parquet")
    )

    if not files:
        raise FileNotFoundError(
            f"No Parquet files found in "
            f"{bulk_dir}"
        )

    for path in files:
        print(
            f"\nReading file: {path.name}"
        )

        parquet_file = pq.ParquetFile(
            path
        )

        print(
            f"Row groups: "
            f"{parquet_file.metadata.num_row_groups}"
        )

        for row_group_index in range(
            parquet_file.metadata.num_row_groups
        ):
            table = (
                parquet_file.read_row_group(
                    row_group_index,
                    columns=list(columns),
                )
            )

            df = table.to_pandas()

            yield (
                path,
                row_group_index,
                df,
            )


def run_bulk_import(
    *,
    bulk_dir: str | Path = DEFAULT_BULK_DIR,
    db_batch_size: int = DEFAULT_DB_BATCH_SIZE,
) -> int:
    """
    Import filtered historical traffic data from
    Parquet bulk exports into PostgreSQL.

    Only road IDs currently present in road_segments
    are kept.

    PostgreSQL upserts make this operation idempotent.
    """
    if db_batch_size <= 0:
        raise ValueError(
            "db_batch_size must be greater than 0"
        )

    with get_db_session() as session:
        eligible_road_ids = set(
            get_all_road_ids(
                session
            )
        )

    if not eligible_road_ids:
        raise RuntimeError(
            "No eligible roads found in "
            "road_segments."
        )

    print(
        f"Eligible roads: "
        f"{len(eligible_road_ids)}"
    )

    total_source_rows = 0
    total_eligible_rows = 0
    total_upserted_rows = 0

    current_file: Path | None = None
    file_source_rows = 0
    file_eligible_rows = 0
    file_upserted_rows = 0

    for (
        path,
        row_group_index,
        raw_df,
    ) in iter_bulk_parquet_chunks(
        bulk_dir=bulk_dir,
    ):
        if (
            current_file is not None
            and path != current_file
        ):
            print(
                f"\nCompleted "
                f"{current_file.name}"
            )

            print(
                f"  Source rows  : "
                f"{file_source_rows:,}"
            )

            print(
                f"  Eligible rows: "
                f"{file_eligible_rows:,}"
            )

            print(
                f"  Upserted rows: "
                f"{file_upserted_rows:,}"
            )

            file_source_rows = 0
            file_eligible_rows = 0
            file_upserted_rows = 0

        current_file = path

        source_rows = len(
            raw_df
        )

        total_source_rows += (
            source_rows
        )

        file_source_rows += (
            source_rows
        )

        df = _normalize_bulk_chunk(
            raw_df
        )

        df = df[
            df["iu_ac"].isin(
                eligible_road_ids
            )
        ].copy()

        eligible_rows = len(
            df
        )

        total_eligible_rows += (
            eligible_rows
        )

        file_eligible_rows += (
            eligible_rows
        )

        print(
            f"  Row group "
            f"{row_group_index + 1}: "
            f"{source_rows:,} source -> "
            f"{eligible_rows:,} eligible"
        )

        if df.empty:
            continue

        with get_db_session() as session:
            upserted = (
                upsert_traffic_dataframe(
                    session,
                    df,
                    batch_size=db_batch_size,
                )
            )

        total_upserted_rows += (
            upserted
        )

        file_upserted_rows += (
            upserted
        )

    if current_file is not None:
        print(
            f"\nCompleted "
            f"{current_file.name}"
        )

        print(
            f"  Source rows  : "
            f"{file_source_rows:,}"
        )

        print(
            f"  Eligible rows: "
            f"{file_eligible_rows:,}"
        )

        print(
            f"  Upserted rows: "
            f"{file_upserted_rows:,}"
        )

    print(
        "\n========================================"
    )

    print(
        "Bulk import completed."
    )

    print(
        f"Source rows   : "
        f"{total_source_rows:,}"
    )

    print(
        f"Eligible rows : "
        f"{total_eligible_rows:,}"
    )

    print(
        f"Upserted rows : "
        f"{total_upserted_rows:,}"
    )

    print(
        "========================================"
    )

    return total_upserted_rows