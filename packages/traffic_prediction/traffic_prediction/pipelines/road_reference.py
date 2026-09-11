from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import delete

from traffic_prediction.storage.database import get_db_session
from traffic_prediction.storage.models import RoadSegment
from traffic_prediction.storage.repositories import upsert_road_segments

DEFAULT_REFERENCE_PATH = Path(
    "data/raw/reference/road_reference.parquet"
)


def sync_road_reference(
    reference_path: str | Path = DEFAULT_REFERENCE_PATH,
) -> int:
    reference_path = Path(reference_path)

    df = pd.read_parquet(reference_path)

    required_columns = [
        "iu_ac",
        "libelle",
        "latitude",
        "longitude",
        "road_length_m",
        "geo_shape",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing road reference columns: {missing_columns}"
        )

    roads = (
        df[required_columns]
        .drop_duplicates(subset=["iu_ac"])
        .copy()
    )

    roads["iu_ac"] = roads["iu_ac"].astype(str)

    rows = roads.to_dict(orient="records")

    eligible_ids = set(
        roads["iu_ac"].astype(str)
    )

    with get_db_session() as session:
        upsert_road_segments(
            session,
            rows,
        )

        if eligible_ids:
            session.execute(
                delete(RoadSegment).where(
                    RoadSegment.iu_ac.not_in(
                        eligible_ids
                    )
                )
            )

    return len(rows)