from __future__ import annotations

from collections.abc import Iterable

from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from traffic_prediction.storage.models import (
    Prediction,
    TrafficObservation,
)
import pandas as pd


def upsert_traffic_observations(
    session: Session,
    rows: Iterable[dict],
) -> None:
    rows = list(rows)

    if not rows:
        return

    statement = insert(
        TrafficObservation
    ).values(rows)

    statement = statement.on_conflict_do_update(
        constraint="uq_traffic_observation",
        set_={
            "q": statement.excluded.q,
            "k": statement.excluded.k,
        },
    )

    session.execute(statement)

def get_recent_traffic_dataframe(
    session: Session,
    hours: int = 48,
) -> pd.DataFrame:
    latest_timestamp = get_latest_traffic_timestamp(
        session
    )

    if latest_timestamp is None:
        return pd.DataFrame()

    start_timestamp = (
        pd.Timestamp(latest_timestamp)
        - pd.Timedelta(hours=hours)
    )

    statement = (
        select(
            TrafficObservation.iu_ac,
            TrafficObservation.timestamp_utc,
            TrafficObservation.q,
            TrafficObservation.k,
        )
        .where(
            TrafficObservation.timestamp_utc
            >= start_timestamp.to_pydatetime()
        )
        .order_by(
            TrafficObservation.timestamp_utc,
            TrafficObservation.iu_ac,
        )
    )

    rows = session.execute(
        statement
    ).mappings().all()

    return pd.DataFrame(rows)


def insert_predictions(
    session: Session,
    rows: Iterable[dict],
) -> None:
    rows = list(rows)

    if not rows:
        return

    statement = insert(
        Prediction
    ).values(rows)

    statement = statement.on_conflict_do_update(
        constraint="uq_prediction",
        set_={
            "predicted_k":
                statement.excluded.predicted_k,
            "prediction_timestamp_utc":
                statement.excluded.prediction_timestamp_utc,
        },
    )

    session.execute(statement)


def get_latest_traffic_observations(
    session: Session,
    iu_ac: str,
    limit: int = 24,
) -> list[TrafficObservation]:
    statement = (
        select(TrafficObservation)
        .where(
            TrafficObservation.iu_ac == iu_ac
        )
        .order_by(
            TrafficObservation.timestamp_utc.desc()
        )
        .limit(limit)
    )

    return list(
        session.scalars(statement)
    )

def upsert_traffic_dataframe(
    session: Session,
    df: pd.DataFrame,
    batch_size: int = 5000,
) -> int:
    required_columns = {
        "iu_ac",
        "timestamp_utc",
        "q",
        "k",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing traffic columns: {sorted(missing)}"
        )

    records = df[
        [
            "iu_ac",
            "timestamp_utc",
            "q",
            "k",
        ]
    ].copy()

    records["timestamp_utc"] = pd.to_datetime(
        records["timestamp_utc"],
        utc=True,
    )

    # PostgreSQL NULL instead of pandas NaN
    records = records.astype(object).where(
        pd.notna(records),
        None,
    )

    total_rows = len(records)

    for start in range(
        0,
        total_rows,
        batch_size,
    ):
        batch = records.iloc[
            start:start + batch_size
        ]

        rows = batch.to_dict(
            orient="records"
        )

        upsert_traffic_observations(
            session,
            rows,
        )

        # Flush each batch to PostgreSQL.
        session.flush()

        print(
            f"Upserted "
            f"{min(start + batch_size, total_rows)}"
            f"/{total_rows}"
        )

    return total_rows

def get_latest_traffic_timestamp(
    session: Session,
) -> datetime | None:
    statement = select(
        func.max(
            TrafficObservation.timestamp_utc
        )
    )

    return session.scalar(statement)

def get_latest_predictions(
    session: Session,
) -> list[Prediction]:
    latest_target = session.scalar(
        select(
            func.max(
                Prediction.target_timestamp_utc
            )
        )
    )

    if latest_target is None:
        return []

    statement = (
        select(Prediction)
        .where(
            Prediction.target_timestamp_utc
            == latest_target
        )
        .order_by(
            Prediction.iu_ac
        )
    )

    return list(
        session.scalars(statement)
    )


def get_latest_prediction_for_road(
    session: Session,
    iu_ac: str,
) -> Prediction | None:
    statement = (
        select(Prediction)
        .where(
            Prediction.iu_ac == iu_ac
        )
        .order_by(
            Prediction.target_timestamp_utc.desc()
        )
        .limit(1)
    )

    return session.scalar(statement)