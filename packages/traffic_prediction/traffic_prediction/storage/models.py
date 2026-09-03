from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
)


class Base(DeclarativeBase):
    pass


class TrafficObservation(Base):
    __tablename__ = "traffic_observations"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    iu_ac: Mapped[str] = mapped_column(
        String,
        nullable=False,
        index=True,
    )

    timestamp_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    q: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    k: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    __table_args__ = (
        UniqueConstraint(
            "iu_ac",
            "timestamp_utc",
            name="uq_traffic_observation",
        ),
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    iu_ac: Mapped[str] = mapped_column(
        String,
        nullable=False,
        index=True,
    )

    prediction_timestamp_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    target_timestamp_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    predicted_k: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    model_version: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "iu_ac",
            "target_timestamp_utc",
            "model_version",
            name="uq_prediction",
        ),
    )