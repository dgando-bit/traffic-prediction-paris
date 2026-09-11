from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    Integer,
    String,
    Text,
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

class RoadSegment(Base):
    __tablename__ = "road_segments"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    iu_ac: Mapped[str] = mapped_column(
        String,
        nullable=False,
        unique=True,
        index=True,
    )

    libelle: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )

    latitude: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    longitude: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    road_length_m: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    geo_shape: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
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

    horizon_hours: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        server_default="1",
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
        CheckConstraint(
            "horizon_hours > 0",
            name="ck_prediction_horizon_positive",
        ),
        UniqueConstraint(
            "iu_ac",
            "prediction_timestamp_utc",
            "horizon_hours",
            "model_version",
            name="uq_prediction",
        ),
    )