from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from shared.config import get_settings


def build_database_url() -> str:
    settings = get_settings()

    return (
        f"postgresql+psycopg://"
        f"{settings.postgres_user}:"
        f"{settings.postgres_password}@"
        f"{settings.postgres_host}:"
        f"{settings.postgres_port}/"
        f"{settings.postgres_db}"
    )


def create_db_engine() -> Engine:
    return create_engine(
        build_database_url(),
        pool_pre_ping=True,
    )


ENGINE = create_db_engine()

SessionLocal = sessionmaker(
    bind=ENGINE,
    autoflush=False,
    autocommit=False,
)


@contextmanager
def get_db_session() -> Iterator[Session]:
    session = SessionLocal()

    try:
        yield session
        session.commit()

    except Exception:
        session.rollback()
        raise

    finally:
        session.close()