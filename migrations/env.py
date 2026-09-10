from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine
from sqlalchemy import pool

from traffic_prediction.storage.database import build_database_url
from traffic_prediction.storage.models import Base


# Alembic Config object.
config = context.config


# Configure Python logging from alembic.ini.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)


# Metadata used by Alembic autogenerate.
target_metadata = Base.metadata


def get_database_url() -> str:
    """
    Return the PostgreSQL URL used by the application.
    """
    return build_database_url()


def run_migrations_offline() -> None:
    """
    Run migrations without creating a DB connection.

    Alembic generates SQL statements directly from the URL.
    """
    context.configure(
        url=get_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={
            "paramstyle": "named",
        },
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations using a real database connection.
    """
    connectable = create_engine(
        get_database_url(),
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()