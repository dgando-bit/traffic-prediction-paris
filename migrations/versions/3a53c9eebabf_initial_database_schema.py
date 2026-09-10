"""initial database schema

Revision ID: 3a53c9eebabf
Revises:
Create Date: 2026-09-10 10:07:17.109050

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "3a53c9eebabf"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "traffic_observations",
        sa.Column(
            "id",
            sa.Integer(),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "iu_ac",
            sa.String(),
            nullable=False,
        ),
        sa.Column(
            "timestamp_utc",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "q",
            sa.Float(),
            nullable=True,
        ),
        sa.Column(
            "k",
            sa.Float(),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "iu_ac",
            "timestamp_utc",
            name="uq_traffic_observation",
        ),
    )

    op.create_index(
        op.f("ix_traffic_observations_iu_ac"),
        "traffic_observations",
        ["iu_ac"],
        unique=False,
    )

    op.create_index(
        op.f("ix_traffic_observations_timestamp_utc"),
        "traffic_observations",
        ["timestamp_utc"],
        unique=False,
    )

    op.create_table(
        "road_segments",
        sa.Column(
            "id",
            sa.Integer(),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "iu_ac",
            sa.String(),
            nullable=False,
        ),
        sa.Column(
            "libelle",
            sa.String(),
            nullable=True,
        ),
        sa.Column(
            "latitude",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "longitude",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "road_length_m",
            sa.Float(),
            nullable=True,
        ),
        sa.Column(
            "geo_shape",
            sa.Text(),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("iu_ac"),
    )

    op.create_index(
        op.f("ix_road_segments_iu_ac"),
        "road_segments",
        ["iu_ac"],
        unique=True,
    )

    op.create_table(
        "predictions",
        sa.Column(
            "id",
            sa.Integer(),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "iu_ac",
            sa.String(),
            nullable=False,
        ),
        sa.Column(
            "prediction_timestamp_utc",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "target_timestamp_utc",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "predicted_k",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "model_version",
            sa.String(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "iu_ac",
            "target_timestamp_utc",
            "model_version",
            name="uq_prediction",
        ),
    )

    op.create_index(
        op.f("ix_predictions_iu_ac"),
        "predictions",
        ["iu_ac"],
        unique=False,
    )

    op.create_index(
        op.f("ix_predictions_prediction_timestamp_utc"),
        "predictions",
        ["prediction_timestamp_utc"],
        unique=False,
    )

    op.create_index(
        op.f("ix_predictions_target_timestamp_utc"),
        "predictions",
        ["target_timestamp_utc"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_predictions_target_timestamp_utc"),
        table_name="predictions",
    )

    op.drop_index(
        op.f("ix_predictions_prediction_timestamp_utc"),
        table_name="predictions",
    )

    op.drop_index(
        op.f("ix_predictions_iu_ac"),
        table_name="predictions",
    )

    op.drop_table("predictions")

    op.drop_index(
        op.f("ix_road_segments_iu_ac"),
        table_name="road_segments",
    )

    op.drop_table("road_segments")

    op.drop_index(
        op.f("ix_traffic_observations_timestamp_utc"),
        table_name="traffic_observations",
    )

    op.drop_index(
        op.f("ix_traffic_observations_iu_ac"),
        table_name="traffic_observations",
    )

    op.drop_table("traffic_observations")