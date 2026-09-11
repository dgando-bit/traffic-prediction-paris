"""add prediction horizon

Revision ID: 60ae55549c67
Revises: 3a53c9eebabf
Create Date: 2026-09-11 11:31:10.164981

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "60ae55549c67"
down_revision: Union[str, None] = "3a53c9eebabf"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Existing predictions were all +1h predictions.
    op.add_column(
        "predictions",
        sa.Column(
            "horizon_hours",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("1"),
        ),
    )

    op.create_check_constraint(
        "ck_prediction_horizon_positive",
        "predictions",
        "horizon_hours > 0",
    )

    # Replace the old uniqueness rule.
    op.drop_constraint(
        "uq_prediction",
        "predictions",
        type_="unique",
    )

    op.create_unique_constraint(
        "uq_prediction",
        "predictions",
        [
            "iu_ac",
            "prediction_timestamp_utc",
            "horizon_hours",
            "model_version",
        ],
    )

    op.create_index(
        "ix_predictions_horizon_hours",
        "predictions",
        ["horizon_hours"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_predictions_horizon_hours",
        table_name="predictions",
    )

    op.drop_constraint(
        "uq_prediction",
        "predictions",
        type_="unique",
    )

    op.create_unique_constraint(
        "uq_prediction",
        "predictions",
        [
            "iu_ac",
            "target_timestamp_utc",
            "model_version",
        ],
    )

    op.drop_constraint(
        "ck_prediction_horizon_positive",
        "predictions",
        type_="check",
    )

    op.drop_column(
        "predictions",
        "horizon_hours",
    )