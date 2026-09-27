"""Reserve an isolated artifact attempt on each index version.

Revision ID: 2b7f4a9c1d06
Revises: d4e5f6a7b8c9
Create Date: 2026-09-27
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "2b7f4a9c1d06"
down_revision: Union[str, Sequence[str], None] = "d4e5f6a7b8c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Keep a database default while older API and worker binaries may still insert rows.
    op.add_column(
        "index_versions",
        sa.Column(
            "artifact_attempt",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )
    with op.batch_alter_table("index_versions") as batch:
        batch.create_check_constraint(
            "ck_index_versions_artifact_attempt", "artifact_attempt >= 0"
        )


def downgrade() -> None:
    # A value above zero means new-format artifacts could exist outside PostgreSQL.
    positive_attempts = op.get_bind().execute(
        sa.text("SELECT COUNT(*) FROM index_versions WHERE artifact_attempt > 0")
    ).scalar_one()
    if positive_attempts:
        raise RuntimeError(
            "cannot drop artifact_attempt while new-format index artifacts may exist"
        )
    with op.batch_alter_table("index_versions") as batch:
        batch.drop_constraint("ck_index_versions_artifact_attempt", type_="check")
        batch.drop_column("artifact_attempt")
