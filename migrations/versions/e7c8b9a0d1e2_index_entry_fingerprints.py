"""Add index entry validation fingerprints.

Revision ID: e7c8b9a0d1e2
Revises: 9a9e51c8d4f0
Create Date: 2026-07-26
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e7c8b9a0d1e2"
down_revision: Union[str, Sequence[str], None] = "9a9e51c8d4f0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    for column_name in (
        "manifest_entries_sha256",
        "validated_pg_entries_sha256",
        "validated_qdrant_entries_sha256",
    ):
        op.add_column(
            "index_versions",
            sa.Column(column_name, sa.String(length=64), nullable=True),
        )
        op.execute(f"UPDATE index_versions SET {column_name} = ''")
        with op.batch_alter_table("index_versions") as batch:
            batch.alter_column(
                column_name,
                existing_type=sa.String(length=64),
                nullable=False,
            )


def downgrade() -> None:
    with op.batch_alter_table("index_versions") as batch:
        batch.drop_column("validated_qdrant_entries_sha256")
        batch.drop_column("validated_pg_entries_sha256")
        batch.drop_column("manifest_entries_sha256")
