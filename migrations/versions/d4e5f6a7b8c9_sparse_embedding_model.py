"""Record the sparse embedding model for each index version.

Revision ID: d4e5f6a7b8c9
Revises: f8d9e0a1b2c3
Create Date: 2026-09-25
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, Sequence[str], None] = "f8d9e0a1b2c3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Historical per-index overrides were not persisted, so preserve uncertainty.
    op.add_column(
        "index_versions",
        sa.Column("sparse_embedding_model", sa.String(length=255), nullable=True),
    )
    op.execute("UPDATE index_versions SET sparse_embedding_model = ''")
    with op.batch_alter_table("index_versions") as batch:
        batch.alter_column(
            "sparse_embedding_model",
            existing_type=sa.String(length=255),
            nullable=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("index_versions") as batch:
        batch.drop_column("sparse_embedding_model")
