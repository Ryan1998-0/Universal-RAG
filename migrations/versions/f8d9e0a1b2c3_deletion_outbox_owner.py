"""Bind deletion outbox records to their creator.

Revision ID: f8d9e0a1b2c3
Revises: e7c8b9a0d1e2
Create Date: 2026-07-26
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f8d9e0a1b2c3"
down_revision: Union[str, Sequence[str], None] = "e7c8b9a0d1e2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "deletion_outbox",
        sa.Column("created_by_user_id", sa.String(length=36), nullable=True),
    )
    op.execute(
        """
        UPDATE deletion_outbox
        SET created_by_user_id = (
            SELECT documents.owner_user_id
            FROM documents
            WHERE documents.id = deletion_outbox.resource_id
              AND documents.tenant_id = deletion_outbox.tenant_id
        )
        """
    )
    with op.batch_alter_table("deletion_outbox") as batch:
        batch.alter_column(
            "created_by_user_id",
            existing_type=sa.String(length=36),
            nullable=False,
        )
        batch.create_foreign_key(
            "fk_deletion_outbox_created_by_user",
            "users",
            ["created_by_user_id"],
            ["id"],
            ondelete="RESTRICT",
        )


def downgrade() -> None:
    with op.batch_alter_table("deletion_outbox") as batch:
        batch.drop_constraint(
            "fk_deletion_outbox_created_by_user",
            type_="foreignkey",
        )
        batch.drop_column("created_by_user_id")
