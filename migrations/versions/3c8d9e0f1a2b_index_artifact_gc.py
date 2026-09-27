"""Add durable cleanup state for isolated index artifact attempts.

Revision ID: 3c8d9e0f1a2b
Revises: 2b7f4a9c1d06
Create Date: 2026-09-27
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "3c8d9e0f1a2b"
down_revision: Union[str, Sequence[str], None] = "2b7f4a9c1d06"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "index_artifact_gc",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("tenant_id", sa.String(length=36), nullable=False),
        sa.Column("knowledge_base_id", sa.String(length=36), nullable=False),
        sa.Column("index_version_id", sa.String(length=36), nullable=False),
        sa.Column("artifact_attempt", sa.Integer(), nullable=False),
        sa.Column("qdrant_collection", sa.String(length=255), nullable=False),
        sa.Column("manifest_object_key", sa.String(length=1024), nullable=False),
        sa.Column(
            "state",
            sa.String(length=30),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "max_attempts",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("20"),
        ),
        sa.Column(
            "worker_task_id",
            sa.String(length=255),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column("dispatch_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("last_dispatched_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "lease_owner",
            sa.String(length=160),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_checked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "error_code",
            sa.String(length=80),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column("error_detail", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "tenant_id",
            "knowledge_base_id",
            "index_version_id",
            "artifact_attempt",
            name="uq_index_artifact_gc_target",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["knowledge_base_id"],
            ["knowledge_bases.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "knowledge_base_id", "index_version_id"],
            [
                "index_versions.tenant_id",
                "index_versions.knowledge_base_id",
                "index_versions.id",
            ],
            name="fk_index_artifact_gc_scoped_index",
            ondelete="RESTRICT",
        ),
        sa.CheckConstraint(
            "artifact_attempt > 0",
            name="ck_index_artifact_gc_attempt",
        ),
        sa.CheckConstraint(
            "state IN ('pending', 'running', 'retry_wait', 'completed', 'dead')",
            name="ck_index_artifact_gc_state",
        ),
    )
    op.create_index(
        "ix_index_artifact_gc_state_next",
        "index_artifact_gc",
        ["state", "next_attempt_at"],
    )


def downgrade() -> None:
    remaining = op.get_bind().execute(
        sa.text("SELECT COUNT(*) FROM index_artifact_gc")
    ).scalar_one()
    if remaining:
        raise RuntimeError(
            "cannot drop index_artifact_gc while durable cleanup evidence remains"
        )
    op.drop_index("ix_index_artifact_gc_state_next", table_name="index_artifact_gc")
    op.drop_table("index_artifact_gc")
