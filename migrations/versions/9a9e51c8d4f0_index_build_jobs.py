"""Add durable index build jobs.

Revision ID: 9a9e51c8d4f0
Revises: 7152560432ec
Create Date: 2026-07-26
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "9a9e51c8d4f0"
down_revision: Union[str, Sequence[str], None] = "7152560432ec"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "index_versions",
        sa.Column("manifest_object_key", sa.String(length=1024), nullable=True),
    )
    op.execute("UPDATE index_versions SET manifest_object_key = ''")
    with op.batch_alter_table("index_versions") as batch:
        batch.alter_column(
            "manifest_object_key",
            existing_type=sa.String(length=1024),
            nullable=False,
        )
    for column in (
        sa.Column("worker_task_id", sa.String(length=255), nullable=True),
        sa.Column("dispatch_count", sa.Integer(), nullable=True),
        sa.Column("last_dispatched_at", sa.DateTime(timezone=True), nullable=True),
    ):
        op.add_column("deletion_outbox", column)
    op.execute(
        "UPDATE deletion_outbox SET worker_task_id = '', dispatch_count = 0"
    )
    with op.batch_alter_table("deletion_outbox") as batch:
        batch.alter_column(
            "worker_task_id",
            existing_type=sa.String(length=255),
            nullable=False,
        )
        batch.alter_column(
            "dispatch_count",
            existing_type=sa.Integer(),
            nullable=False,
        )
    op.create_table(
        "index_build_jobs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("tenant_id", sa.String(length=36), nullable=False),
        sa.Column("knowledge_base_id", sa.String(length=36), nullable=False),
        sa.Column("index_version_id", sa.String(length=36), nullable=False),
        sa.Column("created_by_user_id", sa.String(length=36), nullable=False),
        sa.Column("expected_active_index_id", sa.String(length=36), nullable=True),
        sa.Column("expected_generation", sa.Integer(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("selected_document_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("stage", sa.String(length=40), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("worker_task_id", sa.String(length=255), nullable=False),
        sa.Column("dispatch_count", sa.Integer(), nullable=False),
        sa.Column("last_dispatched_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lease_owner", sa.String(length=160), nullable=False),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(length=80), nullable=False),
        sa.Column("error_class", sa.String(length=120), nullable=False),
        sa.Column("error_detail", sa.Text(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'retry_wait', 'succeeded', 'dead', 'cancelled')",
            name="ck_index_build_jobs_status",
        ),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"], ["users.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["expected_active_index_id"], ["index_versions.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["index_version_id"], ["index_versions.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["knowledge_base_id"], ["knowledge_bases.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["tenant_id", "knowledge_base_id", "index_version_id"],
            ["index_versions.tenant_id", "index_versions.knowledge_base_id", "index_versions.id"],
            name="fk_index_build_jobs_scoped_index",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "tenant_id",
            "idempotency_key",
            name="uq_index_build_jobs_tenant_idempotency",
        ),
    )
    op.create_index(
        "ix_index_build_jobs_tenant_status",
        "index_build_jobs",
        ["tenant_id", "status"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_index_build_jobs_tenant_status", table_name="index_build_jobs")
    op.drop_table("index_build_jobs")
    with op.batch_alter_table("deletion_outbox") as batch:
        batch.drop_column("last_dispatched_at")
        batch.drop_column("dispatch_count")
        batch.drop_column("worker_task_id")
    with op.batch_alter_table("index_versions") as batch:
        batch.drop_column("manifest_object_key")
