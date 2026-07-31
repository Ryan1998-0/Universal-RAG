"""Add recoverable production workflow state.

Revision ID: 7152560432ec
Revises: c3e4f5a6b7c8
Create Date: 2026-07-26
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "7152560432ec"
down_revision: Union[str, Sequence[str], None] = "c3e4f5a6b7c8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    _add_required_columns()
    _add_scoped_constraints()
    _create_workflow_tables()


def downgrade() -> None:
    op.drop_index("ix_upload_sessions_tenant_state", table_name="upload_sessions")
    op.drop_table("upload_sessions")
    op.drop_index(
        "ix_index_activation_events_tenant_kb",
        table_name="index_activation_events",
    )
    op.drop_table("index_activation_events")
    op.drop_index("ix_deletion_outbox_state_next", table_name="deletion_outbox")
    op.drop_table("deletion_outbox")

    with op.batch_alter_table("chunks") as batch:
        batch.drop_constraint("fk_chunks_scoped_document", type_="foreignkey")
        batch.drop_constraint("fk_chunks_scoped_document_version", type_="foreignkey")
        batch.drop_constraint("fk_chunks_scoped_index", type_="foreignkey")
    with op.batch_alter_table("index_documents") as batch:
        batch.drop_constraint("fk_index_documents_scoped_document", type_="foreignkey")
        batch.drop_constraint(
            "fk_index_documents_scoped_document_version",
            type_="foreignkey",
        )
        batch.drop_constraint("fk_index_documents_scoped_index", type_="foreignkey")

    with op.batch_alter_table("ingestion_jobs") as batch:
        batch.drop_constraint("ck_ingestion_jobs_status", type_="check")
        batch.drop_constraint(
            "uq_ingestion_jobs_tenant_idempotency",
            type_="unique",
        )
        for column_name in (
            "next_attempt_at",
            "last_dispatched_at",
            "dispatch_count",
            "heartbeat_at",
            "lease_expires_at",
            "lease_owner",
            "error_class",
            "pipeline_fingerprint",
            "idempotency_key",
            "stage",
        ):
            batch.drop_column(column_name)

    with op.batch_alter_table("document_versions") as batch:
        batch.drop_constraint(
            "uq_document_versions_tenant_document_id",
            type_="unique",
        )
    with op.batch_alter_table("documents") as batch:
        batch.drop_constraint("uq_documents_tenant_kb_id", type_="unique")
        batch.drop_column("deleted_at")
    with op.batch_alter_table("index_versions") as batch:
        batch.drop_constraint("uq_index_versions_tenant_kb_id", type_="unique")
        for column_name in (
            "gc_after",
            "retired_at",
            "published_at",
            "validated_at",
            "validated_qdrant_point_count",
            "validated_pg_chunk_count",
            "expected_vector_count",
            "expected_chunk_count",
            "expected_document_count",
            "manifest_sha256",
        ):
            batch.drop_column(column_name)
    with op.batch_alter_table("knowledge_bases") as batch:
        batch.drop_constraint("uq_knowledge_bases_tenant_id", type_="unique")
        batch.drop_column("activation_generation")


def _add_required_columns() -> None:
    op.execute("UPDATE index_versions SET status = 'ready' WHERE status = 'active'")
    op.add_column(
        "knowledge_bases",
        sa.Column("activation_generation", sa.Integer(), nullable=True),
    )
    op.execute("UPDATE knowledge_bases SET activation_generation = 0")
    with op.batch_alter_table("knowledge_bases") as batch:
        batch.alter_column("activation_generation", existing_type=sa.Integer(), nullable=False)
        batch.create_unique_constraint(
            "uq_knowledge_bases_tenant_id",
            ["tenant_id", "id"],
        )

    required_index_columns = (
        sa.Column("manifest_sha256", sa.String(length=64), nullable=True),
        sa.Column("expected_document_count", sa.Integer(), nullable=True),
        sa.Column("expected_chunk_count", sa.Integer(), nullable=True),
        sa.Column("expected_vector_count", sa.Integer(), nullable=True),
        sa.Column("validated_pg_chunk_count", sa.Integer(), nullable=True),
        sa.Column("validated_qdrant_point_count", sa.Integer(), nullable=True),
    )
    for column in required_index_columns:
        op.add_column("index_versions", column)
    op.add_column(
        "index_versions",
        sa.Column("validated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "index_versions",
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "index_versions",
        sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "index_versions",
        sa.Column("gc_after", sa.DateTime(timezone=True), nullable=True),
    )
    op.execute(
        "UPDATE index_versions SET manifest_sha256 = '', "
        "expected_document_count = 0, expected_chunk_count = 0, "
        "expected_vector_count = 0, validated_pg_chunk_count = 0, "
        "validated_qdrant_point_count = 0"
    )
    with op.batch_alter_table("index_versions") as batch:
        batch.alter_column("manifest_sha256", existing_type=sa.String(length=64), nullable=False)
        for column_name in (
            "expected_document_count",
            "expected_chunk_count",
            "expected_vector_count",
            "validated_pg_chunk_count",
            "validated_qdrant_point_count",
        ):
            batch.alter_column(column_name, existing_type=sa.Integer(), nullable=False)
        batch.create_unique_constraint(
            "uq_index_versions_tenant_kb_id",
            ["tenant_id", "knowledge_base_id", "id"],
        )

    op.add_column(
        "documents",
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    with op.batch_alter_table("documents") as batch:
        batch.create_unique_constraint(
            "uq_documents_tenant_kb_id",
            ["tenant_id", "knowledge_base_id", "id"],
        )
    with op.batch_alter_table("document_versions") as batch:
        batch.create_unique_constraint(
            "uq_document_versions_tenant_document_id",
            ["tenant_id", "document_id", "id"],
        )

    for column in (
        sa.Column("stage", sa.String(length=40), nullable=True),
        sa.Column("idempotency_key", sa.String(length=128), nullable=True),
        sa.Column("pipeline_fingerprint", sa.String(length=128), nullable=True),
        sa.Column("error_class", sa.String(length=120), nullable=True),
        sa.Column("lease_owner", sa.String(length=160), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_dispatched_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("dispatch_count", sa.Integer(), nullable=True),
    ):
        op.add_column("ingestion_jobs", column)
    op.execute(
        "UPDATE ingestion_jobs SET status = 'queued', stage = 'queued', "
        "idempotency_key = id, pipeline_fingerprint = 'legacy-v1', "
        "error_class = '', lease_owner = '', dispatch_count = 0"
    )
    with op.batch_alter_table("ingestion_jobs") as batch:
        for column_name, column_type in (
            ("stage", sa.String(length=40)),
            ("idempotency_key", sa.String(length=128)),
            ("pipeline_fingerprint", sa.String(length=128)),
            ("error_class", sa.String(length=120)),
            ("lease_owner", sa.String(length=160)),
        ):
            batch.alter_column(column_name, existing_type=column_type, nullable=False)
        batch.alter_column(
            "dispatch_count",
            existing_type=sa.Integer(),
            nullable=False,
        )
        batch.create_unique_constraint(
            "uq_ingestion_jobs_tenant_idempotency",
            ["tenant_id", "idempotency_key"],
        )
        batch.create_check_constraint(
            "ck_ingestion_jobs_status",
            "status IN ('queued', 'running', 'retry_wait', 'succeeded', 'dead', 'cancelled')",
        )


def _add_scoped_constraints() -> None:
    with op.batch_alter_table("index_documents") as batch:
        batch.create_foreign_key(
            "fk_index_documents_scoped_index",
            "index_versions",
            ["tenant_id", "knowledge_base_id", "index_version_id"],
            ["tenant_id", "knowledge_base_id", "id"],
            ondelete="CASCADE",
        )
        batch.create_foreign_key(
            "fk_index_documents_scoped_document",
            "documents",
            ["tenant_id", "knowledge_base_id", "document_id"],
            ["tenant_id", "knowledge_base_id", "id"],
            ondelete="CASCADE",
        )
        batch.create_foreign_key(
            "fk_index_documents_scoped_document_version",
            "document_versions",
            ["tenant_id", "document_id", "document_version_id"],
            ["tenant_id", "document_id", "id"],
            ondelete="CASCADE",
        )
    with op.batch_alter_table("chunks") as batch:
        batch.create_foreign_key(
            "fk_chunks_scoped_index",
            "index_versions",
            ["tenant_id", "knowledge_base_id", "index_version_id"],
            ["tenant_id", "knowledge_base_id", "id"],
            ondelete="CASCADE",
        )
        batch.create_foreign_key(
            "fk_chunks_scoped_document",
            "documents",
            ["tenant_id", "knowledge_base_id", "document_id"],
            ["tenant_id", "knowledge_base_id", "id"],
            ondelete="CASCADE",
        )
        batch.create_foreign_key(
            "fk_chunks_scoped_document_version",
            "document_versions",
            ["tenant_id", "document_id", "document_version_id"],
            ["tenant_id", "document_id", "id"],
            ondelete="CASCADE",
        )


def _create_workflow_tables() -> None:
    op.create_table(
        "deletion_outbox",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("tenant_id", sa.String(length=36), nullable=False),
        sa.Column("knowledge_base_id", sa.String(length=36), nullable=False),
        sa.Column("resource_type", sa.String(length=80), nullable=False),
        sa.Column("resource_id", sa.String(length=36), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("object_keys_json", sa.JSON(), nullable=False),
        sa.Column("vector_scopes_json", sa.JSON(), nullable=False),
        sa.Column("state", sa.String(length=30), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("lease_owner", sa.String(length=160), nullable=False),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(length=80), nullable=False),
        sa.Column("error_detail", sa.Text(), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "state IN ('pending', 'running', 'retry_wait', 'completed', 'dead')",
            name="ck_deletion_outbox_state",
        ),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "tenant_id",
            "idempotency_key",
            name="uq_deletion_outbox_tenant_idempotency",
        ),
    )
    op.create_index(
        "ix_deletion_outbox_state_next",
        "deletion_outbox",
        ["state", "next_attempt_at"],
        unique=False,
    )

    op.create_table(
        "index_activation_events",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("tenant_id", sa.String(length=36), nullable=False),
        sa.Column("knowledge_base_id", sa.String(length=36), nullable=False),
        sa.Column("previous_index_version_id", sa.String(length=36), nullable=True),
        sa.Column("new_index_version_id", sa.String(length=36), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["knowledge_base_id"], ["knowledge_bases.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["new_index_version_id"], ["index_versions.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["previous_index_version_id"], ["index_versions.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "knowledge_base_id",
            "generation",
            name="uq_index_activation_events_kb_generation",
        ),
    )
    op.create_index(
        "ix_index_activation_events_tenant_kb",
        "index_activation_events",
        ["tenant_id", "knowledge_base_id"],
        unique=False,
    )

    op.create_table(
        "upload_sessions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("tenant_id", sa.String(length=36), nullable=False),
        sa.Column("knowledge_base_id", sa.String(length=36), nullable=False),
        sa.Column("owner_user_id", sa.String(length=36), nullable=False),
        sa.Column("folder_id", sa.String(length=36), nullable=True),
        sa.Column("target_document_id", sa.String(length=36), nullable=True),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("declared_mime_type", sa.String(length=255), nullable=False),
        sa.Column("expected_size_bytes", sa.Integer(), nullable=False),
        sa.Column("expected_sha256", sa.String(length=64), nullable=False),
        sa.Column("object_key", sa.String(length=1024), nullable=False),
        sa.Column("state", sa.String(length=30), nullable=False),
        sa.Column("consumed_document_version_id", sa.String(length=36), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "state IN ('reserved', 'uploading', 'uploaded', 'verified', 'consumed', 'expired', 'aborted')",
            name="ck_upload_sessions_state",
        ),
        sa.ForeignKeyConstraint(
            ["consumed_document_version_id"],
            ["document_versions.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["knowledge_base_id"], ["knowledge_bases.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["folder_id"], ["folders.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["owner_user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["target_document_id"], ["documents.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "tenant_id",
            "idempotency_key",
            name="uq_upload_sessions_tenant_idempotency",
        ),
    )
    op.create_index(
        "ix_upload_sessions_tenant_state",
        "upload_sessions",
        ["tenant_id", "state"],
        unique=False,
    )
