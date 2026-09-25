from datetime import datetime, timezone
from typing import Callable, Optional
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from rag_demo.production.config import DEFAULT_SPARSE_EMBEDDING_MODEL


def new_id() -> str:
    return str(uuid4())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class UserIdentity(Base):
    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("tenant_id", "subject", name="uq_users_tenant_subject"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    subject: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(160), nullable=False, default="")
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Membership(Base):
    __tablename__ = "memberships"
    __table_args__ = (
        UniqueConstraint("tenant_id", "user_id", name="uq_memberships_tenant_user"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String(40), nullable=False, default="member")
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class KnowledgeBaseRecord(Base):
    __tablename__ = "knowledge_bases"
    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_knowledge_bases_tenant_name"),
        UniqueConstraint("tenant_id", "id", name="uq_knowledge_bases_tenant_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    owner_user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    profile: Mapped[str] = mapped_column(String(64), nullable=False, default="default")
    visibility: Mapped[str] = mapped_column(String(20), nullable=False, default="private")
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    activation_generation: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    active_index_version_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey(
            "index_versions.id",
            name="fk_knowledge_bases_active_index_version",
            use_alter=True,
            ondelete="SET NULL",
        ),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )


class IndexVersionRecord(Base):
    __tablename__ = "index_versions"
    __table_args__ = (
        UniqueConstraint(
            "knowledge_base_id",
            "version_number",
            name="uq_index_versions_kb_number",
        ),
        UniqueConstraint(
            "tenant_id",
            "knowledge_base_id",
            "id",
            name="uq_index_versions_tenant_kb_id",
        ),
        Index("ix_index_versions_tenant_kb_status", "tenant_id", "knowledge_base_id", "status"),
        CheckConstraint(
            "status IN ('building', 'validating', 'ready', 'active', 'failed', 'retired')",
            name="ck_index_versions_status",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
    )
    knowledge_base_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
    )
    created_by_user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="building")
    embedding_model: Mapped[str] = mapped_column(String(255), nullable=False)
    sparse_embedding_model: Mapped[str] = mapped_column(
        String(255), nullable=False, default=DEFAULT_SPARSE_EMBEDDING_MODEL
    )
    embedding_dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    reranker_model: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    chunk_schema_version: Mapped[str] = mapped_column(String(100), nullable=False)
    qdrant_collection: Mapped[str] = mapped_column(String(255), nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    manifest_sha256: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    manifest_object_key: Mapped[str] = mapped_column(
        String(1024),
        nullable=False,
        default="",
    )
    manifest_entries_sha256: Mapped[str] = mapped_column(
        String(64), nullable=False, default=""
    )
    validated_pg_entries_sha256: Mapped[str] = mapped_column(
        String(64), nullable=False, default=""
    )
    validated_qdrant_entries_sha256: Mapped[str] = mapped_column(
        String(64), nullable=False, default=""
    )
    expected_document_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    expected_chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    expected_vector_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    validated_pg_chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    validated_qdrant_point_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    metadata_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    validated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    retired_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    gc_after: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    activated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class IndexActivationEventRecord(Base):
    __tablename__ = "index_activation_events"
    __table_args__ = (
        UniqueConstraint(
            "knowledge_base_id",
            "generation",
            name="uq_index_activation_events_kb_generation",
        ),
        Index(
            "ix_index_activation_events_tenant_kb",
            "tenant_id",
            "knowledge_base_id",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
    )
    knowledge_base_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
    )
    previous_index_version_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("index_versions.id", ondelete="SET NULL"),
        nullable=True,
    )
    new_index_version_id: Mapped[str] = mapped_column(
        ForeignKey("index_versions.id", ondelete="RESTRICT"),
        nullable=False,
    )
    generation: Mapped[int] = mapped_column(Integer, nullable=False)
    activated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class FolderRecord(Base):
    __tablename__ = "folders"
    __table_args__ = (
        UniqueConstraint("knowledge_base_id", "name", name="uq_folders_kb_name"),
        Index("ix_folders_tenant_kb", "tenant_id", "knowledge_base_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
    )
    knowledge_base_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class DocumentRecord(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("tenant_id", "source_id", name="uq_documents_tenant_source"),
        UniqueConstraint(
            "tenant_id",
            "knowledge_base_id",
            "id",
            name="uq_documents_tenant_kb_id",
        ),
        Index("ix_documents_tenant_kb_status", "tenant_id", "knowledge_base_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
    )
    knowledge_base_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
    )
    folder_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("folders.id", ondelete="SET NULL"),
        nullable=True,
    )
    owner_user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
    )
    source_id: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="uploaded")
    current_version_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey(
            "document_versions.id",
            name="fk_documents_current_version",
            use_alter=True,
            ondelete="SET NULL",
        ),
        nullable=True,
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )


class DocumentVersionRecord(Base):
    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint("document_id", "version_number", name="uq_document_versions_number"),
        UniqueConstraint(
            "tenant_id",
            "document_id",
            "id",
            name="uq_document_versions_tenant_document_id",
        ),
        Index("ix_document_versions_tenant_document", "tenant_id", "document_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    object_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    extracted_object_key: Mapped[str] = mapped_column(String(1024), nullable=False, default="")
    mime_type: Mapped[str] = mapped_column(String(255), nullable=False, default="application/octet-stream")
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    page_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    parser_version: Mapped[str] = mapped_column(String(100), nullable=False)
    chunk_schema_version: Mapped[str] = mapped_column(String(100), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="uploaded")
    metadata_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class IndexDocumentRecord(Base):
    __tablename__ = "index_documents"
    __table_args__ = (
        UniqueConstraint(
            "index_version_id",
            "document_version_id",
            name="uq_index_documents_version_document",
        ),
        Index(
            "ix_index_documents_tenant_kb_index",
            "tenant_id",
            "knowledge_base_id",
            "index_version_id",
        ),
        CheckConstraint(
            "status IN ('pending', 'indexed', 'ready', 'failed')",
            name="ck_index_documents_status",
        ),
        ForeignKeyConstraint(
            ["tenant_id", "knowledge_base_id", "index_version_id"],
            [
                "index_versions.tenant_id",
                "index_versions.knowledge_base_id",
                "index_versions.id",
            ],
            name="fk_index_documents_scoped_index",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["tenant_id", "knowledge_base_id", "document_id"],
            ["documents.tenant_id", "documents.knowledge_base_id", "documents.id"],
            name="fk_index_documents_scoped_document",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["tenant_id", "document_id", "document_version_id"],
            [
                "document_versions.tenant_id",
                "document_versions.document_id",
                "document_versions.id",
            ],
            name="fk_index_documents_scoped_document_version",
            ondelete="CASCADE",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    knowledge_base_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False
    )
    index_version_id: Mapped[str] = mapped_column(
        ForeignKey("index_versions.id", ondelete="CASCADE"), nullable=False
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    document_version_id: Mapped[str] = mapped_column(
        ForeignKey("document_versions.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="pending")
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class UploadSessionRecord(Base):
    __tablename__ = "upload_sessions"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "idempotency_key",
            name="uq_upload_sessions_tenant_idempotency",
        ),
        Index("ix_upload_sessions_tenant_state", "tenant_id", "state"),
        CheckConstraint(
            "state IN ('reserved', 'uploading', 'uploaded', 'verified', 'consumed', 'expired', 'aborted')",
            name="ck_upload_sessions_state",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
    )
    knowledge_base_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
    )
    owner_user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    folder_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("folders.id", ondelete="SET NULL"),
        nullable=True,
    )
    target_document_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
    )
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    declared_mime_type: Mapped[str] = mapped_column(String(255), nullable=False)
    expected_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    expected_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    object_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    state: Mapped[str] = mapped_column(String(30), nullable=False, default="reserved")
    consumed_document_version_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("document_versions.id", ondelete="SET NULL"),
        nullable=True,
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )


class IngestionJobRecord(Base):
    __tablename__ = "ingestion_jobs"
    __table_args__ = (
        Index("ix_ingestion_jobs_tenant_status", "tenant_id", "status"),
        UniqueConstraint(
            "tenant_id",
            "idempotency_key",
            name="uq_ingestion_jobs_tenant_idempotency",
        ),
        CheckConstraint(
            "status IN ('queued', 'running', 'retry_wait', 'succeeded', 'dead', 'cancelled')",
            name="ck_ingestion_jobs_status",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
    )
    document_version_id: Mapped[str] = mapped_column(
        ForeignKey("document_versions.id", ondelete="CASCADE"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="queued")
    stage: Mapped[str] = mapped_column(String(40), nullable=False, default="queued")
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    pipeline_fingerprint: Mapped[str] = mapped_column(String(128), nullable=False)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    worker_task_id: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    dispatch_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_dispatched_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    error_code: Mapped[str] = mapped_column(String(80), nullable=False, default="")
    error_detail: Mapped[str] = mapped_column(Text, nullable=False, default="")
    error_class: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    lease_owner: Mapped[str] = mapped_column(String(160), nullable=False, default="")
    lease_expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    next_attempt_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )


class IndexBuildJobRecord(Base):
    __tablename__ = "index_build_jobs"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "idempotency_key",
            name="uq_index_build_jobs_tenant_idempotency",
        ),
        Index("ix_index_build_jobs_tenant_status", "tenant_id", "status"),
        CheckConstraint(
            "status IN ('queued', 'running', 'retry_wait', 'succeeded', 'dead', 'cancelled')",
            name="ck_index_build_jobs_status",
        ),
        ForeignKeyConstraint(
            ["tenant_id", "knowledge_base_id", "index_version_id"],
            [
                "index_versions.tenant_id",
                "index_versions.knowledge_base_id",
                "index_versions.id",
            ],
            name="fk_index_build_jobs_scoped_index",
            ondelete="CASCADE",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
    )
    knowledge_base_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
    )
    index_version_id: Mapped[str] = mapped_column(
        ForeignKey("index_versions.id", ondelete="CASCADE"),
        nullable=False,
    )
    created_by_user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
    )
    expected_active_index_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("index_versions.id", ondelete="SET NULL"),
        nullable=True,
    )
    expected_generation: Mapped[int] = mapped_column(Integer, nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    selected_document_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="queued")
    stage: Mapped[str] = mapped_column(String(40), nullable=False, default="queued")
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    worker_task_id: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    dispatch_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_dispatched_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    lease_owner: Mapped[str] = mapped_column(String(160), nullable=False, default="")
    lease_expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    heartbeat_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    next_attempt_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    error_code: Mapped[str] = mapped_column(String(80), nullable=False, default="")
    error_class: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    error_detail: Mapped[str] = mapped_column(Text, nullable=False, default="")
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )


class ChunkRecord(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint(
            "index_version_id",
            "chunk_key",
            name="uq_chunks_index_chunk_key",
        ),
        Index("ix_chunks_tenant_kb_index", "tenant_id", "knowledge_base_id", "index_version_id"),
        Index("ix_chunks_tenant_document_version", "tenant_id", "document_version_id"),
        ForeignKeyConstraint(
            ["tenant_id", "knowledge_base_id", "index_version_id"],
            [
                "index_versions.tenant_id",
                "index_versions.knowledge_base_id",
                "index_versions.id",
            ],
            name="fk_chunks_scoped_index",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["tenant_id", "knowledge_base_id", "document_id"],
            ["documents.tenant_id", "documents.knowledge_base_id", "documents.id"],
            name="fk_chunks_scoped_document",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["tenant_id", "document_id", "document_version_id"],
            [
                "document_versions.tenant_id",
                "document_versions.document_id",
                "document_versions.id",
            ],
            name="fk_chunks_scoped_document_version",
            ondelete="CASCADE",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    knowledge_base_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    document_version_id: Mapped[str] = mapped_column(
        ForeignKey("document_versions.id", ondelete="CASCADE"), nullable=False
    )
    index_version_id: Mapped[str] = mapped_column(
        ForeignKey("index_versions.id", ondelete="CASCADE"), nullable=False
    )
    chunk_key: Mapped[str] = mapped_column(String(255), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    page_start: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    page_end: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    title: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    qdrant_point_id: Mapped[str] = mapped_column(String(100), nullable=False)
    metadata_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ConversationRecord(Base):
    __tablename__ = "conversations"
    __table_args__ = (
        Index("ix_conversations_tenant_updated", "tenant_id", "updated_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    knowledge_base_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False, default="新對話")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class MessageRecord(Base):
    __tablename__ = "messages"
    __table_args__ = (
        Index("ix_messages_tenant_conversation", "tenant_id", "conversation_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
    )
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class AnswerRunRecord(Base):
    __tablename__ = "answer_runs"
    __table_args__ = (
        Index("ix_answer_runs_tenant_created", "tenant_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
    )
    knowledge_base_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
    )
    conversation_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
    )
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    pipeline_version: Mapped[str] = mapped_column(String(100), nullable=False)
    index_version_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("index_versions.id", ondelete="SET NULL"),
        nullable=True,
    )
    request_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    prompt_version: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    retrieval_needed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    source_ids_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    timings_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    retrieval_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class CitationRecord(Base):
    __tablename__ = "citations"
    __table_args__ = (
        Index("ix_citations_tenant_run", "tenant_id", "answer_run_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
    )
    answer_run_id: Mapped[str] = mapped_column(
        ForeignKey("answer_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    document_version_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("document_versions.id", ondelete="SET NULL"),
        nullable=True,
    )
    chunk_record_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("chunks.id", ondelete="SET NULL"),
        nullable=True,
    )
    chunk_id: Mapped[str] = mapped_column(String(255), nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    page: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    quote_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class FeedbackRecord(Base):
    __tablename__ = "feedback"
    __table_args__ = (
        UniqueConstraint("user_id", "answer_run_id", name="uq_feedback_user_run"),
        Index("ix_feedback_tenant_created", "tenant_id", "created_at"),
        CheckConstraint("rating IN (-1, 0, 1)", name="ck_feedback_rating"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    answer_run_id: Mapped[str] = mapped_column(
        ForeignKey("answer_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    comment: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class AuditEventRecord(Base):
    __tablename__ = "audit_events"
    __table_args__ = (
        Index("ix_audit_events_tenant_created", "tenant_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
    )
    subject: Mapped[str] = mapped_column(String(255), nullable=False)
    action: Mapped[str] = mapped_column(String(120), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(80), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(255), nullable=False)
    request_id: Mapped[str] = mapped_column(String(64), nullable=False)
    outcome: Mapped[str] = mapped_column(String(40), nullable=False)
    details_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class DeletionOutboxRecord(Base):
    __tablename__ = "deletion_outbox"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "idempotency_key",
            name="uq_deletion_outbox_tenant_idempotency",
        ),
        Index("ix_deletion_outbox_state_next", "state", "next_attempt_at"),
        CheckConstraint(
            "state IN ('pending', 'running', 'retry_wait', 'completed', 'dead')",
            name="ck_deletion_outbox_state",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
    )
    knowledge_base_id: Mapped[str] = mapped_column(String(36), nullable=False)
    created_by_user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
    )
    resource_type: Mapped[str] = mapped_column(String(80), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(36), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    object_keys_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    vector_scopes_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    state: Mapped[str] = mapped_column(String(30), nullable=False, default="pending")
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=20)
    worker_task_id: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    dispatch_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_dispatched_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    lease_owner: Mapped[str] = mapped_column(String(160), nullable=False, default="")
    lease_expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    next_attempt_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    error_code: Mapped[str] = mapped_column(String(80), nullable=False, default="")
    error_detail: Mapped[str] = mapped_column(Text, nullable=False, default="")
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )


def create_database_engine(database_url: str):
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    return create_engine(
        database_url,
        pool_pre_ping=True,
        connect_args=connect_args,
    )


def create_session_factory(engine) -> Callable:
    return sessionmaker(bind=engine, expire_on_commit=False)
