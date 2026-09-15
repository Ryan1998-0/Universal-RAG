from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import timedelta
from time import perf_counter
from typing import Optional, Sequence
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import and_, delete, func, or_, select

from rag_demo.production.database import (
    ChunkRecord,
    DocumentRecord,
    DocumentVersionRecord,
    IndexBuildJobRecord,
    IndexDocumentRecord,
    IndexVersionRecord,
    KnowledgeBaseRecord,
    UserIdentity,
    new_id,
    utc_now,
)
from rag_demo.production.index_manifest import index_entries_sha256
from rag_demo.production.repository import (
    AccessDeniedError,
    ConcurrentPublishError,
    InvalidServiceStateError,
    ResourceNotFoundError,
)
from rag_demo.retrieval_scope import RetrievalScope
from rag_demo.observability import TimingTrace, configure_logging, elapsed_ms


_IDEMPOTENCY_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{8,128}$")


class IndexBuildValidationError(ValueError):
    pass


@dataclass(frozen=True)
class IndexBuildReservation:
    job_id: str
    index_version_id: str
    status: str
    duplicate: bool


@dataclass(frozen=True)
class IndexDocumentArtifact:
    document_id: str
    document_version_id: str
    source_id: str
    filename: str
    extracted_object_key: str


@dataclass(frozen=True)
class IndexBuildWorkItem:
    job_id: str
    tenant_id: str
    knowledge_base_id: str
    index_version_id: str
    expected_active_index_id: Optional[str]
    expected_generation: int
    attempt: int
    max_attempts: int
    documents: tuple[IndexDocumentArtifact, ...]


class SqlAlchemyIndexingRepository:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def reserve_build(
        self,
        *,
        principal,
        authorized,
        idempotency_key: str,
        document_ids: Sequence[str],
        settings,
    ) -> IndexBuildReservation:
        if not authorized.can_write:
            raise AccessDeniedError("knowledge base write access is denied")
        clean_key = str(idempotency_key or "").strip()
        if not _IDEMPOTENCY_PATTERN.fullmatch(clean_key):
            raise IndexBuildValidationError(
                "Idempotency-Key must contain 8 to 128 safe characters"
            )
        selected = sorted(set(
            str(document_id or "").strip() for document_id in document_ids
            if str(document_id or "").strip()
        ))
        if not selected or len(selected) > 500:
            raise IndexBuildValidationError("select between 1 and 500 documents")
        selected_hash = hashlib.sha256(
            json.dumps(selected, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        with self.session_factory.begin() as session:
            existing = session.scalar(select(IndexBuildJobRecord).where(
                IndexBuildJobRecord.tenant_id == principal.tenant_id,
                IndexBuildJobRecord.idempotency_key == clean_key,
            ))
            if existing is not None:
                if (
                    existing.knowledge_base_id != authorized.id
                    or existing.selected_document_hash != selected_hash
                ):
                    raise InvalidServiceStateError(
                        "idempotency key was already used for another index build"
                    )
                return IndexBuildReservation(
                    job_id=existing.id,
                    index_version_id=existing.index_version_id,
                    status=existing.status,
                    duplicate=True,
                )

            knowledge_base = session.scalar(
                select(KnowledgeBaseRecord)
                .where(
                    KnowledgeBaseRecord.id == authorized.id,
                    KnowledgeBaseRecord.tenant_id == principal.tenant_id,
                    KnowledgeBaseRecord.active.is_(True),
                )
                .with_for_update()
            )
            if knowledge_base is None:
                raise ResourceNotFoundError("knowledge base was not found")
            documents = list(session.scalars(
                select(DocumentRecord)
                .where(
                    DocumentRecord.id.in_(selected),
                    DocumentRecord.tenant_id == principal.tenant_id,
                    DocumentRecord.knowledge_base_id == authorized.id,
                    DocumentRecord.deleted_at.is_(None),
                )
                .with_for_update()
            ))
            if len(documents) != len(selected):
                raise ResourceNotFoundError("one or more documents were not found")

            chosen_versions = {}
            for document in documents:
                version = session.scalar(
                    select(DocumentVersionRecord)
                    .where(
                        DocumentVersionRecord.tenant_id == principal.tenant_id,
                        DocumentVersionRecord.document_id == document.id,
                        DocumentVersionRecord.status.in_({"parsed", "ready"}),
                        DocumentVersionRecord.extracted_object_key != "",
                    )
                    .order_by(DocumentVersionRecord.version_number.desc())
                    .limit(1)
                )
                if version is None:
                    raise InvalidServiceStateError(
                        "all selected documents must finish ingestion before indexing"
                    )
                chosen_versions[document.id] = version

            next_number = int(session.scalar(
                select(func.max(IndexVersionRecord.version_number)).where(
                    IndexVersionRecord.knowledge_base_id == authorized.id
                )
            ) or 0) + 1
            index_version = IndexVersionRecord(
                id=new_id(),
                tenant_id=principal.tenant_id,
                knowledge_base_id=authorized.id,
                created_by_user_id=authorized.user_id,
                version_number=next_number,
                status="building",
                embedding_model=settings.embedding_model,
                embedding_dimensions=settings.embedding_dimensions,
                reranker_model=settings.reranker_model,
                chunk_schema_version=settings.chunk_schema_version,
                qdrant_collection=settings.qdrant_collection,
                expected_document_count=len(documents),
                metadata_json={
                    "selected_document_ids": selected,
                    "selected_document_hash": selected_hash,
                },
            )
            session.add(index_version)
            session.flush()
            for document in documents:
                version = chosen_versions[document.id]
                session.add(IndexDocumentRecord(
                    tenant_id=principal.tenant_id,
                    knowledge_base_id=authorized.id,
                    index_version_id=index_version.id,
                    document_id=document.id,
                    document_version_id=version.id,
                    status="pending",
                    chunk_count=0,
                ))
            job = IndexBuildJobRecord(
                id=new_id(),
                tenant_id=principal.tenant_id,
                knowledge_base_id=authorized.id,
                index_version_id=index_version.id,
                created_by_user_id=authorized.user_id,
                expected_active_index_id=knowledge_base.active_index_version_id,
                expected_generation=knowledge_base.activation_generation,
                idempotency_key=clean_key,
                selected_document_hash=selected_hash,
                status="queued",
                stage="queued",
            )
            session.add(job)
            return IndexBuildReservation(
                job_id=job.id,
                index_version_id=index_version.id,
                status=job.status,
                duplicate=False,
            )

    def claim(
        self,
        *,
        job_id: str,
        worker_id: str,
        lease_seconds: int = 1200,
    ) -> Optional[IndexBuildWorkItem]:
        now = utc_now()
        with self.session_factory.begin() as session:
            job = session.scalar(
                select(IndexBuildJobRecord)
                .where(IndexBuildJobRecord.id == job_id)
                .with_for_update()
            )
            if job is None or job.status in {"succeeded", "dead", "cancelled"}:
                return None
            knowledge_base = session.scalar(
                select(KnowledgeBaseRecord)
                .where(
                    KnowledgeBaseRecord.id == job.knowledge_base_id,
                    KnowledgeBaseRecord.tenant_id == job.tenant_id,
                )
                .with_for_update()
            )
            if knowledge_base is None:
                raise RuntimeError("index build knowledge base disappeared")
            if knowledge_base.active_index_version_id == job.index_version_id:
                job.status = "succeeded"
                job.stage = "published"
                job.completed_at = now
                job.lease_owner = ""
                job.lease_expires_at = None
                return None
            if (
                job.status == "running"
                and job.lease_expires_at is not None
                and _as_utc(job.lease_expires_at) > now
                and job.lease_owner != worker_id
            ):
                return None
            if (
                job.status == "retry_wait"
                and job.next_attempt_at is not None
                and _as_utc(job.next_attempt_at) > now
            ):
                return None
            target = session.scalar(select(IndexVersionRecord).where(
                IndexVersionRecord.id == job.index_version_id,
                IndexVersionRecord.tenant_id == job.tenant_id,
                IndexVersionRecord.knowledge_base_id == job.knowledge_base_id,
            ))
            if target is None:
                raise RuntimeError("index build target disappeared")
            rows = session.execute(
                select(IndexDocumentRecord, DocumentRecord, DocumentVersionRecord)
                .join(DocumentRecord, DocumentRecord.id == IndexDocumentRecord.document_id)
                .join(
                    DocumentVersionRecord,
                    DocumentVersionRecord.id == IndexDocumentRecord.document_version_id,
                )
                .where(
                    IndexDocumentRecord.tenant_id == job.tenant_id,
                    IndexDocumentRecord.knowledge_base_id == job.knowledge_base_id,
                    IndexDocumentRecord.index_version_id == job.index_version_id,
                    DocumentRecord.deleted_at.is_(None),
                )
                .order_by(DocumentRecord.id)
            ).all()
            if len(rows) != target.expected_document_count:
                raise InvalidServiceStateError("index document membership is incomplete")
            artifacts = []
            for _membership, document, version in rows:
                if not version.extracted_object_key:
                    raise InvalidServiceStateError("canonical document artifact is missing")
                artifacts.append(IndexDocumentArtifact(
                    document_id=document.id,
                    document_version_id=version.id,
                    source_id=document.source_id,
                    filename=document.name,
                    extracted_object_key=version.extracted_object_key,
                ))

            job.status = "running"
            job.stage = "preparing"
            job.attempt += 1
            job.lease_owner = worker_id
            job.lease_expires_at = now + timedelta(
                seconds=max(300, int(lease_seconds))
            )
            job.heartbeat_at = now
            job.started_at = job.started_at or now
            job.next_attempt_at = None
            job.error_code = ""
            job.error_class = ""
            job.error_detail = ""
            target.status = "building"
            return IndexBuildWorkItem(
                job_id=job.id,
                tenant_id=job.tenant_id,
                knowledge_base_id=job.knowledge_base_id,
                index_version_id=job.index_version_id,
                expected_active_index_id=job.expected_active_index_id,
                expected_generation=job.expected_generation,
                attempt=job.attempt,
                max_attempts=job.max_attempts,
                documents=tuple(artifacts),
            )

    def dispatchable_job_ids(
        self,
        *,
        limit: int = 50,
        redispatch_after_seconds: int = 120,
    ) -> list[str]:
        now = utc_now()
        stale = now - timedelta(seconds=max(30, int(redispatch_after_seconds)))
        with self.session_factory() as session:
            return list(session.scalars(
                select(IndexBuildJobRecord.id)
                .where(or_(
                    and_(
                        IndexBuildJobRecord.status == "queued",
                        or_(
                            IndexBuildJobRecord.last_dispatched_at.is_(None),
                            IndexBuildJobRecord.last_dispatched_at <= stale,
                        ),
                    ),
                    and_(
                        IndexBuildJobRecord.status == "retry_wait",
                        or_(
                            IndexBuildJobRecord.next_attempt_at.is_(None),
                            IndexBuildJobRecord.next_attempt_at <= now,
                        ),
                    ),
                    and_(
                        IndexBuildJobRecord.status == "running",
                        IndexBuildJobRecord.lease_expires_at.is_not(None),
                        IndexBuildJobRecord.lease_expires_at <= now,
                    ),
                ))
                .order_by(IndexBuildJobRecord.created_at, IndexBuildJobRecord.id)
                .limit(max(1, min(int(limit), 500)))
            ))

    def record_dispatch(self, *, job_id: str, task_id: str) -> None:
        with self.session_factory.begin() as session:
            job = session.get(IndexBuildJobRecord, job_id)
            if job is None:
                raise ResourceNotFoundError("index build job was not found")
            job.worker_task_id = str(task_id)[:255]
            job.dispatch_count += 1
            job.last_dispatched_at = utc_now()

    def heartbeat(self, *, job_id: str, worker_id: str, stage: str) -> None:
        now = utc_now()
        with self.session_factory.begin() as session:
            job = session.scalar(
                select(IndexBuildJobRecord)
                .where(
                    IndexBuildJobRecord.id == job_id,
                    IndexBuildJobRecord.status == "running",
                    IndexBuildJobRecord.lease_owner == worker_id,
                )
                .with_for_update()
            )
            if job is None:
                raise RuntimeError("index build lease was lost")
            job.stage = str(stage)[:40]
            job.heartbeat_at = now
            job.lease_expires_at = now + timedelta(minutes=20)

    def reset_candidate(self, *, work: IndexBuildWorkItem, worker_id: str) -> None:
        with self.session_factory.begin() as session:
            self._require_lease(session, work.job_id, worker_id)
            session.execute(delete(ChunkRecord).where(
                ChunkRecord.tenant_id == work.tenant_id,
                ChunkRecord.knowledge_base_id == work.knowledge_base_id,
                ChunkRecord.index_version_id == work.index_version_id,
            ))
            memberships = list(session.scalars(select(IndexDocumentRecord).where(
                IndexDocumentRecord.tenant_id == work.tenant_id,
                IndexDocumentRecord.knowledge_base_id == work.knowledge_base_id,
                IndexDocumentRecord.index_version_id == work.index_version_id,
            )))
            for membership in memberships:
                membership.status = "pending"
                membership.chunk_count = 0
            target = session.get(IndexVersionRecord, work.index_version_id)
            target.status = "building"
            target.chunk_count = 0
            target.expected_chunk_count = 0
            target.expected_vector_count = 0
            target.validated_pg_chunk_count = 0
            target.validated_qdrant_point_count = 0
            target.validated_at = None
            target.manifest_sha256 = ""
            target.manifest_object_key = ""

    def persist_chunk_batch(
        self,
        *,
        work: IndexBuildWorkItem,
        worker_id: str,
        chunks: Sequence[dict],
        point_ids: Sequence[str],
    ) -> None:
        if len(chunks) != len(point_ids):
            raise ValueError("chunk and point counts must match")
        with self.session_factory.begin() as session:
            self._require_lease(session, work.job_id, worker_id)
            for chunk, point_id in zip(chunks, point_ids):
                content = str(chunk["content"])
                metadata = {
                    key: value
                    for key, value in chunk.items()
                    if key not in {"content", "tenant_id", "knowledge_base_id"}
                }
                page = _page_number(chunk.get("page"))
                session.add(ChunkRecord(
                    id=str(chunk["chunk_record_id"]),
                    tenant_id=work.tenant_id,
                    knowledge_base_id=work.knowledge_base_id,
                    document_id=str(chunk["document_id"]),
                    document_version_id=str(chunk["document_version_id"]),
                    index_version_id=work.index_version_id,
                    chunk_key=str(chunk["id"]),
                    ordinal=int(chunk["ordinal"]),
                    page_start=page,
                    page_end=page,
                    title=str(chunk.get("title") or "")[:500],
                    content=content,
                    content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    qdrant_point_id=str(point_id),
                    metadata_json=metadata,
                ))

    def finalize_candidate(
        self,
        *,
        work: IndexBuildWorkItem,
        worker_id: str,
        document_chunk_counts: dict[str, int],
        chunk_count: int,
        manifest_sha256: str,
        manifest_object_key: str,
    ) -> None:
        with self.session_factory.begin() as session:
            self._require_lease(session, work.job_id, worker_id)
            memberships = list(session.scalars(select(IndexDocumentRecord).where(
                IndexDocumentRecord.tenant_id == work.tenant_id,
                IndexDocumentRecord.knowledge_base_id == work.knowledge_base_id,
                IndexDocumentRecord.index_version_id == work.index_version_id,
            )))
            if len(memberships) != len(document_chunk_counts):
                raise InvalidServiceStateError("manifest document count is inconsistent")
            for membership in memberships:
                count = int(document_chunk_counts.get(membership.document_id, -1))
                if count <= 0:
                    raise InvalidServiceStateError("every indexed document requires chunks")
                membership.chunk_count = count
                membership.status = "ready"
            if sum(document_chunk_counts.values()) != int(chunk_count):
                raise InvalidServiceStateError("manifest chunk count is inconsistent")
            target = session.get(IndexVersionRecord, work.index_version_id)
            target.status = "validating"
            target.chunk_count = int(chunk_count)
            target.expected_document_count = len(memberships)
            target.expected_chunk_count = int(chunk_count)
            target.expected_vector_count = int(chunk_count)
            target.manifest_sha256 = str(manifest_sha256)
            target.manifest_object_key = str(manifest_object_key)
            target.metadata_json = {
                **dict(target.metadata_json or {}),
                "document_chunk_counts": dict(document_chunk_counts),
            }

    def complete(self, *, job_id: str, worker_id: str) -> None:
        now = utc_now()
        with self.session_factory.begin() as session:
            job = self._require_lease(session, job_id, worker_id)
            job.status = "succeeded"
            job.stage = "published"
            job.completed_at = now
            job.heartbeat_at = now
            job.lease_owner = ""
            job.lease_expires_at = None
            job.next_attempt_at = None

    def fail(
        self,
        *,
        job_id: str,
        worker_id: str,
        error_code: str,
        error_class: str,
        error_detail: str,
        permanent: bool,
    ) -> str:
        now = utc_now()
        with self.session_factory.begin() as session:
            job = session.scalar(
                select(IndexBuildJobRecord)
                .where(
                    IndexBuildJobRecord.id == job_id,
                    IndexBuildJobRecord.lease_owner == worker_id,
                )
                .with_for_update()
            )
            if job is None:
                return "lease_lost"
            knowledge_base = session.get(KnowledgeBaseRecord, job.knowledge_base_id)
            if knowledge_base and knowledge_base.active_index_version_id == job.index_version_id:
                job.status = "succeeded"
                job.stage = "published"
                job.completed_at = now
                job.lease_owner = ""
                job.lease_expires_at = None
                return job.status
            terminal = permanent or job.attempt >= job.max_attempts
            job.status = "dead" if terminal else "retry_wait"
            job.stage = "failed"
            job.error_code = str(error_code)[:80]
            job.error_class = str(error_class)[:120]
            job.error_detail = _safe_error_detail(error_detail)
            job.lease_owner = ""
            job.lease_expires_at = None
            job.heartbeat_at = now
            target = session.get(IndexVersionRecord, job.index_version_id)
            if terminal:
                job.completed_at = now
                job.next_attempt_at = None
                if target is not None:
                    target.status = "failed"
            else:
                delay = min(3600, 60 * (2 ** max(0, job.attempt - 1)))
                job.next_attempt_at = now + timedelta(seconds=delay)
                if target is not None:
                    target.status = "building"
            return job.status

    def get_status(self, *, principal, job_id: str) -> dict:
        with self.session_factory() as session:
            row = session.execute(
                select(IndexBuildJobRecord, UserIdentity.subject)
                .join(UserIdentity, UserIdentity.id == IndexBuildJobRecord.created_by_user_id)
                .where(
                    IndexBuildJobRecord.id == job_id,
                    IndexBuildJobRecord.tenant_id == principal.tenant_id,
                )
            ).one_or_none()
            if row is None or row.subject != principal.subject:
                raise ResourceNotFoundError("index build job was not found")
            job = row[0]
            return {
                "id": job.id,
                "knowledge_base_id": job.knowledge_base_id,
                "index_version_id": job.index_version_id,
                "status": job.status,
                "stage": job.stage,
                "attempt": job.attempt,
                "max_attempts": job.max_attempts,
                "error_code": job.error_code,
                "created_at": job.created_at.isoformat(),
                "updated_at": job.updated_at.isoformat(),
            }

    @staticmethod
    def _require_lease(session, job_id: str, worker_id: str):
        job = session.scalar(
            select(IndexBuildJobRecord)
            .where(
                IndexBuildJobRecord.id == job_id,
                IndexBuildJobRecord.status == "running",
                IndexBuildJobRecord.lease_owner == worker_id,
            )
            .with_for_update()
        )
        if job is None:
            raise RuntimeError("index build lease was lost")
        return job


class IndexingService:
    def __init__(
        self,
        *,
        repository: SqlAlchemyIndexingRepository,
        tenant_repository,
        object_storage,
        vector_repository,
        embedding_runtime,
        batch_size: int = 64,
        max_canonical_bytes: int = 200 * 1024 * 1024,
    ):
        self.repository = repository
        self.tenant_repository = tenant_repository
        self.object_storage = object_storage
        self.vector_repository = vector_repository
        self.embedding_runtime = embedding_runtime
        self.batch_size = max(1, min(int(batch_size), 512))
        self.max_canonical_bytes = int(max_canonical_bytes)

    def process(self, *, job_id: str, worker_id: str) -> dict:
        configure_logging()
        started_at = perf_counter()
        trace = TimingTrace(run_id=str(job_id), component="indexing_service")
        work = self.repository.claim(job_id=job_id, worker_id=worker_id)
        if work is None:
            return {"job_id": job_id, "status": "not_claimed"}
        published = False
        try:
            scope = RetrievalScope(
                tenant_id=work.tenant_id,
                knowledge_base_id=work.knowledge_base_id,
                index_version_id=work.index_version_id,
            )
            with trace.stage("index.prepare"):
                self.repository.reset_candidate(work=work, worker_id=worker_id)
                self.vector_repository.delete_index(scope)
                prepared = []
                document_counts = {}
                for document in work.documents:
                    canonical = self._load_canonical(work, document)
                    raw_chunks = canonical.get("chunks")
                    if not isinstance(raw_chunks, list) or not raw_chunks:
                        raise IndexBuildValidationError(
                            "canonical document contains no indexable chunks"
                        )
                    document_counts[document.document_id] = len(raw_chunks)
                    for ordinal, raw in enumerate(raw_chunks):
                        if not isinstance(raw, dict) or not str(raw.get("content") or "").strip():
                            raise IndexBuildValidationError("canonical chunk is invalid")
                        chunk_key = f"{document.document_version_id}::{ordinal}"
                        chunk_record_id = str(uuid5(
                            NAMESPACE_URL,
                            "/".join((
                                work.tenant_id,
                                work.knowledge_base_id,
                                work.index_version_id,
                                chunk_key,
                            )),
                        ))
                        prepared.append({
                            **raw,
                            "id": chunk_key,
                            "chunk_id": chunk_key,
                            "chunk_record_id": chunk_record_id,
                            "ordinal": ordinal,
                            "tenant_id": work.tenant_id,
                            "knowledge_base_id": work.knowledge_base_id,
                            "index_version_id": work.index_version_id,
                            "document_id": document.document_id,
                            "document_version_id": document.document_version_id,
                            "source_id": document.source_id,
                            "source": document.source_id,
                            "filename": document.filename,
                        })

            if not prepared:
                raise IndexBuildValidationError("index contains no chunks")
            self.repository.heartbeat(
                job_id=work.job_id,
                worker_id=worker_id,
                stage="embedding",
            )
            with trace.stage("index.embedding_upsert"):
                manifest_chunks = []
                for start in range(0, len(prepared), self.batch_size):
                    batch = prepared[start : start + self.batch_size]
                    texts = [str(chunk["content"]) for chunk in batch]
                    dense = self.embedding_runtime.embed_documents(texts)
                    sparse = self.embedding_runtime.sparse_documents(texts)
                    point_ids = self.vector_repository.upsert_chunks(
                        scope=scope,
                        chunks=batch,
                        vectors=dense,
                        sparse_vectors=sparse,
                    )
                    self.repository.persist_chunk_batch(
                        work=work,
                        worker_id=worker_id,
                        chunks=batch,
                        point_ids=point_ids,
                    )
                    for chunk, point_id in zip(batch, point_ids):
                        manifest_chunks.append({
                            "chunk_id": chunk["id"],
                            "chunk_record_id": chunk["chunk_record_id"],
                            "document_id": chunk["document_id"],
                            "document_version_id": chunk["document_version_id"],
                            "content_sha256": hashlib.sha256(
                                str(chunk["content"]).encode("utf-8")
                            ).hexdigest(),
                            "qdrant_point_id": point_id,
                        })
                    self.repository.heartbeat(
                        job_id=work.job_id,
                        worker_id=worker_id,
                        stage="embedding",
                    )

            with trace.stage("index.validate_publish"):
                manifest = {
                "schema_version": "immutable-index-manifest-v1",
                "tenant_id": work.tenant_id,
                "knowledge_base_id": work.knowledge_base_id,
                "index_version_id": work.index_version_id,
                "document_versions": [
                    {
                        "document_id": document.document_id,
                        "document_version_id": document.document_version_id,
                        "source_id": document.source_id,
                        "chunk_count": document_counts[document.document_id],
                    }
                    for document in work.documents
                ],
                "chunks": manifest_chunks,
            }
                manifest_body = json.dumps(
                    manifest,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                manifest_entries_digest = index_entries_sha256(manifest_chunks)
                stored_manifest = self.object_storage.put_index_manifest(
                    tenant_id=work.tenant_id,
                    knowledge_base_id=work.knowledge_base_id,
                    index_version_id=work.index_version_id,
                    body=manifest_body,
                )
                self.repository.finalize_candidate(
                    work=work,
                    worker_id=worker_id,
                    document_chunk_counts=document_counts,
                    chunk_count=len(prepared),
                    manifest_sha256=stored_manifest.sha256,
                    manifest_object_key=stored_manifest.key,
                )
                point_count = self.vector_repository.count_index(scope)
                qdrant_entries_digest = self.vector_repository.index_entries_sha256(scope)
                self.tenant_repository.validate_index_version(
                    tenant_id=work.tenant_id,
                    knowledge_base_id=work.knowledge_base_id,
                    index_version_id=work.index_version_id,
                    manifest_sha256=stored_manifest.sha256,
                    manifest_object_key=stored_manifest.key,
                    manifest_entries_sha256=manifest_entries_digest,
                    qdrant_point_count=point_count,
                    qdrant_entries_sha256=qdrant_entries_digest,
                )
                self.tenant_repository.publish_index_version(
                    tenant_id=work.tenant_id,
                    knowledge_base_id=work.knowledge_base_id,
                    index_version_id=work.index_version_id,
                    expected_active_index_id=work.expected_active_index_id,
                    expected_generation=work.expected_generation,
                )
            published = True
            self.repository.complete(job_id=work.job_id, worker_id=worker_id)
            total_ms = elapsed_ms(started_at)
            trace.record("index.total", total_ms)
            trace_payload = trace.as_dict()
            return {
                "job_id": work.job_id,
                "status": "succeeded",
                "index_version_id": work.index_version_id,
                "document_count": len(work.documents),
                "chunk_count": len(prepared),
                "timings": {"totalMs": total_ms, "stages": trace_payload["stages"]},
                "timing_trace": trace_payload,
            }
        except (IndexBuildValidationError, InvalidServiceStateError, ConcurrentPublishError) as exc:
            status = self.repository.fail(
                job_id=work.job_id,
                worker_id=worker_id,
                error_code="INDEX_BUILD_REJECTED",
                error_class=exc.__class__.__name__,
                error_detail=str(exc),
                permanent=True,
            )
            total_ms = elapsed_ms(started_at)
            trace.record("index.total", total_ms, status="failed", error_class=exc.__class__.__name__)
            trace_payload = trace.as_dict()
            return {
                "job_id": work.job_id,
                "status": status,
                "timings": {"totalMs": total_ms, "stages": trace_payload["stages"]},
                "timing_trace": trace_payload,
            }
        except Exception as exc:
            if published:
                total_ms = elapsed_ms(started_at)
                trace.record("index.total", total_ms, status="failed", error_class=exc.__class__.__name__)
                trace_payload = trace.as_dict()
                return {
                    "job_id": work.job_id,
                    "status": "published_pending_reconciliation",
                    "timings": {"totalMs": total_ms, "stages": trace_payload["stages"]},
                    "timing_trace": trace_payload,
                }
            status = self.repository.fail(
                job_id=work.job_id,
                worker_id=worker_id,
                error_code="INDEX_BUILD_DEPENDENCY_FAILED",
                error_class=exc.__class__.__name__,
                error_detail=str(exc),
                permanent=False,
            )
            total_ms = elapsed_ms(started_at)
            trace.record("index.total", total_ms, status="failed", error_class=exc.__class__.__name__)
            trace_payload = trace.as_dict()
            return {
                "job_id": work.job_id,
                "status": status,
                "timings": {"totalMs": total_ms, "stages": trace_payload["stages"]},
                "timing_trace": trace_payload,
            }

    def _load_canonical(
        self,
        work: IndexBuildWorkItem,
        document: IndexDocumentArtifact,
    ) -> dict:
        payload = self.object_storage.download_bytes(
            document.extracted_object_key,
            max_bytes=self.max_canonical_bytes,
        )
        try:
            canonical = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise IndexBuildValidationError("canonical document JSON is invalid") from exc
        if not isinstance(canonical, dict) or canonical.get("schema_version") != "canonical-document-v1":
            raise IndexBuildValidationError("canonical document schema is unsupported")
        expected = {
            "tenant_id": work.tenant_id,
            "knowledge_base_id": work.knowledge_base_id,
            "document_id": document.document_id,
            "document_version_id": document.document_version_id,
        }
        for field, value in expected.items():
            if str(canonical.get(field) or "") != value:
                raise IndexBuildValidationError(
                    f"canonical document {field} is outside the build scope"
                )
        return canonical


def _page_number(value) -> int:
    match = re.search(r"\d+", str(value or ""))
    return int(match.group(0)) if match else 0


def _safe_error_detail(value: str) -> str:
    clean = " ".join(str(value or "").split())
    clean = clean.replace("/Users/", "/redacted/").replace("/app/", "/redacted/")
    return clean[:1000]


def _as_utc(value):
    if value.tzinfo is None:
        return value.replace(tzinfo=utc_now().tzinfo)
    return value
