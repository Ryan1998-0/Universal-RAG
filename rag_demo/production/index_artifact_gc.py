from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import and_, or_, select

from rag_demo.production.database import (
    IndexArtifactGcRecord,
    IndexBuildJobRecord,
    IndexVersionRecord,
    KnowledgeBaseRecord,
    new_id,
    utc_now,
)
from rag_demo.production.object_storage import index_manifest_object_key
from rag_demo.retrieval_scope import RetrievalScope


class IndexArtifactGcError(RuntimeError):
    pass


@dataclass(frozen=True)
class IndexArtifactGcWorkItem:
    gc_id: str
    tenant_id: str
    knowledge_base_id: str
    index_version_id: str
    artifact_attempt: int
    qdrant_collection: str
    manifest_object_key: str
    attempt: int
    max_attempts: int


def ensure_index_artifact_gc(
    session,
    *,
    tenant_id: str,
    knowledge_base_id: str,
    index_version_id: str,
    artifact_attempt: int,
    qdrant_collection: str,
    manifest_object_key: str = "",
    next_attempt_at: Optional[datetime] = None,
) -> IndexArtifactGcRecord:
    """Create or retain one durable cleanup target for an artifact attempt."""
    if type(artifact_attempt) is not int or artifact_attempt <= 0:
        raise ValueError("index artifact GC requires a positive attempt")
    clean_collection = str(qdrant_collection or "").strip()
    if not clean_collection:
        raise ValueError("qdrant_collection is required for index artifact GC")
    clean_key = str(manifest_object_key or "").strip()
    if not clean_key:
        clean_key = index_manifest_object_key(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            index_version_id=index_version_id,
            artifact_attempt=artifact_attempt,
        )
    if not clean_key.startswith("tenants/"):
        raise ValueError("manifest_object_key must be tenant-scoped")
    scheduled_at = next_attempt_at or utc_now()
    existing = session.scalar(
        select(IndexArtifactGcRecord)
        .where(
            IndexArtifactGcRecord.tenant_id == tenant_id,
            IndexArtifactGcRecord.knowledge_base_id == knowledge_base_id,
            IndexArtifactGcRecord.index_version_id == index_version_id,
            IndexArtifactGcRecord.artifact_attempt == artifact_attempt,
        )
        .with_for_update()
    )
    if existing is not None:
        now = utc_now()
        if (
            existing.state == "running"
            and existing.lease_expires_at is not None
            and _as_utc(existing.lease_expires_at) > now
        ):
            return existing
        existing.qdrant_collection = clean_collection
        existing.manifest_object_key = clean_key
        existing.state = "pending"
        existing.next_attempt_at = _earliest(existing.next_attempt_at, scheduled_at)
        existing.error_code = ""
        existing.error_detail = ""
        existing.completed_at = None
        existing.lease_owner = ""
        existing.lease_expires_at = None
        return existing

    row = IndexArtifactGcRecord(
        id=new_id(),
        tenant_id=str(tenant_id),
        knowledge_base_id=str(knowledge_base_id),
        index_version_id=str(index_version_id),
        artifact_attempt=artifact_attempt,
        qdrant_collection=clean_collection,
        manifest_object_key=clean_key,
        state="pending",
        next_attempt_at=scheduled_at,
    )
    session.add(row)
    return row


class SqlAlchemyIndexArtifactGcRepository:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def claim(
        self,
        *,
        gc_id: str,
        worker_id: str,
        lease_seconds: int = 600,
    ) -> Optional[IndexArtifactGcWorkItem]:
        now = utc_now()
        with self.session_factory.begin() as session:
            row = session.scalar(
                select(IndexArtifactGcRecord)
                .where(IndexArtifactGcRecord.id == gc_id)
                .with_for_update()
            )
            if row is None or row.state == "dead":
                return None
            if row.next_attempt_at is not None and _as_utc(row.next_attempt_at) > now:
                return None
            if (
                row.state == "running"
                and row.lease_expires_at is not None
                and _as_utc(row.lease_expires_at) > now
                and row.lease_owner != worker_id
            ):
                return None
            row.state = "running"
            row.attempt += 1
            row.lease_owner = str(worker_id)
            row.lease_expires_at = now + timedelta(
                seconds=max(120, int(lease_seconds))
            )
            row.next_attempt_at = None
            row.error_code = ""
            row.error_detail = ""
            return _work_item(row)

    def dispatchable_ids(
        self,
        *,
        limit: int = 100,
        redispatch_after_seconds: int = 90,
    ) -> list[str]:
        now = utc_now()
        stale = now - timedelta(seconds=max(30, int(redispatch_after_seconds)))
        due = or_(
            IndexArtifactGcRecord.next_attempt_at.is_(None),
            IndexArtifactGcRecord.next_attempt_at <= now,
        )
        with self.session_factory() as session:
            return list(session.scalars(
                select(IndexArtifactGcRecord.id)
                .where(or_(
                    and_(
                        IndexArtifactGcRecord.state.in_({"pending", "completed"}),
                        due,
                        or_(
                            IndexArtifactGcRecord.last_dispatched_at.is_(None),
                            IndexArtifactGcRecord.last_dispatched_at <= stale,
                        ),
                    ),
                    and_(
                        IndexArtifactGcRecord.state == "retry_wait",
                        due,
                    ),
                    and_(
                        IndexArtifactGcRecord.state == "running",
                        IndexArtifactGcRecord.lease_expires_at.is_not(None),
                        IndexArtifactGcRecord.lease_expires_at <= now,
                    ),
                ))
                .order_by(
                    IndexArtifactGcRecord.next_attempt_at,
                    IndexArtifactGcRecord.created_at,
                    IndexArtifactGcRecord.id,
                )
                .limit(max(1, min(int(limit), 1000)))
            ))

    def record_dispatch(self, *, gc_id: str, task_id: str) -> None:
        with self.session_factory.begin() as session:
            row = session.get(IndexArtifactGcRecord, gc_id)
            if row is None:
                raise IndexArtifactGcError("index artifact GC item was not found")
            row.worker_task_id = str(task_id)[:255]
            row.dispatch_count += 1
            row.last_dispatched_at = utc_now()

    def is_protected(self, work: IndexArtifactGcWorkItem) -> bool:
        now = utc_now()
        with self.session_factory() as session:
            active_index_id = session.scalar(
                select(KnowledgeBaseRecord.active_index_version_id).where(
                    KnowledgeBaseRecord.id == work.knowledge_base_id,
                    KnowledgeBaseRecord.tenant_id == work.tenant_id,
                    KnowledgeBaseRecord.active.is_(True),
                )
            )
            if active_index_id == work.index_version_id:
                active_attempt = session.scalar(
                    select(IndexVersionRecord.artifact_attempt).where(
                        IndexVersionRecord.id == work.index_version_id,
                        IndexVersionRecord.tenant_id == work.tenant_id,
                        IndexVersionRecord.knowledge_base_id == work.knowledge_base_id,
                    )
                )
                if int(active_attempt or 0) == work.artifact_attempt:
                    return True
            running = session.scalar(
                select(IndexBuildJobRecord.id)
                .where(
                    IndexBuildJobRecord.tenant_id == work.tenant_id,
                    IndexBuildJobRecord.knowledge_base_id == work.knowledge_base_id,
                    IndexBuildJobRecord.index_version_id == work.index_version_id,
                    IndexBuildJobRecord.attempt == work.artifact_attempt,
                    IndexBuildJobRecord.status == "running",
                    IndexBuildJobRecord.lease_expires_at.is_not(None),
                    IndexBuildJobRecord.lease_expires_at > now,
                )
            )
            return running is not None

    def complete(
        self,
        *,
        gc_id: str,
        worker_id: str,
        recheck_seconds: int = 3600,
    ) -> None:
        now = utc_now()
        with self.session_factory.begin() as session:
            row = session.scalar(
                select(IndexArtifactGcRecord)
                .where(
                    IndexArtifactGcRecord.id == gc_id,
                    IndexArtifactGcRecord.state == "running",
                    IndexArtifactGcRecord.lease_owner == worker_id,
                )
                .with_for_update()
            )
            if row is None:
                raise IndexArtifactGcError("index artifact GC lease was lost")
            row.state = "completed"
            row.completed_at = now
            row.last_checked_at = now
            row.next_attempt_at = now + timedelta(
                seconds=max(60, int(recheck_seconds))
            )
            row.lease_owner = ""
            row.lease_expires_at = None

    def defer(
        self,
        *,
        gc_id: str,
        worker_id: str,
        seconds: int,
        reason: str,
    ) -> str:
        now = utc_now()
        with self.session_factory.begin() as session:
            row = session.scalar(
                select(IndexArtifactGcRecord)
                .where(
                    IndexArtifactGcRecord.id == gc_id,
                    IndexArtifactGcRecord.state == "running",
                    IndexArtifactGcRecord.lease_owner == worker_id,
                )
                .with_for_update()
            )
            if row is None:
                return "lease_lost"
            row.state = "retry_wait"
            row.next_attempt_at = now + timedelta(seconds=max(60, int(seconds)))
            row.error_code = "GC_DEFERRED"
            row.error_detail = _safe_error_detail(reason)
            row.lease_owner = ""
            row.lease_expires_at = None
            return row.state

    def fail(
        self,
        *,
        gc_id: str,
        worker_id: str,
        error_code: str,
        error_detail: str,
    ) -> str:
        now = utc_now()
        with self.session_factory.begin() as session:
            row = session.scalar(
                select(IndexArtifactGcRecord)
                .where(
                    IndexArtifactGcRecord.id == gc_id,
                    IndexArtifactGcRecord.state == "running",
                    IndexArtifactGcRecord.lease_owner == worker_id,
                )
                .with_for_update()
            )
            if row is None:
                return "lease_lost"
            terminal = row.attempt >= row.max_attempts
            row.state = "dead" if terminal else "retry_wait"
            row.error_code = str(error_code)[:80]
            row.error_detail = _safe_error_detail(error_detail)
            row.lease_owner = ""
            row.lease_expires_at = None
            if terminal:
                row.completed_at = now
                row.next_attempt_at = None
            else:
                delay = min(6 * 3600, 60 * (2 ** max(0, row.attempt - 1)))
                row.next_attempt_at = now + timedelta(seconds=delay)
            return row.state


class IndexArtifactGcService:
    def __init__(self, *, repository, object_storage, vector_repository):
        self.repository = repository
        self.object_storage = object_storage
        self.vector_repository = vector_repository

    def process(self, *, gc_id: str, worker_id: str) -> dict:
        work = self.repository.claim(gc_id=gc_id, worker_id=worker_id)
        if work is None:
            return {"gc_id": gc_id, "status": "not_claimed"}
        try:
            if self.repository.is_protected(work):
                status = self.repository.defer(
                    gc_id=work.gc_id,
                    worker_id=worker_id,
                    seconds=300,
                    reason="active index or valid build lease still owns this attempt",
                )
                return {"gc_id": work.gc_id, "status": status}
            active_collection = str(
                getattr(self.vector_repository, "collection_name", "") or ""
            )
            if active_collection != work.qdrant_collection:
                raise IndexArtifactGcError(
                    "index artifact GC collection does not match the worker adapter"
                )
            scope = RetrievalScope(
                tenant_id=work.tenant_id,
                knowledge_base_id=work.knowledge_base_id,
                index_version_id=work.index_version_id,
                artifact_attempt=work.artifact_attempt,
            )
            self.vector_repository.delete_attempt(scope)
            remaining = self.vector_repository.count_index(scope)
            if remaining:
                raise IndexArtifactGcError(
                    f"{remaining} Qdrant points remain after attempt cleanup"
                )
            self.object_storage.delete(work.manifest_object_key)
            self.repository.complete(gc_id=work.gc_id, worker_id=worker_id)
            return {"gc_id": work.gc_id, "status": "completed"}
        except Exception as exc:
            status = self.repository.fail(
                gc_id=work.gc_id,
                worker_id=worker_id,
                error_code="INDEX_ARTIFACT_GC_FAILED",
                error_detail=str(exc),
            )
            return {"gc_id": work.gc_id, "status": status}


def _work_item(row: IndexArtifactGcRecord) -> IndexArtifactGcWorkItem:
    return IndexArtifactGcWorkItem(
        gc_id=row.id,
        tenant_id=row.tenant_id,
        knowledge_base_id=row.knowledge_base_id,
        index_version_id=row.index_version_id,
        artifact_attempt=int(row.artifact_attempt),
        qdrant_collection=row.qdrant_collection,
        manifest_object_key=row.manifest_object_key,
        attempt=int(row.attempt),
        max_attempts=int(row.max_attempts),
    )


def _earliest(first: Optional[datetime], second: datetime) -> datetime:
    if first is None:
        return second
    return first if _as_utc(first) <= _as_utc(second) else second


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _safe_error_detail(value: object) -> str:
    return str(value or "")[:2000]
