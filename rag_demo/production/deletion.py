from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

from sqlalchemy import and_, or_, select

from rag_demo.production.database import (
    DeletionOutboxRecord,
    DocumentRecord,
    DocumentVersionRecord,
    IndexBuildJobRecord,
    IndexDocumentRecord,
    IngestionJobRecord,
    UserIdentity,
    new_id,
    utc_now,
)
from rag_demo.production.object_storage import extracted_object_key
from rag_demo.production.repository import (
    AccessDeniedError,
    InvalidServiceStateError,
    ResourceNotFoundError,
)
from rag_demo.retrieval_scope import RetrievalScope


_IDEMPOTENCY_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{8,128}$")


@dataclass(frozen=True)
class DeletionReservation:
    outbox_id: str
    document_id: str
    state: str
    duplicate: bool


@dataclass(frozen=True)
class DeletionWorkItem:
    outbox_id: str
    tenant_id: str
    knowledge_base_id: str
    document_id: str
    object_keys: tuple[str, ...]
    vector_scopes: tuple[dict, ...]
    attempt: int
    max_attempts: int


class SqlAlchemyDeletionRepository:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def tombstone_document(
        self,
        *,
        principal,
        authorized,
        document_id: str,
        idempotency_key: str,
    ) -> DeletionReservation:
        if not authorized.can_write:
            raise AccessDeniedError("knowledge base write access is denied")
        clean_key = str(idempotency_key or "").strip()
        if not _IDEMPOTENCY_PATTERN.fullmatch(clean_key):
            raise ValueError("Idempotency-Key must contain 8 to 128 safe characters")
        clean_document_id = str(document_id or "").strip()
        if not clean_document_id:
            raise ValueError("document_id is required")
        now = utc_now()
        with self.session_factory.begin() as session:
            existing = session.scalar(select(DeletionOutboxRecord).where(
                DeletionOutboxRecord.tenant_id == principal.tenant_id,
                DeletionOutboxRecord.idempotency_key == clean_key,
            ))
            if existing is not None:
                if (
                    existing.knowledge_base_id != authorized.id
                    or existing.created_by_user_id != authorized.user_id
                    or existing.resource_type != "document"
                    or existing.resource_id != clean_document_id
                ):
                    raise InvalidServiceStateError(
                        "idempotency key was already used for another deletion"
                    )
                return DeletionReservation(
                    outbox_id=existing.id,
                    document_id=existing.resource_id,
                    state=existing.state,
                    duplicate=True,
                )

            document = session.scalar(
                select(DocumentRecord)
                .where(
                    DocumentRecord.id == clean_document_id,
                    DocumentRecord.tenant_id == principal.tenant_id,
                    DocumentRecord.knowledge_base_id == authorized.id,
                )
                .with_for_update()
            )
            if document is None:
                raise ResourceNotFoundError("document was not found")
            versions = list(session.scalars(
                select(DocumentVersionRecord).where(
                    DocumentVersionRecord.tenant_id == principal.tenant_id,
                    DocumentVersionRecord.document_id == document.id,
                )
            ))
            version_ids = [version.id for version in versions]
            object_keys = []
            for version in versions:
                object_keys.append(version.object_key)
                if version.extracted_object_key:
                    object_keys.append(version.extracted_object_key)
                object_keys.append(extracted_object_key(
                    tenant_id=principal.tenant_id,
                    knowledge_base_id=authorized.id,
                    document_id=document.id,
                    version_id=version.id,
                    artifact_name="canonical.json",
                ))
            memberships = session.execute(
                select(
                    IndexDocumentRecord.index_version_id,
                    IndexDocumentRecord.document_version_id,
                ).where(
                    IndexDocumentRecord.tenant_id == principal.tenant_id,
                    IndexDocumentRecord.knowledge_base_id == authorized.id,
                    IndexDocumentRecord.document_id == document.id,
                )
            ).all()
            vector_scopes = [
                {
                    "tenant_id": principal.tenant_id,
                    "knowledge_base_id": authorized.id,
                    "index_version_id": row.index_version_id,
                    "document_version_id": row.document_version_id,
                }
                for row in memberships
            ]

            document.deleted_at = document.deleted_at or now
            document.status = "deleted"
            if version_ids:
                for job in session.scalars(select(IngestionJobRecord).where(
                    IngestionJobRecord.tenant_id == principal.tenant_id,
                    IngestionJobRecord.document_version_id.in_(version_ids),
                    IngestionJobRecord.status.in_({"queued", "running", "retry_wait"}),
                )):
                    job.status = "cancelled"
                    job.stage = "cancelled"
                    job.completed_at = now
                    job.lease_owner = ""
                    job.lease_expires_at = None
            index_ids = sorted(set(row.index_version_id for row in memberships))
            if index_ids:
                for job in session.scalars(select(IndexBuildJobRecord).where(
                    IndexBuildJobRecord.tenant_id == principal.tenant_id,
                    IndexBuildJobRecord.index_version_id.in_(index_ids),
                    IndexBuildJobRecord.status.in_({"queued", "running", "retry_wait"}),
                )):
                    job.status = "cancelled"
                    job.stage = "cancelled"
                    job.completed_at = now
                    job.lease_owner = ""
                    job.lease_expires_at = None

            outbox = DeletionOutboxRecord(
                id=new_id(),
                tenant_id=principal.tenant_id,
                knowledge_base_id=authorized.id,
                created_by_user_id=authorized.user_id,
                resource_type="document",
                resource_id=document.id,
                idempotency_key=clean_key,
                object_keys_json=sorted(set(key for key in object_keys if key)),
                vector_scopes_json=vector_scopes,
                state="pending",
                next_attempt_at=now + timedelta(seconds=60),
            )
            session.add(outbox)
            return DeletionReservation(
                outbox_id=outbox.id,
                document_id=document.id,
                state=outbox.state,
                duplicate=False,
            )

    def claim(
        self,
        *,
        outbox_id: str,
        worker_id: str,
        lease_seconds: int = 600,
    ) -> Optional[DeletionWorkItem]:
        now = utc_now()
        with self.session_factory.begin() as session:
            row = session.scalar(
                select(DeletionOutboxRecord)
                .where(DeletionOutboxRecord.id == outbox_id)
                .with_for_update()
            )
            if row is None or row.state in {"completed", "dead"}:
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
            row.lease_owner = worker_id
            row.lease_expires_at = now + timedelta(
                seconds=max(120, int(lease_seconds))
            )
            row.next_attempt_at = None
            row.error_code = ""
            row.error_detail = ""
            return DeletionWorkItem(
                outbox_id=row.id,
                tenant_id=row.tenant_id,
                knowledge_base_id=row.knowledge_base_id,
                document_id=row.resource_id,
                object_keys=tuple(row.object_keys_json or []),
                vector_scopes=tuple(row.vector_scopes_json or []),
                attempt=row.attempt,
                max_attempts=row.max_attempts,
            )

    def dispatchable_ids(
        self,
        *,
        limit: int = 100,
        redispatch_after_seconds: int = 90,
    ) -> list[str]:
        now = utc_now()
        stale = now - timedelta(seconds=max(30, int(redispatch_after_seconds)))
        with self.session_factory() as session:
            return list(session.scalars(
                select(DeletionOutboxRecord.id)
                .where(or_(
                    and_(
                        DeletionOutboxRecord.state == "pending",
                        or_(
                            DeletionOutboxRecord.next_attempt_at.is_(None),
                            DeletionOutboxRecord.next_attempt_at <= now,
                        ),
                        or_(
                            DeletionOutboxRecord.last_dispatched_at.is_(None),
                            DeletionOutboxRecord.last_dispatched_at <= stale,
                        ),
                    ),
                    and_(
                        DeletionOutboxRecord.state == "retry_wait",
                        or_(
                            DeletionOutboxRecord.next_attempt_at.is_(None),
                            DeletionOutboxRecord.next_attempt_at <= now,
                        ),
                    ),
                    and_(
                        DeletionOutboxRecord.state == "running",
                        DeletionOutboxRecord.lease_expires_at.is_not(None),
                        DeletionOutboxRecord.lease_expires_at <= now,
                    ),
                ))
                .order_by(DeletionOutboxRecord.created_at, DeletionOutboxRecord.id)
                .limit(max(1, min(int(limit), 1000)))
            ))

    def record_dispatch(self, *, outbox_id: str, task_id: str) -> None:
        with self.session_factory.begin() as session:
            row = session.get(DeletionOutboxRecord, outbox_id)
            if row is None:
                raise ResourceNotFoundError("deletion outbox item was not found")
            row.worker_task_id = str(task_id)[:255]
            row.dispatch_count += 1
            row.last_dispatched_at = utc_now()

    def complete(self, *, outbox_id: str, worker_id: str) -> None:
        now = utc_now()
        with self.session_factory.begin() as session:
            row = session.scalar(
                select(DeletionOutboxRecord)
                .where(
                    DeletionOutboxRecord.id == outbox_id,
                    DeletionOutboxRecord.state == "running",
                    DeletionOutboxRecord.lease_owner == worker_id,
                )
                .with_for_update()
            )
            if row is None:
                raise RuntimeError("deletion outbox lease was lost")
            row.state = "completed"
            row.completed_at = now
            row.lease_owner = ""
            row.lease_expires_at = None

    def fail(
        self,
        *,
        outbox_id: str,
        worker_id: str,
        error_code: str,
        error_detail: str,
    ) -> str:
        now = utc_now()
        with self.session_factory.begin() as session:
            row = session.scalar(
                select(DeletionOutboxRecord)
                .where(
                    DeletionOutboxRecord.id == outbox_id,
                    DeletionOutboxRecord.lease_owner == worker_id,
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

    def get_status(self, *, principal, outbox_id: str) -> dict:
        with self.session_factory() as session:
            result = session.execute(
                select(DeletionOutboxRecord, UserIdentity.subject)
                .join(
                    UserIdentity,
                    UserIdentity.id == DeletionOutboxRecord.created_by_user_id,
                )
                .where(
                    DeletionOutboxRecord.id == outbox_id,
                    DeletionOutboxRecord.tenant_id == principal.tenant_id,
                )
            ).one_or_none()
            if result is None or result.subject != principal.subject:
                raise ResourceNotFoundError("deletion outbox item was not found")
            row = result[0]
            return {
                "id": row.id,
                "resource_type": row.resource_type,
                "resource_id": row.resource_id,
                "state": row.state,
                "attempt": row.attempt,
                "max_attempts": row.max_attempts,
                "error_code": row.error_code,
                "created_at": row.created_at.isoformat(),
                "updated_at": row.updated_at.isoformat(),
            }


class DeletionService:
    def __init__(self, *, repository, object_storage, vector_repository):
        self.repository = repository
        self.object_storage = object_storage
        self.vector_repository = vector_repository

    def process(self, *, outbox_id: str, worker_id: str) -> dict:
        work = self.repository.claim(outbox_id=outbox_id, worker_id=worker_id)
        if work is None:
            return {"outbox_id": outbox_id, "status": "not_claimed"}
        try:
            for key in work.object_keys:
                self.object_storage.delete(key)
            for raw_scope in work.vector_scopes:
                scope = RetrievalScope(
                    tenant_id=str(raw_scope["tenant_id"]),
                    knowledge_base_id=str(raw_scope["knowledge_base_id"]),
                    index_version_id=str(raw_scope["index_version_id"]),
                )
                if (
                    scope.tenant_id != work.tenant_id
                    or scope.knowledge_base_id != work.knowledge_base_id
                ):
                    raise InvalidServiceStateError(
                        "deletion vector scope is outside the outbox tenant"
                    )
                self.vector_repository.delete_document_version(
                    scope=scope,
                    document_version_id=str(raw_scope["document_version_id"]),
                )
            self.repository.complete(outbox_id=work.outbox_id, worker_id=worker_id)
            return {"outbox_id": work.outbox_id, "status": "completed"}
        except Exception as exc:
            status = self.repository.fail(
                outbox_id=work.outbox_id,
                worker_id=worker_id,
                error_code="DELETION_DEPENDENCY_FAILED",
                error_detail=f"{exc.__class__.__name__}: {exc}",
            )
            return {"outbox_id": work.outbox_id, "status": status}


def _safe_error_detail(value: str) -> str:
    clean = " ".join(str(value or "").split())
    clean = clean.replace("/Users/", "/redacted/").replace("/app/", "/redacted/")
    return clean[:1000]


def _as_utc(value):
    if value.tzinfo is None:
        return value.replace(tzinfo=utc_now().tzinfo)
    return value
