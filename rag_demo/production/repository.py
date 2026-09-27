from contextlib import nullcontext
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import logging
import re
from typing import Iterable, List, Optional, Sequence

from sqlalchemy import and_, func, select, text
from sqlalchemy.exc import IntegrityError

from rag_demo.production.auth import Principal
from rag_demo.production.database import (
    AnswerRunRecord,
    AuditEventRecord,
    CitationRecord,
    ChunkRecord,
    ConversationRecord,
    DocumentRecord,
    DocumentVersionRecord,
    FolderRecord,
    IndexBuildJobRecord,
    IndexDocumentRecord,
    IndexActivationEventRecord,
    IndexVersionRecord,
    KnowledgeBaseRecord,
    Membership,
    MessageRecord,
    Tenant,
    IngestionJobRecord,
    UploadSessionRecord,
    UserIdentity,
    new_id,
    utc_now,
)
from rag_demo.production.index_manifest import index_entries_sha256
from rag_demo.production.index_artifact_gc import ensure_index_artifact_gc


class AccessDeniedError(PermissionError):
    pass


class ResourceNotFoundError(LookupError):
    pass


class InvalidServiceStateError(RuntimeError):
    pass


class ConcurrentPublishError(RuntimeError):
    pass


_WITHDRAWN_ANSWER_TEXT = "此歷史回答的來源無法確認或已失效，因此不再顯示。"
_CITATION_LINEAGE_KEY = "citation_lineage_v1"
_CONTEXT_LINEAGE_KEY = "context_lineage_v1"
_CONTEXT_CHUNK_LINEAGE_VERSION = "chunk_record_v1"
_HISTORY_DEPENDENCY_KEY = "history_dependency_v1"
LOGGER = logging.getLogger("rag_demo.production.repository")


@dataclass(frozen=True)
class AuthorizedKnowledgeBase:
    id: str
    profile: str
    user_id: str
    source_ids: List[str]
    active_index_version_id: Optional[str]
    activation_generation: int
    can_write: bool
    embedding_model: Optional[str] = None
    sparse_embedding_model: Optional[str] = None
    embedding_dimensions: Optional[int] = None
    chunk_schema_version: Optional[str] = None
    qdrant_collection: Optional[str] = None
    artifact_attempt: int = 0


@dataclass(frozen=True)
class UploadReservation:
    id: str
    object_key: str
    state: str
    expires_at: datetime
    expected_size_bytes: int
    expected_sha256: str
    declared_mime_type: str


@dataclass(frozen=True)
class CompletedUpload:
    upload_id: str
    document_id: str
    document_version_id: str
    ingestion_job_id: str
    duplicate: bool


class SqlAlchemyTenantRepository:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def ping(self) -> None:
        with self.session_factory() as session:
            session.execute(text("SELECT 1"))
            session.execute(select(Tenant.id).limit(1))

    def list_knowledge_bases(self, principal: Principal) -> list[dict]:
        with self.session_factory() as session:
            user, membership = self._active_identity(session, principal)
            query = select(KnowledgeBaseRecord).where(
                KnowledgeBaseRecord.tenant_id == principal.tenant_id,
                KnowledgeBaseRecord.active.is_(True),
            )
            if not (
                membership.role in {"owner", "admin"}
                or principal.has_role("tenant_admin")
            ):
                query = query.where(
                    (KnowledgeBaseRecord.owner_user_id == user.id)
                    | (KnowledgeBaseRecord.visibility == "tenant")
                )
            rows = list(session.scalars(
                query.order_by(KnowledgeBaseRecord.created_at, KnowledgeBaseRecord.id)
            ))
            return [
                {
                    "id": row.id,
                    "name": row.name,
                    "profile": row.profile,
                    "visibility": row.visibility,
                    "active_index_version_id": row.active_index_version_id,
                    "activation_generation": row.activation_generation,
                    "can_write": (
                        membership.role in {"owner", "admin"}
                        or principal.has_role("tenant_admin")
                        or row.owner_user_id == user.id
                    ),
                    "created_at": row.created_at.isoformat(),
                    "updated_at": row.updated_at.isoformat(),
                }
                for row in rows
            ]

    def create_knowledge_base(
        self,
        *,
        principal: Principal,
        name: str,
        profile: str,
        visibility: str,
    ) -> dict:
        clean_name = " ".join(str(name or "").split())
        clean_profile = str(profile or "default").strip()
        clean_visibility = str(visibility or "private").strip().lower()
        if not clean_name or len(clean_name) > 160:
            raise ValueError("knowledge base name is invalid")
        if not clean_profile or len(clean_profile) > 64:
            raise ValueError("knowledge base profile is invalid")
        if clean_visibility not in {"private", "tenant"}:
            raise ValueError("knowledge base visibility is invalid")
        try:
            with self.session_factory.begin() as session:
                user, _membership = self._active_identity(session, principal)
                existing = session.scalar(select(KnowledgeBaseRecord).where(
                    KnowledgeBaseRecord.tenant_id == principal.tenant_id,
                    KnowledgeBaseRecord.name == clean_name,
                ))
                if existing is not None:
                    raise InvalidServiceStateError(
                        "a knowledge base with this name already exists"
                    )
                row = KnowledgeBaseRecord(
                    tenant_id=principal.tenant_id,
                    owner_user_id=user.id,
                    name=clean_name,
                    profile=clean_profile,
                    visibility=clean_visibility,
                )
                session.add(row)
                session.flush()
                return {
                    "id": row.id,
                    "name": row.name,
                    "profile": row.profile,
                    "visibility": row.visibility,
                    "active_index_version_id": None,
                    "activation_generation": 0,
                    "can_write": True,
                    "created_at": row.created_at.isoformat(),
                    "updated_at": row.updated_at.isoformat(),
                }
        except IntegrityError as exc:
            raise InvalidServiceStateError(
                "a knowledge base with this name already exists"
            ) from exc

    def list_folders(
        self,
        *,
        principal: Principal,
        authorized: AuthorizedKnowledgeBase,
    ) -> list[dict]:
        with self.session_factory() as session:
            rows = list(session.scalars(
                select(FolderRecord)
                .where(
                    FolderRecord.tenant_id == principal.tenant_id,
                    FolderRecord.knowledge_base_id == authorized.id,
                )
                .order_by(FolderRecord.created_at, FolderRecord.id)
            ))
            return [
                {
                    "id": row.id,
                    "name": row.name,
                    "created_at": row.created_at.isoformat(),
                }
                for row in rows
            ]

    def create_folder(
        self,
        *,
        principal: Principal,
        authorized: AuthorizedKnowledgeBase,
        name: str,
    ) -> dict:
        if not authorized.can_write:
            raise AccessDeniedError("knowledge base write access is denied")
        clean_name = " ".join(str(name or "").split())
        if not clean_name or len(clean_name) > 160:
            raise ValueError("folder name is invalid")
        try:
            with self.session_factory.begin() as session:
                existing = session.scalar(select(FolderRecord).where(
                    FolderRecord.tenant_id == principal.tenant_id,
                    FolderRecord.knowledge_base_id == authorized.id,
                    FolderRecord.name == clean_name,
                ))
                if existing is not None:
                    raise InvalidServiceStateError("a folder with this name already exists")
                row = FolderRecord(
                    tenant_id=principal.tenant_id,
                    knowledge_base_id=authorized.id,
                    name=clean_name,
                )
                session.add(row)
                session.flush()
                return {
                    "id": row.id,
                    "name": row.name,
                    "created_at": row.created_at.isoformat(),
                }
        except IntegrityError as exc:
            raise InvalidServiceStateError(
                "a folder with this name already exists"
            ) from exc

    def rename_folder(
        self,
        *,
        principal: Principal,
        authorized: AuthorizedKnowledgeBase,
        folder_id: str,
        name: str,
    ) -> dict:
        if not authorized.can_write:
            raise AccessDeniedError("knowledge base write access is denied")
        clean_name = " ".join(str(name or "").split())
        if not clean_name or len(clean_name) > 160:
            raise ValueError("folder name is invalid")
        try:
            with self.session_factory.begin() as session:
                row = session.scalar(
                    select(FolderRecord)
                    .where(
                        FolderRecord.id == folder_id,
                        FolderRecord.tenant_id == principal.tenant_id,
                        FolderRecord.knowledge_base_id == authorized.id,
                    )
                    .with_for_update()
                )
                if row is None:
                    raise ResourceNotFoundError("folder was not found")
                duplicate = session.scalar(select(FolderRecord.id).where(
                    FolderRecord.tenant_id == principal.tenant_id,
                    FolderRecord.knowledge_base_id == authorized.id,
                    FolderRecord.name == clean_name,
                    FolderRecord.id != row.id,
                ))
                if duplicate is not None:
                    raise InvalidServiceStateError("a folder with this name already exists")
                row.name = clean_name
                session.flush()
                return {
                    "id": row.id,
                    "name": row.name,
                    "created_at": row.created_at.isoformat(),
                }
        except IntegrityError as exc:
            raise InvalidServiceStateError(
                "a folder with this name already exists"
            ) from exc

    def delete_folder(
        self,
        *,
        principal: Principal,
        authorized: AuthorizedKnowledgeBase,
        folder_id: str,
    ) -> dict:
        if not authorized.can_write:
            raise AccessDeniedError("knowledge base write access is denied")
        with self.session_factory.begin() as session:
            row = session.scalar(
                select(FolderRecord)
                .where(
                    FolderRecord.id == folder_id,
                    FolderRecord.tenant_id == principal.tenant_id,
                    FolderRecord.knowledge_base_id == authorized.id,
                )
                .with_for_update()
            )
            if row is None:
                raise ResourceNotFoundError("folder was not found")
            documents = list(session.scalars(
                select(DocumentRecord).where(
                    DocumentRecord.tenant_id == principal.tenant_id,
                    DocumentRecord.knowledge_base_id == authorized.id,
                    DocumentRecord.folder_id == row.id,
                    DocumentRecord.deleted_at.is_(None),
                )
            ))
            for document in documents:
                document.folder_id = None
            session.delete(row)
            return {
                "deleted": True,
                "id": folder_id,
                "moved_document_count": len(documents),
            }

    def move_document(
        self,
        *,
        principal: Principal,
        authorized: AuthorizedKnowledgeBase,
        document_id: str,
        folder_id: Optional[str],
    ) -> dict:
        if not authorized.can_write:
            raise AccessDeniedError("knowledge base write access is denied")
        clean_folder_id = str(folder_id or "").strip() or None
        with self.session_factory.begin() as session:
            document = session.scalar(
                select(DocumentRecord)
                .where(
                    DocumentRecord.id == document_id,
                    DocumentRecord.tenant_id == principal.tenant_id,
                    DocumentRecord.knowledge_base_id == authorized.id,
                    DocumentRecord.deleted_at.is_(None),
                )
                .with_for_update()
            )
            if document is None:
                raise ResourceNotFoundError("document was not found")
            if clean_folder_id:
                folder = session.scalar(select(FolderRecord.id).where(
                    FolderRecord.id == clean_folder_id,
                    FolderRecord.tenant_id == principal.tenant_id,
                    FolderRecord.knowledge_base_id == authorized.id,
                ))
                if folder is None:
                    raise ResourceNotFoundError("folder was not found")
            document.folder_id = clean_folder_id
            session.flush()
            return {
                "id": document.id,
                "source_id": document.source_id,
                "name": document.name,
                "folder_id": document.folder_id,
                "status": document.status,
            }

    def list_documents(
        self,
        *,
        principal: Principal,
        authorized: AuthorizedKnowledgeBase,
    ) -> list[dict]:
        with self.session_factory() as session:
            documents = list(session.scalars(
                select(DocumentRecord)
                .where(
                    DocumentRecord.tenant_id == principal.tenant_id,
                    DocumentRecord.knowledge_base_id == authorized.id,
                    DocumentRecord.deleted_at.is_(None),
                )
                .order_by(DocumentRecord.created_at, DocumentRecord.id)
            ))
            active_membership = {}
            if authorized.active_index_version_id:
                active_membership = {
                    row.document_id: row.document_version_id
                    for row in session.scalars(select(IndexDocumentRecord).where(
                        IndexDocumentRecord.tenant_id == principal.tenant_id,
                        IndexDocumentRecord.knowledge_base_id == authorized.id,
                        IndexDocumentRecord.index_version_id
                        == authorized.active_index_version_id,
                        IndexDocumentRecord.status == "ready",
                    ))
                }
            result = []
            for document in documents:
                latest = session.scalar(
                    select(DocumentVersionRecord)
                    .where(
                        DocumentVersionRecord.tenant_id == principal.tenant_id,
                        DocumentVersionRecord.document_id == document.id,
                    )
                    .order_by(DocumentVersionRecord.version_number.desc())
                    .limit(1)
                )
                result.append({
                    "id": document.id,
                    "source_id": document.source_id,
                    "name": document.name,
                    "folder_id": document.folder_id,
                    "status": document.status,
                    "current_version_id": document.current_version_id,
                    "latest_version": (
                        {
                            "id": latest.id,
                            "version_number": latest.version_number,
                            "status": latest.status,
                            "mime_type": latest.mime_type,
                            "size_bytes": latest.size_bytes,
                            "page_count": latest.page_count,
                            "created_at": latest.created_at.isoformat(),
                        }
                        if latest is not None
                        else None
                    ),
                    "in_active_index": (
                        active_membership.get(document.id)
                        == document.current_version_id
                        and document.current_version_id is not None
                    ),
                    "created_at": document.created_at.isoformat(),
                    "updated_at": document.updated_at.isoformat(),
                })
            return result

    def ensure_conversation(
        self,
        *,
        principal: Principal,
        authorized: AuthorizedKnowledgeBase,
        conversation_id: Optional[str] = None,
    ) -> dict:
        clean_id = str(conversation_id or "").strip()
        with self.session_factory.begin() as session:
            if clean_id:
                row = session.scalar(select(ConversationRecord).where(
                    ConversationRecord.id == clean_id,
                    ConversationRecord.tenant_id == principal.tenant_id,
                    ConversationRecord.user_id == authorized.user_id,
                    ConversationRecord.knowledge_base_id == authorized.id,
                ))
                if row is None:
                    raise ResourceNotFoundError("conversation was not found")
            else:
                row = ConversationRecord(
                    tenant_id=principal.tenant_id,
                    user_id=authorized.user_id,
                    knowledge_base_id=authorized.id,
                    title="新對話",
                )
                session.add(row)
                session.flush()
            return {
                "id": row.id,
                "knowledge_base_id": row.knowledge_base_id,
                "title": row.title,
                "created_at": row.created_at.isoformat(),
                "updated_at": row.updated_at.isoformat(),
            }

    def recent_conversation_messages(
        self,
        *,
        principal: Principal,
        authorized: AuthorizedKnowledgeBase,
        conversation_id: str,
        limit: int = 10,
    ) -> list[dict]:
        with self.session_factory() as session:
            user, membership = self._active_identity(session, principal)
            conversation = session.scalar(select(ConversationRecord.id).where(
                ConversationRecord.id == conversation_id,
                ConversationRecord.tenant_id == principal.tenant_id,
                ConversationRecord.user_id == user.id,
                ConversationRecord.knowledge_base_id == authorized.id,
            ))
            if conversation is None:
                raise ResourceNotFoundError("conversation was not found")
            self._require_knowledge_base_read(
                session,
                principal=principal,
                user=user,
                membership=membership,
                knowledge_base_id=authorized.id,
            )
            rows = list(session.scalars(
                select(MessageRecord)
                .where(
                    MessageRecord.tenant_id == principal.tenant_id,
                    MessageRecord.conversation_id == conversation_id,
                )
                .order_by(MessageRecord.created_at.desc(), MessageRecord.id.desc())
                .limit(max(1, min(int(limit), 50)))
            ))
            answer_runs = self._conversation_answer_runs(
                session,
                tenant_id=principal.tenant_id,
                user_id=user.id,
                knowledge_base_id=authorized.id,
                conversation_id=conversation_id,
            )
            withdrawn_run_ids = self._withdrawn_answer_run_ids(
                session,
                tenant_id=principal.tenant_id,
                knowledge_base_id=authorized.id,
                answer_runs=answer_runs,
            )
            history = []
            for row in reversed(rows):
                run_id = str((row.metadata_json or {}).get("run_id") or "")
                if (
                    not run_id
                    or run_id not in answer_runs
                    or run_id in withdrawn_run_ids
                ):
                    continue
                history.append({
                    "role": row.role,
                    "content": row.content,
                    "run_id": run_id,
                })
            return history

    def list_conversations(self, *, principal: Principal) -> list[dict]:
        with self.session_factory() as session:
            user, membership = self._active_identity(session, principal)
            rows = list(session.execute(
                select(ConversationRecord, KnowledgeBaseRecord)
                .join(
                    KnowledgeBaseRecord,
                    KnowledgeBaseRecord.id == ConversationRecord.knowledge_base_id,
                )
                .where(
                    ConversationRecord.tenant_id == principal.tenant_id,
                    ConversationRecord.user_id == user.id,
                )
                .order_by(ConversationRecord.updated_at.desc(), ConversationRecord.id)
            ))
            items = []
            for row, knowledge_base in rows:
                try:
                    self._require_knowledge_base_read(
                        session,
                        principal=principal,
                        user=user,
                        membership=membership,
                        knowledge_base_id=row.knowledge_base_id,
                        knowledge_base=knowledge_base,
                    )
                except (ResourceNotFoundError, AccessDeniedError):
                    continue
                items.append({
                    "id": row.id,
                    "knowledge_base_id": row.knowledge_base_id,
                    "title": row.title,
                    "created_at": row.created_at.isoformat(),
                    "updated_at": row.updated_at.isoformat(),
                })
            return items

    def get_conversation(
        self,
        *,
        principal: Principal,
        conversation_id: str,
    ) -> dict:
        with self.session_factory() as session:
            user, membership = self._active_identity(session, principal)
            row = session.scalar(select(ConversationRecord).where(
                ConversationRecord.id == conversation_id,
                ConversationRecord.tenant_id == principal.tenant_id,
                ConversationRecord.user_id == user.id,
            ))
            if row is None:
                raise ResourceNotFoundError("conversation was not found")
            self._require_knowledge_base_read(
                session,
                principal=principal,
                user=user,
                membership=membership,
                knowledge_base_id=row.knowledge_base_id,
            )
            messages = list(session.scalars(
                select(MessageRecord)
                .where(
                    MessageRecord.tenant_id == principal.tenant_id,
                    MessageRecord.conversation_id == row.id,
                )
                .order_by(MessageRecord.created_at, MessageRecord.id)
            ))
            answer_runs = self._conversation_answer_runs(
                session,
                tenant_id=principal.tenant_id,
                user_id=user.id,
                knowledge_base_id=row.knowledge_base_id,
                conversation_id=row.id,
            )
            withdrawn_run_ids = self._withdrawn_answer_run_ids(
                session,
                tenant_id=principal.tenant_id,
                knowledge_base_id=row.knowledge_base_id,
                answer_runs=answer_runs,
            )
            return {
                "id": row.id,
                "knowledge_base_id": row.knowledge_base_id,
                "title": row.title,
                "created_at": row.created_at.isoformat(),
                "updated_at": row.updated_at.isoformat(),
                "messages": [
                    self._conversation_message_payload(
                        message, answer_runs, withdrawn_run_ids,
                    )
                    for message in messages
                ],
            }

    @staticmethod
    def _conversation_message_payload(
        message: MessageRecord,
        answer_runs: dict[str, AnswerRunRecord],
        withdrawn_run_ids: set[str],
    ) -> dict:
        run_id = str((message.metadata_json or {}).get("run_id") or "")
        answer_run = answer_runs.get(run_id)
        withdrawn = (
            message.role == "assistant"
            and (answer_run is None or run_id in withdrawn_run_ids)
        )
        payload = {
            "id": message.id,
            "role": message.role,
            "content": _WITHDRAWN_ANSWER_TEXT if withdrawn else message.content,
            "created_at": message.created_at.isoformat(),
        }
        if message.role == "assistant":
            payload["answer_withdrawn"] = withdrawn
        if message.role == "assistant" and answer_run is not None:
            payload["answer_run"] = {
                "run_id": answer_run.id,
                "answer_withdrawn": withdrawn,
                "retrieval": {"needed": answer_run.retrieval_needed},
                "model": {"name": answer_run.model},
                "timings": answer_run.timings_json,
                "evidence_validation": None if withdrawn else (
                    dict((answer_run.retrieval_json or {}).get("evidence_validation") or {})
                    if isinstance(answer_run.retrieval_json, dict)
                    else None
                ),
                "citations": [],
            }
        return payload

    @staticmethod
    def _conversation_answer_runs(
        session,
        *,
        tenant_id: str,
        user_id: str,
        knowledge_base_id: str,
        conversation_id: str,
    ) -> dict[str, AnswerRunRecord]:
        return {
            run.id: run
            for run in session.scalars(
                select(AnswerRunRecord).where(
                    AnswerRunRecord.tenant_id == tenant_id,
                    AnswerRunRecord.user_id == user_id,
                    AnswerRunRecord.knowledge_base_id == knowledge_base_id,
                    AnswerRunRecord.conversation_id == conversation_id,
                )
            )
        }

    @staticmethod
    def _withdrawn_answer_run_ids(
        session,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        answer_runs: dict[str, AnswerRunRecord],
    ) -> set[str]:
        if not answer_runs:
            return set()
        citations_by_run: dict[str, list[CitationRecord]] = {
            run_id: [] for run_id in answer_runs
        }
        withdrawn = set()
        rows = session.execute(
            select(
                CitationRecord,
                DocumentVersionRecord.id,
                DocumentRecord.id,
                DocumentRecord.deleted_at,
                DocumentRecord.status,
            )
            .outerjoin(
                DocumentVersionRecord,
                and_(
                    DocumentVersionRecord.id == CitationRecord.document_version_id,
                    DocumentVersionRecord.tenant_id == tenant_id,
                ),
            )
            .outerjoin(
                DocumentRecord,
                and_(
                    DocumentRecord.id == DocumentVersionRecord.document_id,
                    DocumentRecord.tenant_id == tenant_id,
                    DocumentRecord.knowledge_base_id == knowledge_base_id,
                ),
            )
            .where(
                CitationRecord.tenant_id == tenant_id,
                CitationRecord.answer_run_id.in_(answer_runs),
            )
        )
        for citation, version_id, document_id, deleted_at, status in rows:
            citations_by_run[citation.answer_run_id].append(citation)
            if (
                version_id is None
                or document_id is None
                or deleted_at is not None
                or status == "deleted"
            ):
                withdrawn.add(citation.answer_run_id)
        context_versions_by_run = {}
        all_context_versions = set()
        for run_id, run in answer_runs.items():
            citations = citations_by_run[run_id]
            context_lineage_intact, context_versions = _context_lineage_status(
                run.retrieval_json, run.retrieval_needed, run.pipeline_version,
            )
            if not _context_lineage_chunks_intact(
                session,
                tenant_id=tenant_id,
                knowledge_base_id=knowledge_base_id,
                index_version_id=run.index_version_id or "",
                retrieval_json=run.retrieval_json,
                pipeline_version=run.pipeline_version,
            ):
                context_lineage_intact = False
            context_versions_by_run[run_id] = context_versions
            all_context_versions.update(context_versions)
            if (
                (run.retrieval_needed and not citations)
                or not _citation_lineage_intact(
                    run.retrieval_json, citations, run.pipeline_version,
                )
                or not context_lineage_intact
            ):
                withdrawn.add(run_id)
        available_context_versions = _available_context_version_ids(
            session,
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            version_ids=all_context_versions,
        )
        for run_id, version_ids in context_versions_by_run.items():
            if not version_ids.issubset(available_context_versions):
                withdrawn.add(run_id)

        dependencies: dict[str, tuple[str, ...]] = {}
        dependents: dict[str, list[str]] = {run_id: [] for run_id in answer_runs}
        remaining: dict[str, int] = {}
        for run_id, run in answer_runs.items():
            intact, run_dependencies = _history_dependency_status(
                run.retrieval_json, run.pipeline_version, run_id,
            )
            if not intact or (run_dependencies and not run.conversation_id):
                withdrawn.add(run_id)
            if any(dependency_id not in answer_runs for dependency_id in run_dependencies):
                withdrawn.add(run_id)
            known_dependencies = tuple(
                dependency_id
                for dependency_id in run_dependencies
                if dependency_id in answer_runs
            )
            dependencies[run_id] = known_dependencies
            remaining[run_id] = len(known_dependencies)
            for dependency_id in known_dependencies:
                dependents[dependency_id].append(run_id)

        ready = deque(run_id for run_id, count in remaining.items() if count == 0)
        processed = set()
        while ready:
            run_id = ready.popleft()
            processed.add(run_id)
            if any(dependency_id in withdrawn for dependency_id in dependencies[run_id]):
                withdrawn.add(run_id)
            for dependent_id in dependents[run_id]:
                remaining[dependent_id] -= 1
                if remaining[dependent_id] == 0:
                    ready.append(dependent_id)
        # An unprocessed run is in, or downstream from, a dependency cycle.
        withdrawn.update(answer_runs.keys() - processed)
        return withdrawn

    def delete_conversation(
        self,
        *,
        principal: Principal,
        conversation_id: str,
    ) -> None:
        with self.session_factory.begin() as session:
            user, _membership = self._active_identity(session, principal)
            row = session.scalar(
                select(ConversationRecord)
                .where(
                    ConversationRecord.id == conversation_id,
                    ConversationRecord.tenant_id == principal.tenant_id,
                    ConversationRecord.user_id == user.id,
                )
                .with_for_update()
            )
            if row is None:
                raise ResourceNotFoundError("conversation was not found")
            session.delete(row)

    def get_answer_run(
        self,
        *,
        principal: Principal,
        run_id: str,
    ) -> dict:
        with self.session_factory() as session:
            user, membership = self._active_identity(session, principal)
            run = session.scalar(select(AnswerRunRecord).where(
                AnswerRunRecord.id == run_id,
                AnswerRunRecord.tenant_id == principal.tenant_id,
                AnswerRunRecord.user_id == user.id,
            ))
            if run is None:
                raise ResourceNotFoundError("answer run was not found")
            self._require_knowledge_base_read(
                session,
                principal=principal,
                user=user,
                membership=membership,
                knowledge_base_id=run.knowledge_base_id,
            )
            answer_runs = (
                self._conversation_answer_runs(
                    session,
                    tenant_id=principal.tenant_id,
                    user_id=user.id,
                    knowledge_base_id=run.knowledge_base_id,
                    conversation_id=run.conversation_id,
                )
                if run.conversation_id else {run.id: run}
            )
            answer_runs.setdefault(run.id, run)
            withdrawn_run_ids = self._withdrawn_answer_run_ids(
                session,
                tenant_id=principal.tenant_id,
                knowledge_base_id=run.knowledge_base_id,
                answer_runs=answer_runs,
            )
            citations = list(session.scalars(
                select(CitationRecord)
                .where(
                    CitationRecord.tenant_id == principal.tenant_id,
                    CitationRecord.answer_run_id == run.id,
                )
                .order_by(CitationRecord.rank, CitationRecord.id)
            ))
            evidence = []
            answer_withdrawn = run.id in withdrawn_run_ids
            for citation in citations:
                if answer_withdrawn:
                    evidence.append({"rank": citation.rank, "available": False})
                    continue
                version = (
                    session.scalar(
                        select(DocumentVersionRecord).where(
                            DocumentVersionRecord.id == citation.document_version_id,
                            DocumentVersionRecord.tenant_id == principal.tenant_id,
                        )
                    )
                    if citation.document_version_id
                    else None
                )
                document = (
                    session.scalar(
                        select(DocumentRecord).where(
                            DocumentRecord.id == version.document_id,
                            DocumentRecord.tenant_id == principal.tenant_id,
                            DocumentRecord.knowledge_base_id == run.knowledge_base_id,
                            DocumentRecord.deleted_at.is_(None),
                            DocumentRecord.status != "deleted",
                        )
                    )
                    if version is not None
                    else None
                )
                chunk = (
                    session.scalar(
                        select(ChunkRecord).where(
                            ChunkRecord.id == citation.chunk_record_id,
                            ChunkRecord.tenant_id == principal.tenant_id,
                            ChunkRecord.knowledge_base_id == run.knowledge_base_id,
                            ChunkRecord.document_id == document.id,
                            ChunkRecord.document_version_id == version.id,
                            ChunkRecord.index_version_id == run.index_version_id,
                        )
                    )
                    if document is not None and citation.chunk_record_id and run.index_version_id
                    else None
                )
                if document is None:
                    answer_withdrawn = True
                    evidence.append({"rank": citation.rank, "available": False})
                    continue
                snippet_available = bool(
                    chunk is not None
                    and chunk.content
                    and hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
                    == chunk.content_sha256
                )
                evidence.append({
                    "rank": citation.rank,
                    "available": True,
                    "chunk_id": citation.chunk_id,
                    "chunk_record_id": citation.chunk_record_id,
                    "document_version_id": citation.document_version_id,
                    "document_id": document.id if document is not None else None,
                    "document_name": document.name if document is not None else "",
                    "source_id": document.source_id if document is not None else "",
                    "page": citation.page,
                    "verified": citation.verified,
                    "content_sha256": citation.content_sha256,
                    "snippet_available": snippet_available,
                    "title": chunk.title if snippet_available else "",
                    "content": chunk.content if snippet_available else "",
                })
            if answer_withdrawn:
                evidence = [
                    {"rank": citation.rank, "available": False}
                    for citation in citations
                ]
            return {
                "id": run.id,
                "knowledge_base_id": run.knowledge_base_id,
                "conversation_id": run.conversation_id,
                "index_version_id": run.index_version_id,
                "model": run.model,
                "pipeline_version": run.pipeline_version,
                "question": run.question,
                "answer": _WITHDRAWN_ANSWER_TEXT if answer_withdrawn else run.answer,
                "answer_withdrawn": answer_withdrawn,
                "retrieval_needed": run.retrieval_needed,
                "timings": run.timings_json,
                "evidence_validation": None if answer_withdrawn else (
                    dict((run.retrieval_json or {}).get("evidence_validation") or {})
                    if isinstance(run.retrieval_json, dict)
                    else None
                ),
                "created_at": run.created_at.isoformat(),
                "citations": evidence,
            }

    def resolve_document_version_download(
        self,
        *,
        principal: Principal,
        document_version_id: str,
    ) -> dict:
        with self.session_factory() as session:
            row = session.execute(
                select(DocumentVersionRecord, DocumentRecord)
                .join(DocumentRecord, DocumentRecord.id == DocumentVersionRecord.document_id)
                .where(
                    DocumentVersionRecord.id == document_version_id,
                    DocumentVersionRecord.tenant_id == principal.tenant_id,
                    DocumentRecord.tenant_id == principal.tenant_id,
                    DocumentRecord.deleted_at.is_(None),
                )
            ).one_or_none()
            if row is None:
                raise ResourceNotFoundError("document version was not found")
            version, document = row
            return {
                "knowledge_base_id": document.knowledge_base_id,
                "document_id": document.id,
                "document_version_id": version.id,
                "filename": document.name,
                "mime_type": version.mime_type,
                "size_bytes": version.size_bytes,
                "object_key": version.object_key,
            }

    @staticmethod
    def _active_identity(session, principal: Principal):
        tenant = session.scalar(select(Tenant).where(
            Tenant.id == principal.tenant_id,
            Tenant.active.is_(True),
        ))
        if tenant is None:
            raise AccessDeniedError("tenant is not active")
        user = session.scalar(select(UserIdentity).where(
            UserIdentity.tenant_id == principal.tenant_id,
            UserIdentity.subject == principal.subject,
            UserIdentity.active.is_(True),
        ))
        if user is None:
            raise AccessDeniedError("identity is not an active tenant member")
        membership = session.scalar(select(Membership).where(
            Membership.tenant_id == principal.tenant_id,
            Membership.user_id == user.id,
            Membership.active.is_(True),
        ))
        if membership is None:
            raise AccessDeniedError("identity is not an active tenant member")
        return user, membership

    @staticmethod
    def _require_knowledge_base_read(
        session,
        *,
        principal: Principal,
        user: UserIdentity,
        membership: Membership,
        knowledge_base_id: str,
        knowledge_base: Optional[KnowledgeBaseRecord] = None,
    ) -> KnowledgeBaseRecord:
        row = knowledge_base
        if row is None:
            row = session.scalar(select(KnowledgeBaseRecord).where(
                KnowledgeBaseRecord.id == knowledge_base_id,
                KnowledgeBaseRecord.tenant_id == principal.tenant_id,
                KnowledgeBaseRecord.active.is_(True),
            ))
        if (
            row is None
            or row.id != knowledge_base_id
            or row.tenant_id != principal.tenant_id
            or not row.active
        ):
            raise ResourceNotFoundError("knowledge base was not found")
        if not (
            membership.role in {"owner", "admin"}
            or principal.has_role("tenant_admin")
            or row.owner_user_id == user.id
            or row.visibility == "tenant"
        ):
            raise AccessDeniedError("knowledge base access is denied")
        return row

    def authorize_knowledge_base(
        self,
        principal: Principal,
        knowledge_base_id: str,
        requested_source_ids: Optional[Sequence[str]],
    ) -> AuthorizedKnowledgeBase:
        with self.session_factory() as session:
            user, membership = self._active_identity(session, principal)
            knowledge_base = self._require_knowledge_base_read(
                session,
                principal=principal,
                user=user,
                membership=membership,
                knowledge_base_id=knowledge_base_id,
            )
            can_write = (
                membership.role in {"owner", "admin"}
                or principal.has_role("tenant_admin")
                or knowledge_base.owner_user_id == user.id
            )

            ready_source_ids = []
            if knowledge_base.active_index_version_id:
                active_index = session.scalar(
                    select(IndexVersionRecord).where(
                        IndexVersionRecord.id == knowledge_base.active_index_version_id,
                        IndexVersionRecord.tenant_id == principal.tenant_id,
                        IndexVersionRecord.knowledge_base_id == knowledge_base.id,
                        IndexVersionRecord.status == "ready",
                    )
                )
                if active_index is None:
                    raise InvalidServiceStateError(
                        "knowledge base active index is inconsistent"
                    )
                # A replacement upload changes the document workflow status
                # before its new index is published. The active membership and
                # current version remain the authority for searchable sources.
                ready_source_ids = list(session.scalars(
                    select(DocumentRecord.source_id)
                    .join(
                        IndexDocumentRecord,
                        IndexDocumentRecord.document_id == DocumentRecord.id,
                    )
                    .where(
                        DocumentRecord.tenant_id == principal.tenant_id,
                        DocumentRecord.knowledge_base_id == knowledge_base.id,
                        DocumentRecord.deleted_at.is_(None),
                        DocumentRecord.current_version_id
                        == IndexDocumentRecord.document_version_id,
                        IndexDocumentRecord.tenant_id == principal.tenant_id,
                        IndexDocumentRecord.knowledge_base_id == knowledge_base.id,
                        IndexDocumentRecord.index_version_id == active_index.id,
                        IndexDocumentRecord.status == "ready",
                    )
                    .order_by(DocumentRecord.created_at, DocumentRecord.id)
                ))
            if requested_source_ids is None:
                authorized_source_ids = ready_source_ids
            else:
                requested = list(dict.fromkeys(str(item) for item in requested_source_ids))
                missing = sorted(set(requested) - set(ready_source_ids))
                if missing:
                    raise AccessDeniedError("one or more documents are not accessible")
                authorized_source_ids = requested

            return AuthorizedKnowledgeBase(
                id=knowledge_base.id,
                profile=knowledge_base.profile,
                user_id=user.id,
                source_ids=authorized_source_ids,
                active_index_version_id=knowledge_base.active_index_version_id,
                activation_generation=knowledge_base.activation_generation,
                can_write=can_write,
                embedding_model=active_index.embedding_model if knowledge_base.active_index_version_id else None,
                sparse_embedding_model=(
                    active_index.sparse_embedding_model
                    if knowledge_base.active_index_version_id
                    else None
                ),
                embedding_dimensions=active_index.embedding_dimensions if knowledge_base.active_index_version_id else None,
                chunk_schema_version=active_index.chunk_schema_version if knowledge_base.active_index_version_id else None,
                qdrant_collection=active_index.qdrant_collection if knowledge_base.active_index_version_id else None,
                artifact_attempt=(
                    active_index.artifact_attempt
                    if knowledge_base.active_index_version_id else 0
                ),
            )

    def reserve_upload(
        self,
        *,
        principal: Principal,
        authorized: AuthorizedKnowledgeBase,
        upload_id: str,
        idempotency_key: str,
        filename: str,
        declared_mime_type: str,
        expected_size_bytes: int,
        expected_sha256: str,
        object_key: str,
        expires_at: datetime,
        folder_id: Optional[str] = None,
        target_document_id: Optional[str] = None,
    ) -> UploadReservation:
        if not authorized.can_write:
            raise AccessDeniedError("knowledge base write access is denied")
        with self.session_factory.begin() as session:
            existing = session.scalar(
                select(UploadSessionRecord).where(
                    UploadSessionRecord.tenant_id == principal.tenant_id,
                    UploadSessionRecord.idempotency_key == idempotency_key,
                )
            )
            if existing is not None:
                if (
                    existing.knowledge_base_id != authorized.id
                    or existing.owner_user_id != authorized.user_id
                    or existing.folder_id != folder_id
                    or existing.target_document_id != target_document_id
                    or existing.filename != filename
                    or existing.declared_mime_type != declared_mime_type
                    or existing.expected_size_bytes != expected_size_bytes
                    or existing.expected_sha256 != expected_sha256
                ):
                    raise InvalidServiceStateError(
                        "idempotency key was already used for another upload"
                    )
                return _upload_reservation(existing)

            if folder_id:
                folder = session.scalar(select(FolderRecord.id).where(
                    FolderRecord.id == folder_id,
                    FolderRecord.tenant_id == principal.tenant_id,
                    FolderRecord.knowledge_base_id == authorized.id,
                ))
                if folder is None:
                    raise ResourceNotFoundError("folder was not found")
            if target_document_id:
                target = session.scalar(select(DocumentRecord.id).where(
                    DocumentRecord.id == target_document_id,
                    DocumentRecord.tenant_id == principal.tenant_id,
                    DocumentRecord.knowledge_base_id == authorized.id,
                    DocumentRecord.deleted_at.is_(None),
                ))
                if target is None:
                    raise ResourceNotFoundError("target document was not found")

            record = UploadSessionRecord(
                id=upload_id,
                tenant_id=principal.tenant_id,
                knowledge_base_id=authorized.id,
                owner_user_id=authorized.user_id,
                folder_id=folder_id,
                target_document_id=target_document_id,
                idempotency_key=idempotency_key,
                filename=filename,
                declared_mime_type=declared_mime_type,
                expected_size_bytes=expected_size_bytes,
                expected_sha256=expected_sha256,
                object_key=object_key,
                state="reserved",
                expires_at=expires_at,
            )
            session.add(record)
            session.flush()
            return _upload_reservation(record)

    def complete_upload(
        self,
        *,
        principal: Principal,
        upload_id: str,
        observed_size_bytes: int,
        observed_sha256: str,
        observed_mime_type: str,
        parser_version: str,
        chunk_schema_version: str,
        embedding_model: str,
    ) -> CompletedUpload:
        with self.session_factory.begin() as session:
            upload = session.scalar(
                select(UploadSessionRecord)
                .where(
                    UploadSessionRecord.id == upload_id,
                    UploadSessionRecord.tenant_id == principal.tenant_id,
                )
                .with_for_update()
            )
            if upload is None:
                raise ResourceNotFoundError("upload session was not found")
            user = session.scalar(select(UserIdentity).where(
                UserIdentity.id == upload.owner_user_id,
                UserIdentity.tenant_id == principal.tenant_id,
                UserIdentity.subject == principal.subject,
                UserIdentity.active.is_(True),
            ))
            if user is None:
                raise AccessDeniedError("upload session access is denied")
            if upload.state == "consumed" and upload.consumed_document_version_id:
                job = session.scalar(select(IngestionJobRecord).where(
                    IngestionJobRecord.document_version_id
                    == upload.consumed_document_version_id,
                ))
                version = session.get(
                    DocumentVersionRecord,
                    upload.consumed_document_version_id,
                )
                if job is None or version is None:
                    raise InvalidServiceStateError("consumed upload is inconsistent")
                return CompletedUpload(
                    upload_id=upload.id,
                    document_id=version.document_id,
                    document_version_id=version.id,
                    ingestion_job_id=job.id,
                    duplicate=True,
                )
            if upload.state not in {"reserved", "uploaded", "verified"}:
                raise InvalidServiceStateError("upload session cannot be completed")
            if _as_utc(upload.expires_at) <= utc_now():
                upload.state = "expired"
                session.flush()
                session.commit()
                raise InvalidServiceStateError("upload session has expired")
            if (
                int(observed_size_bytes) != int(upload.expected_size_bytes)
                or str(observed_sha256 or "").lower() != upload.expected_sha256
            ):
                raise InvalidServiceStateError("uploaded object metadata does not match reservation")
            if observed_mime_type and upload.declared_mime_type and (
                observed_mime_type != upload.declared_mime_type
            ):
                raise InvalidServiceStateError("uploaded object MIME type does not match reservation")

            document = (
                session.scalar(
                    select(DocumentRecord)
                    .where(
                        DocumentRecord.id == upload.target_document_id,
                        DocumentRecord.tenant_id == principal.tenant_id,
                        DocumentRecord.knowledge_base_id == upload.knowledge_base_id,
                        DocumentRecord.deleted_at.is_(None),
                    )
                    .with_for_update()
                )
                if upload.target_document_id
                else None
            )
            if upload.target_document_id and document is None:
                raise ResourceNotFoundError("target document was not found")
            if document is None:
                document = DocumentRecord(
                    id=new_id(),
                    tenant_id=principal.tenant_id,
                    knowledge_base_id=upload.knowledge_base_id,
                    folder_id=upload.folder_id,
                    owner_user_id=upload.owner_user_id,
                    source_id=f"document-{upload.id}",
                    name=upload.filename,
                    status="processing",
                )
                session.add(document)
                session.flush()
                version_number = 1
            else:
                max_version = session.scalar(
                    select(func.max(DocumentVersionRecord.version_number)).where(
                        DocumentVersionRecord.document_id == document.id,
                    )
                ) or 0
                version_number = int(max_version) + 1
                document.status = "processing"

            version = DocumentVersionRecord(
                id=new_id(),
                tenant_id=principal.tenant_id,
                document_id=document.id,
                version_number=version_number,
                sha256=upload.expected_sha256,
                object_key=upload.object_key,
                mime_type=upload.declared_mime_type,
                size_bytes=upload.expected_size_bytes,
                parser_version=parser_version,
                chunk_schema_version=chunk_schema_version,
                embedding_model=embedding_model,
                status="uploaded",
            )
            job = IngestionJobRecord(
                id=new_id(),
                tenant_id=principal.tenant_id,
                document_version_id=version.id,
                status="queued",
                stage="queued",
                idempotency_key=f"upload:{upload.id}",
                pipeline_fingerprint=(
                    f"{parser_version}:{chunk_schema_version}:{embedding_model}"
                )[:128],
            )
            session.add_all([version, job])
            upload.state = "consumed"
            upload.consumed_document_version_id = version.id
            return CompletedUpload(
                upload_id=upload.id,
                document_id=document.id,
                document_version_id=version.id,
                ingestion_job_id=job.id,
                duplicate=False,
            )

    def get_upload_object_key(self, *, principal: Principal, upload_id: str) -> str:
        with self.session_factory() as session:
            row = session.execute(
                select(UploadSessionRecord.object_key, UserIdentity.subject)
                .join(UserIdentity, UserIdentity.id == UploadSessionRecord.owner_user_id)
                .where(
                    UploadSessionRecord.id == upload_id,
                    UploadSessionRecord.tenant_id == principal.tenant_id,
                )
            ).one_or_none()
            if row is None or row.subject != principal.subject:
                raise ResourceNotFoundError("upload session was not found")
            return str(row.object_key)

    def get_upload_reservation(
        self,
        *,
        principal: Principal,
        upload_id: str,
    ) -> UploadReservation:
        with self.session_factory.begin() as session:
            row = session.execute(
                select(UploadSessionRecord, UserIdentity.subject)
                .join(UserIdentity, UserIdentity.id == UploadSessionRecord.owner_user_id)
                .where(
                    UploadSessionRecord.id == upload_id,
                    UploadSessionRecord.tenant_id == principal.tenant_id,
                )
                .with_for_update()
            ).one_or_none()
            if row is None or row.subject != principal.subject:
                raise ResourceNotFoundError("upload session was not found")
            upload = row[0]
            if upload.state not in {"reserved", "uploading", "uploaded"}:
                raise InvalidServiceStateError("upload session cannot receive content")
            if _as_utc(upload.expires_at) <= utc_now():
                upload.state = "expired"
                session.flush()
                session.commit()
                raise InvalidServiceStateError("upload session has expired")
            if upload.state == "reserved":
                upload.state = "uploading"
            return _upload_reservation(upload)

    def mark_upload_stored(
        self,
        *,
        principal: Principal,
        upload_id: str,
        observed_size_bytes: int,
        observed_sha256: str,
        observed_mime_type: str,
    ) -> None:
        with self.session_factory.begin() as session:
            row = session.execute(
                select(UploadSessionRecord, UserIdentity.subject)
                .join(UserIdentity, UserIdentity.id == UploadSessionRecord.owner_user_id)
                .where(
                    UploadSessionRecord.id == upload_id,
                    UploadSessionRecord.tenant_id == principal.tenant_id,
                )
                .with_for_update()
            ).one_or_none()
            if row is None or row.subject != principal.subject:
                raise ResourceNotFoundError("upload session was not found")
            upload = row[0]
            if upload.state not in {"uploading", "uploaded"}:
                raise InvalidServiceStateError("upload session cannot be marked uploaded")
            if _as_utc(upload.expires_at) <= utc_now():
                upload.state = "expired"
                session.flush()
                session.commit()
                raise InvalidServiceStateError("upload session has expired")
            if (
                int(observed_size_bytes) != int(upload.expected_size_bytes)
                or str(observed_sha256 or "").lower() != upload.expected_sha256
                or str(observed_mime_type or "").lower()
                != upload.declared_mime_type.lower()
            ):
                raise InvalidServiceStateError(
                    "uploaded object metadata does not match reservation"
                )
            upload.state = "uploaded"

    def expired_upload_cleanup_candidates(
        self,
        *,
        limit: int = 100,
        grace_seconds: int = 300,
    ) -> list[dict]:
        cutoff = utc_now() - timedelta(seconds=max(0, int(grace_seconds)))
        with self.session_factory.begin() as session:
            rows = list(session.scalars(
                select(UploadSessionRecord)
                .where(
                    UploadSessionRecord.state.in_({
                        "reserved",
                        "uploading",
                        "uploaded",
                        "verified",
                        "expired",
                    }),
                    UploadSessionRecord.expires_at <= cutoff,
                )
                .order_by(UploadSessionRecord.expires_at, UploadSessionRecord.id)
                .limit(max(1, min(int(limit), 1000)))
                .with_for_update(skip_locked=True)
            ))
            candidates = []
            for upload in rows:
                upload.state = "expired"
                candidates.append({
                    "upload_id": upload.id,
                    "object_key": upload.object_key,
                })
            return candidates

    def mark_upload_cleanup_complete(self, *, upload_id: str, object_key: str) -> None:
        with self.session_factory.begin() as session:
            upload = session.scalar(
                select(UploadSessionRecord)
                .where(
                    UploadSessionRecord.id == upload_id,
                    UploadSessionRecord.object_key == object_key,
                )
                .with_for_update()
            )
            if upload is None:
                raise ResourceNotFoundError("upload session was not found")
            if upload.state == "aborted":
                return
            if upload.state != "expired":
                raise InvalidServiceStateError(
                    "only an expired upload can be cleaned up"
                )
            upload.state = "aborted"

    def record_ingestion_dispatch(self, *, job_id: str, task_id: str) -> None:
        with self.session_factory.begin() as session:
            job = session.get(IngestionJobRecord, job_id)
            if job is None:
                raise ResourceNotFoundError("ingestion job was not found")
            job.worker_task_id = str(task_id)[:255]
            job.dispatch_count += 1
            job.last_dispatched_at = utc_now()

    def get_ingestion_job(self, *, principal: Principal, job_id: str) -> dict:
        with self.session_factory() as session:
            row = session.execute(
                select(
                    IngestionJobRecord,
                    DocumentVersionRecord,
                    DocumentRecord,
                    UserIdentity,
                )
                .join(
                    DocumentVersionRecord,
                    DocumentVersionRecord.id == IngestionJobRecord.document_version_id,
                )
                .join(DocumentRecord, DocumentRecord.id == DocumentVersionRecord.document_id)
                .join(UserIdentity, UserIdentity.id == DocumentRecord.owner_user_id)
                .where(
                    IngestionJobRecord.id == job_id,
                    IngestionJobRecord.tenant_id == principal.tenant_id,
                    UserIdentity.subject == principal.subject,
                    UserIdentity.active.is_(True),
                )
            ).one_or_none()
            if row is None:
                raise ResourceNotFoundError("ingestion job was not found")
            job, version, document, _user = row
            return {
                "id": job.id,
                "document_id": document.id,
                "document_version_id": version.id,
                "status": job.status,
                "stage": job.stage,
                "attempt": job.attempt,
                "max_attempts": job.max_attempts,
                "error_code": job.error_code,
                "created_at": job.created_at.isoformat(),
                "updated_at": job.updated_at.isoformat(),
            }

    def validate_index_version(
        self,
        *,
        job_id: str,
        worker_id: str,
        attempt: int,
        tenant_id: str,
        knowledge_base_id: str,
        index_version_id: str,
        manifest_sha256: str,
        manifest_object_key: str,
        manifest_entries_sha256: str,
        qdrant_point_count: int,
        qdrant_entries_sha256: str,
    ) -> None:
        clean_manifest = str(manifest_sha256 or "").lower()
        if len(clean_manifest) != 64 or any(char not in "0123456789abcdef" for char in clean_manifest):
            raise ValueError("manifest_sha256 must be a lowercase SHA-256 digest")
        clean_manifest_key = str(manifest_object_key or "").strip()
        if not clean_manifest_key.startswith("tenants/"):
            raise ValueError("manifest_object_key must be a tenant-scoped object key")
        clean_manifest_entries = _require_sha256(
            manifest_entries_sha256,
            "manifest_entries_sha256",
        )
        clean_qdrant_entries = _require_sha256(
            qdrant_entries_sha256,
            "qdrant_entries_sha256",
        )
        with self.session_factory.begin() as session:
            now = utc_now()
            job = session.scalar(
                select(IndexBuildJobRecord)
                .where(
                    IndexBuildJobRecord.id == job_id,
                    IndexBuildJobRecord.tenant_id == tenant_id,
                    IndexBuildJobRecord.knowledge_base_id == knowledge_base_id,
                    IndexBuildJobRecord.index_version_id == index_version_id,
                    IndexBuildJobRecord.status == "running",
                    IndexBuildJobRecord.lease_owner == worker_id,
                )
                .with_for_update()
            )
            if (
                job is None
                or job.lease_expires_at is None
                or _as_utc(job.lease_expires_at) <= now
                or job.attempt != int(attempt)
            ):
                raise InvalidServiceStateError(
                    "index build lease or attempt was lost before validation"
                )
            target = session.scalar(
                select(IndexVersionRecord)
                .where(
                    IndexVersionRecord.id == index_version_id,
                    IndexVersionRecord.tenant_id == tenant_id,
                    IndexVersionRecord.knowledge_base_id == knowledge_base_id,
                    IndexVersionRecord.status.in_({"building", "validating"}),
                )
                .with_for_update()
            )
            if target is None:
                raise InvalidServiceStateError("index is not in a validatable state")
            # Legacy candidates keep artifact_attempt=0 while the old writer
            # is enabled. A phase-2 writer must persist its positive attempt
            # and match it to the durable job attempt before this method can
            # be used for that format.
            if target.artifact_attempt > 0 and target.artifact_attempt != int(attempt):
                raise InvalidServiceStateError(
                    "index artifact attempt does not match the build attempt"
                )
            if target.manifest_sha256 and target.manifest_sha256 != clean_manifest:
                raise InvalidServiceStateError("index manifest digest changed before validation")
            if (
                target.manifest_object_key
                and target.manifest_object_key != clean_manifest_key
            ):
                raise InvalidServiceStateError("index manifest object changed before validation")
            membership_count = session.scalar(
                select(func.count(IndexDocumentRecord.id)).where(
                    IndexDocumentRecord.tenant_id == tenant_id,
                    IndexDocumentRecord.knowledge_base_id == knowledge_base_id,
                    IndexDocumentRecord.index_version_id == index_version_id,
                    IndexDocumentRecord.status == "ready",
                )
            ) or 0
            pg_chunk_count = session.scalar(
                select(func.count(ChunkRecord.id)).where(
                    ChunkRecord.tenant_id == tenant_id,
                    ChunkRecord.knowledge_base_id == knowledge_base_id,
                    ChunkRecord.index_version_id == index_version_id,
                )
            ) or 0
            pg_entries = [
                {
                    "qdrant_point_id": row.qdrant_point_id,
                    "chunk_id": row.chunk_key,
                    "chunk_record_id": row.id,
                    "document_id": row.document_id,
                    "document_version_id": row.document_version_id,
                    "content": row.content,
                    "content_sha256": row.content_sha256,
                }
                for row in session.scalars(
                    select(ChunkRecord).where(
                        ChunkRecord.tenant_id == tenant_id,
                        ChunkRecord.knowledge_base_id == knowledge_base_id,
                        ChunkRecord.index_version_id == index_version_id,
                    )
                )
            ]
            try:
                pg_entries_sha256 = index_entries_sha256(pg_entries)
            except ValueError as exc:
                raise InvalidServiceStateError(
                    "PostgreSQL index entries are internally inconsistent"
                ) from exc
            observed = (
                int(membership_count),
                int(pg_chunk_count),
                int(qdrant_point_count),
            )
            expected = (
                int(target.expected_document_count),
                int(target.expected_chunk_count),
                int(target.expected_vector_count),
            )
            if observed != expected or int(target.chunk_count) != int(pg_chunk_count):
                raise InvalidServiceStateError(
                    "index manifest counts do not match persisted artifacts"
                )
            if not (
                clean_manifest_entries
                == pg_entries_sha256
                == clean_qdrant_entries
            ):
                raise InvalidServiceStateError(
                    "index manifest entries do not match PostgreSQL and Qdrant"
                )
            target.status = "ready"
            target.manifest_sha256 = clean_manifest
            target.manifest_object_key = clean_manifest_key
            target.manifest_entries_sha256 = clean_manifest_entries
            target.validated_pg_entries_sha256 = pg_entries_sha256
            target.validated_qdrant_entries_sha256 = clean_qdrant_entries
            target.validated_pg_chunk_count = int(pg_chunk_count)
            target.validated_qdrant_point_count = int(qdrant_point_count)
            target.validated_at = utc_now()

    def publish_index_version(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        index_version_id: str,
        expected_active_index_id: Optional[str] = None,
        expected_generation: Optional[int] = None,
        grace_period_seconds: int = 600,
        session=None,
    ) -> None:
        """Atomically publish one validated immutable index version."""
        with (nullcontext(session) if session is not None else self.session_factory.begin()) as session:
            knowledge_base = session.scalar(
                select(KnowledgeBaseRecord)
                .where(
                    KnowledgeBaseRecord.id == knowledge_base_id,
                    KnowledgeBaseRecord.tenant_id == tenant_id,
                    KnowledgeBaseRecord.active.is_(True),
                )
                .with_for_update()
            )
            if knowledge_base is None:
                raise ResourceNotFoundError("knowledge base was not found")
            if (
                expected_active_index_id is not None
                and knowledge_base.active_index_version_id != expected_active_index_id
            ):
                raise ConcurrentPublishError("active index changed before publish")
            if (
                expected_generation is not None
                and knowledge_base.activation_generation != expected_generation
            ):
                raise ConcurrentPublishError("activation generation changed before publish")
            target = session.scalar(
                select(IndexVersionRecord)
                .where(
                    IndexVersionRecord.id == index_version_id,
                    IndexVersionRecord.tenant_id == tenant_id,
                    IndexVersionRecord.knowledge_base_id == knowledge_base_id,
                    IndexVersionRecord.status == "ready",
                    IndexVersionRecord.validated_at.is_not(None),
                )
                .with_for_update()
            )
            if target is None:
                raise InvalidServiceStateError(
                    "only a ready index version can be published"
                )
            if (
                len(target.manifest_sha256) != 64
                or not target.manifest_object_key.startswith("tenants/")
                or target.expected_chunk_count != target.validated_pg_chunk_count
                or target.expected_vector_count != target.validated_qdrant_point_count
                or len(target.manifest_entries_sha256) != 64
                or target.manifest_entries_sha256
                != target.validated_pg_entries_sha256
                or target.manifest_entries_sha256
                != target.validated_qdrant_entries_sha256
            ):
                raise InvalidServiceStateError("index validation evidence is incomplete")

            old_index_id = knowledge_base.active_index_version_id
            if old_index_id and old_index_id != target.id:
                old_index = session.scalar(
                    select(IndexVersionRecord)
                    .where(
                        IndexVersionRecord.id == old_index_id,
                        IndexVersionRecord.tenant_id == tenant_id,
                        IndexVersionRecord.knowledge_base_id == knowledge_base_id,
                    )
                    .with_for_update()
                )
                if old_index is not None:
                    old_index.retired_at = utc_now()
                    old_index.gc_after = utc_now() + timedelta(
                        seconds=max(60, int(grace_period_seconds))
                    )
                    if int(old_index.artifact_attempt or 0) > 0:
                        ensure_index_artifact_gc(
                            session,
                            tenant_id=old_index.tenant_id,
                            knowledge_base_id=old_index.knowledge_base_id,
                            index_version_id=old_index.id,
                            artifact_attempt=int(old_index.artifact_attempt),
                            qdrant_collection=old_index.qdrant_collection,
                            manifest_object_key=old_index.manifest_object_key,
                            next_attempt_at=old_index.gc_after,
                        )

            now = utc_now()
            memberships = list(session.scalars(
                select(IndexDocumentRecord).where(
                    IndexDocumentRecord.tenant_id == tenant_id,
                    IndexDocumentRecord.knowledge_base_id == knowledge_base_id,
                    IndexDocumentRecord.index_version_id == target.id,
                    IndexDocumentRecord.status == "ready",
                )
            ))
            if len(memberships) != target.expected_document_count:
                raise InvalidServiceStateError(
                    "index document membership changed after validation"
                )
            for membership in memberships:
                document = session.scalar(
                    select(DocumentRecord)
                    .where(
                        DocumentRecord.id == membership.document_id,
                        DocumentRecord.tenant_id == tenant_id,
                        DocumentRecord.knowledge_base_id == knowledge_base_id,
                        DocumentRecord.deleted_at.is_(None),
                    )
                    .with_for_update()
                )
                version = session.scalar(
                    select(DocumentVersionRecord).where(
                        DocumentVersionRecord.id == membership.document_version_id,
                        DocumentVersionRecord.tenant_id == tenant_id,
                        DocumentVersionRecord.document_id == membership.document_id,
                    )
                )
                if document is None or version is None:
                    raise InvalidServiceStateError(
                        "index references a missing document version"
                    )
                document.current_version_id = version.id
                document.status = "ready"
                version.status = "ready"
            target.published_at = target.published_at or now
            target.activated_at = target.activated_at or now
            knowledge_base.active_index_version_id = target.id
            knowledge_base.activation_generation += 1
            session.add(IndexActivationEventRecord(
                tenant_id=tenant_id,
                knowledge_base_id=knowledge_base_id,
                previous_index_version_id=old_index_id,
                new_index_version_id=target.id,
                generation=knowledge_base.activation_generation,
                activated_at=now,
            ))

    def record_answer_run(
        self,
        principal: Principal,
        authorized: AuthorizedKnowledgeBase,
        question: str,
        model: str,
        result: dict,
        pipeline_version: str,
        request_id: str = "",
        conversation_id: Optional[str] = None,
    ) -> None:
        with self.session_factory.begin() as session:
            conversation = None
            if conversation_id:
                conversation = session.scalar(
                    select(ConversationRecord)
                    .where(
                        ConversationRecord.id == conversation_id,
                        ConversationRecord.tenant_id == principal.tenant_id,
                        ConversationRecord.user_id == authorized.user_id,
                        ConversationRecord.knowledge_base_id == authorized.id,
                    )
                    .with_for_update()
                )
                if conversation is None:
                    raise ResourceNotFoundError("conversation was not found")
            history_manifest = result.get("_history_dependency_v1")
            history_intact, history_run_ids = _history_dependency_status(
                {_HISTORY_DEPENDENCY_KEY: history_manifest},
                pipeline_version,
                str(result["run_id"]),
            )
            if not history_intact or (history_run_ids and conversation is None):
                raise InvalidServiceStateError("answer history dependency is incomplete")
            if history_run_ids:
                prior_runs = self._conversation_answer_runs(
                    session,
                    tenant_id=principal.tenant_id,
                    user_id=authorized.user_id,
                    knowledge_base_id=authorized.id,
                    conversation_id=conversation.id,
                )
                if (
                    not set(history_run_ids).issubset(prior_runs)
                    or set(history_run_ids) & self._withdrawn_answer_run_ids(
                        session,
                        tenant_id=principal.tenant_id,
                        knowledge_base_id=authorized.id,
                        answer_runs=prior_runs,
                    )
                ):
                    raise InvalidServiceStateError("answer history dependency is unavailable")
            citation_records = [
                CitationRecord(
                    tenant_id=principal.tenant_id,
                    answer_run_id=str(result["run_id"]),
                    chunk_id=str(citation.get("id") or ""),
                    rank=int(citation.get("rank") or 0),
                    page=str(citation.get("page") or ""),
                    content_sha256=str(citation.get("content_sha256") or ""),
                    document_version_id=(
                        str(citation.get("document_version_id"))
                        if citation.get("document_version_id")
                        else None
                    ),
                    chunk_record_id=(
                        str(citation.get("chunk_record_id"))
                        if citation.get("chunk_record_id")
                        else None
                    ),
                    verified=bool(citation.get("verified", False)),
                )
                for citation in result.get("citations") or []
            ]
            retrieval_record = _safe_retrieval_record(
                result.get("retrieval"),
                result.get("evidence_validation"),
            )
            retrieval_record[_CITATION_LINEAGE_KEY] = _citation_lineage(
                citation_records
            )
            try:
                context_lineage = _context_lineage_for_result(result.get("retrieval"))
                retrieval_record[_CONTEXT_LINEAGE_KEY] = context_lineage
                context_versions = {
                    entry["document_version_id"]
                    for entry in context_lineage["contexts"]
                }
                available_context_versions = _available_context_version_ids(
                    session,
                    tenant_id=principal.tenant_id,
                    knowledge_base_id=authorized.id,
                    version_ids=context_versions,
                )
                if not context_versions.issubset(available_context_versions):
                    raise InvalidServiceStateError("final context document version is unavailable")
                if not _context_lineage_chunks_intact(
                    session,
                    tenant_id=principal.tenant_id,
                    knowledge_base_id=authorized.id,
                    index_version_id=authorized.active_index_version_id or "",
                    retrieval_json=retrieval_record,
                    pipeline_version=pipeline_version,
                ):
                    raise InvalidServiceStateError(
                        "final context chunk lineage is unavailable"
                    )
            except InvalidServiceStateError as exc:
                LOGGER.warning(
                    "answer_run.context_lineage_incomplete run_id=%s reason=%s",
                    str(result.get("run_id") or ""),
                    str(exc),
                )
                raise
            retrieval_record[_HISTORY_DEPENDENCY_KEY] = {
                "run_ids": list(history_run_ids),
                "complete": True,
            }
            run = AnswerRunRecord(
                id=str(result["run_id"]),
                tenant_id=principal.tenant_id,
                user_id=authorized.user_id,
                knowledge_base_id=authorized.id,
                conversation_id=conversation.id if conversation is not None else None,
                model=model,
                pipeline_version=pipeline_version,
                index_version_id=authorized.active_index_version_id,
                request_id=request_id,
                question=question,
                answer=str(result.get("answer") or ""),
                retrieval_needed=bool(result.get("retrieval", {}).get("needed")),
                source_ids_json=list(authorized.source_ids),
                timings_json=dict(result.get("timings") or {}),
                retrieval_json=retrieval_record,
            )
            session.add(run)
            if conversation is not None:
                message_created_at = utc_now()
                previous_user_count = session.scalar(
                    select(func.count(MessageRecord.id)).where(
                        MessageRecord.tenant_id == principal.tenant_id,
                        MessageRecord.conversation_id == conversation.id,
                        MessageRecord.role == "user",
                    )
                ) or 0
                session.add_all([
                    MessageRecord(
                        tenant_id=principal.tenant_id,
                        conversation_id=conversation.id,
                        role="user",
                        content=question,
                        metadata_json={"run_id": run.id},
                        created_at=message_created_at,
                    ),
                    MessageRecord(
                        tenant_id=principal.tenant_id,
                        conversation_id=conversation.id,
                        role="assistant",
                        content=run.answer,
                        metadata_json={
                            "run_id": run.id,
                            "confidence": str(result.get("confidence") or ""),
                            "evidence_validation": dict(
                                result.get("evidence_validation") or {}
                            ),
                        },
                        created_at=message_created_at + timedelta(microseconds=1),
                    ),
                ])
                if int(previous_user_count) == 0:
                    conversation.title = " ".join(question.split())[:80] or "新對話"
                conversation.updated_at = utc_now()
            session.add_all(citation_records)

    def record_audit_event(
        self,
        principal: Principal,
        action: str,
        resource_type: str,
        resource_id: str,
        request_id: str,
        outcome: str,
        details: Optional[dict] = None,
    ) -> None:
        with self.session_factory.begin() as session:
            session.add(AuditEventRecord(
                tenant_id=principal.tenant_id,
                subject=principal.subject,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                request_id=request_id,
                outcome=outcome,
                details_json=dict(details or {}),
            ))


def _citation_lineage(citations: Iterable[CitationRecord]) -> list[dict]:
    entries = [
        {
            "rank": int(citation.rank),
            "document_version_id": str(citation.document_version_id or ""),
            "chunk_record_id": str(citation.chunk_record_id or ""),
            "chunk_id": str(citation.chunk_id or ""),
            "content_sha256": str(citation.content_sha256 or ""),
            "page": str(citation.page or ""),
            "verified": bool(citation.verified),
        }
        for citation in citations
    ]
    return sorted(entries, key=lambda entry: (
        entry["rank"],
        entry["document_version_id"],
        entry["chunk_record_id"],
        entry["chunk_id"],
        entry["content_sha256"],
        entry["page"],
        entry["verified"],
    ))


def _citation_lineage_intact(
    retrieval_json: dict,
    citations: Iterable[CitationRecord],
    pipeline_version: str = "",
) -> bool:
    if not isinstance(retrieval_json, dict) or _CITATION_LINEAGE_KEY not in retrieval_json:
        return not _lineage_required(pipeline_version)
    return retrieval_json[_CITATION_LINEAGE_KEY] == _citation_lineage(citations)


def _lineage_required(pipeline_version: str) -> bool:
    version = re.fullmatch(r"canonical-v(\d+)", str(pipeline_version or ""))
    return bool(version and int(version.group(1)) >= 2)


def _history_dependency_status(
    retrieval_json: dict,
    pipeline_version: str,
    run_id: str,
) -> tuple[bool, tuple[str, ...]]:
    required = _lineage_required(pipeline_version)
    if not isinstance(retrieval_json, dict) or _HISTORY_DEPENDENCY_KEY not in retrieval_json:
        return not required, ()  # Older runs cannot prove their history dependencies.
    manifest = retrieval_json[_HISTORY_DEPENDENCY_KEY]
    if not isinstance(manifest, dict) or manifest.get("complete") is not True:
        return False, ()
    run_ids = manifest.get("run_ids")
    if not isinstance(run_ids, list) or len(run_ids) > 10:
        return False, ()
    if any(
        not isinstance(dependency_id, str)
        or not dependency_id
        or dependency_id != dependency_id.strip()
        or dependency_id == run_id
        for dependency_id in run_ids
    ) or len(set(run_ids)) != len(run_ids):
        return False, ()
    return True, tuple(run_ids)


def _context_lineage_for_result(raw_retrieval) -> dict:
    if not isinstance(raw_retrieval, dict):
        raise InvalidServiceStateError("retrieval metadata is missing")
    needed = bool(raw_retrieval.get("needed"))
    contexts = raw_retrieval.get("contexts")
    if not isinstance(contexts, list):
        raise InvalidServiceStateError("final context list is missing")
    if contexts and not needed:
        raise InvalidServiceStateError("non-retrieval answer has final contexts")
    raw_contexts = raw_retrieval.get("raw_contexts")
    raw_by_id: dict[str, list[dict[str, str]]] = {}
    if isinstance(raw_contexts, list):
        for raw_context in raw_contexts:
            if not isinstance(raw_context, dict):
                continue
            raw_id = str(raw_context.get("id") or "").strip()
            if raw_id:
                raw_by_id.setdefault(raw_id, []).append(
                    {
                        "document_version_id": str(
                            raw_context.get("documentVersionId") or ""
                        ).strip(),
                        "chunk_record_id": str(
                            raw_context.get("chunkRecordId") or ""
                        ).strip(),
                        "content_sha256": str(
                            raw_context.get("contentSha256") or ""
                        ).strip().lower(),
                    }
                )
    entries = []
    for context in contexts:
        if not isinstance(context, dict):
            raise InvalidServiceStateError("final context is malformed")
        context_id = str(context.get("id") or "").strip()
        if not context_id:
            raise InvalidServiceStateError("final context identifier is missing")
        version_id = str(context.get("documentVersionId") or "").strip()
        chunk_record_id = str(context.get("chunkRecordId") or "").strip()
        content_sha256 = str(context.get("contentSha256") or "").strip().lower()
        raw_candidates = []
        raw_id = re.sub(r"::(?:focus|fine-evidence)-[1-9]\d*$", "", context_id)
        raw_candidates = raw_by_id.get(raw_id, [])
        if not version_id:
            raw_versions = {
                candidate["document_version_id"]
                for candidate in raw_candidates
                if candidate["document_version_id"]
            }
            if len(raw_versions) != 1:
                raise InvalidServiceStateError(
                    "final context document version cannot be resolved uniquely"
                )
            version_id = next(iter(raw_versions))
        if not chunk_record_id:
            raw_chunk_ids = {
                candidate["chunk_record_id"]
                for candidate in raw_candidates
                if candidate["chunk_record_id"]
            }
            if len(raw_chunk_ids) == 1:
                chunk_record_id = next(iter(raw_chunk_ids))
        if not content_sha256:
            raw_hashes = {
                candidate["content_sha256"]
                for candidate in raw_candidates
                if candidate["content_sha256"]
            }
            if len(raw_hashes) == 1:
                content_sha256 = next(iter(raw_hashes))
        if not content_sha256:
            content = str(context.get("content") or "")
            if content:
                content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
        entries.append({
            "context_id": context_id,
            "document_version_id": version_id,
            "chunk_record_id": chunk_record_id,
            "content_sha256": content_sha256,
        })
    return {
        "chunk_lineage": _CONTEXT_CHUNK_LINEAGE_VERSION,
        "state": (
            "retrieved_contexts" if entries
            else "retrieval_no_context" if needed
            else "no_retrieval"
        ),
        "contexts": entries,
    }


def _context_lineage_status(
    retrieval_json: dict,
    retrieval_needed: bool,
    pipeline_version: str = "",
) -> tuple[bool, set[str]]:
    if not isinstance(retrieval_json, dict) or _CONTEXT_LINEAGE_KEY not in retrieval_json:
        return not _lineage_required(pipeline_version), set()
    lineage = retrieval_json[_CONTEXT_LINEAGE_KEY]
    if not isinstance(lineage, dict) or not isinstance(lineage.get("contexts"), list):
        return False, set()
    entries = lineage["contexts"]
    expected_state = (
        "retrieved_contexts" if entries
        else "retrieval_no_context" if retrieval_needed
        else "no_retrieval"
    )
    if (
        lineage.get("state") != expected_state
        or (entries and not retrieval_needed)
        or retrieval_json.get("needed") is not retrieval_needed
        or not isinstance(retrieval_json.get("context_ids"), list)
    ):
        return False, set()
    context_ids = []
    version_ids = set()
    chunk_lineage_version = lineage.get("chunk_lineage")
    if _lineage_required(pipeline_version) and chunk_lineage_version not in (
        None,
        _CONTEXT_CHUNK_LINEAGE_VERSION,
    ):
        return False, set()
    for entry in entries:
        if not isinstance(entry, dict):
            return False, set()
        context_id = entry.get("context_id")
        version_id = entry.get("document_version_id")
        if (
            not isinstance(context_id, str) or not context_id.strip()
            or not isinstance(version_id, str) or not version_id.strip()
        ):
            return False, set()
        if (
            _lineage_required(pipeline_version)
            and chunk_lineage_version == _CONTEXT_CHUNK_LINEAGE_VERSION
        ):
            chunk_record_id = entry.get("chunk_record_id")
            content_sha256 = entry.get("content_sha256")
            if (
                not isinstance(chunk_record_id, str)
                or not chunk_record_id.strip()
                or not isinstance(content_sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", content_sha256.lower())
            ):
                return False, set()
        context_ids.append(context_id)
        version_ids.add(version_id)
    if retrieval_json["context_ids"] != context_ids:
        return False, set()
    return True, version_ids


def _context_lineage_chunks_intact(
    session,
    *,
    tenant_id: str,
    knowledge_base_id: str,
    index_version_id: str,
    retrieval_json: dict,
    pipeline_version: str,
) -> bool:
    """Verify the chunks that supplied new answer Contexts still exist unchanged."""
    if not _lineage_required(pipeline_version):
        return True
    if not isinstance(retrieval_json, dict):
        return False
    lineage = retrieval_json.get(_CONTEXT_LINEAGE_KEY)
    if not isinstance(lineage, dict):
        return False
    chunk_lineage_version = lineage.get("chunk_lineage")
    if chunk_lineage_version is None:
        # Runs written before chunk-level Context lineage remain readable under
        # the documented compatibility limit; they cannot prove chunk state.
        return True
    if chunk_lineage_version != _CONTEXT_CHUNK_LINEAGE_VERSION:
        return False
    entries = lineage.get("contexts")
    if not isinstance(entries, list):
        return False
    if not entries:
        return True
    clean_ids = {
        str(entry.get("chunk_record_id") or "").strip()
        for entry in entries
        if isinstance(entry, dict)
    }
    if not index_version_id or not clean_ids or "" in clean_ids:
        return False
    rows = list(session.scalars(
        select(ChunkRecord).where(
            ChunkRecord.id.in_(clean_ids),
            ChunkRecord.tenant_id == tenant_id,
            ChunkRecord.knowledge_base_id == knowledge_base_id,
            ChunkRecord.index_version_id == index_version_id,
        )
    ))
    chunks_by_id = {row.id: row for row in rows}
    for entry in entries:
        if not isinstance(entry, dict):
            return False
        chunk_id = str(entry.get("chunk_record_id") or "").strip()
        version_id = str(entry.get("document_version_id") or "").strip()
        content_sha256 = str(entry.get("content_sha256") or "").strip().lower()
        chunk = chunks_by_id.get(chunk_id)
        if (
            chunk is None
            or chunk.document_version_id != version_id
            or str(chunk.content_sha256 or "").strip().lower() != content_sha256
        ):
            return False
    return True


def _available_context_version_ids(
    session,
    *,
    tenant_id: str,
    knowledge_base_id: str,
    version_ids: set[str],
) -> set[str]:
    if not version_ids:
        return set()
    return set(session.scalars(
        select(DocumentVersionRecord.id)
        .join(
            DocumentRecord,
            and_(
                DocumentRecord.id == DocumentVersionRecord.document_id,
                DocumentRecord.tenant_id == tenant_id,
                DocumentRecord.knowledge_base_id == knowledge_base_id,
                DocumentRecord.deleted_at.is_(None),
                DocumentRecord.status != "deleted",
            ),
        )
        .where(
            DocumentVersionRecord.id.in_(version_ids),
            DocumentVersionRecord.tenant_id == tenant_id,
        )
    ))


def _safe_retrieval_record(raw_retrieval, raw_evidence_validation=None) -> dict:
    raw = raw_retrieval if isinstance(raw_retrieval, dict) else {}
    contexts = raw.get("contexts") if isinstance(raw.get("contexts"), list) else []
    retrieval = {
        "needed": bool(raw.get("needed")),
        "context_ids": [
            str(context.get("id") or "")
            for context in contexts
            if isinstance(context, dict)
        ],
    }
    if isinstance(raw_evidence_validation, dict):
        retrieval["evidence_validation"] = {
            "sufficient": bool(raw_evidence_validation.get("sufficient")),
            "status": str(raw_evidence_validation.get("status") or ""),
            "valid_citations": [
                int(rank)
                for rank in raw_evidence_validation.get("valid_citations") or []
                if str(rank).isdigit()
            ],
            "invalid_citations": [
                int(rank)
                for rank in raw_evidence_validation.get("invalid_citations") or []
                if str(rank).isdigit()
            ],
            "uncited_claims": [
                str(claim)[:500]
                for claim in raw_evidence_validation.get("uncited_claims") or []
                if str(claim).strip()
            ][:20],
            "unsupported_claims": [
                str(claim)[:500]
                for claim in raw_evidence_validation.get("unsupported_claims") or []
                if str(claim).strip()
            ][:20],
            "reason": str(raw_evidence_validation.get("reason") or ""),
        }
    return retrieval


def _upload_reservation(record: UploadSessionRecord) -> UploadReservation:
    return UploadReservation(
        id=record.id,
        object_key=record.object_key,
        state=record.state,
        expires_at=record.expires_at,
        expected_size_bytes=record.expected_size_bytes,
        expected_sha256=record.expected_sha256,
        declared_mime_type=record.declared_mime_type,
    )


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _require_sha256(value: str, field_name: str) -> str:
    clean = str(value or "").strip().lower()
    if len(clean) != 64 or any(
        character not in "0123456789abcdef" for character in clean
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return clean
