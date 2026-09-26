from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Sequence

from sqlalchemy import func, select, text
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


class AccessDeniedError(PermissionError):
    pass


class ResourceNotFoundError(LookupError):
    pass


class InvalidServiceStateError(RuntimeError):
    pass


class ConcurrentPublishError(RuntimeError):
    pass


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
            conversation = session.scalar(select(ConversationRecord.id).where(
                ConversationRecord.id == conversation_id,
                ConversationRecord.tenant_id == principal.tenant_id,
                ConversationRecord.user_id == authorized.user_id,
                ConversationRecord.knowledge_base_id == authorized.id,
            ))
            if conversation is None:
                raise ResourceNotFoundError("conversation was not found")
            rows = list(session.scalars(
                select(MessageRecord)
                .where(
                    MessageRecord.tenant_id == principal.tenant_id,
                    MessageRecord.conversation_id == conversation_id,
                )
                .order_by(MessageRecord.created_at.desc(), MessageRecord.id.desc())
                .limit(max(1, min(int(limit), 50)))
            ))
            return [
                {"role": row.role, "content": row.content}
                for row in reversed(rows)
            ]

    def list_conversations(self, *, principal: Principal) -> list[dict]:
        with self.session_factory() as session:
            user, _membership = self._active_identity(session, principal)
            rows = list(session.scalars(
                select(ConversationRecord)
                .where(
                    ConversationRecord.tenant_id == principal.tenant_id,
                    ConversationRecord.user_id == user.id,
                )
                .order_by(ConversationRecord.updated_at.desc(), ConversationRecord.id)
            ))
            return [
                {
                    "id": row.id,
                    "knowledge_base_id": row.knowledge_base_id,
                    "title": row.title,
                    "created_at": row.created_at.isoformat(),
                    "updated_at": row.updated_at.isoformat(),
                }
                for row in rows
            ]

    def get_conversation(
        self,
        *,
        principal: Principal,
        conversation_id: str,
    ) -> dict:
        with self.session_factory() as session:
            user, _membership = self._active_identity(session, principal)
            row = session.scalar(select(ConversationRecord).where(
                ConversationRecord.id == conversation_id,
                ConversationRecord.tenant_id == principal.tenant_id,
                ConversationRecord.user_id == user.id,
            ))
            if row is None:
                raise ResourceNotFoundError("conversation was not found")
            messages = list(session.scalars(
                select(MessageRecord)
                .where(
                    MessageRecord.tenant_id == principal.tenant_id,
                    MessageRecord.conversation_id == row.id,
                )
                .order_by(MessageRecord.created_at, MessageRecord.id)
            ))
            run_ids = {
                str((message.metadata_json or {}).get("run_id") or "")
                for message in messages
                if message.role == "assistant"
            }
            run_ids.discard("")
            answer_runs = {
                answer_run.id: answer_run
                for answer_run in session.scalars(
                    select(AnswerRunRecord).where(
                        AnswerRunRecord.id.in_(run_ids),
                        AnswerRunRecord.tenant_id == principal.tenant_id,
                        AnswerRunRecord.user_id == user.id,
                        AnswerRunRecord.conversation_id == row.id,
                    )
                )
            } if run_ids else {}
            return {
                "id": row.id,
                "knowledge_base_id": row.knowledge_base_id,
                "title": row.title,
                "created_at": row.created_at.isoformat(),
                "updated_at": row.updated_at.isoformat(),
                "messages": [self._conversation_message_payload(message, answer_runs)
                             for message in messages],
            }

    @staticmethod
    def _conversation_message_payload(
        message: MessageRecord,
        answer_runs: dict[str, AnswerRunRecord],
    ) -> dict:
        payload = {
            "id": message.id,
            "role": message.role,
            "content": message.content,
            "created_at": message.created_at.isoformat(),
        }
        run_id = str((message.metadata_json or {}).get("run_id") or "")
        answer_run = answer_runs.get(run_id)
        if message.role == "assistant" and answer_run is not None:
            payload["answer_run"] = {
                "run_id": answer_run.id,
                "retrieval": {"needed": answer_run.retrieval_needed},
                "model": {"name": answer_run.model},
                "timings": answer_run.timings_json,
                "evidence_validation": (
                    dict((answer_run.retrieval_json or {}).get("evidence_validation") or {})
                    if isinstance(answer_run.retrieval_json, dict)
                    else None
                ),
                "citations": [],
            }
        return payload

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
            user, _membership = self._active_identity(session, principal)
            run = session.scalar(select(AnswerRunRecord).where(
                AnswerRunRecord.id == run_id,
                AnswerRunRecord.tenant_id == principal.tenant_id,
                AnswerRunRecord.user_id == user.id,
            ))
            if run is None:
                raise ResourceNotFoundError("answer run was not found")
            citations = list(session.scalars(
                select(CitationRecord)
                .where(
                    CitationRecord.tenant_id == principal.tenant_id,
                    CitationRecord.answer_run_id == run.id,
                )
                .order_by(CitationRecord.rank, CitationRecord.id)
            ))
            evidence = []
            for citation in citations:
                chunk = (
                    session.scalar(
                        select(ChunkRecord).where(
                            ChunkRecord.id == citation.chunk_record_id,
                            ChunkRecord.tenant_id == principal.tenant_id,
                            ChunkRecord.knowledge_base_id == run.knowledge_base_id,
                        )
                    )
                    if citation.chunk_record_id
                    else None
                )
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
                        )
                    )
                    if version is not None
                    else None
                )
                evidence.append({
                    "rank": citation.rank,
                    "chunk_id": citation.chunk_id,
                    "chunk_record_id": citation.chunk_record_id,
                    "document_version_id": citation.document_version_id,
                    "document_id": document.id if document is not None else None,
                    "document_name": document.name if document is not None else "",
                    "source_id": document.source_id if document is not None else "",
                    "page": citation.page,
                    "verified": citation.verified,
                    "content_sha256": citation.content_sha256,
                    "title": chunk.title if chunk is not None else "",
                    "content": chunk.content if chunk is not None else "",
                })
            return {
                "id": run.id,
                "knowledge_base_id": run.knowledge_base_id,
                "conversation_id": run.conversation_id,
                "index_version_id": run.index_version_id,
                "model": run.model,
                "pipeline_version": run.pipeline_version,
                "question": run.question,
                "answer": run.answer,
                "retrieval_needed": run.retrieval_needed,
                "timings": run.timings_json,
                "evidence_validation": (
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

    def authorize_knowledge_base(
        self,
        principal: Principal,
        knowledge_base_id: str,
        requested_source_ids: Optional[Sequence[str]],
    ) -> AuthorizedKnowledgeBase:
        with self.session_factory() as session:
            tenant = session.scalar(
                select(Tenant).where(
                    Tenant.id == principal.tenant_id,
                    Tenant.active.is_(True),
                )
            )
            if tenant is None:
                raise AccessDeniedError("tenant is not active")

            user = session.scalar(
                select(UserIdentity).where(
                    UserIdentity.tenant_id == principal.tenant_id,
                    UserIdentity.subject == principal.subject,
                    UserIdentity.active.is_(True),
                )
            )
            if user is None:
                raise AccessDeniedError("identity is not an active tenant member")

            membership = session.scalar(
                select(Membership).where(
                    Membership.tenant_id == principal.tenant_id,
                    Membership.user_id == user.id,
                    Membership.active.is_(True),
                )
            )
            if membership is None:
                raise AccessDeniedError("identity is not an active tenant member")

            knowledge_base = session.scalar(
                select(KnowledgeBaseRecord).where(
                    KnowledgeBaseRecord.id == knowledge_base_id,
                    KnowledgeBaseRecord.tenant_id == principal.tenant_id,
                    KnowledgeBaseRecord.active.is_(True),
                )
            )
            if knowledge_base is None:
                raise ResourceNotFoundError("knowledge base was not found")

            can_read = (
                membership.role in {"owner", "admin"}
                or principal.has_role("tenant_admin")
                or knowledge_base.owner_user_id == user.id
                or knowledge_base.visibility == "tenant"
            )
            if not can_read:
                raise AccessDeniedError("knowledge base access is denied")
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
                    or existing.filename != filename
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
    ) -> None:
        """Atomically publish one validated immutable index version."""
        with self.session_factory.begin() as session:
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
                retrieval_json=_safe_retrieval_record(
                    result.get("retrieval"),
                    result.get("evidence_validation"),
                ),
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
            for citation in result.get("citations") or []:
                session.add(CitationRecord(
                    tenant_id=principal.tenant_id,
                    answer_run_id=run.id,
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
                ))

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


def _safe_retrieval_record(raw_retrieval, raw_evidence_validation=None) -> dict:
    retrieval = dict(raw_retrieval or {})
    contexts = list(retrieval.pop("contexts", []) or [])
    retrieval["context_ids"] = [
        str(context.get("id") or "")
        for context in contexts
        if isinstance(context, dict)
    ]
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
