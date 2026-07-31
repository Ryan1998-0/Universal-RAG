import hashlib
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import jwt
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import select

from rag_demo.production.api import create_app
from rag_demo.production.config import ProductionSettings
from rag_demo.production.database import (
    AnswerRunRecord,
    AuditEventRecord,
    Base,
    ChunkRecord,
    DocumentRecord,
    DocumentVersionRecord,
    IndexDocumentRecord,
    IndexActivationEventRecord,
    IndexVersionRecord,
    KnowledgeBaseRecord,
    Membership,
    Tenant,
    UploadSessionRecord,
    UserIdentity,
    create_database_engine,
    create_session_factory,
)
from rag_demo.production.auth import Principal
from rag_demo.production.repository import (
    ConcurrentPublishError,
    InvalidServiceStateError,
    SqlAlchemyTenantRepository,
)
from rag_demo.production.rate_limit import RateLimitDecision
from rag_demo.production.object_storage import ObjectInfo
from rag_demo.production.index_manifest import index_entries_sha256
from rag_demo.production.upload_service import UploadService


SECRET = "production-api-test-secret-with-32-characters"


class FakePipeline:
    def __init__(self):
        self.calls = []

    def run(self, request):
        self.calls.append(request)
        return {
            "schema_version": "rag-agent-response-v1",
            "run_id": f"run-{len(self.calls)}",
            "answer": "這是經過伺服器檢索後產生的回答。來源：[1]",
            "confidence": "high",
            "citations": [
                {
                    "id": "chunk-a-1",
                    "rank": 1,
                    "title": "測試文件",
                    "page": "1",
                    "source": "a-ready",
                    "run_id": f"run-{len(self.calls)}",
                    "content_sha256": "a" * 64,
                    "document_version_id": "version-a",
                }
            ],
            "grounding_warnings": [],
            "retrieval": {
                "server_generated": True,
                "needed": True,
                "reason": "需要文件證據",
                "query": request.question,
                "contexts": [{"content": "不得公開的完整內部內容"}],
                "evidence_evaluation": {
                    "sufficient": True,
                    "confidence": "high",
                    "reason": "證據充足",
                    "private_debug": "must not be returned",
                },
            },
            "model": {"provider": "ollama", "name": "qwen2.5:7b"},
            "timings": {"totalMs": 12.5},
        }


class DenyRateLimiter:
    def allow(self, principal, action):
        return RateLimitDecision(False, remaining=0, retry_after_seconds=37)


class FailingProbe:
    def ping(self):
        raise RuntimeError("dependency unavailable")


class FakeUploadStorage:
    def __init__(self):
        self.reservations = {}
        self.objects = {}
        self.downloads = {}

    def ping(self):
        return None

    def presign_upload(self, *, key, content_type, expected_sha256, expires_seconds):
        self.reservations[key] = {
            "content_type": content_type,
            "sha256": expected_sha256,
        }
        return {
            "url": f"https://upload.invalid/{key}",
            "headers": {
                "Content-Type": content_type,
                "x-amz-meta-sha256": expected_sha256,
            },
            "expires_in": expires_seconds,
        }

    def head(self, key):
        return self.objects[key]

    def put_reserved_upload(
        self,
        *,
        key,
        body,
        content_type,
        size_bytes,
        sha256,
    ):
        payload = body.read()
        self.reservations[key] = {
            "content_type": content_type,
            "sha256": sha256,
            "payload": payload,
        }
        info = ObjectInfo(
            key=key,
            size_bytes=size_bytes,
            content_type=content_type,
            sha256=sha256,
        )
        self.objects[key] = info
        self.downloads[key] = payload
        return info

    def download_bytes(self, key, max_bytes):
        payload = self.downloads[key]
        if len(payload) > max_bytes:
            raise RuntimeError("object too large")
        return payload


class FakeTaskDispatcher:
    def __init__(self):
        self.jobs = []
        self.index_jobs = []

    def dispatch_ingestion(self, job_id):
        self.jobs.append(job_id)
        return f"task-{job_id}"

    def dispatch_index_build(self, job_id):
        self.index_jobs.append(job_id)
        return f"index-task-{job_id}"


class ProductionApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        database_url = f"sqlite+pysqlite:///{self.temp_dir.name}/api-test.sqlite3"
        self.settings = ProductionSettings(
            RAG_ENV="test",
            RAG_DATABASE_URL=database_url,
            RAG_AUTH_MODE="dev_hs256",
            RAG_DEV_JWT_SECRET=SECRET,
            RAG_MODEL="ollama:qwen2.5:7b",
            RAG_ALLOWED_MODELS="ollama:qwen2.5:7b",
            RAG_ENABLE_API_DOCS=True,
        )
        self.engine = create_database_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.session_factory = create_session_factory(self.engine)
        self.repository = SqlAlchemyTenantRepository(self.session_factory)
        self.pipeline = FakePipeline()
        self.upload_storage = FakeUploadStorage()
        self.dispatcher = FakeTaskDispatcher()
        self.upload_service = UploadService(
            repository=self.repository,
            object_storage=self.upload_storage,
            settings=self.settings,
            dispatcher=self.dispatcher,
        )
        self._seed_database()
        self.client = TestClient(create_app(
            settings=self.settings,
            repository=self.repository,
            pipeline=self.pipeline,
            upload_service=self.upload_service,
        ))

    def tearDown(self):
        self.client.close()
        self.engine.dispose()
        self.temp_dir.cleanup()

    def _seed_database(self):
        with self.session_factory.begin() as session:
            session.add_all([
                Tenant(id="tenant-a", name="Tenant A"),
                Tenant(id="tenant-b", name="Tenant B"),
            ])
            session.add_all([
                UserIdentity(
                    id="user-a",
                    tenant_id="tenant-a",
                    subject="subject-a",
                    display_name="User A",
                ),
                UserIdentity(
                    id="user-b",
                    tenant_id="tenant-b",
                    subject="subject-b",
                    display_name="User B",
                ),
            ])
            session.add_all([
                Membership(
                    id="membership-a",
                    tenant_id="tenant-a",
                    user_id="user-a",
                    role="member",
                ),
                Membership(
                    id="membership-b",
                    tenant_id="tenant-b",
                    user_id="user-b",
                    role="member",
                ),
            ])
            kb_a = KnowledgeBaseRecord(
                id="kb-a",
                tenant_id="tenant-a",
                owner_user_id="user-a",
                name="KB A",
                profile="ifrs17",
                visibility="private",
            )
            kb_b = KnowledgeBaseRecord(
                id="kb-b",
                tenant_id="tenant-b",
                owner_user_id="user-b",
                name="KB B",
                profile="labor-law",
                visibility="private",
            )
            session.add_all([kb_a, kb_b])
            session.flush()

            index_a = IndexVersionRecord(
                id="index-a",
                tenant_id="tenant-a",
                knowledge_base_id="kb-a",
                created_by_user_id="user-a",
                version_number=1,
                status="ready",
                embedding_model="test-embedding",
                embedding_dimensions=2,
                reranker_model="test-reranker",
                chunk_schema_version="test-v1",
                qdrant_collection="rag_chunks",
                chunk_count=1,
                manifest_sha256="a" * 64,
                expected_document_count=1,
                expected_chunk_count=1,
                expected_vector_count=1,
                validated_pg_chunk_count=1,
                validated_qdrant_point_count=1,
                validated_at=datetime.now(timezone.utc),
            )
            index_b = IndexVersionRecord(
                id="index-b",
                tenant_id="tenant-b",
                knowledge_base_id="kb-b",
                created_by_user_id="user-b",
                version_number=1,
                status="ready",
                embedding_model="test-embedding",
                embedding_dimensions=2,
                reranker_model="test-reranker",
                chunk_schema_version="test-v1",
                qdrant_collection="rag_chunks",
                chunk_count=1,
                manifest_sha256="b" * 64,
                expected_document_count=1,
                expected_chunk_count=1,
                expected_vector_count=1,
                validated_pg_chunk_count=1,
                validated_qdrant_point_count=1,
                validated_at=datetime.now(timezone.utc),
            )
            document_a_ready = DocumentRecord(
                    id="document-a-ready",
                    tenant_id="tenant-a",
                    knowledge_base_id="kb-a",
                    owner_user_id="user-a",
                    source_id="a-ready",
                    name="Ready A.pdf",
                    status="ready",
                )
            document_a_uploaded = DocumentRecord(
                    id="document-a-uploaded",
                    tenant_id="tenant-a",
                    knowledge_base_id="kb-a",
                    owner_user_id="user-a",
                    source_id="a-uploaded",
                    name="Uploaded A.pdf",
                    status="uploaded",
                )
            document_b_ready = DocumentRecord(
                    id="document-b-ready",
                    tenant_id="tenant-b",
                    knowledge_base_id="kb-b",
                    owner_user_id="user-b",
                    source_id="b-ready",
                    name="Ready B.pdf",
                    status="ready",
                )
            session.add_all([
                index_a,
                index_b,
                document_a_ready,
                document_a_uploaded,
                document_b_ready,
            ])
            session.flush()

            version_a = DocumentVersionRecord(
                id="version-a",
                tenant_id="tenant-a",
                document_id="document-a-ready",
                version_number=1,
                sha256="a" * 64,
                object_key="tenants/tenant-a/a.pdf",
                extracted_object_key=(
                    "tenants/tenant-a/knowledge-bases/kb-a/documents/document-a-ready"
                    "/versions/version-a/extracted/canonical.json"
                ),
                mime_type="application/pdf",
                parser_version="test-v1",
                chunk_schema_version="test-v1",
                embedding_model="test-embedding",
                status="ready",
            )
            version_b = DocumentVersionRecord(
                id="version-b",
                tenant_id="tenant-b",
                document_id="document-b-ready",
                version_number=1,
                sha256="b" * 64,
                object_key="tenants/tenant-b/b.pdf",
                mime_type="application/pdf",
                parser_version="test-v1",
                chunk_schema_version="test-v1",
                embedding_model="test-embedding",
                status="ready",
            )
            session.add_all([version_a, version_b])
            session.flush()

            document_a_ready.current_version_id = version_a.id
            document_b_ready.current_version_id = version_b.id
            kb_a.active_index_version_id = index_a.id
            kb_b.active_index_version_id = index_b.id
            session.add_all([
                IndexDocumentRecord(
                    id="index-document-a",
                    tenant_id="tenant-a",
                    knowledge_base_id="kb-a",
                    index_version_id="index-a",
                    document_id="document-a-ready",
                    document_version_id="version-a",
                    status="ready",
                    chunk_count=1,
                ),
                IndexDocumentRecord(
                    id="index-document-b",
                    tenant_id="tenant-b",
                    knowledge_base_id="kb-b",
                    index_version_id="index-b",
                    document_id="document-b-ready",
                    document_version_id="version-b",
                    status="ready",
                    chunk_count=1,
                ),
            ])

    def _token(self, tenant_id="tenant-a", subject="subject-a"):
        now = datetime.now(timezone.utc)
        return jwt.encode(
            {
                "sub": subject,
                "tenant_id": tenant_id,
                "roles": ["member"],
                "aud": "ifrs17-rag-dev",
                "iat": now,
                "exp": now + timedelta(minutes=5),
            },
            SECRET,
            algorithm="HS256",
        )

    def _headers(self, **kwargs):
        return {"Authorization": f"Bearer {self._token(**kwargs)}"}

    def test_requires_bearer_token(self):
        response = self.client.post("/v1/ask", json={
            "question": "什麼是 CSM？",
            "knowledge_base_id": "kb-a",
        })

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], "AUTH_REQUIRED")
        self.assertEqual(response.headers["www-authenticate"], "Bearer")

    def test_expired_token_is_rejected(self):
        now = datetime.now(timezone.utc)
        expired = jwt.encode(
            {
                "sub": "subject-a",
                "tenant_id": "tenant-a",
                "aud": "ifrs17-rag-dev",
                "iat": now - timedelta(minutes=10),
                "exp": now - timedelta(minutes=5),
            },
            SECRET,
            algorithm="HS256",
        )

        response = self.client.post(
            "/v1/ask",
            headers={"Authorization": f"Bearer {expired}"},
            json={"question": "什麼是 CSM？", "knowledge_base_id": "kb-a"},
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], "INVALID_TOKEN")

    def test_authorized_request_forces_ready_tenant_sources_and_hides_contexts(self):
        response = self.client.post(
            "/v1/ask",
            headers=self._headers(),
            json={"question": "什麼是 CSM？", "knowledge_base_id": "kb-a"},
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.pipeline.calls[0].profile, "ifrs17")
        self.assertEqual(self.pipeline.calls[0].source_ids, ["a-ready"])
        self.assertFalse(self.pipeline.calls[0].persist_conversation)
        self.assertEqual(self.pipeline.calls[0].retrieval_scope.tenant_id, "tenant-a")
        self.assertEqual(self.pipeline.calls[0].retrieval_scope.knowledge_base_id, "kb-a")
        self.assertNotIn("contexts", response.json()["retrieval"])
        self.assertNotIn(
            "private_debug",
            response.json()["retrieval"]["evidence_evaluation"],
        )
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")

        with self.session_factory() as session:
            answer_runs = list(session.scalars(select(AnswerRunRecord)))
            audit_events = list(session.scalars(select(AuditEventRecord)))
        self.assertEqual(len(answer_runs), 1)
        self.assertEqual(answer_runs[0].tenant_id, "tenant-a")
        self.assertEqual(answer_runs[0].source_ids_json, ["a-ready"])
        self.assertEqual(len(audit_events), 1)
        self.assertEqual(audit_events[0].outcome, "success")

    def test_answer_run_and_source_content_are_protected_and_traceable(self):
        source_payload = b"%PDF-1.4\nsource\n%%EOF"
        self.upload_storage.downloads["tenants/tenant-a/a.pdf"] = source_payload
        answer = self.client.post(
            "/v1/ask",
            headers=self._headers(),
            json={"question": "什麼是 CSM？", "knowledge_base_id": "kb-a"},
        )
        self.assertEqual(answer.status_code, 200, answer.text)
        run_id = answer.json()["run_id"]
        citation = answer.json()["citations"][0]
        self.assertEqual(citation["details_url"], f"/v1/answer-runs/{run_id}")
        self.assertEqual(
            citation["source_url"],
            "/v1/document-versions/version-a/content",
        )

        details = self.client.get(
            f"/v1/answer-runs/{run_id}",
            headers=self._headers(),
        )
        self.assertEqual(details.status_code, 200, details.text)
        self.assertEqual(details.json()["citations"][0]["document_name"], "Ready A.pdf")
        self.assertEqual(
            details.json()["citations"][0]["source_url"],
            "/v1/document-versions/version-a/content",
        )

        source = self.client.get(
            "/v1/document-versions/version-a/content",
            headers=self._headers(),
        )
        self.assertEqual(source.status_code, 200, source.text)
        self.assertEqual(source.content, source_payload)
        self.assertIn("inline", source.headers["content-disposition"])
        self.assertEqual(source.headers["cache-control"], "private, no-store")

        cross_tenant = self.client.get(
            f"/v1/answer-runs/{run_id}",
            headers=self._headers(tenant_id="tenant-b", subject="subject-b"),
        )
        self.assertEqual(cross_tenant.status_code, 404)

    def test_cross_tenant_knowledge_base_is_not_disclosed(self):
        response = self.client.post(
            "/v1/ask",
            headers=self._headers(),
            json={"question": "秘密是什麼？", "knowledge_base_id": "kb-b"},
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "KNOWLEDGE_BASE_NOT_FOUND")
        self.assertEqual(self.pipeline.calls, [])

    def test_inactive_tenant_is_denied_before_pipeline(self):
        with self.session_factory.begin() as session:
            session.get(Tenant, "tenant-a").active = False

        response = self.client.post(
            "/v1/ask",
            headers=self._headers(),
            json={"question": "什麼是 CSM？", "knowledge_base_id": "kb-a"},
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.pipeline.calls, [])

    def test_cross_tenant_or_unready_source_is_denied(self):
        for source_id in ("b-ready", "a-uploaded"):
            with self.subTest(source_id=source_id):
                response = self.client.post(
                    "/v1/ask",
                    headers=self._headers(),
                    json={
                        "question": "秘密是什麼？",
                        "knowledge_base_id": "kb-a",
                        "source_ids": [source_id],
                    },
                )
                self.assertEqual(response.status_code, 403)
                self.assertEqual(response.json()["error"]["code"], "ACCESS_DENIED")
        self.assertEqual(self.pipeline.calls, [])

    def test_explicit_empty_source_selection_stays_empty(self):
        response = self.client.post(
            "/v1/ask",
            headers=self._headers(),
            json={
                "question": "2+2 是多少？",
                "knowledge_base_id": "kb-a",
                "source_ids": [],
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.pipeline.calls[0].source_ids, [])

    def test_resource_apis_create_and_list_knowledge_bases_folders_and_documents(self):
        initial = self.client.get("/v1/knowledge-bases", headers=self._headers())
        self.assertEqual(initial.status_code, 200, initial.text)
        self.assertEqual([item["id"] for item in initial.json()["items"]], ["kb-a"])

        created = self.client.post(
            "/v1/knowledge-bases",
            headers=self._headers(),
            json={"name": "勞基法", "profile": "labor-law", "visibility": "private"},
        )
        self.assertEqual(created.status_code, 201, created.text)
        self.assertTrue(created.json()["can_write"])

        folder = self.client.post(
            "/v1/knowledge-bases/kb-a/folders",
            headers=self._headers(),
            json={"name": "會計準則"},
        )
        self.assertEqual(folder.status_code, 201, folder.text)
        folder_id = folder.json()["id"]
        moved = self.client.put(
            "/v1/knowledge-bases/kb-a/documents/document-a-ready/folder",
            headers=self._headers(),
            json={"folder_id": folder_id},
        )
        self.assertEqual(moved.status_code, 200, moved.text)
        self.assertEqual(moved.json()["folder_id"], folder_id)
        renamed = self.client.put(
            f"/v1/knowledge-bases/kb-a/folders/{folder_id}",
            headers=self._headers(),
            json={"name": "IFRS 17 準則"},
        )
        self.assertEqual(renamed.status_code, 200, renamed.text)
        self.assertEqual(renamed.json()["name"], "IFRS 17 準則")
        folders = self.client.get(
            "/v1/knowledge-bases/kb-a/folders",
            headers=self._headers(),
        )
        self.assertEqual(folders.json()["items"][0]["name"], "IFRS 17 準則")

        documents = self.client.get(
            "/v1/knowledge-bases/kb-a/documents",
            headers=self._headers(),
        )
        self.assertEqual(documents.status_code, 200, documents.text)
        self.assertEqual(
            {item["id"] for item in documents.json()["items"]},
            {"document-a-ready", "document-a-uploaded"},
        )
        deleted = self.client.delete(
            f"/v1/knowledge-bases/kb-a/folders/{folder_id}",
            headers=self._headers(),
        )
        self.assertEqual(deleted.status_code, 200, deleted.text)
        self.assertEqual(deleted.json()["moved_document_count"], 1)
        refreshed = self.client.get(
            "/v1/knowledge-bases/kb-a/documents",
            headers=self._headers(),
        )
        moved_document = next(
            item for item in refreshed.json()["items"] if item["id"] == "document-a-ready"
        )
        self.assertIsNone(moved_document["folder_id"])

    def test_conversation_history_is_tenant_scoped_and_reused_by_pipeline(self):
        created = self.client.post(
            "/v1/conversations",
            headers=self._headers(),
            json={"knowledge_base_id": "kb-a"},
        )
        self.assertEqual(created.status_code, 201, created.text)
        conversation_id = created.json()["id"]
        first = self.client.post(
            "/v1/ask",
            headers=self._headers(),
            json={
                "question": "第一個問題",
                "knowledge_base_id": "kb-a",
                "conversation_id": conversation_id,
            },
        )
        second = self.client.post(
            "/v1/ask",
            headers=self._headers(),
            json={
                "question": "那第二個呢？",
                "knowledge_base_id": "kb-a",
                "conversation_id": conversation_id,
            },
        )
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(second.json()["conversation_id"], conversation_id)
        self.assertEqual(
            [message["role"] for message in self.pipeline.calls[-1].history],
            ["user", "assistant"],
        )
        loaded = self.client.get(
            f"/v1/conversations/{conversation_id}",
            headers=self._headers(),
        )
        messages = loaded.json()["messages"]
        self.assertEqual(len(messages), 4)
        assistant_messages = [message for message in messages if message["role"] == "assistant"]
        self.assertEqual(
            [message["answer_run"]["run_id"] for message in assistant_messages],
            [first.json()["run_id"], second.json()["run_id"]],
        )
        self.assertTrue(all(message["answer_run"]["model"]["name"] for message in assistant_messages))
        cross_tenant = self.client.get(
            f"/v1/conversations/{conversation_id}",
            headers=self._headers(tenant_id="tenant-b", subject="subject-b"),
        )
        self.assertEqual(cross_tenant.status_code, 404)
        deleted = self.client.delete(
            f"/v1/conversations/{conversation_id}",
            headers=self._headers(),
        )
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(
            self.client.get(
                f"/v1/conversations/{conversation_id}",
                headers=self._headers(),
            ).status_code,
            404,
        )

    def test_document_version_not_in_active_index_is_not_authorized(self):
        with self.session_factory.begin() as session:
            replacement = DocumentVersionRecord(
                id="version-a-2",
                tenant_id="tenant-a",
                document_id="document-a-ready",
                version_number=2,
                sha256="c" * 64,
                object_key="tenants/tenant-a/a-v2.pdf",
                mime_type="application/pdf",
                parser_version="test-v1",
                chunk_schema_version="test-v1",
                embedding_model="test-embedding",
                status="ready",
            )
            session.add(replacement)
            session.flush()
            session.get(DocumentRecord, "document-a-ready").current_version_id = replacement.id

        response = self.client.post(
            "/v1/ask",
            headers=self._headers(),
            json={
                "question": "Use the replacement",
                "knowledge_base_id": "kb-a",
                "source_ids": ["a-ready"],
            },
        )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.pipeline.calls, [])

    def test_index_validation_and_publish_are_atomic_and_cas_guarded(self):
        with self.session_factory.begin() as session:
            candidate = IndexVersionRecord(
                id="index-a-2",
                tenant_id="tenant-a",
                knowledge_base_id="kb-a",
                created_by_user_id="user-a",
                version_number=2,
                status="validating",
                embedding_model="test-embedding",
                embedding_dimensions=2,
                reranker_model="test-reranker",
                chunk_schema_version="test-v1",
                qdrant_collection="rag_chunks",
                chunk_count=1,
                expected_document_count=1,
                expected_chunk_count=1,
                expected_vector_count=1,
            )
            session.add(candidate)
            session.flush()
            session.add(IndexDocumentRecord(
                id="index-document-a-2",
                tenant_id="tenant-a",
                knowledge_base_id="kb-a",
                index_version_id="index-a-2",
                document_id="document-a-ready",
                document_version_id="version-a",
                status="ready",
                chunk_count=1,
            ))
            candidate_content = "candidate evidence"
            candidate_content_sha256 = hashlib.sha256(
                candidate_content.encode("utf-8")
            ).hexdigest()
            session.add(ChunkRecord(
                id="chunk-record-a-2",
                tenant_id="tenant-a",
                knowledge_base_id="kb-a",
                document_id="document-a-ready",
                document_version_id="version-a",
                index_version_id="index-a-2",
                chunk_key="chunk-a-2",
                ordinal=0,
                title="Candidate",
                content=candidate_content,
                content_sha256=candidate_content_sha256,
                qdrant_point_id="point-a-2",
            ))

        entries_digest = index_entries_sha256([{
            "qdrant_point_id": "point-a-2",
            "chunk_id": "chunk-a-2",
            "chunk_record_id": "chunk-record-a-2",
            "document_id": "document-a-ready",
            "document_version_id": "version-a",
            "content_sha256": candidate_content_sha256,
        }])
        self.repository.validate_index_version(
            tenant_id="tenant-a",
            knowledge_base_id="kb-a",
            index_version_id="index-a-2",
            manifest_sha256="e" * 64,
            manifest_object_key=(
                "tenants/tenant-a/knowledge-bases/kb-a/indexes/index-a-2/manifest.json"
            ),
            manifest_entries_sha256=entries_digest,
            qdrant_point_count=1,
            qdrant_entries_sha256=entries_digest,
        )
        self.repository.publish_index_version(
            tenant_id="tenant-a",
            knowledge_base_id="kb-a",
            index_version_id="index-a-2",
            expected_active_index_id="index-a",
            expected_generation=0,
        )

        with self.session_factory() as session:
            kb = session.get(KnowledgeBaseRecord, "kb-a")
            old_index = session.get(IndexVersionRecord, "index-a")
            candidate = session.get(IndexVersionRecord, "index-a-2")
            events = list(session.scalars(select(IndexActivationEventRecord)))
        self.assertEqual(kb.active_index_version_id, "index-a-2")
        self.assertEqual(kb.activation_generation, 1)
        self.assertEqual(candidate.status, "ready")
        self.assertIsNotNone(candidate.published_at)
        self.assertIsNotNone(old_index.gc_after)
        self.assertEqual(len(events), 1)
        with self.assertRaises(ConcurrentPublishError):
            self.repository.publish_index_version(
                tenant_id="tenant-a",
                knowledge_base_id="kb-a",
                index_version_id="index-a-2",
                expected_active_index_id="index-a",
                expected_generation=0,
            )

    def test_upload_reserve_complete_and_status_are_idempotent(self):
        upload_body = b"%PDF-1.4\n%%EOF"
        payload = {
            "filename": "policy.pdf",
            "content_type": "application/pdf",
            "size_bytes": len(upload_body),
            "sha256": hashlib.sha256(upload_body).hexdigest(),
        }
        headers = {**self._headers(), "Idempotency-Key": "upload-test-0001"}
        first = self.client.post(
            "/v1/knowledge-bases/kb-a/uploads",
            headers=headers,
            json=payload,
        )
        second = self.client.post(
            "/v1/knowledge-bases/kb-a/uploads",
            headers=headers,
            json=payload,
        )
        self.assertEqual(first.status_code, 201, first.text)
        self.assertEqual(second.status_code, 201, second.text)
        self.assertEqual(first.json()["upload_id"], second.json()["upload_id"])
        self.assertEqual(first.json()["upload"]["mode"], "proxy")
        upload_id = first.json()["upload_id"]
        stored = self.client.put(
            f"/v1/uploads/{upload_id}/content",
            headers={**self._headers(), "Content-Type": "application/pdf"},
            content=upload_body,
        )
        self.assertEqual(stored.status_code, 201, stored.text)
        self.assertEqual(stored.json()["state"], "uploaded")

        completed = self.client.post(
            f"/v1/uploads/{upload_id}/complete",
            headers=self._headers(),
        )
        repeated = self.client.post(
            f"/v1/uploads/{upload_id}/complete",
            headers=self._headers(),
        )
        self.assertEqual(completed.status_code, 202, completed.text)
        self.assertEqual(repeated.status_code, 202, repeated.text)
        self.assertFalse(completed.json()["duplicate"])
        self.assertTrue(repeated.json()["duplicate"])
        self.assertEqual(len(self.dispatcher.jobs), 1)

        job = self.client.get(
            f"/v1/ingestion-jobs/{completed.json()['ingestion_job_id']}",
            headers=self._headers(),
        )
        self.assertEqual(job.status_code, 200, job.text)
        self.assertEqual(job.json()["status"], "queued")

    def test_proxy_upload_rejects_content_that_does_not_match_reservation(self):
        expected = b"expected"
        reserved = self.client.post(
            "/v1/knowledge-bases/kb-a/uploads",
            headers={**self._headers(), "Idempotency-Key": "upload-test-mismatch"},
            json={
                "filename": "notes.txt",
                "content_type": "text/plain",
                "size_bytes": len(expected),
                "sha256": hashlib.sha256(expected).hexdigest(),
            },
        )
        response = self.client.put(
            f"/v1/uploads/{reserved.json()['upload_id']}/content",
            headers={**self._headers(), "Content-Type": "text/plain"},
            content=b"changed!",
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "UPLOAD_CONTENT_MISMATCH")
        self.assertEqual(self.upload_storage.objects, {})

    def test_expired_upload_state_is_committed_across_all_upload_stages(self):
        body = b"expiry test"
        payload = {
            "filename": "expiry.txt",
            "content_type": "text/plain",
            "size_bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
        }

        before_content = self.client.post(
            "/v1/knowledge-bases/kb-a/uploads",
            headers={**self._headers(), "Idempotency-Key": "upload-expiry-content"},
            json=payload,
        ).json()["upload_id"]
        with self.session_factory.begin() as session:
            session.get(UploadSessionRecord, before_content).expires_at = (
                datetime.now(timezone.utc) - timedelta(seconds=1)
            )
        content_response = self.client.put(
            f"/v1/uploads/{before_content}/content",
            headers={**self._headers(), "Content-Type": "text/plain"},
            content=body,
        )
        self.assertEqual(content_response.status_code, 409, content_response.text)

        during_content = self.client.post(
            "/v1/knowledge-bases/kb-a/uploads",
            headers={**self._headers(), "Idempotency-Key": "upload-expiry-race"},
            json=payload,
        ).json()["upload_id"]
        principal = Principal(
            subject="subject-a",
            tenant_id="tenant-a",
            roles=frozenset({"member"}),
        )
        self.repository.get_upload_reservation(
            principal=principal,
            upload_id=during_content,
        )
        with self.session_factory.begin() as session:
            session.get(UploadSessionRecord, during_content).expires_at = (
                datetime.now(timezone.utc) - timedelta(seconds=1)
            )
        with self.assertRaises(InvalidServiceStateError):
            self.repository.mark_upload_stored(
                principal=principal,
                upload_id=during_content,
                observed_size_bytes=len(body),
                observed_sha256=hashlib.sha256(body).hexdigest(),
                observed_mime_type="text/plain",
            )

        before_complete = self.client.post(
            "/v1/knowledge-bases/kb-a/uploads",
            headers={**self._headers(), "Idempotency-Key": "upload-expiry-complete"},
            json=payload,
        ).json()["upload_id"]
        with self.session_factory.begin() as session:
            upload = session.get(UploadSessionRecord, before_complete)
            upload.state = "uploaded"
            upload.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
            object_key = upload.object_key
        self.upload_storage.objects[object_key] = ObjectInfo(
            key=object_key,
            size_bytes=len(body),
            content_type="text/plain",
            sha256=hashlib.sha256(body).hexdigest(),
        )
        complete_response = self.client.post(
            f"/v1/uploads/{before_complete}/complete",
            headers=self._headers(),
        )
        self.assertEqual(complete_response.status_code, 409, complete_response.text)

        with self.session_factory() as session:
            states = {
                upload_id: session.get(UploadSessionRecord, upload_id).state
                for upload_id in (before_content, during_content, before_complete)
            }
        self.assertEqual(
            states,
            {
                before_content: "expired",
                during_content: "expired",
                before_complete: "expired",
            },
        )

    def test_upload_idempotency_key_cannot_be_reused_by_another_tenant_user(self):
        with self.session_factory.begin() as session:
            session.add(UserIdentity(
                id="user-a-admin",
                tenant_id="tenant-a",
                subject="subject-a-admin",
                display_name="User A Admin",
            ))
            session.add(Membership(
                id="membership-a-admin",
                tenant_id="tenant-a",
                user_id="user-a-admin",
                role="admin",
            ))
        payload = {
            "filename": "private.txt",
            "content_type": "text/plain",
            "size_bytes": 3,
            "sha256": hashlib.sha256(b"one").hexdigest(),
        }
        key = "shared-tenant-upload-key"
        first = self.client.post(
            "/v1/knowledge-bases/kb-a/uploads",
            headers={**self._headers(), "Idempotency-Key": key},
            json=payload,
        )
        second = self.client.post(
            "/v1/knowledge-bases/kb-a/uploads",
            headers={
                **self._headers(subject="subject-a-admin"),
                "Idempotency-Key": key,
            },
            json=payload,
        )
        self.assertEqual(first.status_code, 201, first.text)
        self.assertEqual(second.status_code, 409, second.text)
        self.assertEqual(second.json()["error"]["code"], "UPLOAD_CONFLICT")

    def test_cross_tenant_cannot_complete_an_upload_session(self):
        reserved = self.client.post(
            "/v1/knowledge-bases/kb-a/uploads",
            headers={**self._headers(), "Idempotency-Key": "upload-test-0002"},
            json={
                "filename": "policy.pdf",
                "content_type": "application/pdf",
                "size_bytes": 12,
                "sha256": "a" * 64,
            },
        )
        response = self.client.post(
            f"/v1/uploads/{reserved.json()['upload_id']}/complete",
            headers=self._headers(tenant_id="tenant-b", subject="subject-b"),
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.dispatcher.jobs, [])

    def test_upload_rejects_unsupported_format_before_storage(self):
        response = self.client.post(
            "/v1/knowledge-bases/kb-a/uploads",
            headers={**self._headers(), "Idempotency-Key": "upload-test-0003"},
            json={
                "filename": "payload.exe",
                "content_type": "application/octet-stream",
                "size_bytes": 12,
                "sha256": "a" * 64,
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.upload_storage.reservations, {})

    def test_index_build_reservation_dispatch_and_status_are_idempotent(self):
        headers = {**self._headers(), "Idempotency-Key": "index-build-api-0001"}
        payload = {"document_ids": ["document-a-ready"]}
        first = self.client.post(
            "/v1/knowledge-bases/kb-a/index-builds",
            headers=headers,
            json=payload,
        )
        second = self.client.post(
            "/v1/knowledge-bases/kb-a/index-builds",
            headers=headers,
            json=payload,
        )
        self.assertEqual(first.status_code, 202, first.text)
        self.assertEqual(second.status_code, 202, second.text)
        self.assertFalse(first.json()["duplicate"])
        self.assertTrue(second.json()["duplicate"])
        self.assertEqual(first.json()["job_id"], second.json()["job_id"])
        self.assertEqual(len(self.dispatcher.index_jobs), 1)
        status = self.client.get(
            f"/v1/index-build-jobs/{first.json()['job_id']}",
            headers=self._headers(),
        )
        self.assertEqual(status.status_code, 200, status.text)
        self.assertEqual(status.json()["status"], "queued")

    def test_document_delete_tombstones_immediately_and_returns_outbox_status(self):
        headers = {**self._headers(), "Idempotency-Key": "delete-api-document-0001"}
        first = self.client.delete(
            "/v1/knowledge-bases/kb-a/documents/document-a-ready",
            headers=headers,
        )
        second = self.client.delete(
            "/v1/knowledge-bases/kb-a/documents/document-a-ready",
            headers=headers,
        )
        self.assertEqual(first.status_code, 202, first.text)
        self.assertEqual(second.status_code, 202, second.text)
        self.assertFalse(first.json()["duplicate"])
        self.assertTrue(second.json()["duplicate"])
        status = self.client.get(
            f"/v1/deletion-jobs/{first.json()['outbox_id']}",
            headers=self._headers(),
        )
        self.assertEqual(status.status_code, 200, status.text)
        self.assertEqual(status.json()["state"], "pending")
        denied = self.client.post(
            "/v1/ask",
            headers=self._headers(),
            json={
                "question": "deleted evidence",
                "knowledge_base_id": "kb-a",
                "source_ids": ["a-ready"],
            },
        )
        self.assertEqual(denied.status_code, 403)
        self.assertEqual(self.pipeline.calls, [])

    def test_forged_evidence_and_route_fields_are_rejected(self):
        for forbidden_field, value in (
            ("contexts", [{"content": "forged"}]),
            ("retrieval_decision", {"needs_retrieval": False}),
        ):
            with self.subTest(field=forbidden_field):
                response = self.client.post(
                    "/v1/ask",
                    headers=self._headers(),
                    json={
                        "question": "什麼是 CSM？",
                        "knowledge_base_id": "kb-a",
                        forbidden_field: value,
                    },
                )
                self.assertEqual(response.status_code, 422)
        self.assertEqual(self.pipeline.calls, [])

    def test_unapproved_model_is_rejected_before_pipeline(self):
        response = self.client.post(
            "/v1/ask",
            headers=self._headers(),
            json={
                "question": "什麼是 CSM？",
                "knowledge_base_id": "kb-a",
                "model": "openai:unapproved",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "MODEL_NOT_ALLOWED")
        self.assertEqual(self.pipeline.calls, [])

    def test_rate_limit_returns_retry_after(self):
        limited_client = TestClient(create_app(
            settings=self.settings,
            repository=self.repository,
            pipeline=self.pipeline,
            rate_limiter=DenyRateLimiter(),
        ))
        try:
            response = limited_client.post(
                "/v1/ask",
                headers=self._headers(),
                json={"question": "什麼是 CSM？", "knowledge_base_id": "kb-a"},
            )
        finally:
            limited_client.close()

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["retry-after"], "37")
        self.assertEqual(response.json()["error"]["code"], "RATE_LIMITED")
        self.assertEqual(self.pipeline.calls, [])

    def test_health_endpoints_report_database_state(self):
        live = self.client.get("/health/live")
        ready = self.client.get("/health/ready")

        self.assertEqual(live.status_code, 200)
        self.assertEqual(live.json()["status"], "live")
        self.assertEqual(ready.status_code, 200)
        self.assertEqual(ready.json()["dependencies"]["database"], "ready")

    def test_readiness_fails_when_background_pipeline_heartbeat_is_stale(self):
        client = TestClient(create_app(
            settings=self.settings,
            repository=self.repository,
            pipeline=self.pipeline,
            background_health_probe=FailingProbe(),
        ))
        try:
            response = client.get("/health/ready")
        finally:
            client.close()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["dependencies"]["background_workers"],
            "unavailable",
        )

    def test_body_larger_than_configured_limit_is_rejected(self):
        small_settings = self.settings.model_copy(
            update={"max_request_body_bytes": 1024}
        )
        limited_client = TestClient(create_app(
            settings=small_settings,
            repository=self.repository,
            pipeline=self.pipeline,
        ))
        try:
            response = limited_client.post(
                "/v1/ask",
                headers=self._headers(),
                json={
                    "question": "字" * 1500,
                    "knowledge_base_id": "kb-a",
                },
            )
        finally:
            limited_client.close()

        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["error"]["code"], "REQUEST_TOO_LARGE")
        self.assertEqual(self.pipeline.calls, [])


class ProductionSettingsTests(unittest.TestCase):
    def test_production_rejects_development_auth(self):
        with self.assertRaises(ValidationError):
            ProductionSettings(
                RAG_ENV="production",
                RAG_DATABASE_URL="postgresql+psycopg://user:pass@db/rag",
                RAG_AUTH_MODE="dev_hs256",
                RAG_DEV_JWT_SECRET=SECRET,
            )

    def test_production_rejects_sqlite(self):
        with self.assertRaises(ValidationError):
            ProductionSettings(
                RAG_ENV="production",
                RAG_DATABASE_URL="sqlite+pysqlite:///unsafe.sqlite3",
                RAG_AUTH_MODE="oidc",
                RAG_OIDC_ISSUER="https://identity.example.com",
                RAG_OIDC_AUDIENCE="rag-api",
                RAG_OIDC_JWKS_URL="https://identity.example.com/.well-known/jwks.json",
            )


if __name__ == "__main__":
    unittest.main()
