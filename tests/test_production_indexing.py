import json
import hashlib
import tempfile
import unittest
from types import SimpleNamespace

from sqlalchemy import func, select

from rag_demo.production.auth import Principal
from rag_demo.production.config import DEFAULT_SPARSE_EMBEDDING_MODEL
from rag_demo.production.database import (
    Base,
    ChunkRecord,
    DocumentRecord,
    DocumentVersionRecord,
    IndexBuildJobRecord,
    IndexVersionRecord,
    KnowledgeBaseRecord,
    Membership,
    Tenant,
    UserIdentity,
    create_database_engine,
    create_session_factory,
)
from rag_demo.production.indexing import IndexingService, SqlAlchemyIndexingRepository
from rag_demo.production.index_manifest import (
    index_entries_sha256 as compute_index_entries_sha256,
)
from rag_demo.production.object_storage import StoredObject
from rag_demo.production.repository import SqlAlchemyTenantRepository


class FakeEmbeddingRuntime:
    def embed_documents(self, texts):
        return [[float(len(text)), 1.0] for text in texts]

    def sparse_documents(self, texts):
        return [{"indices": [1], "values": [1.0]} for _ in texts]


class FakeVectorRepository:
    def __init__(self, count_offset=0, corrupt_payload=False):
        self.points = {}
        self.count_offset = count_offset
        self.corrupt_payload = corrupt_payload

    def delete_index(self, scope):
        self.points.pop(scope.index_version_id, None)

    def upsert_chunks(self, *, scope, chunks, vectors, sparse_vectors):
        bucket = self.points.setdefault(scope.index_version_id, {})
        result = []
        for chunk in chunks:
            point_id = f"point-{chunk['id']}"
            bucket[point_id] = dict(chunk)
            result.append(point_id)
        return result

    def count_index(self, scope):
        return len(self.points.get(scope.index_version_id, {})) + self.count_offset

    def index_entries_sha256(self, scope):
        entries = []
        for point_id, chunk in self.points.get(scope.index_version_id, {}).items():
            digest = hashlib.sha256(str(chunk["content"]).encode()).hexdigest()
            if self.corrupt_payload:
                digest = "0" * 64
            entries.append({
                "qdrant_point_id": point_id,
                "chunk_id": chunk["id"],
                "chunk_record_id": chunk["chunk_record_id"],
                "document_id": chunk["document_id"],
                "document_version_id": chunk["document_version_id"],
                "content_sha256": digest,
            })
        return compute_index_entries_sha256(entries)


class FakeIndexStorage:
    def __init__(self, canonical):
        self.canonical = canonical
        self.manifests = {}

    def download_bytes(self, key, max_bytes):
        payload = json.dumps(self.canonical, ensure_ascii=False).encode()
        if len(payload) > max_bytes:
            raise RuntimeError("canonical artifact too large")
        return payload

    def put_index_manifest(
        self,
        *,
        tenant_id,
        knowledge_base_id,
        index_version_id,
        body,
    ):
        import hashlib

        key = (
            f"tenants/{tenant_id}/knowledge-bases/{knowledge_base_id}"
            f"/indexes/{index_version_id}/manifest.json"
        )
        payload = bytes(body)
        self.manifests[key] = payload
        return StoredObject(
            key=key,
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )


class ProductionIndexingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        database_url = f"sqlite+pysqlite:///{self.temp_dir.name}/indexing.sqlite3"
        self.engine = create_database_engine(database_url)
        self.addCleanup(self.engine.dispose)
        Base.metadata.create_all(self.engine)
        self.session_factory = create_session_factory(self.engine)
        self.tenant_repository = SqlAlchemyTenantRepository(self.session_factory)
        self.indexing_repository = SqlAlchemyIndexingRepository(self.session_factory)
        self.principal = Principal(
            subject="subject-a",
            tenant_id="tenant-a",
            roles=frozenset(),
        )
        self.settings = SimpleNamespace(
            embedding_model="test-dense",
            sparse_embedding_model=DEFAULT_SPARSE_EMBEDDING_MODEL,
            embedding_dimensions=2,
            reranker_model="test-reranker",
            chunk_schema_version="test-v1",
            qdrant_collection="rag_chunks",
        )
        self._seed()
        self.authorized = self.tenant_repository.authorize_knowledge_base(
            self.principal,
            "kb-a",
            None,
        )
        self.canonical = {
            "schema_version": "canonical-document-v1",
            "tenant_id": "tenant-a",
            "knowledge_base_id": "kb-a",
            "document_id": "document-a",
            "document_version_id": "version-a",
            "chunks": [
                {"id": "source-a::1", "content": "第一段可驗證內容", "page": "1"},
                {"id": "source-a::2", "content": "第二段可驗證內容", "page": "2"},
            ],
        }

    def _seed(self):
        with self.session_factory.begin() as session:
            session.add(Tenant(id="tenant-a", name="Tenant A"))
            session.add(UserIdentity(
                id="user-a",
                tenant_id="tenant-a",
                subject="subject-a",
                display_name="User A",
            ))
            session.add(Membership(
                id="membership-a",
                tenant_id="tenant-a",
                user_id="user-a",
                role="admin",
            ))
            session.add(KnowledgeBaseRecord(
                id="kb-a",
                tenant_id="tenant-a",
                owner_user_id="user-a",
                name="KB A",
            ))
            session.add(DocumentRecord(
                id="document-a",
                tenant_id="tenant-a",
                knowledge_base_id="kb-a",
                owner_user_id="user-a",
                source_id="source-a",
                name="policy.txt",
                status="parsed",
            ))
            session.add(DocumentVersionRecord(
                id="version-a",
                tenant_id="tenant-a",
                document_id="document-a",
                version_number=1,
                sha256="a" * 64,
                object_key="tenants/tenant-a/uploads/original.txt",
                extracted_object_key=(
                    "tenants/tenant-a/knowledge-bases/kb-a/documents/document-a"
                    "/versions/version-a/extracted/canonical.json"
                ),
                mime_type="text/plain",
                size_bytes=100,
                parser_version="canonical-v1",
                chunk_schema_version="test-v1",
                embedding_model="test-dense",
                status="parsed",
            ))

    def _reserve(self, key="index-build-0001"):
        return self.indexing_repository.reserve_build(
            principal=self.principal,
            authorized=self.authorized,
            idempotency_key=key,
            document_ids=["document-a"],
            settings=self.settings,
        )

    def _service(self, storage=None, vectors=None):
        return IndexingService(
            repository=self.indexing_repository,
            tenant_repository=self.tenant_repository,
            object_storage=storage or FakeIndexStorage(self.canonical),
            vector_repository=vectors or FakeVectorRepository(),
            embedding_runtime=FakeEmbeddingRuntime(),
            batch_size=1,
        )

    def test_reservation_is_idempotent_for_the_same_document_set(self):
        first = self._reserve()
        second = self._reserve()
        self.assertFalse(first.duplicate)
        self.assertTrue(second.duplicate)
        self.assertEqual(first.job_id, second.job_id)
        self.assertEqual(first.index_version_id, second.index_version_id)

    def test_build_persists_manifest_validates_counts_and_atomically_activates(self):
        reservation = self._reserve()
        storage = FakeIndexStorage(self.canonical)
        result = self._service(storage=storage).process(
            job_id=reservation.job_id,
            worker_id="worker-a",
        )
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["chunk_count"], 2)
        self.assertEqual(len(storage.manifests), 1)
        with self.session_factory() as session:
            job = session.get(IndexBuildJobRecord, reservation.job_id)
            index = session.get(IndexVersionRecord, reservation.index_version_id)
            kb = session.get(KnowledgeBaseRecord, "kb-a")
            document = session.get(DocumentRecord, "document-a")
            version = session.get(DocumentVersionRecord, "version-a")
            chunk_count = session.scalar(select(func.count(ChunkRecord.id)))
        self.assertEqual(job.status, "succeeded")
        self.assertEqual(index.status, "ready")
        self.assertEqual(index.expected_chunk_count, 2)
        self.assertTrue(index.manifest_object_key.endswith("/manifest.json"))
        self.assertEqual(kb.active_index_version_id, reservation.index_version_id)
        self.assertEqual(kb.activation_generation, 1)
        self.assertEqual(document.current_version_id, "version-a")
        self.assertEqual(document.status, "ready")
        self.assertEqual(version.status, "ready")
        self.assertEqual(chunk_count, 2)
        authorized = self.tenant_repository.authorize_knowledge_base(
            self.principal,
            "kb-a",
            None,
        )
        self.assertEqual(authorized.source_ids, ["source-a"])

    def test_qdrant_count_mismatch_fails_closed_without_activation(self):
        reservation = self._reserve("index-build-count-mismatch")
        result = self._service(vectors=FakeVectorRepository(count_offset=1)).process(
            job_id=reservation.job_id,
            worker_id="worker-a",
        )
        self.assertEqual(result["status"], "dead")
        with self.session_factory() as session:
            job = session.get(IndexBuildJobRecord, reservation.job_id)
            kb = session.get(KnowledgeBaseRecord, "kb-a")
        self.assertEqual(job.error_code, "INDEX_BUILD_REJECTED")
        self.assertIsNone(kb.active_index_version_id)

    def test_same_count_with_corrupted_qdrant_payload_fails_closed(self):
        reservation = self._reserve("index-build-payload-mismatch")
        result = self._service(
            vectors=FakeVectorRepository(corrupt_payload=True)
        ).process(
            job_id=reservation.job_id,
            worker_id="worker-a",
        )
        self.assertEqual(result["status"], "dead")
        with self.session_factory() as session:
            job = session.get(IndexBuildJobRecord, reservation.job_id)
            kb = session.get(KnowledgeBaseRecord, "kb-a")
        self.assertEqual(job.error_code, "INDEX_BUILD_REJECTED")
        self.assertIsNone(kb.active_index_version_id)

    def test_canonical_scope_mismatch_is_rejected(self):
        reservation = self._reserve("index-build-scope-mismatch")
        bad = dict(self.canonical)
        bad["tenant_id"] = "tenant-b"
        result = self._service(storage=FakeIndexStorage(bad)).process(
            job_id=reservation.job_id,
            worker_id="worker-a",
        )
        self.assertEqual(result["status"], "dead")
        with self.session_factory() as session:
            self.assertIsNone(
                session.get(KnowledgeBaseRecord, "kb-a").active_index_version_id
            )


if __name__ == "__main__":
    unittest.main()
