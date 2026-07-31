import tempfile
import unittest
from datetime import timedelta

from rag_demo.production.auth import Principal
from rag_demo.production.database import (
    Base,
    DeletionOutboxRecord,
    DocumentRecord,
    DocumentVersionRecord,
    IndexDocumentRecord,
    IndexVersionRecord,
    IngestionJobRecord,
    KnowledgeBaseRecord,
    Membership,
    Tenant,
    UserIdentity,
    create_database_engine,
    create_session_factory,
    utc_now,
)
from rag_demo.production.deletion import DeletionService, SqlAlchemyDeletionRepository
from rag_demo.production.ingestion import SqlAlchemyIngestionRepository
from rag_demo.production.repository import (
    InvalidServiceStateError,
    ResourceNotFoundError,
    SqlAlchemyTenantRepository,
)


class FakeDeletionStorage:
    def __init__(self, fail=False):
        self.fail = fail
        self.deleted = []

    def delete(self, key):
        if self.fail:
            raise RuntimeError("S3 unavailable")
        self.deleted.append(key)


class FakeDeletionVectors:
    def __init__(self):
        self.deleted = []

    def delete_document_version(self, *, scope, document_version_id):
        self.deleted.append((scope, document_version_id))


class ProductionDeletionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        database_url = f"sqlite+pysqlite:///{self.temp_dir.name}/deletion.sqlite3"
        self.engine = create_database_engine(database_url)
        self.addCleanup(self.engine.dispose)
        Base.metadata.create_all(self.engine)
        self.session_factory = create_session_factory(self.engine)
        self.tenant_repository = SqlAlchemyTenantRepository(self.session_factory)
        self.repository = SqlAlchemyDeletionRepository(self.session_factory)
        self.principal = Principal("subject-a", "tenant-a", frozenset())
        self._seed()
        self.authorized = self.tenant_repository.authorize_knowledge_base(
            self.principal,
            "kb-a",
            None,
        )

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
            kb = KnowledgeBaseRecord(
                id="kb-a",
                tenant_id="tenant-a",
                owner_user_id="user-a",
                name="KB A",
            )
            session.add(kb)
            session.flush()
            index = IndexVersionRecord(
                id="index-a",
                tenant_id="tenant-a",
                knowledge_base_id="kb-a",
                created_by_user_id="user-a",
                version_number=1,
                status="ready",
                embedding_model="test",
                embedding_dimensions=2,
                reranker_model="test",
                chunk_schema_version="test",
                qdrant_collection="rag_chunks",
                chunk_count=1,
                manifest_sha256="a" * 64,
                manifest_object_key=(
                    "tenants/tenant-a/knowledge-bases/kb-a/indexes/index-a/manifest.json"
                ),
                expected_document_count=1,
                expected_chunk_count=1,
                expected_vector_count=1,
                validated_pg_chunk_count=1,
                validated_qdrant_point_count=1,
                validated_at=utc_now(),
            )
            document = DocumentRecord(
                id="document-a",
                tenant_id="tenant-a",
                knowledge_base_id="kb-a",
                owner_user_id="user-a",
                source_id="source-a",
                name="policy.txt",
                status="ready",
            )
            session.add_all([index, document])
            session.flush()
            version = DocumentVersionRecord(
                id="version-a",
                tenant_id="tenant-a",
                document_id="document-a",
                version_number=1,
                sha256="b" * 64,
                object_key="tenants/tenant-a/uploads/original.txt",
                extracted_object_key=(
                    "tenants/tenant-a/knowledge-bases/kb-a/documents/document-a"
                    "/versions/version-a/extracted/canonical.json"
                ),
                mime_type="text/plain",
                size_bytes=100,
                parser_version="test",
                chunk_schema_version="test",
                embedding_model="test",
                status="ready",
            )
            session.add(version)
            session.flush()
            document.current_version_id = version.id
            kb.active_index_version_id = index.id
            session.add(IndexDocumentRecord(
                id="membership-index-a",
                tenant_id="tenant-a",
                knowledge_base_id="kb-a",
                index_version_id="index-a",
                document_id="document-a",
                document_version_id="version-a",
                status="ready",
                chunk_count=1,
            ))

    def _reserve(self):
        return self.repository.tombstone_document(
            principal=self.principal,
            authorized=self.authorized,
            document_id="document-a",
            idempotency_key="delete-document-0001",
        )

    def test_tombstone_is_immediate_and_reservation_is_idempotent(self):
        first = self._reserve()
        second = self._reserve()
        self.assertFalse(first.duplicate)
        self.assertTrue(second.duplicate)
        self.assertEqual(first.outbox_id, second.outbox_id)
        with self.session_factory() as session:
            document = session.get(DocumentRecord, "document-a")
            outbox = session.get(DeletionOutboxRecord, first.outbox_id)
        self.assertEqual(document.status, "deleted")
        self.assertIsNotNone(document.deleted_at)
        self.assertIn("tenants/tenant-a/uploads/original.txt", outbox.object_keys_json)
        self.assertEqual(len(outbox.vector_scopes_json), 1)
        authorized = self.tenant_repository.authorize_knowledge_base(
            self.principal,
            "kb-a",
            None,
        )
        self.assertEqual(authorized.source_ids, [])

    def test_outbox_cleanup_is_idempotent_across_object_and_vector_stores(self):
        reservation = self._reserve()
        with self.session_factory.begin() as session:
            session.get(DeletionOutboxRecord, reservation.outbox_id).next_attempt_at = (
                utc_now() - timedelta(seconds=1)
            )
        storage = FakeDeletionStorage()
        vectors = FakeDeletionVectors()
        service = DeletionService(
            repository=self.repository,
            object_storage=storage,
            vector_repository=vectors,
        )
        first = service.process(outbox_id=reservation.outbox_id, worker_id="worker-a")
        second = service.process(outbox_id=reservation.outbox_id, worker_id="worker-b")
        self.assertEqual(first["status"], "completed")
        self.assertEqual(second["status"], "not_claimed")
        self.assertGreaterEqual(len(storage.deleted), 2)
        self.assertEqual(vectors.deleted[0][1], "version-a")

    def test_cleanup_dependency_failure_is_retried_without_restoring_document(self):
        reservation = self._reserve()
        with self.session_factory.begin() as session:
            session.get(DeletionOutboxRecord, reservation.outbox_id).next_attempt_at = (
                utc_now() - timedelta(seconds=1)
            )
        result = DeletionService(
            repository=self.repository,
            object_storage=FakeDeletionStorage(fail=True),
            vector_repository=FakeDeletionVectors(),
        ).process(outbox_id=reservation.outbox_id, worker_id="worker-a")
        self.assertEqual(result["status"], "retry_wait")
        with self.session_factory() as session:
            document = session.get(DocumentRecord, "document-a")
            outbox = session.get(DeletionOutboxRecord, reservation.outbox_id)
        self.assertEqual(document.status, "deleted")
        self.assertEqual(outbox.state, "retry_wait")
        self.assertIsNotNone(outbox.next_attempt_at)

    def test_tombstone_revokes_a_running_ingestion_lease_before_publish(self):
        with self.session_factory.begin() as session:
            session.add(IngestionJobRecord(
                id="ingestion-a",
                tenant_id="tenant-a",
                document_version_id="version-a",
                status="queued",
                stage="queued",
                idempotency_key="ingestion-delete-race",
                pipeline_fingerprint="test",
            ))
        ingestion = SqlAlchemyIngestionRepository(self.session_factory)
        self.assertIsNotNone(
            ingestion.claim(job_id="ingestion-a", worker_id="worker-a")
        )
        self._reserve()
        with self.assertRaises(RuntimeError):
            ingestion.complete(
                job_id="ingestion-a",
                worker_id="worker-a",
                extracted_object_key="tenants/tenant-a/kb-a/canonical.json",
                page_count=1,
                metadata={},
            )
        with self.session_factory() as session:
            job = session.get(IngestionJobRecord, "ingestion-a")
            document = session.get(DocumentRecord, "document-a")
        self.assertEqual(job.status, "cancelled")
        self.assertEqual(document.status, "deleted")

    def test_deletion_idempotency_and_status_are_creator_scoped(self):
        reservation = self._reserve()
        with self.session_factory.begin() as session:
            session.add(UserIdentity(
                id="user-b",
                tenant_id="tenant-a",
                subject="subject-b",
                display_name="User B",
            ))
            session.add(Membership(
                id="membership-b",
                tenant_id="tenant-a",
                user_id="user-b",
                role="admin",
            ))
        other = Principal("subject-b", "tenant-a", frozenset())
        authorized = self.tenant_repository.authorize_knowledge_base(
            other,
            "kb-a",
            None,
        )
        with self.assertRaises(InvalidServiceStateError):
            self.repository.tombstone_document(
                principal=other,
                authorized=authorized,
                document_id="document-a",
                idempotency_key="delete-document-0001",
            )
        with self.assertRaises(ResourceNotFoundError):
            self.repository.get_status(
                principal=other,
                outbox_id=reservation.outbox_id,
            )


if __name__ == "__main__":
    unittest.main()
