import hashlib
import tempfile
import unittest
from datetime import timedelta

from rag_demo.production.auth import Principal
from rag_demo.production.database import (
    Base,
    KnowledgeBaseRecord,
    Membership,
    Tenant,
    UploadSessionRecord,
    UserIdentity,
    create_database_engine,
    create_session_factory,
    utc_now,
)
from rag_demo.production.repository import SqlAlchemyTenantRepository
from rag_demo.production.upload_cleanup import UploadCleanupService


class FakeStorage:
    def __init__(self, fail=False):
        self.fail = fail
        self.deleted = []

    def delete(self, key):
        if self.fail:
            raise RuntimeError("storage unavailable")
        self.deleted.append(key)


class UploadCleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        url = f"sqlite+pysqlite:///{self.temp_dir.name}/cleanup.sqlite3"
        self.engine = create_database_engine(url)
        self.addCleanup(self.engine.dispose)
        Base.metadata.create_all(self.engine)
        self.session_factory = create_session_factory(self.engine)
        self.repository = SqlAlchemyTenantRepository(self.session_factory)
        self.principal = Principal("subject-a", "tenant-a", frozenset())
        with self.session_factory.begin() as session:
            session.add(Tenant(id="tenant-a", name="Tenant A"))
            session.add(UserIdentity(
                id="user-a",
                tenant_id="tenant-a",
                subject="subject-a",
                display_name="User A",
            ))
            session.add(Membership(
                id="member-a",
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
        authorized = self.repository.authorize_knowledge_base(
            self.principal,
            "kb-a",
            None,
        )
        self.repository.reserve_upload(
            principal=self.principal,
            authorized=authorized,
            upload_id="upload-a",
            idempotency_key="cleanup-upload-a",
            filename="notes.txt",
            declared_mime_type="text/plain",
            expected_size_bytes=3,
            expected_sha256=hashlib.sha256(b"one").hexdigest(),
            object_key=(
                "tenants/tenant-a/knowledge-bases/kb-a/uploads/upload-a/original.txt"
            ),
            expires_at=utc_now() - timedelta(hours=1),
        )

    def test_expired_object_is_deleted_then_session_is_aborted(self):
        storage = FakeStorage()
        result = UploadCleanupService(self.repository, storage).sweep(
            grace_seconds=0
        )
        self.assertEqual(result["cleaned"], ["upload-a"])
        self.assertEqual(len(storage.deleted), 1)
        with self.session_factory() as session:
            self.assertEqual(session.get(UploadSessionRecord, "upload-a").state, "aborted")

    def test_storage_failure_keeps_expired_state_for_retry(self):
        result = UploadCleanupService(
            self.repository,
            FakeStorage(fail=True),
        ).sweep(grace_seconds=0)
        self.assertEqual(result["cleaned"], [])
        self.assertEqual(result["failed"][0]["upload_id"], "upload-a")
        with self.session_factory() as session:
            self.assertEqual(session.get(UploadSessionRecord, "upload-a").state, "expired")


if __name__ == "__main__":
    unittest.main()
