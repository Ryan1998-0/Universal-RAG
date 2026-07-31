import tempfile
import unittest

from sqlalchemy import select

from rag_demo.production.database import (
    Base,
    KnowledgeBaseRecord,
    Membership,
    Tenant,
    UserIdentity,
    create_database_engine,
    create_session_factory,
)
from rag_demo.production.provision import ProvisionRequest, provision_identity


class ProvisionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.engine = create_database_engine(
            f"sqlite+pysqlite:///{self.temp_dir.name}/provision.sqlite3"
        )
        Base.metadata.create_all(self.engine)
        self.session_factory = create_session_factory(self.engine)

    def tearDown(self):
        self.engine.dispose()
        self.temp_dir.cleanup()

    def test_provision_is_idempotent_and_updates_membership(self):
        request = ProvisionRequest(
            tenant_id="tenant-a",
            tenant_name="Tenant A",
            subject="oidc-subject-a",
            display_name="Ryan",
            role="owner",
            knowledge_base_name="主要知識庫",
        )
        first = provision_identity(self.session_factory, request)
        second = provision_identity(
            self.session_factory,
            ProvisionRequest(**{**request.__dict__, "role": "admin"}),
        )
        self.assertTrue(first["tenant"]["created"])
        self.assertTrue(first["knowledge_base"]["created"])
        self.assertFalse(second["tenant"]["created"])
        self.assertFalse(second["knowledge_base"]["created"])
        self.assertEqual(second["membership"]["role"], "admin")
        with self.session_factory() as session:
            self.assertEqual(len(list(session.scalars(select(Tenant)))), 1)
            self.assertEqual(len(list(session.scalars(select(UserIdentity)))), 1)
            self.assertEqual(len(list(session.scalars(select(Membership)))), 1)
            self.assertEqual(len(list(session.scalars(select(KnowledgeBaseRecord)))), 1)

    def test_provision_rejects_invalid_role(self):
        with self.assertRaises(ValueError):
            provision_identity(
                self.session_factory,
                ProvisionRequest(
                    tenant_id="tenant-a",
                    tenant_name="Tenant A",
                    subject="subject-a",
                    display_name="Ryan",
                    role="superuser",
                ),
            )


if __name__ == "__main__":
    unittest.main()
