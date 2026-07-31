import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_VERSION_TABLE = "alembic_version"
EXPECTED_PRODUCTION_TABLES = {
    "answer_runs",
    "audit_events",
    "chunks",
    "citations",
    "conversations",
    "deletion_outbox",
    "document_versions",
    "documents",
    "feedback",
    "folders",
    "index_activation_events",
    "index_build_jobs",
    "index_documents",
    "index_versions",
    "ingestion_jobs",
    "knowledge_bases",
    "memberships",
    "messages",
    "tenants",
    "upload_sessions",
    "users",
}


class ProductionMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.database_path = (
            Path(self.temp_directory.name) / "production-migrations.sqlite3"
        )
        self.database_url = f"sqlite+pysqlite:///{self.database_path}"
        self.alembic_config = Config(str(PROJECT_ROOT / "alembic.ini"))
        self.head_revision = ScriptDirectory.from_config(
            self.alembic_config
        ).get_current_head()

    def _table_names(self):
        with closing(sqlite3.connect(self.database_path)) as connection:
            rows = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        return {name for (name,) in rows if not name.startswith("sqlite_")}

    def _version_rows(self):
        with closing(sqlite3.connect(self.database_path)) as connection:
            return [
                version
                for (version,) in connection.execute(
                    "SELECT version_num FROM alembic_version ORDER BY version_num"
                ).fetchall()
            ]

    def _assert_schema_is_at_head(self):
        table_names = self._table_names()
        business_tables = table_names - {ALEMBIC_VERSION_TABLE}

        self.assertEqual(len(business_tables), 21)
        self.assertSetEqual(business_tables, EXPECTED_PRODUCTION_TABLES)
        self.assertSetEqual(
            table_names,
            EXPECTED_PRODUCTION_TABLES | {ALEMBIC_VERSION_TABLE},
        )
        self.assertEqual(self._version_rows(), [self.head_revision])

    def test_upgrade_check_downgrade_and_reupgrade_are_reproducible(self):
        with patch.dict(os.environ, {"RAG_DATABASE_URL": self.database_url}):
            command.upgrade(self.alembic_config, "head")
            self._assert_schema_is_at_head()

            command.check(self.alembic_config)

            command.downgrade(self.alembic_config, "base")
            self.assertSetEqual(self._table_names(), {ALEMBIC_VERSION_TABLE})
            self.assertEqual(self._version_rows(), [])

            command.upgrade(self.alembic_config, "head")
            self._assert_schema_is_at_head()


if __name__ == "__main__":
    unittest.main()
