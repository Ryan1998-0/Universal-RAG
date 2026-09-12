import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / ".local" / "agent_memory.sqlite3"
DEFAULT_PROFILE = os.getenv("RAG_PROFILE", "default")


class ConversationStore:
    def __init__(self, db_path=None):
        configured_path = db_path or os.getenv("RAG_MEMORY_DB") or DEFAULT_DB_PATH
        self.db_path = Path(configured_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    @contextmanager
    def _connection(self):
        """Open, transact and close a SQLite connection for one operation."""

        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self):
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT '新對話',
                    profile TEXT NOT NULL DEFAULT 'default',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                    ON messages(conversation_id, id);

                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL UNIQUE,
                    category TEXT NOT NULL DEFAULT 'explicit',
                    source_conversation_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(source_conversation_id) REFERENCES conversations(id) ON DELETE SET NULL
                );
                """
            )

    def create_conversation(
        self,
        profile=DEFAULT_PROFILE,
        title="新對話",
        conversation_id=None,
    ):
        conversation_id = str(conversation_id or uuid4().hex)
        now = _utc_now()
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO conversations(id, title, profile, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (conversation_id, _clean_title(title), str(profile or DEFAULT_PROFILE), now, now),
            )
        return self.get_conversation(conversation_id)

    def ensure_conversation(self, conversation_id=None, profile=DEFAULT_PROFILE):
        if conversation_id:
            conversation = self.get_conversation(conversation_id)
            if conversation is not None:
                return conversation
        return self.create_conversation(profile=profile)

    def list_conversations(self, limit=30):
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT c.id, c.title, c.profile, c.created_at, c.updated_at,
                       COUNT(m.id) AS message_count
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.id
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                LIMIT ?
                """,
                (max(1, min(int(limit), 100)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_conversation(self, conversation_id):
        with self._connection() as connection:
            row = connection.execute(
                "SELECT id, title, profile, created_at, updated_at FROM conversations WHERE id = ?",
                (str(conversation_id),),
            ).fetchone()
        return dict(row) if row else None

    def get_conversation_with_messages(self, conversation_id):
        conversation = self.get_conversation(conversation_id)
        if conversation is None:
            return None
        conversation["messages"] = self.get_messages(conversation_id)
        return conversation

    def delete_conversation(self, conversation_id):
        clean_id = str(conversation_id or "").strip()
        if not clean_id:
            return False
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM conversations WHERE id = ?",
                (clean_id,),
            )
        return cursor.rowcount > 0

    def add_message(self, conversation_id, role, content, metadata=None):
        clean_content = str(content or "").strip()
        if role not in {"user", "assistant"}:
            raise ValueError("role must be user or assistant")
        if not clean_content:
            raise ValueError("message content is required")
        if self.get_conversation(conversation_id) is None:
            raise ValueError("conversation not found")

        now = _utc_now()
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO messages(conversation_id, role, content, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(conversation_id),
                    role,
                    clean_content[:20000],
                    json.dumps(metadata or {}, ensure_ascii=False),
                    now,
                ),
            )
            if role == "user":
                first_user_count = connection.execute(
                    "SELECT COUNT(*) FROM messages WHERE conversation_id = ? AND role = 'user'",
                    (str(conversation_id),),
                ).fetchone()[0]
                if first_user_count == 1:
                    connection.execute(
                        "UPDATE conversations SET title = ? WHERE id = ?",
                        (_clean_title(clean_content), str(conversation_id)),
                    )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, str(conversation_id)),
            )
            message_id = cursor.lastrowid
        return self.get_message(message_id)

    def clear_messages(self, conversation_id):
        """Delete messages while retaining the conversation and its memories."""

        clean_id = str(conversation_id or "").strip()
        if not clean_id:
            return False
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM messages WHERE conversation_id = ?",
                (clean_id,),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (_utc_now(), clean_id),
            )
        return cursor.rowcount > 0

    def get_message(self, message_id):
        with self._connection() as connection:
            row = connection.execute(
                "SELECT id, conversation_id, role, content, metadata_json, created_at FROM messages WHERE id = ?",
                (int(message_id),),
            ).fetchone()
        return _message_dict(row) if row else None

    def get_messages(self, conversation_id, limit=None):
        params = [str(conversation_id)]
        limit_sql = ""
        if limit is not None:
            limit_sql = " LIMIT ?"
            params.append(max(1, min(int(limit), 200)))
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT id, conversation_id, role, content, metadata_json, created_at
                FROM messages WHERE conversation_id = ? ORDER BY id ASC
                """ + limit_sql,
                tuple(params),
            ).fetchall()
        return [_message_dict(row) for row in rows]

    def get_recent_messages(self, conversation_id, limit=10):
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT id, conversation_id, role, content, metadata_json, created_at
                FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?
                """,
                (str(conversation_id), max(1, min(int(limit), 30))),
            ).fetchall()
        return [_message_dict(row) for row in reversed(rows)]

    def remember_explicit_statement(self, text, conversation_id=None):
        content = _explicit_memory_content(text)
        if not content:
            return None
        now = _utc_now()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO memories(content, category, source_conversation_id, created_at, updated_at)
                VALUES (?, 'explicit', ?, ?, ?)
                ON CONFLICT(content) DO UPDATE SET updated_at = excluded.updated_at
                """,
                (content, conversation_id, now, now),
            )
            row = connection.execute(
                "SELECT id, content, category, source_conversation_id, created_at, updated_at FROM memories WHERE content = ?",
                (content,),
            ).fetchone()
        return dict(row) if row else None

    def list_memories(self, limit=12):
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT id, content, category, source_conversation_id, created_at, updated_at
                FROM memories ORDER BY updated_at DESC LIMIT ?
                """,
                (max(1, min(int(limit), 50)),),
            ).fetchall()
        return [dict(row) for row in rows]


def _message_dict(row):
    item = dict(row)
    try:
        item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
    except json.JSONDecodeError:
        item["metadata"] = {}
        item.pop("metadata_json", None)
    return item


def _explicit_memory_content(text):
    clean_text = str(text or "").strip()
    for prefix in ("請幫我記住", "幫我記住", "請記住", "記住"):
        if clean_text.startswith(prefix):
            content = clean_text[len(prefix):].lstrip("：:，, ").strip()
            return content[:1000] if content else None
    return None


def _clean_title(text):
    clean_text = " ".join(str(text or "新對話").split())
    return clean_text[:36] or "新對話"


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
