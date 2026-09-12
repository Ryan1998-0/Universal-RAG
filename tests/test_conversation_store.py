import tempfile
import unittest
from pathlib import Path

from rag_demo.conversation_store import ConversationStore


class ConversationStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = ConversationStore(Path(self.temporary_directory.name) / "memory.sqlite3")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_persists_conversation_messages_and_title(self):
        conversation = self.store.create_conversation(profile="ifrs17")
        self.store.add_message(conversation["id"], "user", "今天星期幾？")
        self.store.add_message(conversation["id"], "assistant", "今天是星期一。")

        loaded = self.store.get_conversation_with_messages(conversation["id"])

        self.assertEqual(loaded["title"], "今天星期幾？")
        self.assertEqual([item["role"] for item in loaded["messages"]], ["user", "assistant"])
        self.assertEqual(self.store.list_conversations()[0]["message_count"], 2)

    def test_explicit_memory_survives_across_conversations(self):
        conversation = self.store.create_conversation()
        remembered = self.store.remember_explicit_statement(
            "請記住我的名字是 Ryan",
            conversation["id"],
        )

        self.assertEqual(remembered["content"], "我的名字是 Ryan")
        self.assertEqual(self.store.list_memories()[0]["content"], "我的名字是 Ryan")

    def test_delete_conversation_cascades_messages_but_preserves_memory(self):
        conversation = self.store.create_conversation()
        self.store.add_message(conversation["id"], "user", "請記住我的專案代碼是 9421")
        self.store.add_message(conversation["id"], "assistant", "我記住了。")
        self.store.remember_explicit_statement(
            "請記住我的專案代碼是 9421",
            conversation["id"],
        )

        deleted = self.store.delete_conversation(conversation["id"])

        self.assertTrue(deleted)
        self.assertIsNone(self.store.get_conversation(conversation["id"]))
        self.assertEqual(self.store.get_messages(conversation["id"]), [])
        self.assertEqual(self.store.list_memories()[0]["content"], "我的專案代碼是 9421")
        self.assertIsNone(self.store.list_memories()[0]["source_conversation_id"])
        self.assertFalse(self.store.delete_conversation(conversation["id"]))

    def test_clear_messages_keeps_conversation_and_memories(self):
        conversation = self.store.create_conversation()
        self.store.add_message(conversation["id"], "user", "第一則")
        self.store.remember_explicit_statement("請記住代碼 9421", conversation["id"])

        self.assertTrue(self.store.clear_messages(conversation["id"]))
        self.assertEqual(self.store.get_messages(conversation["id"]), [])
        self.assertIsNotNone(self.store.get_conversation(conversation["id"]))
        self.assertEqual(self.store.list_memories()[0]["content"], "代碼 9421")


if __name__ == "__main__":
    unittest.main()
