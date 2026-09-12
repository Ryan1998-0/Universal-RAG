import tempfile
import unittest
from pathlib import Path

from rag_demo.conversation_store import ConversationStore


class LangChainMemoryTests(unittest.TestCase):
    def test_sqlite_history_adapter_round_trip(self):
        try:
            from langchain_core.messages import AIMessage, HumanMessage
            from langchain_core.runnables import RunnableLambda
            from rag_demo.langchain_memory import (
                SqliteChatMessageHistory,
                wrap_with_sqlite_message_history,
            )
        except ImportError:
            self.skipTest("LangChain is optional")

        with tempfile.TemporaryDirectory() as directory:
            store = ConversationStore(Path(directory) / "memory.sqlite3")
            conversation = store.create_conversation()
            history = SqliteChatMessageHistory(store, conversation["id"])
            history.add_messages([
                HumanMessage(content="你好"),
                AIMessage(content="你好，有什麼可以協助？"),
            ])
            self.assertEqual(
                [item.type for item in history.messages],
                ["human", "ai"],
            )
            self.assertTrue(store.clear_messages(conversation["id"]))

            chain = RunnableLambda(
                lambda value: AIMessage(
                    content=f"history_count={len(value['history'])}"
                )
            )
            chain_with_history = wrap_with_sqlite_message_history(chain, store)
            chain_with_history.invoke(
                {"messages": [HumanMessage(content="第一輪")]},
                config={"configurable": {"session_id": "langchain-session"}},
            )
            second = chain_with_history.invoke(
                {"messages": [HumanMessage(content="第二輪")]},
                config={"configurable": {"session_id": "langchain-session"}},
            )
            self.assertEqual(second.content, "history_count=2")
            self.assertEqual(
                len(store.get_messages("langchain-session")),
                4,
            )


if __name__ == "__main__":
    unittest.main()
