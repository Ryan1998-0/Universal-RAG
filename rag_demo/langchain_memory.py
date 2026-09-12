"""LangChain message-history adapters backed by the project's SQLite store."""

from __future__ import annotations

from typing import Any

from rag_demo.conversation_store import ConversationStore

try:  # LangChain is optional for the local demo.
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.messages import AIMessage, HumanMessage
except ImportError:  # pragma: no cover - exercised without the optional extra
    BaseChatMessageHistory = object  # type: ignore[assignment,misc]
    AIMessage = HumanMessage = None  # type: ignore[assignment]


class SqliteChatMessageHistory(BaseChatMessageHistory):
    """Expose ConversationStore messages as LangChain chat messages."""

    def __init__(
        self,
        store: ConversationStore,
        session_id: str,
        max_messages: int = 20,
    ) -> None:
        if AIMessage is None or HumanMessage is None:
            raise RuntimeError(
                "LangChain core is required for SqliteChatMessageHistory."
            )
        self.store = store
        self.session_id = str(session_id)
        self.max_messages = max(1, min(int(max_messages), 200))

    @property
    def messages(self) -> list[Any]:
        result = []
        for message in self.store.get_recent_messages(
            self.session_id,
            limit=self.max_messages,
        ):
            content = str(message.get("content") or "")
            if message.get("role") == "user":
                result.append(HumanMessage(content=content))
            elif message.get("role") == "assistant":
                result.append(AIMessage(content=content))
        return result

    def add_message(self, message: Any) -> None:
        role = _role_for_message(message)
        if role is None:
            raise ValueError("Only human and AI messages can be persisted.")
        self.store.add_message(
            self.session_id,
            role,
            str(getattr(message, "content", "") or ""),
        )

    def add_messages(self, messages: list[Any]) -> None:
        for message in messages:
            self.add_message(message)

    def clear(self) -> None:
        self.store.clear_messages(self.session_id)


def _role_for_message(message: Any) -> str | None:
    if HumanMessage is not None and isinstance(message, HumanMessage):
        return "user"
    if AIMessage is not None and isinstance(message, AIMessage):
        return "assistant"
    role = str(getattr(message, "type", "") or "").strip().lower()
    return {"human": "user", "ai": "assistant", "assistant": "assistant"}.get(role)


def build_sqlite_history_factory(
    store: ConversationStore | None = None,
    *,
    profile: str = "default",
    max_messages: int = 20,
):
    """Create a factory for LangChain's RunnableWithMessageHistory."""

    if AIMessage is None or HumanMessage is None:
        raise RuntimeError("LangChain core is required for message history.")
    conversation_store = store or ConversationStore()

    def get_history(session_id: str) -> SqliteChatMessageHistory:
        stable_session_id = str(session_id or "").strip()
        if not stable_session_id:
            raise ValueError("LangChain session_id is required for persistent memory.")
        conversation = conversation_store.get_conversation(stable_session_id)
        if conversation is None:
            conversation = conversation_store.create_conversation(
                profile=profile,
                conversation_id=stable_session_id,
            )
        return SqliteChatMessageHistory(
            conversation_store,
            conversation["id"],
            max_messages=max_messages,
        )

    return get_history


def wrap_with_sqlite_message_history(
    runnable: Any,
    store: ConversationStore | None = None,
    *,
    profile: str = "default",
    max_messages: int = 20,
) -> Any:
    """Wrap a LangChain Runnable with persistent SQLite message history."""

    try:
        from langchain_core.runnables.history import RunnableWithMessageHistory
    except ImportError as exc:
        raise RuntimeError(
            "LangChain core is required for RunnableWithMessageHistory."
        ) from exc

    return RunnableWithMessageHistory(
        runnable,
        build_sqlite_history_factory(
            store,
            profile=profile,
            max_messages=max_messages,
        ),
        input_messages_key="messages",
        history_messages_key="history",
    )
