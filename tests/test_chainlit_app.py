from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rag_demo import chainlit_app as ui
from rag_demo.access_control import AccessPolicyStore
from rag_demo.conversation_store import ConversationStore
from rag_demo.rag_pipeline import RagPipeline


@pytest.fixture
def session(monkeypatch, tmp_path):
    values = {"settings": {"principal": "staff-demo"}}
    monkeypatch.setattr(ui.cl, "user_session", SimpleNamespace(get=lambda key, default=None: values.get(key, default)))
    policy = AccessPolicyStore(tmp_path / "acl.json")
    policy.set_policy("private", ["finance"])
    monkeypatch.setattr(ui.backend, "_access_policy_store", lambda: policy)
    monkeypatch.setattr(ui, "profile_data", lambda profile: {"sources": [{"source_id": "public"}, {"source_id": "private"}]})
    return values


def config(**overrides):
    return {"model": ui.backend.DEFAULT_MODEL, "profile": "default", "top_k": 3, "source_ids": [], **overrides}


def test_acl_filters_sources_before_pipeline(session, monkeypatch):
    pipeline = Mock()
    monkeypatch.setattr(ui.backend, "_rag_pipeline", lambda: pipeline)
    ui.run_question("測試", config(source_ids=["private", "public"]), "")
    assert pipeline.run.call_args.args[0].source_ids == ["public"]


def test_denied_selection_stays_empty_not_all_documents(session, monkeypatch):
    pipeline = Mock()
    monkeypatch.setattr(ui.backend, "_rag_pipeline", lambda: pipeline)
    ui.run_question("測試", config(source_ids=["private"]), "")
    assert pipeline.run.call_args.args[0].source_ids == []


def test_invalid_model_rejected_before_pipeline(session, monkeypatch):
    pipeline = Mock()
    monkeypatch.setattr(ui.backend, "_rag_pipeline", lambda: pipeline)
    with pytest.raises(ValueError, match="not allowed"):
        ui.run_question("測試", config(model="arbitrary:model"), "")
    pipeline.run.assert_not_called()


def test_upload_registers_existing_document_pipeline(monkeypatch, tmp_path):
    path = tmp_path / "hello.txt"
    path.write_text("測試文件")
    document = SimpleNamespace(metadata={"source_id": "test"}, chunks=[{"content": "測試文件"}], embeddings=[])
    store = Mock()
    store.import_document.return_value = document
    register = Mock()
    monkeypatch.setattr(ui.backend, "_document_store", lambda: store)
    monkeypatch.setattr(ui, "register_uploaded_document", register)
    assert ui.import_file(SimpleNamespace(path=str(path), name="hello.txt", mime="text/plain"), "uncategorized") == document.metadata
    assert store.import_document.call_args.args[1] == path.read_bytes()
    register.assert_called_once_with(source=document.metadata, chunks=document.chunks, embeddings=document.embeddings)


def test_pipeline_keeps_sqlite_conversation(session, monkeypatch, tmp_path):
    store = ConversationStore(tmp_path / "chat.sqlite")
    pipeline = RagPipeline(conversation_store=store)
    monkeypatch.setattr(ui.backend, "_rag_pipeline", lambda: pipeline)
    result = ui.run_question("今天幾號？", config(), "")
    conversation = store.get_conversation_with_messages(result["conversation_id"])
    assert [m["role"] for m in conversation["messages"]] == ["user", "assistant"]
    assert conversation["messages"][-1]["content"] == result["answer"]
    again = ui.run_question("現在幾點？", config(), result["conversation_id"])
    assert again["conversation_id"] == result["conversation_id"]
    assert len(store.get_messages(result["conversation_id"])) == 4
