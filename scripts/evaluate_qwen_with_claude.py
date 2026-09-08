#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_demo.config import RagConfig  # noqa: E402
from rag_demo.conversation_store import ConversationStore  # noqa: E402
from rag_demo.document_pipeline import DocumentStore  # noqa: E402
from rag_demo.hybrid_retrieval import get_hybrid_retriever  # noqa: E402
from rag_demo.rag_pipeline import build_grounded_answer_request, normalize_contexts  # noqa: E402
from rag_demo.reference_evaluation import evaluate_qwen_with_claude  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Send the exact grounded RAG prompt to Claude, normalize the Claude answer "
            "to 100, and score an existing Qwen conversation answer."
        )
    )
    parser.add_argument("--conversation-id", required=True)
    parser.add_argument("--folder", default="Gmail 精簡版")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--claude-model", default="sonnet")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    store = ConversationStore()
    conversation = store.get_conversation_with_messages(args.conversation_id)
    if conversation is None:
        raise SystemExit(f"Conversation not found: {args.conversation_id}")
    question, qwen_answer, history = latest_answer_pair(conversation["messages"])

    source_ids = source_ids_for_folder(args.folder)
    if not source_ids:
        raise SystemExit(f"No indexed documents found in folder: {args.folder}")
    settings = RagConfig.from_env().normalized()
    retrieval = get_hybrid_retriever(args.profile).retrieve(
        question=question,
        retrieval_query=question,
        source_ids=source_ids,
        top_k=settings.hybrid_top_k,
        candidate_k=settings.hybrid_candidate_k,
    )
    contexts = normalize_contexts(
        retrieval.get("contexts"),
        max_contexts=settings.hybrid_max_top_k,
    )
    if not contexts:
        raise SystemExit("Retrieval returned no contexts; refusing to create a reference score.")

    answer_request = build_grounded_answer_request(
        question=question,
        contexts=contexts,
        history=history,
        memories=[item["content"] for item in store.list_memories(limit=12)],
    )
    evaluation = evaluate_qwen_with_claude(
        question=question,
        final_prompt=answer_request["prompt"],
        system_prompt=answer_request["system"],
        qwen_answer=qwen_answer,
        contexts=contexts,
        claude_model=args.claude_model,
    )

    output_path = args.output or default_output_path(args.conversation_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": "qwen-claude-benchmark-artifact-v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "conversation_id": args.conversation_id,
        "profile": args.profile,
        "folder": args.folder,
        "source_ids": source_ids,
        "question": question,
        "final_request": answer_request,
        "contexts": contexts,
        "qwen_answer": qwen_answer,
        "evaluation": evaluation,
    }
    output_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(output_path)


def latest_answer_pair(messages):
    for assistant_index in range(len(messages) - 1, -1, -1):
        if messages[assistant_index].get("role") != "assistant":
            continue
        for user_index in range(assistant_index - 1, -1, -1):
            if messages[user_index].get("role") == "user":
                return (
                    str(messages[user_index].get("content") or "").strip(),
                    str(messages[assistant_index].get("content") or "").strip(),
                    list(messages[:user_index]),
                )
    raise SystemExit("Conversation does not contain a completed user/assistant answer pair.")


def source_ids_for_folder(folder_name: str):
    return [
        document["source_id"]
        for document in DocumentStore().list_documents()
        if str(document.get("folder_name") or "") == str(folder_name)
    ]


def default_output_path(conversation_id: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        PROJECT_ROOT
        / "evals"
        / "qwen_claude"
        / "runs"
        / f"{stamp}-{conversation_id[:8]}.json"
    )


if __name__ == "__main__":
    main()
