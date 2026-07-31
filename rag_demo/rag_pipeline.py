import hashlib
import os
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Dict, List, Optional, Sequence
from uuid import uuid4

from rag_demo.config import RagConfig
from rag_demo.conversation_store import ConversationStore
from rag_demo.general_answer import (
    CURRENT_DATETIME_REASON,
    answer_current_datetime_question,
    build_no_retrieval_answer_prompt,
    build_qwen_direct_system_prompt,
    build_qwen_rag_system_prompt,
)
from rag_demo.hybrid_retrieval import evaluate_retrieval_evidence, get_hybrid_retriever
from rag_demo.model_providers import ask_model, parse_model_spec
from rag_demo.query_rewriter import QueryRewriteDecision, decide_and_rewrite_query_for_retrieval
from rag_demo.retrieval_scope import RetrievalScope
from rag_demo.threshold_evidence import resolve_threshold_evidence, render_threshold_answer


AGENT_RESPONSE_SCHEMA = "rag-agent-response-v1"
MAX_QUESTION_CHARACTERS = 20_000
MAX_SOURCE_IDS = 500
_PROFILE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class RagPipelineInputError(ValueError):
    """Raised when a caller violates the canonical server-side API contract."""


@dataclass(frozen=True)
class RagPipelineRequest:
    question: str
    model: str
    profile: str = "default"
    source_ids: Optional[Sequence[str]] = None
    top_k: Optional[int] = None
    conversation_id: str = ""
    persist_conversation: bool = True
    retrieval_scope: Optional[RetrievalScope] = None
    history: Optional[Sequence[dict]] = None

    @classmethod
    def from_payload(
        cls,
        payload: object,
        default_model: str,
        default_profile: str,
        allowed_models: Optional[Sequence[str]] = None,
        persist_conversation: bool = True,
    ) -> "RagPipelineRequest":
        if not isinstance(payload, dict):
            raise RagPipelineInputError("request body must be a JSON object")

        question = str(payload.get("question") or "").strip()
        if not question:
            raise RagPipelineInputError("question is required")
        if len(question) > MAX_QUESTION_CHARACTERS:
            raise RagPipelineInputError(
                f"question exceeds {MAX_QUESTION_CHARACTERS} characters"
            )

        model = _model_string(payload.get("model"), default_model=default_model)
        model_allowlist = set(allowed_models or [default_model])
        if model not in model_allowlist:
            raise RagPipelineInputError("requested model is not allowed")

        profile = str(payload.get("profile") or default_profile).strip()
        if not _PROFILE_PATTERN.fullmatch(profile):
            raise RagPipelineInputError("profile is invalid")

        source_ids = _source_ids_from_payload(payload)
        raw_top_k = payload.get("top_k")
        top_k = None if raw_top_k in (None, "") else _positive_int(raw_top_k, "top_k")
        conversation_id = str(payload.get("conversation_id") or "").strip()
        if len(conversation_id) > 128:
            raise RagPipelineInputError("conversation_id is too long")

        # Contexts, scores and retrieval decisions are intentionally not part of
        # this contract. The server must derive all evidence itself.
        return cls(
            question=question,
            model=model,
            profile=profile,
            source_ids=source_ids,
            top_k=top_k,
            conversation_id=conversation_id,
            persist_conversation=persist_conversation,
        )


class RagPipeline:
    def __init__(
        self,
        conversation_store=None,
        settings_factory: Callable[[], RagConfig] = RagConfig.from_env,
        retriever_factory: Callable[[str], object] = get_hybrid_retriever,
        route_fn: Callable[..., object] = decide_and_rewrite_query_for_retrieval,
        evidence_fn: Callable[..., dict] = evaluate_retrieval_evidence,
        ask_model_fn: Callable[..., str] = ask_model,
        datetime_answer_fn: Callable[[str], Optional[str]] = answer_current_datetime_question,
        run_id_fn: Optional[Callable[[], str]] = None,
    ):
        self._conversation_store = conversation_store
        self._settings_factory = settings_factory
        self._retriever_factory = retriever_factory
        self._route_fn = route_fn
        self._evidence_fn = evidence_fn
        self._ask_model_fn = ask_model_fn
        self._datetime_answer_fn = datetime_answer_fn
        self._run_id_fn = run_id_fn or (lambda: uuid4().hex)

    def run(self, request: RagPipelineRequest) -> dict:
        started_at = perf_counter()
        run_id = self._run_id_fn()
        stage_timings: Dict[str, float] = {}
        settings = self._settings_factory().normalized()
        top_k = max(
            1,
            min(
                int(request.top_k or settings.hybrid_top_k),
                settings.hybrid_max_top_k,
            ),
        )

        history: List[dict] = [
            dict(message)
            for message in (request.history or [])[-10:]
            if isinstance(message, dict)
            and str(message.get("role") or "") in {"user", "assistant"}
        ]
        memories: List[str] = []
        remembered = None
        conversation_id = ""
        store = None
        if request.persist_conversation:
            store = self._store()
            conversation = store.ensure_conversation(
                request.conversation_id,
                profile=request.profile,
            )
            conversation_id = conversation["id"]
            history = store.get_recent_messages(conversation_id, limit=10)
            remembered = store.remember_explicit_statement(
                request.question,
                conversation_id,
            )
            memories = [item["content"] for item in store.list_memories(limit=12)]
            store.add_message(conversation_id, "user", request.question)

        contexts: List[dict] = []
        evidence_evaluation = None
        retrieval_decision = None
        answer = ""

        direct_system_answer = self._datetime_answer_fn(request.question)
        if direct_system_answer is not None:
            retrieval_decision = {
                "needs_retrieval": False,
                "reason": CURRENT_DATETIME_REASON,
                "retrieval_query": request.question,
            }
            answer = direct_system_answer
            stage_timings["routeMs"] = 0.0
            stage_timings["retrieveMs"] = 0.0
            stage_timings["generateMs"] = 0.0
        else:
            route_started = perf_counter()
            retrieval_decision = self._route(
                request.question,
                request.model,
                history,
            )
            stage_timings["routeMs"] = _elapsed_ms(route_started)

            if retrieval_decision["needs_retrieval"]:
                retrieval_started = perf_counter()
                retrieval_result = self._retriever_factory(request.profile).retrieve(
                    question=request.question,
                    retrieval_query=retrieval_decision["retrieval_query"],
                    source_ids=request.source_ids,
                    top_k=top_k,
                    candidate_k=settings.hybrid_candidate_k,
                    retrieval_scope=request.retrieval_scope,
                )
                contexts = normalize_contexts(
                    retrieval_result.get("contexts"),
                    max_contexts=settings.hybrid_max_top_k,
                )
                stage_timings["retrieveMs"] = _elapsed_ms(retrieval_started)
            else:
                stage_timings["retrieveMs"] = 0.0

            if contexts:
                evidence_evaluation = self._evidence_fn(
                    contexts,
                    settings=settings,
                    question=request.question,
                )

            generation_started = perf_counter()
            if retrieval_decision["needs_retrieval"] and (
                not contexts
                or (evidence_evaluation and not evidence_evaluation["sufficient"])
            ):
                answer = (
                    "我已嘗試檢索，但目前可用資料與問題的相關性不足，"
                    "無法根據知識庫可靠回答。請改寫問題或選擇其他文件後再試。"
                )
            elif contexts:
                answer = answer_from_contexts(
                    question=request.question,
                    contexts=contexts,
                    model=request.model,
                    history=history,
                    memories=memories,
                    ask_model_fn=self._ask_model_fn,
                )
            else:
                answer = self._ask_model_fn(
                    prompt_with_history(
                        build_no_retrieval_answer_prompt(
                            request.question,
                            retrieval_decision["reason"],
                        ),
                        history,
                    ),
                    model=request.model,
                    system=build_qwen_direct_system_prompt(memories=memories),
                )
            stage_timings["generateMs"] = _elapsed_ms(generation_started)

        if store is not None:
            store.add_message(
                conversation_id,
                "assistant",
                answer,
                metadata={
                    "run_id": run_id,
                    "retrieval_needed": retrieval_decision["needs_retrieval"],
                    "remembered": bool(remembered),
                },
            )

        spec = parse_model_spec(request.model)
        evidence_is_sufficient = (
            not evidence_evaluation or evidence_evaluation["sufficient"]
        )
        stage_timings["totalMs"] = _elapsed_ms(started_at)
        grounding_warnings = (
            []
            if evidence_is_sufficient
            else [evidence_evaluation["reason"]]
        )
        citations = []
        if evidence_is_sufficient and contexts:
            citations, citation_warnings = citations_from_answer(
                answer,
                contexts,
                run_id=run_id,
            )
            grounding_warnings.extend(citation_warnings)

        return {
            "schema_version": AGENT_RESPONSE_SCHEMA,
            "run_id": run_id,
            "profile": request.profile,
            "conversation_id": conversation_id,
            "answer": answer,
            "confidence": (
                evidence_evaluation["confidence"]
                if evidence_evaluation
                else "medium"
            ),
            "citations": citations,
            "grounding_warnings": grounding_warnings,
            "retrieval": {
                "server_generated": True,
                "needed": retrieval_decision["needs_retrieval"],
                "reason": retrieval_decision["reason"],
                "query": retrieval_decision["retrieval_query"],
                "contexts": contexts,
                "evidence_evaluation": evidence_evaluation,
            },
            "model": {"provider": spec.provider, "name": spec.model},
            "timings": stage_timings,
        }

    def _store(self):
        if self._conversation_store is None:
            self._conversation_store = ConversationStore()
        return self._conversation_store

    def _route(self, question: str, model: str, history: Sequence[dict]) -> dict:
        try:
            decision = self._route_fn(
                question,
                model=model,
                conversation_context=conversation_context(history),
            )
        except Exception as exc:
            decision = QueryRewriteDecision(
                needs_retrieval=True,
                reason=(
                    "檢索判斷失敗，為避免漏掉文件證據，保守改走檢索："
                    f"{type(exc).__name__}"
                ),
                retrieval_query=question,
            )
        return {
            "needs_retrieval": bool(decision.needs_retrieval),
            "reason": str(decision.reason or ""),
            "retrieval_query": str(decision.retrieval_query or question).strip(),
        }


def allowed_models_from_env(default_model: str) -> List[str]:
    configured = os.getenv("RAG_ALLOWED_MODELS", "")
    models = [item.strip() for item in configured.split(",") if item.strip()]
    if default_model not in models:
        models.insert(0, default_model)
    return list(dict.fromkeys(models))


def normalize_contexts(raw_contexts: object, max_contexts: Optional[int] = None) -> List[dict]:
    if not isinstance(raw_contexts, list):
        return []
    limit = max_contexts or RagConfig.from_env().normalized().hybrid_max_top_k
    contexts = []
    for index, raw in enumerate(raw_contexts[:limit], start=1):
        if not isinstance(raw, dict):
            continue
        content = str(raw.get("content") or "").strip()
        if not content:
            continue
        contexts.append({
            "id": str(raw.get("id") or f"context-{index}"),
            "rank": _safe_rank(raw.get("rank"), index),
            "title": str(raw.get("title") or "Untitled"),
            "source": str(raw.get("source") or ""),
            "page": str(raw.get("page") or ""),
            "content": content[:2600],
            "score": raw.get("score") or 0,
            "bm25Score": raw.get("bm25Score") or 0,
            "embeddingScore": raw.get("embeddingScore") or 0,
            "rerankScore": raw.get("rerankScore") or raw.get("score") or 0,
            "matchedTerms": list(raw.get("matchedTerms") or [])[:20],
            "documentVersionId": str(raw.get("documentVersionId") or ""),
            "chunkRecordId": str(raw.get("chunkRecordId") or ""),
            "indexVersionId": str(raw.get("indexVersionId") or ""),
        })
    return contexts


def answer_from_contexts(
    question: str,
    contexts: List[dict],
    model: str,
    history=None,
    memories=None,
    ask_model_fn: Callable[..., str] = ask_model,
) -> str:
    threshold_evidence = resolve_threshold_evidence(
        question,
        contexts,
        previous_user_question=latest_user_question(history),
    )
    if threshold_evidence is not None:
        return render_threshold_answer(threshold_evidence)

    context_text = "\n\n".join(
        f"[{context['rank']}] {context['title']} {context['page']}\n{context['content']}"
        for context in contexts
    )
    prompt = f"""{rag_conversation_context(history)}

### 使用者問題
{question}

### 檢索資料
{context_text}

### 回答要求
請使用繁體中文回答。
只能根據檢索資料回答；如果資料不足，請明確說資料不足。
最近對話只用於解析代名詞，不是證據；不可重複或延續舊助理回答。
目前問題中的數字與邊界條件優先；逐一核對「以上、未滿、以下」後再回答，不可套用相鄰區間。
問題若要求公式，必須逐字列出檢索資料中的公式，不可只列計算範例。
回答要精簡，但要保留關鍵原因。
最後用「來源：」列出用到的 rank，例如 [1], [2]。
"""
    return ask_model_fn(
        prompt,
        model=model,
        system=build_qwen_rag_system_prompt(memories=memories),
    )


def prompt_with_history(prompt: str, history: Sequence[dict]) -> str:
    if not history:
        return prompt
    return f"""{conversation_context(history)}

{prompt}"""


def conversation_context(history: Sequence[dict]) -> str:
    if not history:
        return "### 最近對話\n無"
    lines = []
    for message in history[-10:]:
        role = "使用者" if message.get("role") == "user" else "助理"
        content = str(message.get("content") or "").strip()[:2000]
        if content:
            lines.append(f"{role}：{content}")
    return "### 最近對話\n" + ("\n".join(lines) if lines else "無")


def rag_conversation_context(history: Sequence[dict]) -> str:
    if not history:
        return "### 最近使用者問題（只供代名詞解析）\n無"
    lines = []
    for message in history[-10:]:
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "").strip()[:1000]
        if content:
            lines.append(f"使用者：{content}")
    return "### 最近使用者問題（只供代名詞解析）\n" + (
        "\n".join(lines[-5:]) if lines else "無"
    )


def latest_user_question(history: Sequence[dict]) -> str:
    for message in reversed(history or []):
        if message.get("role") == "user":
            return str(message.get("content") or "").strip()[:1000]
    return ""


def citations_from_answer(
    answer: str,
    contexts: Sequence[dict],
    run_id: str,
) -> tuple:
    context_by_rank = {
        int(context.get("rank") or index): context
        for index, context in enumerate(contexts, start=1)
    }
    referenced_ranks = list(
        dict.fromkeys(int(rank) for rank in re.findall(r"\[(\d+)\]", str(answer)))
    )
    valid_ranks = [rank for rank in referenced_ranks if rank in context_by_rank]
    invalid_ranks = [rank for rank in referenced_ranks if rank not in context_by_rank]
    warnings = []
    if invalid_ranks:
        warnings.append(
            "回答包含無效來源標記："
            + ", ".join(f"[{rank}]" for rank in invalid_ranks)
        )
    if not referenced_ranks:
        warnings.append("回答未提供可驗證的來源標記。")

    citations = []
    for rank in valid_ranks:
        context = context_by_rank[rank]
        content = str(context.get("content") or "")
        citations.append({
            "id": context["id"],
            "rank": context["rank"],
            "title": context["title"],
            "page": context["page"],
            "source": context.get("source", ""),
            "run_id": run_id,
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "document_version_id": str(context.get("documentVersionId") or ""),
            "chunk_record_id": str(context.get("chunkRecordId") or ""),
            "verified": True,
        })
    return citations, warnings


def _model_string(model_payload: object, default_model: str) -> str:
    if isinstance(model_payload, dict):
        provider = str(model_payload.get("provider") or "").strip()
        name = str(model_payload.get("name") or "").strip()
        if not provider or not name:
            return default_model
        return f"{provider}:{name}"
    return str(model_payload or default_model).strip() or default_model


def _source_ids_from_payload(payload: dict) -> Optional[List[str]]:
    if "source_ids" not in payload:
        return None
    raw_source_ids = payload.get("source_ids")
    if not isinstance(raw_source_ids, list):
        raise RagPipelineInputError("source_ids must be an array")
    source_ids = list(
        dict.fromkeys(
            str(source_id).strip()
            for source_id in raw_source_ids
            if str(source_id).strip()
        )
    )
    if len(source_ids) > MAX_SOURCE_IDS:
        raise RagPipelineInputError(f"source_ids exceeds {MAX_SOURCE_IDS} items")
    return source_ids


def _positive_int(value: object, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RagPipelineInputError(f"{field_name} must be an integer") from exc
    if parsed <= 0:
        raise RagPipelineInputError(f"{field_name} must be greater than zero")
    return parsed


def _safe_rank(value: object, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 2)
