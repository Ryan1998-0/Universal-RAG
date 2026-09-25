import hashlib
import os
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Dict, List, Optional, Sequence
from uuid import uuid4

from rag_demo.config import RagConfig
from rag_demo.conversation_store import ConversationStore
from rag_demo.evidence_focus import focus_retrieved_evidence
from rag_demo.evidence_validation import normalized_refusal_answer, validate_answer_evidence
from rag_demo.fine_evidence import retrieve_fine_evidence
from rag_demo.general_answer import (
    CURRENT_DATETIME_REASON,
    answer_current_datetime_question,
    build_no_retrieval_answer_prompt,
    build_qwen_direct_system_prompt,
    build_qwen_rag_system_prompt,
)
from rag_demo.hybrid_retrieval import evaluate_retrieval_evidence, get_hybrid_retriever
from rag_demo.model_gateway import ModelGateway, resolve_model_for_node
from rag_demo.model_providers import ask_model, parse_model_spec
from rag_demo.observability import TimingTrace, elapsed_ms
from rag_demo.query_rewriter import QueryRewriteDecision, decide_and_rewrite_query_for_retrieval
from rag_demo.reference_evaluation import (
    claude_reference_evaluation_enabled,
    evaluate_qwen_with_claude,
    unavailable_reference_evaluation,
)
from rag_demo.retrieval_planner import build_retrieval_plan
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
    request_id: str = ""

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
        quality_evaluator_fn: Callable[..., dict] = evaluate_qwen_with_claude,
        quality_evaluation_enabled_fn: Callable[[], bool] = claude_reference_evaluation_enabled,
        datetime_answer_fn: Callable[[str], Optional[str]] = answer_current_datetime_question,
        run_id_fn: Optional[Callable[[], str]] = None,
        model_gateway: Optional[ModelGateway] = None,
    ):
        self._conversation_store = conversation_store
        self._settings_factory = settings_factory
        self._retriever_factory = retriever_factory
        self._route_fn = route_fn
        self._evidence_fn = evidence_fn
        self._ask_model_fn = ask_model_fn
        self._quality_evaluator_fn = quality_evaluator_fn
        self._quality_evaluation_enabled_fn = quality_evaluation_enabled_fn
        self._datetime_answer_fn = datetime_answer_fn
        self._run_id_fn = run_id_fn or (lambda: uuid4().hex)
        self._model_gateway = model_gateway

    def _resolve_model(self, node: str, requested_model: Optional[str] = None) -> str:
        if self._model_gateway is not None:
            return self._model_gateway.resolve(node, requested_model=requested_model)
        return resolve_model_for_node(node, requested_model=requested_model)

    def _invoke_model(
        self,
        prompt: str,
        *,
        node: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        trace: Optional[TimingTrace] = None,
    ) -> str:
        if self._model_gateway is not None:
            return self._model_gateway.invoke(
                prompt,
                node=node,
                model=model,
                system=system,
                trace=trace,
            )
        resolved_model = self._resolve_model(node, requested_model=model)
        started_at = perf_counter()
        try:
            return self._ask_model_fn(prompt, model=resolved_model, system=system)
        finally:
            if trace is not None:
                trace.record(
                    "model." + str(node).replace("_", "."),
                    elapsed_ms(started_at),
                    model=resolved_model,
                )

    def run(self, request: RagPipelineRequest) -> dict:
        started_at = perf_counter()
        run_id = self._run_id_fn()
        trace = TimingTrace(
            run_id=run_id,
            request_id=str(request.request_id or ""),
            component="rag_pipeline",
        )
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
        generation_model = self._resolve_model("generation", request.model)
        isolated_subagent = parse_model_spec(generation_model).provider == "codex"
        remembered = None
        conversation_id = ""
        store = None
        conversation_started = perf_counter()
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
        trace.record(
            "conversation.load",
            elapsed_ms(conversation_started),
            status="completed" if request.persist_conversation else "skipped",
            persisted=bool(request.persist_conversation),
        )

        # A Codex subagent is intentionally stateless per question.  The
        # application may still persist the conversation for the UI, but no
        # prior messages or long-term memories are passed to routing or answer
        # generation for this model provider.
        model_history = [] if isolated_subagent else history
        model_memories = [] if isolated_subagent else memories

        contexts: List[dict] = []
        raw_contexts: List[dict] = []
        evidence_focus = {
            "enabled": False,
            "candidateCount": 0,
            "selectedCount": 0,
        }
        evidence_evaluation = None
        quality_evaluation = None
        grounded_request: Dict[str, str] = {}
        evidence_validation = None
        retrieval_decision = None
        retrieval_plan = None
        retrieval_timings: Dict[str, float] = {}
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
            trace.record("query.route", 0.0, status="skipped", reason="direct_answer")
            trace.record("retrieval", 0.0, status="skipped", reason="direct_answer")
            trace.record("generation", 0.0, status="completed", reason="direct_answer")
        else:
            route_started = perf_counter()
            route_model = self._resolve_model("query_rewrite", request.model)
            route_model_started = perf_counter()
            retrieval_decision = self._route(
                request.question,
                route_model,
                model_history,
            )
            trace.record(
                "model.query_rewrite",
                elapsed_ms(route_model_started),
                model=route_model,
            )
            stage_timings["routeMs"] = _elapsed_ms(route_started)
            trace.record(
                "query.route",
                stage_timings["routeMs"],
                model=route_model,
                needs_retrieval=bool(retrieval_decision["needs_retrieval"]),
            )

            if retrieval_decision["needs_retrieval"]:
                retrieval_plan = build_retrieval_plan(
                    question=request.question,
                    primary_query=retrieval_decision["retrieval_query"],
                    rewritten_query=retrieval_decision["retrieval_query"],
                    query_variants=retrieval_decision.get("query_variants") or (),
                    sub_questions=retrieval_decision.get("sub_questions") or (),
                    max_variants=(
                        settings.multi_query_max_variants
                        if settings.multi_query_enabled
                        else 1
                    ),
                )
                retrieval_started = perf_counter()
                retrieval_result = self._retriever_factory(request.profile).retrieve(
                    question=request.question,
                    retrieval_query=retrieval_plan.primary_query,
                    query_variants=retrieval_plan.query_variants,
                    evidence_query=retrieval_plan.evidence_query,
                    source_ids=request.source_ids,
                    top_k=top_k,
                    candidate_k=settings.hybrid_candidate_k,
                    retrieval_scope=request.retrieval_scope,
                )
                retrieval_node_timings = dict(retrieval_result.get("timings") or {})
                retrieval_timings = retrieval_node_timings
                for timing_key, node_name in (
                    ("bm25Ms", "retrieval.bm25"),
                    ("embeddingMs", "retrieval.embedding"),
                    ("fusionMs", "retrieval.fusion"),
                    ("rerankMs", "retrieval.rerank"),
                    ("totalMs", "retrieval.total"),
                ):
                    value = retrieval_node_timings.get(timing_key)
                    if isinstance(value, (int, float)):
                        trace.record(node_name, float(value))
                contexts = normalize_contexts(
                    retrieval_result.get("contexts"),
                    max_contexts=settings.hybrid_max_top_k,
                )
                if contexts and settings.fine_evidence_enabled:
                    raw_contexts = [dict(context) for context in contexts]
                    focus_started = perf_counter()
                    contexts, evidence_focus = retrieve_fine_evidence(
                        # Fine-grained semantic matching must use the user's
                        # actual question. The expanded evidence query is
                        # useful for recall, but dilutes cosine similarity when
                        # it contains several sub-queries and metadata terms.
                        question=request.question,
                        contexts=contexts,
                        settings=settings,
                        chunk_fraction=settings.fine_evidence_chunk_fraction,
                    )
                    stage_timings["focusMs"] = _elapsed_ms(focus_started)
                    trace.record("evidence.focus", stage_timings["focusMs"])
                elif contexts and settings.evidence_focus_enabled:
                    raw_contexts = [dict(context) for context in contexts]
                    focus_started = perf_counter()
                    contexts, evidence_focus = focus_retrieved_evidence(
                        question=(
                            retrieval_plan.evidence_query
                            if retrieval_plan is not None
                            else request.question
                        ),
                        contexts=contexts,
                        settings=settings,
                    )
                    stage_timings["focusMs"] = _elapsed_ms(focus_started)
                    trace.record("evidence.focus", stage_timings["focusMs"])
                else:
                    stage_timings["focusMs"] = 0.0
                    trace.record("evidence.focus", 0.0, status="skipped")
                stage_timings["retrieveMs"] = _elapsed_ms(retrieval_started)
                trace.record(
                    "retrieval",
                    stage_timings["retrieveMs"],
                    context_count=len(contexts),
                )
            else:
                stage_timings["retrieveMs"] = 0.0
                stage_timings["focusMs"] = 0.0
                trace.record("retrieval", 0.0, status="skipped", reason="route_no_retrieval")
                trace.record("evidence.focus", 0.0, status="skipped")

            evidence_started = perf_counter()
            if contexts:
                evidence_query = (
                    retrieval_plan.evidence_query
                    if retrieval_plan is not None
                    else request.question
                )
                evidence_evaluation = self._evidence_fn(
                    contexts,
                    settings=settings,
                    question=evidence_query,
                )
                if raw_contexts and evidence_focus.get("enabled"):
                    raw_evaluation = self._evidence_fn(
                        raw_contexts,
                        settings=settings,
                        question=evidence_query,
                    )
                    if (
                        raw_evaluation.get("sufficient")
                        and not evidence_evaluation.get("sufficient")
                    ):
                        contexts = raw_contexts
                        evidence_evaluation = raw_evaluation
                        evidence_focus["fallback"] = "quality_guard"
            trace.record(
                "evidence.gate",
                elapsed_ms(evidence_started),
                status="completed" if contexts else "skipped",
                sufficient=(
                    bool(evidence_evaluation.get("sufficient"))
                    if isinstance(evidence_evaluation, dict)
                    else None
                ),
            )

            generation_started = perf_counter()
            if retrieval_decision["needs_retrieval"] and (
                not contexts
                or (evidence_evaluation and not evidence_evaluation["sufficient"])
            ):
                evidence_reason = (
                    evidence_evaluation.get("reason", "")
                    if evidence_evaluation
                    else "檢索器沒有返回可用片段。"
                )
                answer = (
                    "根據目前檢索資料無法確認。"
                    f"{evidence_reason}請補充或選擇含有直接答案的文件後再試。"
                )
            elif contexts:
                evidence_validation = {}
                answer = answer_from_contexts(
                    question=request.question,
                    contexts=contexts,
                    model=generation_model,
                    history=model_history,
                    memories=model_memories,
                    ask_model_fn=(
                        lambda prompt, model, system=None: self._invoke_model(
                            prompt,
                            node="generation",
                            model=model,
                            system=system,
                            trace=trace,
                        )
                    ),
                    capture_request=grounded_request,
                    capture_validation=evidence_validation,
                )
            else:
                answer = self._invoke_model(
                    prompt_with_history(
                        build_no_retrieval_answer_prompt(
                            request.question,
                            retrieval_decision["reason"],
                        ),
                        model_history,
                    ),
                    model=generation_model,
                    system=build_qwen_direct_system_prompt(memories=model_memories),
                    node="generation",
                    trace=trace,
                )
            stage_timings["generateMs"] = _elapsed_ms(generation_started)
            generation_refused = bool(
                retrieval_decision["needs_retrieval"]
                and (
                    not contexts
                    or (evidence_evaluation and not evidence_evaluation["sufficient"])
                )
            )
            trace.record(
                "generation",
                stage_timings["generateMs"],
                status="completed",
                model=generation_model,
                model_called=not generation_refused,
            )

        requested_model = parse_model_spec(generation_model)
        if (
            grounded_request
            and requested_model.provider == "ollama"
            and self._quality_evaluation_enabled_fn()
        ):
            evaluation_started = perf_counter()
            try:
                quality_evaluation = self._quality_evaluator_fn(
                    question=request.question,
                    final_prompt=grounded_request["prompt"],
                    system_prompt=grounded_request["system"],
                    qwen_answer=answer,
                    contexts=contexts,
                )
            except Exception as exc:
                quality_evaluation = unavailable_reference_evaluation(
                    f"{type(exc).__name__}: {str(exc)[:300]}"
                )
            stage_timings["evaluateMs"] = _elapsed_ms(evaluation_started)
            trace.record("evaluation", stage_timings["evaluateMs"], status="completed")
        else:
            stage_timings["evaluateMs"] = 0.0
            trace.record("evaluation", 0.0, status="skipped")

        persist_started = perf_counter()
        if store is not None:
            store.add_message(
                conversation_id,
                "assistant",
                answer,
                metadata={
                    "run_id": run_id,
                    "retrieval_needed": retrieval_decision["needs_retrieval"],
                    "remembered": bool(remembered),
                    "qwen_quality_score": (
                        quality_evaluation.get("candidate", {}).get("score")
                        if quality_evaluation
                        else None
                    ),
                    "evidence_validation": dict(evidence_validation or {}),
                },
            )
        trace.record(
            "conversation.persist",
            elapsed_ms(persist_started),
            status="completed" if store is not None else "skipped",
            persisted=bool(store is not None),
        )

        spec = requested_model
        evidence_is_sufficient = (
            not evidence_evaluation or evidence_evaluation["sufficient"]
        )
        stage_timings["totalMs"] = _elapsed_ms(started_at)
        trace.record("total", stage_timings["totalMs"])
        trace_payload = trace.as_dict()
        stage_timings["stages"] = trace_payload["stages"]
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
            "request_id": str(request.request_id or ""),
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
            "evidence_validation": evidence_validation,
            "retrieval": {
                "server_generated": True,
                "needed": retrieval_decision["needs_retrieval"],
                "reason": retrieval_decision["reason"],
                "query": retrieval_decision["retrieval_query"],
                "queries": list(retrieval_plan.query_variants) if retrieval_plan else [],
                "sub_questions": list(retrieval_plan.sub_questions) if retrieval_plan else [],
                "rewrite_semantic_validation": retrieval_decision.get(
                    "rewrite_semantic_validation",
                    {
                        "status": "skipped",
                        "similarity": None,
                        "accepted": True,
                        "threshold": None,
                        "reason": "",
                    },
                ),
                "intent_labels": list(retrieval_plan.intent_labels) if retrieval_plan else [],
                "contexts": contexts,
                "raw_contexts": raw_contexts,
                "evidence_focus": evidence_focus,
                "evidence_evaluation": evidence_evaluation,
                "timings": retrieval_timings,
            },
            "model": {"provider": spec.provider, "name": spec.model},
            "quality_evaluation": quality_evaluation,
            "timings": stage_timings,
            "timing_trace": trace_payload,
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
                conversation_context=conversation_context(history) if history else "",
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
            "sub_questions": list(getattr(decision, "sub_questions", ()) or ()),
            "query_variants": list(getattr(decision, "query_variants", ()) or ()),
            "rewrite_semantic_validation": {
                "status": str(getattr(decision, "semantic_validation_status", "skipped") or "skipped"),
                "similarity": getattr(decision, "semantic_similarity", None),
                "accepted": bool(getattr(decision, "semantic_accepted", True)),
                "threshold": getattr(decision, "semantic_validation_threshold", None),
                "reason": str(getattr(decision, "semantic_validation_reason", "") or ""),
            },
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
    capture_request: Optional[dict] = None,
    capture_validation: Optional[dict] = None,
) -> str:
    model = resolve_model_for_node(
        "generation",
        requested_model=model,
        prefer_requested=bool(model),
    )
    threshold_evidence = resolve_threshold_evidence(
        question,
        contexts,
        previous_user_question=latest_user_question(history),
    )
    if threshold_evidence is not None:
        if capture_validation is not None:
            capture_validation.update({
                "sufficient": True,
                "status": "threshold",
                "valid_citations": [],
                "invalid_citations": [],
                "uncited_claims": [],
                "unsupported_claims": [],
                "reason": "由可驗證的門檻證據直接產生答案。",
            })
        return render_threshold_answer(threshold_evidence)

    request = build_grounded_answer_request(
        question=question,
        contexts=contexts,
        history=history,
        memories=memories,
    )
    if capture_request is not None:
        capture_request.update(request)
    answer = ask_model_fn(
        request["prompt"],
        model=model,
        system=request["system"],
    )
    validation = validate_answer_evidence(answer, contexts)
    if capture_validation is not None:
        capture_validation.update(validation)
    return enforce_grounded_answer_contract(answer, contexts)


def build_grounded_answer_request(
    question: str,
    contexts: List[dict],
    history=None,
    memories=None,
) -> Dict[str, str]:
    ordered_contexts = attention_order_contexts(contexts)
    context_text = "\n\n".join(
        "\n".join(
            [
                f'<evidence rank="{context["rank"]}">',
                f'標題：{context["title"]}',
                f'頁碼：{context["page"]}',
                "內容：",
                context["content"],
                "</evidence>",
            ]
        )
        for context in ordered_contexts
    )
    prompt = f"""{rag_conversation_context(history)}

### 唯一允許引用的本次檢索證據
<trusted_evidence>
{context_text}
</trusted_evidence>

### 目前使用者問題
{question}

### 強制回答契約
請使用繁體中文回答。
只能根據 <trusted_evidence> 回答；如果資料不足，請明確回答「根據目前檢索資料無法確認」，並列出缺少的證據。
最近對話只用於解析代名詞，不是證據；不可重複或延續舊助理回答。
目前問題中的數字與邊界條件優先；逐一核對「以上、未滿、以下」後再回答，不可套用相鄰區間。
問題若要求公式，必須逐字列出檢索資料中的公式，不可只列計算範例。
回答要精簡，但要保留關鍵原因。
每一個包含事實、數字、日期、條件、程序或結論的句子，都必須緊接直接支持它的 rank。
最後用「來源：」列出實際使用的 rank，例如 [1], [2]；禁止列出沒有直接支持答案的來源。

再次確認目前問題：{question}
"""
    return {
        "prompt": prompt,
        "system": build_qwen_rag_system_prompt(memories=memories),
    }


def attention_order_contexts(contexts: Sequence[dict]) -> List[dict]:
    """Place the strongest passages near prompt edges without duplicating them.

    The hosted Ollama API does not expose per-token attention bias. Edge-aware
    ordering is therefore an input-level mitigation: rank 1 is placed first,
    rank 2 last, rank 3 second, and so on.
    """

    ranked = [dict(context) for context in contexts]
    left = []
    right = []
    for index, context in enumerate(ranked):
        if index % 2 == 0:
            left.append(context)
        else:
            right.insert(0, context)
    return [*left, *right]


def enforce_grounded_answer_contract(answer: str, contexts: Sequence[dict]) -> str:
    """Fail closed when a generated factual answer has no valid evidence marker."""

    text = str(answer or "").strip()
    validation = validate_answer_evidence(text, contexts)
    if validation["status"] == "refused":
        return normalized_refusal_answer(text) or "根據目前檢索資料無法確認。"
    if validation["sufficient"]:
        return text

    return (
        "根據目前檢索資料無法確認。模型產生的答案沒有通過來源約束檢查，"
        "因此系統未顯示未受證據支持的內容。"
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
