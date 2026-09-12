#!/usr/bin/env python3
"""Run a deterministic 100-question MultiHop-RAG Qwen A/B benchmark.

The unoptimized version uses hard 600-character chunks, raw 50/50 BM25 and
dense score addition, and the existing lexical reranker.  The optimized
version retrieves 256-token children, fuses the two rankings with RRF over a
100-candidate pool, routes simple questions directly to top 5, uses a
Cross-Encoder for complex questions, and expands selected children to their
1024-token parent evidence before calling Qwen.

The benchmark is intentionally stateless: no conversation history or long-
term memory is passed to the answer model.  A fixed seed makes the 100-query
sample reproducible from the full 2,556-query MultiHop-RAG set.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_demo.answer_quality import evaluate_answer_quality
from rag_demo.chunk_strategies import CHUNK_STRATEGY_DYNAMIC, CHUNK_STRATEGY_HARD, split_text
from rag_demo.config import RagConfig
from rag_demo.cross_encoder import DEFAULT_CROSS_ENCODER_MODEL, CrossEncoderReranker
from rag_demo.embeddings import DEFAULT_EMBEDDING_MODEL, embed_chunks, embed_query
from rag_demo.hybrid_retrieval import Bm25Index, reciprocal_rank_fusion, rerank_candidates, tokenize_bm25
from rag_demo.model_providers import ask_model
from rag_demo.parent_child import build_parent_child_index, expand_child_contexts
from rag_demo.query_complexity import classify_query_complexity
from rag_demo.rag_pipeline import build_grounded_answer_request, enforce_grounded_answer_contract


DEFAULT_DATA = ROOT / "RAG測試題庫" / "01_MultiHop-RAG" / "data"
DEFAULT_QUERIES = DEFAULT_DATA / "MultiHopRAG.json"
DEFAULT_CORPUS = DEFAULT_DATA / "corpus.json"
DEFAULT_SAMPLE = ROOT / "evals" / "multihop_rag_qwen" / "sample-100.json"
DEFAULT_OUTPUT = ROOT / "evals" / "multihop_rag_qwen" / "qwen-ab-results.json"
DEFAULT_REPORT = ROOT / "evals" / "multihop_rag_qwen" / "qwen-ab-report.md"
DEFAULT_MODEL = "ollama:qwen2.5:7b"


def _normalise_embeddings(embeddings: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _dense_hits(
    query: str,
    matrix: np.ndarray,
    top_k: int,
    cache: dict[str, np.ndarray],
    model: str,
) -> list[dict[str, Any]]:
    vector = cache.get(query)
    if vector is None:
        vector = np.asarray(embed_query(query, model_name=model), dtype=np.float32)
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        cache[query] = vector
    scores = matrix @ vector
    order = np.argsort(-scores)[: min(max(1, int(top_k)), len(scores))]
    return [{"index": int(index), "score": float(scores[index])} for index in order]


def _raw_fusion(
    bm25_hits: Sequence[dict[str, Any]],
    dense_hits: Sequence[dict[str, Any]],
    keyword_weight: float = 0.5,
    embedding_weight: float = 0.5,
) -> list[dict[str, Any]]:
    candidates: dict[int, dict[str, Any]] = {}
    for hit in bm25_hits:
        item = candidates.setdefault(
            int(hit["index"]),
            {"index": int(hit["index"]), "bm25_score": 0.0, "embedding_score": 0.0, "matched_terms": []},
        )
        item["bm25_score"] = float(hit.get("score", 0.0))
        item["matched_terms"] = list(hit.get("matched_terms") or [])
    for hit in dense_hits:
        item = candidates.setdefault(
            int(hit["index"]),
            {"index": int(hit["index"]), "bm25_score": 0.0, "embedding_score": 0.0, "matched_terms": []},
        )
        item["embedding_score"] = float(hit.get("score", 0.0))
    total = max(1e-12, float(keyword_weight) + float(embedding_weight))
    keyword_weight /= total
    embedding_weight /= total
    for item in candidates.values():
        item["fusion_score"] = keyword_weight * item["bm25_score"] + embedding_weight * item["embedding_score"]
        item["score"] = item["fusion_score"]
    return sorted(candidates.values(), key=lambda item: item["fusion_score"], reverse=True)


def _hard_chunks(corpus: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for doc_index, document in enumerate(corpus, start=1):
        body = str(document.get("body") or "").strip()
        if not body:
            continue
        title = str(document.get("title") or f"document-{doc_index}").strip()
        source = str(document.get("source") or "MultiHop-RAG").strip()
        url = str(document.get("url") or "").strip()
        for chunk_index, piece in enumerate(
            split_text(body, 600, 600, strategy=CHUNK_STRATEGY_HARD, overlap_tokens=0),
            start=1,
        ):
            chunks.append(
                {
                    "id": f"multihop-hard-{doc_index:04d}-{chunk_index:04d}",
                    "source_id": url or source,
                    "source": source,
                    "parent_title": "MultiHop-RAG",
                    "title": title,
                    "page": source,
                    "content": piece,
                    "chunk_level": "single",
                }
            )
    return chunks


def _parent_child_index(corpus: Sequence[dict[str, Any]]):
    units = []
    for doc_index, document in enumerate(corpus, start=1):
        body = str(document.get("body") or "").strip()
        if not body:
            continue
        title = str(document.get("title") or f"document-{doc_index}").strip()
        source = str(document.get("source") or "MultiHop-RAG").strip()
        url = str(document.get("url") or "").strip()
        units.append(
            {
                "title": title,
                "page": source,
                "source": source,
                "source_id": url or source,
                "content": body,
            }
        )
    return build_parent_child_index(
        units,
        source_id="multihop-rag",
        filename="MultiHop-RAG",
        source_type="json-corpus",
        extraction_method="dataset-body",
        parent_size_tokens=1024,
        child_size_tokens=256,
        parent_overlap_tokens=0,
        child_overlap_tokens=0,
        strategy=CHUNK_STRATEGY_DYNAMIC,
    )


def _load_sample(
    queries_path: Path,
    sample_path: Path,
    sample_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = json.loads(queries_path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) != 2556:
        raise ValueError(f"MultiHop-RAG query file must contain 2556 records, got {len(data) if isinstance(data, list) else 'invalid'}")
    if sample_size <= 0 or sample_size > len(data):
        raise ValueError("sample_size must be between 1 and the full dataset size")
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(data)), sample_size))
    items = []
    for index in indices:
        original = dict(data[index])
        question_type = str(original.get("question_type") or "unknown")
        gold = str(original.get("answer") or "").strip()
        original.update(
            {
                "id": f"multihop-{index + 1:04d}",
                "dataset_index": index,
                "answerable": question_type != "null_query" and gold.casefold() != "insufficient information.",
            }
        )
        items.append(original)
    payload = {
        "dataset": "yixuantt/MultiHopRAG",
        "dataset_total": len(data),
        "sample_size": len(items),
        "sample_seed": seed,
        "indices": indices,
        "items": items,
    }
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return items, payload


def _compact(text: object) -> str:
    return re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", str(text or "").casefold())


def _answer_is_refusal(answer: str) -> bool:
    text = str(answer or "").casefold()
    markers = (
        "根據目前檢索資料無法確認",
        "目前檢索資料不足",
        "資料不足",
        "insufficient information",
        "cannot determine",
        "cannot be determined",
        "not enough information",
    )
    return any(marker in text for marker in markers)


def _answer_matches_gold(answer: str, gold: str) -> bool:
    if _answer_is_refusal(answer):
        return False
    expected = str(gold or "").strip()
    if not expected:
        return False
    if expected.casefold() in {"yes", "no"}:
        match = re.search(r"\b(yes|no)\b", str(answer or "").casefold())
        if match:
            return match.group(1) == expected.casefold()
        match = re.search(r"(?:答案|結論)\s*[：:]?\s*(是|否)", str(answer or ""))
        return bool(match and ((match.group(1) == "是") == (expected.casefold() == "yes")))
    expected_compact = _compact(expected)
    answer_compact = _compact(answer)
    if expected_compact and expected_compact in answer_compact:
        return True
    expected_tokens = set(re.findall(r"[a-z0-9]+", expected.casefold()))
    answer_tokens = set(re.findall(r"[a-z0-9]+", str(answer or "").casefold()))
    expected_tokens = {token for token in expected_tokens if len(token) > 1}
    return bool(expected_tokens) and expected_tokens.issubset(answer_tokens)


def _fact_recall(contexts: Sequence[dict[str, Any]], evidence_list: Sequence[dict[str, Any]]) -> tuple[int, int, list[str]]:
    context_text = "\n".join(str(context.get("content") or "") for context in contexts)
    context_tokens = set(tokenize_bm25(context_text))
    matched = []
    for evidence in evidence_list or []:
        fact = str(evidence.get("fact") or "").strip() if isinstance(evidence, dict) else str(evidence).strip()
        if not fact:
            continue
        fact_tokens = set(tokenize_bm25(fact))
        lexical = len(fact_tokens & context_tokens) / max(1, len(fact_tokens))
        if _compact(fact) in _compact(context_text) or lexical >= 0.45:
            matched.append(fact)
    return len(matched), len([e for e in evidence_list or [] if str((e or {}).get("fact") if isinstance(e, dict) else e).strip()]), matched


def _contexts_from_candidates(
    selected: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
    branch: str,
) -> list[dict[str, Any]]:
    contexts = []
    for rank, candidate in enumerate(selected, start=1):
        chunk = chunks[int(candidate["index"])]
        contexts.append(
            {
                "id": str(chunk.get("id") or candidate["index"]),
                "rank": rank,
                "title": str(chunk.get("title") or "MultiHop-RAG"),
                "page": str(chunk.get("page") or ""),
                "source": str(chunk.get("source_id") or chunk.get("source") or "multihop-rag"),
                "content": str(chunk.get("content") or ""),
                "branch": branch,
                "score": round(float(candidate.get("rerank_score", candidate.get("fusion_score", candidate.get("rrf_score", 0.0)))), 8),
                "bm25Score": round(float(candidate.get("bm25_score", 0.0)), 8),
                "embeddingScore": round(float(candidate.get("embedding_score", 0.0)), 8),
                "fusionScore": round(float(candidate.get("fusion_score", candidate.get("rrf_score", 0.0))), 8),
                "rerankScore": round(float(candidate.get("rerank_score", candidate.get("fusion_score", candidate.get("rrf_score", 0.0)))), 8),
                "matchedTerms": list(candidate.get("matched_terms") or []),
            }
        )
    return contexts


def _retrieve_unoptimized(
    question: str,
    chunks: Sequence[dict[str, Any]],
    matrix: np.ndarray,
    bm25: Bm25Index,
    cache: dict[str, np.ndarray],
    embedding_model: str,
    settings: RagConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bm25_hits = bm25.search(question, top_k=100)
    dense_hits = _dense_hits(question, matrix, 100, cache, embedding_model)
    candidates = _raw_fusion(bm25_hits, dense_hits, 0.5, 0.5)[:100]
    token_sets = [set(tokenize_bm25(str(chunk.get("content") or ""))) for chunk in chunks]
    selected = rerank_candidates(
        candidates,
        chunks,
        question,
        top_k=5,
        settings=settings,
        chunk_token_sets=token_sets,
    )
    return _contexts_from_candidates(selected, chunks, "unoptimized"), {
        "retrieval_query": question,
        "rewrite": {"status": "disabled", "accepted": True},
        "complexity": {"label": "unoptimized-always-rerank", "is_complex": True},
        "fusion": "raw_weighted_addition",
        "candidate_count": len(candidates),
        "rerank_applied": True,
        "retrieval_children": 0,
        "evidence_parents": len(selected),
    }


def _retrieve_optimized(
    question: str,
    index,
    matrix: np.ndarray,
    bm25: Bm25Index,
    cache: dict[str, np.ndarray],
    embedding_model: str,
    cross_encoder: CrossEncoderReranker,
    settings: RagConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # MultiHop questions are already carefully authored English queries.  The
    # benchmark keeps the exact text so query rewriting cannot leak gold data.
    retrieval_query = question
    bm25_hits = bm25.search(retrieval_query, top_k=100)
    dense_hits = _dense_hits(retrieval_query, matrix, 100, cache, embedding_model)
    candidates = reciprocal_rank_fusion(bm25_hits, dense_hits, rrf_k=settings.hybrid_rrf_k)[:100]
    decision = classify_query_complexity(question, threshold=settings.query_complexity_threshold)
    if decision.is_complex:
        documents = [index.children[int(candidate["index"])].get("content", "") for candidate in candidates]
        scores = cross_encoder.score(question, documents)
        ranked = [dict(candidate, rerank_score=float(score)) for candidate, score in zip(candidates, scores)]
        ranked.sort(key=lambda item: (item["rerank_score"], item.get("rrf_score", item.get("fusion_score", 0.0))), reverse=True)
        rerank_applied = True
    else:
        ranked = [dict(candidate, rerank_score=float(candidate.get("rrf_score", candidate.get("fusion_score", 0.0)))) for candidate in candidates]
        rerank_applied = False
    selected = ranked[:5]
    child_records = []
    for candidate in selected:
        child = dict(index.children[int(candidate["index"])])
        child.update(candidate)
        child_records.append(child)
    parents = expand_child_contexts(child_records, parent_by_id=index.parent_by_id, limit=5)
    candidate_by_child = {str(child.get("id")): candidate for child, candidate in zip(child_records, selected)}
    contexts = []
    for rank, parent in enumerate(parents, start=1):
        candidate = candidate_by_child.get(str(parent.get("retrieval_child_id")), {})
        contexts.append(
            {
                "id": str(parent.get("id") or parent.get("retrieval_child_id") or ""),
                "rank": rank,
                "title": str(parent.get("title") or "MultiHop-RAG"),
                "page": str(parent.get("page") or ""),
                "source": str(parent.get("source_id") or parent.get("source") or "multihop-rag"),
                "content": str(parent.get("content") or ""),
                "branch": "optimized-parent-evidence",
                "score": round(float(candidate.get("rerank_score", candidate.get("rrf_score", 0.0))), 8),
                "bm25Score": round(float(candidate.get("bm25_score", 0.0)), 8),
                "embeddingScore": round(float(candidate.get("embedding_score", 0.0)), 8),
                "fusionScore": round(float(candidate.get("rrf_score", candidate.get("fusion_score", 0.0))), 8),
                "rerankScore": round(float(candidate.get("rerank_score", candidate.get("rrf_score", 0.0))), 8),
                "childChunkId": str(parent.get("retrieval_child_id") or ""),
                "parentChunkId": str(parent.get("id") or ""),
                "matchedTerms": list(candidate.get("matched_terms") or []),
            }
        )
    return contexts, {
        "retrieval_query": retrieval_query,
        "rewrite": {"status": "identity_english_benchmark", "accepted": True, "similarity": 1.0},
        "complexity": decision.as_dict(),
        "fusion": "RRF",
        "candidate_count": len(candidates),
        "rerank_applied": rerank_applied,
        "retrieval_children": len(selected),
        "evidence_parents": len(contexts),
    }


def _quality_case(item: dict[str, Any], answer: str, contexts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    evidence_list = item.get("evidence_list") or []
    matched, total, matched_facts = _fact_recall(contexts, evidence_list)
    gold = str(item.get("answer") or "").strip()
    answerable = bool(item.get("answerable"))
    correct = _answer_matches_gold(answer, gold) if answerable else _answer_is_refusal(answer)
    answer_in_chunks = (
        _answer_matches_gold(answer, gold) and bool(_compact(gold) in _compact("\n".join(c.get("content", "") for c in contexts)))
        if answerable and gold.casefold() not in {"yes", "no"}
        else (matched == total if answerable else matched == 0)
    )
    grounding = evaluate_answer_quality(
        question=str(item.get("query") or ""),
        answer=answer,
        contexts=contexts,
        expected_facts=(),
        embedding_fn=None,
    )
    return {
        "answer_correct": correct,
        "answer_in_retrieved_chunks": answer_in_chunks,
        "evidence_fact_recall": round(matched / max(1, total), 4) if total else None,
        "evidence_facts_matched": matched,
        "evidence_facts_total": total,
        "matched_facts": matched_facts,
        "refused": _answer_is_refusal(answer),
        "hallucination": grounding["hallucination"],
        "answer_grounding": grounding["answer_in_retrieved_chunks"],
    }


def _summary(items: Sequence[dict[str, Any]], cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    answerable = [case for case, item in zip(cases, items) if item.get("answerable")]
    unanswerable = [case for case, item in zip(cases, items) if not item.get("answerable")]
    latencies = [float(case["timings_ms"]["total"]) for case in cases]
    fact_numerator = sum(case["quality"]["evidence_facts_matched"] for case in answerable)
    fact_denominator = sum(case["quality"]["evidence_facts_total"] for case in answerable)
    hallucination_free = sum(bool(case["quality"]["hallucination"]["hallucination_free"]) for case in cases)
    return {
        "question_count": len(cases),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "answer_accuracy": round(sum(bool(case["quality"]["answer_correct"]) for case in answerable) / max(1, len(answerable)), 4),
        "answer_in_retrieved_chunks_rate": round(sum(bool(case["quality"]["answer_in_retrieved_chunks"]) for case in answerable) / max(1, len(answerable)), 4),
        "evidence_fact_recall": round(fact_numerator / max(1, fact_denominator), 4),
        "safe_refusal_rate": round(sum(bool(case["quality"]["refused"]) for case in unanswerable) / max(1, len(unanswerable)), 4),
        "hallucination_free_rate": round(hallucination_free / max(1, len(cases)), 4),
        "hallucination_count": len(cases) - hallucination_free,
        "average_total_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "p95_total_ms": round(sorted(latencies)[int(round((len(latencies) - 1) * 0.95))], 2) if latencies else 0.0,
        "complex_count": sum(case["retrieval"]["complexity"].get("label") == "complex" for case in cases),
        "rerank_count": sum(bool(case["retrieval"].get("rerank_applied")) for case in cases),
    }


def run_version(
    *,
    version: str,
    label: str,
    items: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
    matrix: np.ndarray,
    bm25: Bm25Index,
    embedding_model: str,
    model: str,
    settings: RagConfig,
    optimized: bool,
    parent_index=None,
    cross_encoder=None,
) -> dict[str, Any]:
    query_cache: dict[str, np.ndarray] = {}
    cases = []
    for position, item in enumerate(items, start=1):
        question = str(item.get("query") or "").strip()
        print(f"[{label} {position}/{len(items)}] {item.get('id')} -> {model}", flush=True)
        started = time.perf_counter()
        if optimized:
            contexts, retrieval = _retrieve_optimized(
                question,
                parent_index,
                matrix,
                bm25,
                query_cache,
                embedding_model,
                cross_encoder,
                settings,
            )
        else:
            contexts, retrieval = _retrieve_unoptimized(
                question,
                chunks,
                matrix,
                bm25,
                query_cache,
                embedding_model,
                settings,
            )
        retrieval_ms = (time.perf_counter() - started) * 1000.0
        request = build_grounded_answer_request(
            question=question,
            contexts=contexts,
            history=[],
            memories=[],
        )
        generation_started = time.perf_counter()
        raw_answer = ask_model(request["prompt"], model=model, system=request["system"])
        answer = enforce_grounded_answer_contract(raw_answer, contexts)
        generation_ms = (time.perf_counter() - generation_started) * 1000.0
        cases.append(
            {
                "id": item.get("id"),
                "dataset_index": item.get("dataset_index"),
                "question_type": item.get("question_type"),
                "question": question,
                "gold_answer": item.get("answer"),
                "answerable": bool(item.get("answerable")),
                "raw_answer": raw_answer,
                "answer": answer,
                "retrieval": retrieval,
                "contexts": contexts,
                "timings_ms": {
                    "retrieval": round(retrieval_ms, 2),
                    "generation": round(generation_ms, 2),
                    "total": round(retrieval_ms + generation_ms, 2),
                },
                "quality": _quality_case(item, answer, contexts),
            }
        )
    return {
        "name": label,
        "version": version,
        "model": model,
        "configuration": {
            "chunking": "hard 600 characters, no overlap" if not optimized else "1024-token parent / 256-token child, no overlap",
            "query_rewrite": "disabled" if not optimized else "identity (English benchmark; no gold leakage)",
            "hybrid_fusion": "raw 50/50 weighted addition" if not optimized else "RRF",
            "candidate_pool": 100,
            "routing": "always basic reranker top 5" if not optimized else "simple direct top 5; complex Cross-Encoder top 5",
            "memory": "none",
        },
        "summary": _summary(items, cases),
        "cases": cases,
    }


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# MultiHop-RAG 100 題 Qwen 無優化版與全優化版比較",
        "",
        "本次從完整 2,556 題 MultiHop-RAG 以固定 seed 抽出 100 題。兩個版本使用相同的 609 份文件、Embedding 與 Qwen 2.5 7B；每題不傳入對話歷史或長期記憶。",
        "",
        f"- 模型：`{payload['model']}`",
        f"- 抽樣：完整 `{payload['dataset_total']}` 題，seed `{payload['sample_seed']}`，本次 `{payload['sample_size']}` 題",
        f"- 文件數：`{payload['corpus_documents']}`",
        f"- 執行時間：`{payload['run_at']}`",
        "",
        "## 主要指標",
        "",
        "| 版本 | 答案正確率 | 答案在檢索切片中 | 證據 fact recall | 安全拒答率 | 無幻覺率 | 平均總時間 ms | P95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for version in payload["versions"]:
        summary = version["summary"]
        lines.append(
            f"| {version['name']} | {summary['answer_accuracy']:.1%} | {summary['answer_in_retrieved_chunks_rate']:.1%} | "
            f"{summary['evidence_fact_recall']:.1%} | {summary['safe_refusal_rate']:.1%} | {summary['hallucination_free_rate']:.1%} | "
            f"{summary['average_total_ms']} | {summary['p95_total_ms']} |"
        )
    lines.extend(["", "## 設定差異", "", "| 版本 | 分塊 | 問題重寫 | 混合檢索 | 候選池 | 路由與重排 | 記憶 |", "| --- | --- | --- | --- | ---: | --- | --- |"])
    for version in payload["versions"]:
        config = version["configuration"]
        lines.append(
            f"| {version['name']} | {config['chunking']} | {config['query_rewrite']} | {config['hybrid_fusion']} | "
            f"{config['candidate_pool']} | {config['routing']} | {config['memory']} |"
        )
    lines.extend(
        [
            "",
            "## 指標定義",
            "",
            "- 答案正確率：可回答題中，模型答案符合 MultiHop-RAG gold answer 的比例；Yes／No 題保留方向判斷。",
            "- 答案在檢索切片中：答案與 gold answer 均能由本版本送入模型的證據支持；Yes／No 題改以該題所有 gold evidence facts 是否找回判定。",
            "- 證據 fact recall：每題 evidence_list 的事實在檢索證據中的找回比例。",
            "- 安全拒答率：資料集標為 null_query 時，模型是否拒答。",
            "- 無幻覺率：回答中的主張、數字與引用均通過 deterministic evidence validation；系統拒答視為安全。",
            "- 全優化版的英文問題保留原始查詢，避免額外 Qwen 改寫請求把模型延遲與資料集答案混在一起；優化差異集中在父子 Chunk、RRF、100 候選池與複雜度分流重排。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--cross-encoder-model", default=DEFAULT_CROSS_ENCODER_MODEL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    items, sample_payload = _load_sample(args.queries.resolve(), args.sample.resolve(), args.sample_size, args.seed)
    corpus = json.loads(args.corpus.resolve().read_text(encoding="utf-8"))
    if not isinstance(corpus, list) or len(corpus) != 609:
        raise ValueError(f"MultiHop-RAG corpus must contain 609 documents, got {len(corpus) if isinstance(corpus, list) else 'invalid'}")

    print(f"Loaded {len(items)} sampled queries from {sample_payload['dataset_total']} and {len(corpus)} corpus documents", flush=True)
    base = RagConfig.from_env().normalized()
    settings = replace(
        base,
        hybrid_candidate_k=100,
        hybrid_max_candidate_k=200,
        hybrid_top_k=5,
        rerank_top_k=5,
        simple_query_top_k=5,
        hybrid_rrf_k=60,
        query_complexity_threshold=2.0,
        rerank_fusion_weight=0.30,
        rerank_bm25_weight=0.32,
        rerank_embedding_weight=0.26,
        rerank_coverage_weight=0.10,
        rerank_phrase_weight=0.02,
    ).normalized()

    hard_chunks = _hard_chunks(corpus)
    parent_child = _parent_child_index(corpus)
    print(f"Built {len(hard_chunks)} hard chunks, {len(parent_child.parents)} parent chunks, {len(parent_child.children)} child chunks", flush=True)
    hard_embeddings = _normalise_embeddings(embed_chunks(hard_chunks, model_name=args.embedding_model))
    child_embeddings = _normalise_embeddings(embed_chunks(parent_child.children, model_name=args.embedding_model))
    hard_bm25 = Bm25Index(hard_chunks, k1=settings.hybrid_bm25_k1, b=settings.hybrid_bm25_b)
    child_bm25 = Bm25Index(parent_child.children, k1=settings.hybrid_bm25_k1, b=settings.hybrid_bm25_b)
    cross_encoder = CrossEncoderReranker(args.cross_encoder_model)

    unoptimized = run_version(
        version="unoptimized",
        label="無優化版",
        items=items,
        chunks=hard_chunks,
        matrix=hard_embeddings,
        bm25=hard_bm25,
        embedding_model=args.embedding_model,
        model=args.model,
        settings=settings,
        optimized=False,
    )
    optimized = run_version(
        version="optimized",
        label="全優化版",
        items=items,
        chunks=parent_child.children,
        matrix=child_embeddings,
        bm25=child_bm25,
        embedding_model=args.embedding_model,
        model=args.model,
        settings=settings,
        optimized=True,
        parent_index=parent_child,
        cross_encoder=cross_encoder,
    )
    payload = {
        "schema_version": "multihop-rag-qwen-ab-v1",
        "dataset": "yixuantt/MultiHopRAG",
        "dataset_total": sample_payload["dataset_total"],
        "sample_size": sample_payload["sample_size"],
        "sample_seed": sample_payload["sample_seed"],
        "corpus_documents": len(corpus),
        "hard_chunk_count": len(hard_chunks),
        "parent_chunk_count": len(parent_child.parents),
        "child_chunk_count": len(parent_child.children),
        "embedding_model": args.embedding_model,
        "cross_encoder_model": args.cross_encoder_model,
        "model": args.model,
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "versions": [unoptimized, optimized],
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.resolve().write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "report": str(args.report.resolve())}, ensure_ascii=False))
    for version in payload["versions"]:
        print(json.dumps({"version": version["name"], **version["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
