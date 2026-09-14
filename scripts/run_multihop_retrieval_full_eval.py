#!/usr/bin/env python3
"""Run the full MultiHop-RAG retrieval-only comparison.

This benchmark deliberately stops after evidence retrieval.  It never builds
an answer prompt and never calls a generation model.  The two branches match
the existing MultiHop-RAG A/B benchmark:

* 無優化版: hard 600-character chunks, raw 50/50 BM25+dense addition, and
  the basic lexical/semantic reranker.
* 全優化版: sentence-aware 1024-token parents with 256-token children, RRF
  over 100 candidates, complexity routing, Cross-Encoder reranking for
  complex questions, and parent evidence expansion.

Gold evidence is used only to calculate retrieval recall after each question.
The output stores compact rankings and aggregate metrics, not the retrieved
document text, so a full 2,556-question run remains reviewable.
"""

from __future__ import annotations

import argparse
import json
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

from rag_demo.config import RagConfig
from rag_demo.cross_encoder import DEFAULT_CROSS_ENCODER_MODEL, CrossEncoderReranker
from rag_demo.embeddings import DEFAULT_EMBEDDING_MODEL, embed_chunks
from rag_demo.hybrid_retrieval import Bm25Index, tokenize_bm25

from scripts.run_multihop_qwen_ab import (
    _dense_hits,
    _hard_chunks,
    _normalise_embeddings,
    _parent_child_index,
    _retrieve_optimized,
    _retrieve_unoptimized,
)


DEFAULT_DATA = ROOT / "RAG測試題庫" / "01_MultiHop-RAG" / "data"
DEFAULT_QUERIES = DEFAULT_DATA / "MultiHopRAG.json"
DEFAULT_CORPUS = DEFAULT_DATA / "corpus.json"
DEFAULT_OUTPUT = ROOT / "evals" / "multihop_rag_retrieval_full" / "retrieval-full-results.json"
DEFAULT_REPORT = ROOT / "evals" / "multihop_rag_retrieval_full" / "retrieval-full-report.md"
DEFAULT_SUMMARY = ROOT / "evals" / "multihop_rag_retrieval_full" / "retrieval-full-summary.json"
DEFAULT_CACHE_DIR = ROOT / "evals" / "multihop_gpt55_retrieval_standard" / "embedding-cache"
DEFAULT_CHECKPOINT_DIR = ROOT / "evals" / "multihop_rag_retrieval_full" / "checkpoints"


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    return ordered[int(round((len(ordered) - 1) * 0.95))]


def _load_items(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 2556:
        raise ValueError(
            "MultiHop-RAG query file must contain 2,556 records; "
            f"got {len(payload) if isinstance(payload, list) else 'invalid'}"
        )
    items: list[dict[str, Any]] = []
    for index, raw in enumerate(payload):
        item = dict(raw)
        question_type = str(item.get("question_type") or "unknown")
        answer = str(item.get("answer") or "").strip()
        item.update(
            {
                "id": f"multihop-{index + 1:04d}",
                "dataset_index": index,
                "answerable": question_type != "null_query"
                and answer.casefold() != "insufficient information.",
            }
        )
        items.append(item)
    return items


def _load_or_build_embeddings(
    chunks: Sequence[dict[str, Any]],
    cache_path: Path,
    *,
    model_name: str,
) -> np.ndarray:
    expected_count = len(chunks)
    if cache_path.is_file():
        try:
            matrix = np.load(cache_path).astype(np.float32)
            if matrix.ndim == 2 and matrix.shape[0] == expected_count:
                print(
                    f"Loaded embedding cache {cache_path} ({matrix.shape[0]} rows)",
                    flush=True,
                )
                return _normalise_embeddings(matrix)
        except Exception as exc:
            print(f"Ignoring invalid embedding cache {cache_path}: {exc}", flush=True)

    print(f"Embedding {expected_count} chunks for {cache_path}...", flush=True)
    matrix = _normalise_embeddings(embed_chunks(chunks, model_name=model_name))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, matrix)
    return matrix


def _compact(value: object) -> str:
    import re

    return re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", str(value or "").casefold())


def _gold_fact_recall(
    contexts: Sequence[dict[str, Any]], item: dict[str, Any]
) -> tuple[int, int, list[int]]:
    """Return matched gold-fact count, total count, and matched indices."""

    context_text = "\n".join(str(context.get("content") or "") for context in contexts)
    context_tokens = set(tokenize_bm25(context_text))
    matched: list[int] = []
    total = 0
    for evidence_index, evidence in enumerate(item.get("evidence_list") or ()):
        if isinstance(evidence, dict):
            fact = str(evidence.get("fact") or "").strip()
        else:
            fact = str(evidence or "").strip()
        if not fact:
            continue
        total += 1
        fact_tokens = set(tokenize_bm25(fact))
        lexical = len(fact_tokens & context_tokens) / max(1, len(fact_tokens))
        if _compact(fact) in _compact(context_text) or lexical >= 0.45:
            matched.append(evidence_index)
    return len(matched), total, matched


def _compact_contexts(contexts: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep ranking metadata while omitting copyrighted document text."""

    fields = (
        "id",
        "rank",
        "title",
        "page",
        "source",
        "branch",
        "score",
        "bm25Score",
        "embeddingScore",
        "fusionScore",
        "rrfScore",
        "rerankScore",
        "childChunkId",
        "parentChunkId",
    )
    return [{key: context[key] for key in fields if key in context} for context in contexts]


def _first_hit_rank(contexts: Sequence[dict[str, Any]], item: dict[str, Any]) -> int | None:
    """Find the first ranked context containing any gold fact."""

    for context in contexts:
        matched, _, _ = _gold_fact_recall([context], item)
        if matched:
            return int(context.get("rank") or 0) or None
    return None


def _retrieval_metrics(
    *,
    item: dict[str, Any],
    contexts: Sequence[dict[str, Any]],
    retrieval: dict[str, Any],
    retrieval_ms: float,
) -> dict[str, Any]:
    matched, total, matched_indices = _gold_fact_recall(contexts, item)
    first_hit = _first_hit_rank(contexts, item)
    answerable = bool(item.get("answerable"))
    return {
        "gold_fact_matched": matched,
        "gold_fact_total": total,
        "gold_fact_recall": round(matched / total, 6) if total else None,
        "any_gold_fact_hit": bool(matched),
        "complete_gold_evidence_hit": bool(total and matched == total),
        "matched_gold_evidence_indices": matched_indices,
        "first_gold_hit_rank": first_hit,
        "has_gold_evidence": bool(total),
        "answerable": answerable,
        "no_gold_evidence_query": not total,
        "context_count": len(contexts),
        "candidate_count": int(retrieval.get("candidate_count") or 0),
        "rerank_applied": bool(retrieval.get("rerank_applied")),
        "complexity_label": str(
            (retrieval.get("complexity") or {}).get("label") or "unknown"
        ),
        "retrieval_children": int(retrieval.get("retrieval_children") or 0),
        "evidence_parents": int(retrieval.get("evidence_parents") or 0),
        "retrieval_ms": round(float(retrieval_ms), 3),
    }


def _run_version(
    *,
    version: str,
    label: str,
    items: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
    matrix: np.ndarray,
    bm25: Bm25Index,
    embedding_model: str,
    settings: RagConfig,
    optimized: bool,
    parent_index: Any = None,
    cross_encoder: CrossEncoderReranker | None = None,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    checkpoint_every: int = 100,
) -> dict[str, Any]:
    query_cache: dict[str, np.ndarray] = {}
    cases: list[dict[str, Any]] = []
    if resume and checkpoint_path and checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("version") == version:
            cases = list(checkpoint.get("cases") or [])
            print(
                f"Resuming {label} at {len(cases) + 1}/{len(items)} from {checkpoint_path}",
                flush=True,
            )

    if len(cases) > len(items):
        raise ValueError(f"Checkpoint for {version} has more cases than the dataset")

    for position, item in enumerate(items[len(cases) :], start=len(cases) + 1):
        question = str(item.get("query") or "").strip()
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
        metrics = _retrieval_metrics(
            item=item,
            contexts=contexts,
            retrieval=retrieval,
            retrieval_ms=retrieval_ms,
        )
        cases.append(
            {
                "id": item["id"],
                "dataset_index": item["dataset_index"],
                "question_type": item.get("question_type"),
                "question": question,
                "answerable": bool(item.get("answerable")),
                "retrieval": {
                    **retrieval,
                    **metrics,
                },
                "contexts": _compact_contexts(contexts),
            }
        )
        if position == 1 or position % 50 == 0 or position == len(items):
            print(
                f"[{label}] {position}/{len(items)} | "
                f"{metrics['retrieval_ms']:.1f} ms | "
                f"gold {metrics['gold_fact_matched']}/{metrics['gold_fact_total']} | "
                f"{metrics['complexity_label']}",
                flush=True,
            )
        if checkpoint_path and (
            position % max(1, checkpoint_every) == 0 or position == len(items)
        ):
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text(
                json.dumps(
                    {"version": version, "question_count": len(items), "cases": cases},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

    return {
        "name": label,
        "version": version,
        "configuration": {
            "chunking": (
                "hard 600-character chunks, no overlap"
                if not optimized
                else "dynamic 1024-token parent / 256-token child, no overlap"
            ),
            "query_rewrite": (
                "disabled"
                if not optimized
                else "identity for authored English benchmark (semantic guard retained in production)"
            ),
            "hybrid_fusion": (
                "raw 50/50 BM25+dense weighted addition" if not optimized else "RRF, k=60"
            ),
            "candidate_pool": 100,
            "routing": (
                "always basic reranker top 5"
                if not optimized
                else "simple direct top 5; complex Cross-Encoder top 5"
            ),
            "evidence": "retrieval only; no answer model, prompt, memory or generation",
        },
        "summary": _summarize(cases),
        "cases": cases,
    }


def _summarize(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(cases)
    answerable = [case for case in cases if case["answerable"]]
    evidence_cases = [
        case for case in cases if case["retrieval"]["has_gold_evidence"]
    ]
    matched = sum(case["retrieval"]["gold_fact_matched"] for case in cases)
    gold_total = sum(case["retrieval"]["gold_fact_total"] for case in cases)
    latencies = [case["retrieval"]["retrieval_ms"] for case in cases]
    hit_ranks = [
        case["retrieval"]["first_gold_hit_rank"]
        for case in evidence_cases
        if case["retrieval"]["first_gold_hit_rank"] is not None
    ]
    return {
        "question_count": total,
        "answerable_count": len(answerable),
        "null_query_count": total - len(answerable),
        "questions_with_gold_evidence": len(evidence_cases),
        "no_gold_evidence_query_count": sum(
            not case["retrieval"]["has_gold_evidence"] for case in cases
        ),
        "gold_fact_matched": matched,
        "gold_fact_total": gold_total,
        "weighted_gold_fact_recall": round(matched / gold_total, 6)
        if gold_total
        else 0.0,
        "any_gold_fact_hit_rate": round(
            sum(case["retrieval"]["any_gold_fact_hit"] for case in evidence_cases)
            / max(1, len(evidence_cases)),
            6,
        ),
        "complete_gold_evidence_hit_rate": round(
            sum(case["retrieval"]["complete_gold_evidence_hit"] for case in evidence_cases)
            / max(1, len(evidence_cases)),
            6,
        ),
        "first_hit_rate": round(len(hit_ranks) / max(1, len(evidence_cases)), 6),
        "mean_first_hit_rank": round(statistics.mean(hit_ranks), 4) if hit_ranks else None,
        "mean_context_count": round(
            statistics.mean(case["retrieval"]["context_count"] for case in cases), 4
        )
        if cases
        else 0.0,
        "mean_candidate_count": round(
            statistics.mean(case["retrieval"]["candidate_count"] for case in cases), 4
        )
        if cases
        else 0.0,
        "complex_count": sum(
            case["retrieval"]["complexity_label"] == "complex" for case in cases
        ),
        "simple_count": sum(
            case["retrieval"]["complexity_label"] == "simple" for case in cases
        ),
        "rerank_count": sum(case["retrieval"]["rerank_applied"] for case in cases),
        "rerank_rate": round(
            sum(case["retrieval"]["rerank_applied"] for case in cases) / max(1, total),
            6,
        ),
        "average_retrieval_ms": round(statistics.mean(latencies), 3) if latencies else 0.0,
        "p50_retrieval_ms": round(float(np.percentile(latencies, 50)), 3)
        if latencies
        else 0.0,
        "p95_retrieval_ms": round(_p95(latencies), 3),
        "p99_retrieval_ms": round(
            float(np.percentile(latencies, 99)), 3
        )
        if latencies
        else 0.0,
    }


def _breakdown(cases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_type: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_type.setdefault(str(case.get("question_type") or "unknown"), []).append(case)
    rows: list[dict[str, Any]] = []
    for question_type in sorted(by_type):
        grouped = by_type[question_type]
        with_evidence = [
            case for case in grouped if case["retrieval"]["has_gold_evidence"]
        ]
        total_facts = sum(case["retrieval"]["gold_fact_total"] for case in grouped)
        matched_facts = sum(case["retrieval"]["gold_fact_matched"] for case in grouped)
        rows.append(
            {
                "question_type": question_type,
                "question_count": len(grouped),
                "answerable_count": sum(case["answerable"] for case in grouped),
                "gold_evidence_questions": len(with_evidence),
                "weighted_gold_fact_recall": round(matched_facts / total_facts, 6)
                if total_facts
                else None,
                "complete_gold_evidence_hit_rate": round(
                    sum(case["retrieval"]["complete_gold_evidence_hit"] for case in with_evidence)
                    / max(1, len(with_evidence)),
                    6,
                )
                if with_evidence
                else None,
                "any_gold_fact_hit_rate": round(
                    sum(case["retrieval"]["any_gold_fact_hit"] for case in with_evidence)
                    / max(1, len(with_evidence)),
                    6,
                )
                if with_evidence
                else None,
                "rerank_count": sum(case["retrieval"]["rerank_applied"] for case in grouped),
                "average_retrieval_ms": round(
                    statistics.mean(case["retrieval"]["retrieval_ms"] for case in grouped), 3
                ),
                "p95_retrieval_ms": round(
                    _p95([case["retrieval"]["retrieval_ms"] for case in grouped]), 3
                ),
            }
        )
    return rows


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{float(value):.1%}"


def _seconds(value: float | None, decimals: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value) / 1000:.{decimals}f} 秒"


def _render_report(payload: dict[str, Any]) -> str:
    versions = {version["version"]: version for version in payload["versions"]}
    unoptimized = versions["unoptimized"]
    optimized = versions["optimized"]
    us = unoptimized["summary"]
    osummary = optimized["summary"]
    lines = [
        "# MultiHop-RAG 全量檢索評測：無優化版 vs 全優化版",
        "",
        "本報告只執行到證據檢索，沒有建立回答 prompt，也沒有呼叫 Qwen、GPT-5.5 或其他生成模型。",
        "完整題庫的 gold evidence 只在檢索完成後用於計算召回指標；輸出 JSON 僅保留排名與指標，不保存檢索文件全文。",
        "",
        f"- 題庫：`{payload['dataset_total']}` 題 MultiHop-RAG；本次評測 `{payload['evaluated_count']}` 題",
        f"- Corpus：`{payload['corpus_documents']}` 份文件",
        f"- 題型分布：`{payload['question_type_counts']}`",
        f"- Embedding：`{payload['embedding_model']}`",
        f"- Cross-Encoder：`{payload['cross_encoder_model']}`（只在全優化版複雜題啟用）",
        f"- 執行時間：`{payload['run_at']}`",
        "",
        "## 主要結果",
        "",
        "| 版本 | Character recall@5（MultiHop：加權 Gold fact recall） | 任一證據命中 | 平均耗時（秒／題） |",
        "| --- | ---: | ---: | ---: |",
        f"| {unoptimized['name']} | {_pct(us['weighted_gold_fact_recall'])} | {_pct(us['any_gold_fact_hit_rate'])} | {_seconds(us['average_retrieval_ms'])} |",
        f"| {optimized['name']} | {_pct(osummary['weighted_gold_fact_recall'])} | {_pct(osummary['any_gold_fact_hit_rate'])} | {_seconds(osummary['average_retrieval_ms'])} |",
        "",
        "指標分母中的證據題只包含有 gold fact 的題目；null_query 或沒有 gold evidence 的題目另行統計，不把不存在的證據誤算成召回失敗。",
        "",
        "## 差異（全優化版 − 無優化版）",
        "",
        f"- Character recall@5（加權 Gold fact recall）：`{osummary['weighted_gold_fact_recall'] - us['weighted_gold_fact_recall']:+.1%}`",
        f"- 任一證據命中率：`{osummary['any_gold_fact_hit_rate'] - us['any_gold_fact_hit_rate']:+.1%}`",
        f"- 平均耗時差異：`{(osummary['average_retrieval_ms'] - us['average_retrieval_ms']) / 1000:+.3f} 秒`",
        "",
        "## 題型分組",
        "",
        "| 版本 | 題型 | 題數 | Character recall@5（加權 Gold fact recall） | 任一證據命中 | 平均耗時 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for version in (unoptimized, optimized):
        for row in _breakdown(version["cases"]):
            lines.append(
                f"| {version['name']} | `{row['question_type']}` | {row['question_count']} | "
                f"{_pct(row['weighted_gold_fact_recall'])} | {_pct(row['any_gold_fact_hit_rate'])} | "
                f"{_seconds(row['average_retrieval_ms'])} |"
            )
    lines.extend(
        [
            "",
            "## 實作設定",
            "",
            f"- 無優化版：{unoptimized['configuration']['chunking']}；{unoptimized['configuration']['hybrid_fusion']}；{unoptimized['configuration']['routing']}。",
            f"- 全優化版：{optimized['configuration']['chunking']}；{optimized['configuration']['hybrid_fusion']}；{optimized['configuration']['routing']}；命中的子 Chunk 會展開為 parent evidence。",
            "- 兩個版本都使用同一份 corpus、同一個 embedding 模型、同一個 top-5 證據預算與同一批 2,556 題。",
            "- 本次沒有回答模型、沒有對話記憶、沒有外部搜尋、沒有把 gold answer 或 gold evidence 注入檢索查詢。",
            "- Gold fact 命中採完整字串或 BM25 token 覆蓋率至少 45% 的 deterministic heuristic；它衡量檢索召回，不等同於回答正確率或幻覺率。",
            "- MultiHop-RAG 題庫沒有 LegalBench 使用的字元 span 標註，因此本報告的 Character recall@5 欄位以加權 Gold fact recall 對應；LegalBench-RAG 的同名欄位則是字元覆蓋率。",
            "",
        ]
    )
    return "\n".join(lines)


def _summary_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": payload["schema_version"],
        "dataset": payload["dataset"],
        "dataset_total": payload["dataset_total"],
        "evaluated_count": payload["evaluated_count"],
        "corpus_documents": payload["corpus_documents"],
        "question_type_counts": payload["question_type_counts"],
        "run_at": payload["run_at"],
        "embedding_model": payload["embedding_model"],
        "cross_encoder_model": payload["cross_encoder_model"],
        "versions": [
            {
                "name": version["name"],
                "version": version["version"],
                "configuration": version["configuration"],
                "summary": version["summary"],
                "breakdown": _breakdown(version["cases"]),
            }
            for version in payload["versions"]
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    all_items = _load_items(args.queries.resolve())
    items = all_items[: args.limit] if args.limit else all_items
    if not items:
        raise ValueError("--limit must be at least 1 when provided")
    corpus = json.loads(args.corpus.resolve().read_text(encoding="utf-8"))
    if not isinstance(corpus, list) or len(corpus) != 609:
        raise ValueError(
            "MultiHop-RAG corpus must contain 609 documents; "
            f"got {len(corpus) if isinstance(corpus, list) else 'invalid'}"
        )
    print(f"Loaded {len(items)} questions and {len(corpus)} corpus documents", flush=True)

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
    print(
        f"Built {len(hard_chunks)} hard chunks, {len(parent_child.parents)} parents, "
        f"{len(parent_child.children)} children",
        flush=True,
    )
    args.cache_dir.resolve().mkdir(parents=True, exist_ok=True)
    hard_embeddings = _load_or_build_embeddings(
        hard_chunks,
        args.cache_dir.resolve() / "hard-embeddings.npy",
        model_name=args.embedding_model,
    )
    child_embeddings = _load_or_build_embeddings(
        parent_child.children,
        args.cache_dir.resolve() / "child-embeddings.npy",
        model_name=args.embedding_model,
    )
    hard_bm25 = Bm25Index(
        hard_chunks,
        k1=settings.hybrid_bm25_k1,
        b=settings.hybrid_bm25_b,
    )
    child_bm25 = Bm25Index(
        parent_child.children,
        k1=settings.hybrid_bm25_k1,
        b=settings.hybrid_bm25_b,
    )
    cross_encoder = CrossEncoderReranker(args.cross_encoder_model)
    args.checkpoint_dir.resolve().mkdir(parents=True, exist_ok=True)

    unoptimized = _run_version(
        version="unoptimized",
        label="無優化版",
        items=items,
        chunks=hard_chunks,
        matrix=hard_embeddings,
        bm25=hard_bm25,
        embedding_model=args.embedding_model,
        settings=settings,
        optimized=False,
        checkpoint_path=args.checkpoint_dir.resolve() / "unoptimized.json",
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
    )
    optimized = _run_version(
        version="optimized",
        label="全優化版",
        items=items,
        chunks=parent_child.children,
        matrix=child_embeddings,
        bm25=child_bm25,
        embedding_model=args.embedding_model,
        settings=settings,
        optimized=True,
        parent_index=parent_child,
        cross_encoder=cross_encoder,
        checkpoint_path=args.checkpoint_dir.resolve() / "optimized.json",
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
    )

    from collections import Counter

    return {
        "schema_version": "multihop-rag-retrieval-only-v1",
        "dataset": "yixuantt/MultiHopRAG",
        "dataset_total": len(all_items),
        "evaluated_count": len(items),
        "corpus_documents": len(corpus),
        "question_type_counts": dict(
            sorted(Counter(str(item.get("question_type") or "unknown") for item in items).items())
        ),
        "answerable_count": sum(bool(item["answerable"]) for item in items),
        "null_query_count": sum(not bool(item["answerable"]) for item in items),
        "embedding_model": args.embedding_model,
        "cross_encoder_model": args.cross_encoder_model,
        "chunk_counts": {
            "hard": len(hard_chunks),
            "parent": len(parent_child.parents),
            "child": len(parent_child.children),
        },
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "retrieval_only": True,
        "model_calls": 0,
        "versions": [unoptimized, optimized],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional prefix size for a smoke run; omit to evaluate all 2,556 questions.",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--cross-encoder-model", default=DEFAULT_CROSS_ENCODER_MODEL)
    args = parser.parse_args()

    payload = run(args)
    output_path = args.output.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    summary_path = args.summary.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path.write_text(_render_report(payload), encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "report": str(report_path),
                "summary": str(summary_path),
                "model_calls": payload["model_calls"],
                "versions": {
                    version["name"]: version["summary"]
                    for version in payload["versions"]
                },
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
