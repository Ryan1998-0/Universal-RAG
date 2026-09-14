#!/usr/bin/env python3
"""Evaluate the project's unoptimized and optimized retrievers on all 710
public LegalBench-RAG held-out questions.

The public held-out records include a ``document_path``.  This evaluator uses
that value only as a source-document filter, matching the known-document
benchmark setting discussed in the public LegalBench-RAG implementation.  It
does not inject gold spans or answers into the query.  The script stops after
evidence retrieval and writes resumable checkpoints for both versions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_legalbench_public710_sample_eval as public_sample
import run_legalbench_rag_retrieval_full_eval as legacy_eval
from rag_demo.cross_encoder import DEFAULT_CROSS_ENCODER_MODEL, CrossEncoderReranker
from rag_demo.embeddings import DEFAULT_EMBEDDING_MODEL


DEFAULT_ARCHIVE = legacy_eval.DEFAULT_ZIP
DEFAULT_OUTPUT_DIR = ROOT / "evals" / "legalbench_public710_full"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "retrieval-results.json"
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "retrieval-report.md"
DEFAULT_SUMMARY = DEFAULT_OUTPUT_DIR / "retrieval-summary.json"
DEFAULT_CHECKPOINT_DIR = DEFAULT_OUTPUT_DIR / "checkpoints"
DEFAULT_QUERY_CACHE = DEFAULT_OUTPUT_DIR / "cache" / "query-embeddings.npy"

# The hard/child embedding matrices are already built with this project's
# legal embedding model.  Reusing them avoids duplicating roughly 600 MB.
DEFAULT_CHUNK_CACHE_DIR = (
    ROOT / "evals" / "legalbench_rag_retrieval_100_legal_embedding" / "embedding-cache"
)
PUBLIC_MODEL_CARD_URL = (
    "https://huggingface.co/lxyuan/LegalBenchRAG-Ettin-150M-Reranker"
)


def _load_public_items(
    archive_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any], str, str]:
    """Resolve all 710 public query IDs against the local LegalBench archive."""

    all_items, corpus, archive_metadata = public_sample._load_all_archive(archive_path)
    public_ids, public_file_sha256 = public_sample._fetch_public_ids()
    public_document_paths = public_sample._fetch_public_document_paths()
    by_public_id = {
        f"{item['subset']}:{int(item['subset_index']):04d}": item
        for item in all_items
    }
    missing = [query_id for query_id in public_ids if query_id not in by_public_id]
    if missing:
        raise ValueError(f"Public IDs missing from local archive: {missing[:5]}")

    items: list[dict[str, Any]] = []
    for query_id in public_ids:
        item = dict(by_public_id[query_id])
        document_path = str(public_document_paths.get(query_id) or "").strip()
        if not document_path:
            raise ValueError(f"Public document_path missing for {query_id}")
        item["benchmark_document_path"] = document_path
        items.append(item)

    if len(items) != 710:
        raise ValueError(f"Expected 710 public questions, got {len(items)}")
    metadata = {
        **archive_metadata,
        "public_test_count": len(public_ids),
        "public_ids_sha256": public_file_sha256,
        "public_document_count": len(set(public_document_paths.values())),
        "subset_counts": dict(
            sorted(Counter(str(item.get("subset") or "unknown") for item in items).items())
        ),
        "selection_rule": "All 710 query IDs from the public test_predictions.jsonl, sorted lexicographically.",
    }
    return items, corpus, metadata, public_file_sha256, hashlib.sha256(
        archive_path.read_bytes()
    ).hexdigest()


def _format_pct(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "—"


def _format_ms(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "—"


def _add_trimmed_latency(summary: dict[str, Any], cases: Sequence[dict[str, Any]]) -> None:
    """Add a 20-high/20-low trimmed mean for the question-stage latency."""

    if not cases:
        summary["trimmed_average_question_stage_ms"] = None
        summary["trimmed_question_count"] = 0
        summary["trimmed_removed_lowest"] = 0
        summary["trimmed_removed_highest"] = 0
        return

    latencies = sorted(
        float(case.get("timings_ms", {}).get("question_stage", 0.0))
        for case in cases
    )
    trim_each_side = 20
    if len(latencies) > trim_each_side * 2:
        kept = latencies[trim_each_side:-trim_each_side]
    else:
        kept = latencies
    summary["trimmed_average_question_stage_ms"] = round(
        float(np.mean(kept)) if kept else 0.0,
        3,
    )
    summary["trimmed_question_count"] = len(kept)
    summary["trimmed_removed_lowest"] = min(trim_each_side, len(latencies) // 2)
    summary["trimmed_removed_highest"] = min(trim_each_side, len(latencies) // 2)


def _render_report(payload: dict[str, Any]) -> str:
    versions = {version["version"]: version for version in payload["versions"]}
    ordered_versions = [
        versions[version_id]
        for version_id in ("unoptimized", "optimized", "public_ettin")
        if version_id in versions
    ]
    lines = [
        "# LegalBench-RAG 公開 710 題：三種檢索方法比較",
        "",
        "本報告使用公開 held-out 710 題完整測試集，三個版本共用同一批題目與同一份 corpus，只執行證據檢索，不呼叫回答模型。",
        "",
        "## 評測條件",
        "",
        "- 文件範圍：`specified`；每題只在公開測試紀錄提供的 `document_path` 內檢索。這是已知文件的 benchmark 條件。",
        "- 公開題目：710 題；題型分布：" + json.dumps(payload["subset_counts"], ensure_ascii=False),
        f"- Corpus：{payload['corpus_documents']:,} 份文件、{payload['corpus_characters']:,} 字元。",
        f"- Embedding：`{payload['embedding_model']}`。",
        f"- Cross-Encoder：`{payload['cross_encoder_model']}`（優化版與公開 Ettin 版）。",
        f"- 公開版數據來源：[LegalBenchRAG-Ettin-150M-Reranker 模型卡]({PUBLIC_MODEL_CARD_URL})。",
        f"- Reranker device：`{payload.get('reranker_device', 'unknown')}`。",
        f"- 執行時間：`{payload['run_at']}`。",
        "- Gold snippets 只在檢索完成後計算指標，沒有放入查詢；沒有回答模型、對話記憶或外部搜尋。",
        "",
        "## 結果",
        "",
        "自有版本耗時各自移除最高 20 題與最低 20 題，使用剩餘 670 題重新計算；主表保留兩個命中欄位與去除極端值後平均延遲。公開版只引用官方聚合數據，沒有逐題耗時。",
        "",
        "| 版本 | Character recall@5 | 任一證據命中／Hit@5 | 去除最高／最低20題後平均 ms/題 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for version in ordered_versions:
        summary = version["summary"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(version["name"]),
                    _format_pct(summary.get("micro_char_recall")),
                    _format_pct(summary.get("any_gold_span_hit_rate")),
                    _format_ms(summary.get("trimmed_average_question_stage_ms")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 指標定義",
            "",
            "- 任一證據命中：Top 5 中至少涵蓋一個標註證據區段。",
            "- Character recall@5：Top 5 回傳段落覆蓋標註證據字元的比例；自有版本使用本機 exact-span 計算，公開版使用官方模型卡數據。",
            "- 任一證據命中／Hit@5：Top 5 中至少涵蓋一個標註證據區段；公開版沿用官方模型卡的 Hit@5。",
            "",
            "## 題型分組",
            "",
            "| 版本 | 題型 | 題數 | Character recall@5 | 任一證據命中 |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for version in ordered_versions:
        for row in version["breakdown"]:
            lines.append(
                f"| {version['name']} | `{row['subset']}` | {row['question_count']} | "
                f"{_format_pct(row.get('micro_char_recall'))} | "
                f"{_format_pct(row.get('any_gold_span_hit_rate'))} |"
            )
    lines.extend(
        [
            "",
            "## 架構設定",
            "",
            "- 無優化版：硬切 600 字元、無問題改寫、BM25/向量 50/50 直接加權、固定重排 Top 5。",
            "- 優化版：動態 1024-token Parent / 256-token Child、RRF（k=60）、複雜度路由、複雜題 Cross-Encoder 重排 Top 5，命中的 Child 展開為 Parent 證據。",
            "- 公開 Ettin 版：Ettin tokenizer 對齊 384-token 段落、96-token 重疊、BM25 Top 32，再以 LegalBench-RAG Ettin Cross-Encoder 重排 Top 5；本報告直接引用官方 710 題聚合數據。",
            "- 三個版本均限制在同一題的 `document_path`，因此本結果不包含跨文件自動找文件的難度。",
            "- 三個版本都使用同一個 `document_path` 文件範圍；公開 Ettin 版沿用其發表的 BM25 + Cross-Encoder 流程。",
            "",
            "## 檔案",
            "",
            "- 完整逐題結果：[retrieval-results.json](retrieval-results.json)",
            "- 摘要：[retrieval-summary.json](retrieval-summary.json)",
            "- 可續跑 checkpoint：`checkpoints/`",
            "",
        ]
    )
    return "\n".join(lines)


def _summary_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return (
        {
            key: payload[key]
            for key in (
                "schema_version",
                "dataset",
                "public_test_count",
                "public_model_card_url",
                "evaluated_count",
                "subset_counts",
                "public_ids_sha256",
                "archive_sha256",
                "corpus_documents",
                "corpus_characters",
                "chunk_counts",
                "mapping_failures",
                "public_passage_count",
                "public_mapping_failures",
                "embedding_model",
                "cross_encoder_model",
                "reranker_device",
                "run_at",
                "document_scope",
            )
        }
        | {
            "versions": [
                {
                    "name": version["name"],
                    "version": version["version"],
                    "configuration": version["configuration"],
                    "summary": version["summary"],
                    "breakdown": version["breakdown"],
                }
                for version in payload["versions"]
            ]
        }
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    archive_path = args.archive.expanduser().resolve()
    items, corpus, metadata, public_ids_sha256, archive_sha256 = _load_public_items(archive_path)
    print(
        f"Loaded all {len(items)} public questions and {len(corpus)} corpus documents",
        flush=True,
    )
    print(f"Question domains: {metadata['subset_counts']}", flush=True)

    settings = public_sample._settings()
    build_started = time.perf_counter()
    hard_chunks = legacy_eval._build_hard_chunks(corpus)
    parent_child, mapping_failures = legacy_eval._build_parent_child_chunks(corpus)
    build_ms = (time.perf_counter() - build_started) * 1000.0
    print(
        f"Built {len(hard_chunks)} hard chunks, {len(parent_child.parents)} parents, "
        f"{len(parent_child.children)} children; mapping_failures={mapping_failures}",
        flush=True,
    )

    embedding_model = str(args.embedding_model)
    chunk_cache_dir = args.chunk_cache_dir.expanduser().resolve()
    hard_matrix, hard_embedding_ms, hard_cache_hit = legacy_eval._load_or_build_embeddings(
        hard_chunks,
        chunk_cache_dir / "hard-embeddings.npy",
        model_name=embedding_model,
        batch_size=args.batch_size,
        label="hard",
    )
    child_matrix, child_embedding_ms, child_cache_hit = legacy_eval._load_or_build_embeddings(
        parent_child.children,
        chunk_cache_dir / "child-embeddings.npy",
        model_name=embedding_model,
        batch_size=args.batch_size,
        label="child",
    )
    query_matrix, query_embedding_ms, query_cache_hit = legacy_eval._load_or_build_query_embeddings(
        [str(item.get("query") or "") for item in items],
        args.query_cache.expanduser().resolve(),
        model_name=embedding_model,
        batch_size=args.batch_size,
    )

    hard_allowed_indices = public_sample._allowed_indices_by_item(items, hard_chunks)
    child_allowed_indices = public_sample._allowed_indices_by_item(items, parent_child.children)
    if any(not indices for indices in hard_allowed_indices):
        raise ValueError("At least one public question has no hard chunks for its document_path")
    if any(not indices for indices in child_allowed_indices):
        raise ValueError("At least one public question has no child chunks for its document_path")
    print(
        "Document-scoped mode: BM25, dense search, and reranking are restricted "
        "to each question's public document_path",
        flush=True,
    )

    hard_bm25 = legacy_eval.FastBm25Index(
        hard_chunks,
        k1=settings.hybrid_bm25_k1,
        b=settings.hybrid_bm25_b,
    )
    child_bm25 = legacy_eval.FastBm25Index(
        parent_child.children,
        k1=settings.hybrid_bm25_k1,
        b=settings.hybrid_bm25_b,
    )
    hard_token_sets = [
        set(legacy_eval.tokenize_bm25(str(chunk.get("content") or "")))
        for chunk in hard_chunks
    ]

    print("Computing document-scoped dense Top 100 for hard chunks", flush=True)
    hard_dense_hits, hard_dense_ms = public_sample._scoped_dense_hits(
        hard_matrix,
        query_matrix,
        hard_allowed_indices,
        top_k=100,
    )
    print("Computing document-scoped dense Top 100 for child chunks", flush=True)
    child_dense_hits, child_dense_ms = public_sample._scoped_dense_hits(
        child_matrix,
        query_matrix,
        child_allowed_indices,
        top_k=100,
    )

    os.environ["RAG_CROSS_ENCODER_BATCH_SIZE"] = str(max(1, int(args.cross_encoder_batch_size)))
    cross_encoder = CrossEncoderReranker(args.cross_encoder_model)
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    unoptimized = legacy_eval._run_version(
        version="unoptimized",
        label="無優化版",
        items=items,
        chunks=hard_chunks,
        dense_hits=hard_dense_hits,
        bm25=hard_bm25,
        settings=settings,
        optimized=False,
        chunk_token_sets=hard_token_sets,
        dense_search_ms=hard_dense_ms,
        checkpoint_path=checkpoint_dir / "unoptimized.json",
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        allowed_indices_by_item=hard_allowed_indices,
    )
    optimized = legacy_eval._run_version(
        version="optimized",
        label="優化版",
        items=items,
        chunks=parent_child.children,
        dense_hits=child_dense_hits,
        bm25=child_bm25,
        settings=settings,
        optimized=True,
        parents=parent_child.parents,
        cross_encoder=cross_encoder,
        dense_search_ms=child_dense_ms,
        checkpoint_path=checkpoint_dir / "optimized.json",
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        allowed_indices_by_item=child_allowed_indices,
    )

    # The public implementation is reported from its official 710-question
    # aggregate results.  The model card reports Hit@5, Character recall@5,
    # NDCG@10 and MRR@10, but does not publish per-question timings or the
    # complete-span/Rank-1 fields used by this local evaluator.  Do not run a
    # second local 710-question Ettin pass merely to manufacture incompatible
    # timing or metric values.
    public_ettin = {
        "name": "公開 Ettin 版（官方數據）",
        "version": "public_ettin",
        "configuration": (
            "Ettin tokenizer 對齊 384-token 段落、96-token 重疊；BM25 Top 32；"
            "LegalBench-RAG Ettin Cross-Encoder Top 5"
        ),
        "summary": {
            "question_count": 710,
            "macro_char_recall": None,
            "macro_char_precision": None,
            "micro_char_recall": 0.8041,
            "micro_char_precision": None,
            "any_gold_span_hit_rate": 0.8676,
            "complete_gold_span_hit_rate": None,
            "rank1_gold_span_hit_rate": None,
            "first_hit_rate": None,
            "mean_first_hit_rank": None,
            "mean_context_count": None,
            "rerank_count": None,
            "rerank_rate": None,
            "average_question_stage_ms": None,
            "trimmed_average_question_stage_ms": None,
            "trimmed_question_count": 0,
            "trimmed_removed_lowest": 0,
            "trimmed_removed_highest": 0,
            "published_ndcg10": 0.7424,
            "published_mrr10": 0.8191,
            "published_source": PUBLIC_MODEL_CARD_URL,
        },
        "breakdown": [],
        "cases": [],
    }

    versions = [unoptimized, optimized, public_ettin]
    for version in (unoptimized, optimized):
        _add_trimmed_latency(version["summary"], version["cases"])

    model = getattr(cross_encoder, "_model", None)
    if model is not None:
        reranker_device = str(getattr(model, "device", None) or "unknown")
    else:
        configured_device = str(os.getenv("RAG_RERANKER_DEVICE") or "").strip().lower()
        if configured_device and configured_device != "auto":
            reranker_device = configured_device
        else:
            try:
                import torch

                reranker_device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                reranker_device = "unknown"
    payload = {
        "schema_version": "legalbench-rag-public710-three-way-v1",
        "dataset": "ZeroEntropy-AI/legalbenchrag",
        "public_test_count": 710,
        "public_model_card_url": PUBLIC_MODEL_CARD_URL,
        "evaluated_count": len(items),
        "subset_counts": metadata["subset_counts"],
        "public_ids_sha256": public_ids_sha256,
        "archive_sha256": archive_sha256,
        "corpus_documents": len(corpus),
        "corpus_characters": sum(len(text) for text in corpus.values()),
        "chunk_counts": {
            "hard": len(hard_chunks),
            "parent": len(parent_child.parents),
            "child": len(parent_child.children),
        },
        "public_passage_count": None,
        "public_mapping_failures": None,
        "mapping_failures": mapping_failures,
        "build_ms": round(build_ms, 3),
        "public_passage_build_ms": None,
        "embedding_ms": {
            "hard": round(hard_embedding_ms, 3),
            "child": round(child_embedding_ms, 3),
            "queries": round(query_embedding_ms, 3),
        },
        "embedding_cache_hit": {
            "hard": hard_cache_hit,
            "child": child_cache_hit,
            "queries": query_cache_hit,
        },
        "embedding_model": embedding_model,
        "cross_encoder_model": args.cross_encoder_model,
        "reranker_device": reranker_device,
        "batch_size": int(args.batch_size),
        "dense_batch_size": 0,
        "cross_encoder_batch_size": int(args.cross_encoder_batch_size),
        "retrieval_only": True,
        "model_calls": 0,
        "document_scope": "specified",
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "versions": versions,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--query-cache", type=Path, default=DEFAULT_QUERY_CACHE)
    parser.add_argument("--chunk-cache-dir", type=Path, default=DEFAULT_CHUNK_CACHE_DIR)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cross-encoder-batch-size", type=int, default=64)
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
                "question_count": payload["evaluated_count"],
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
