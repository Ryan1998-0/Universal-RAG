#!/usr/bin/env python3
"""Evaluate the published LegalBench-RAG retrieval architecture.

This evaluator intentionally follows the public LegalBench-RAG reranker
recipe rather than the project's previous parent-child/RRF experiment:

* tokenizer-aligned 384-token passages with 96-token overlap;
* BM25 first-stage retrieval of 32 candidates;
* the published ``lxyuan/LegalBenchRAG-Ettin-150M-Reranker`` Cross-Encoder;
* final Top 5 evidence passages.

Two retrieval-only branches are measured on the same first 100 paper-aligned
questions.  The baseline keeps BM25 Top 5 directly; the optimized branch
reranks the BM25 Top 32 with the published Cross-Encoder.  No answer model,
conversation memory, external search, or gold evidence is used during
retrieval.  Gold snippets are consulted only after retrieval for metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rag_demo.cross_encoder import (
    LEGALBENCH_RERANKER_MODEL,
    CrossEncoderReranker,
)
from transformers import PreTrainedTokenizerBase

from run_legalbench_rag_retrieval_full_eval import (
    DEFAULT_ZIP,
    PUBLISHED_COUNTS,
    FastBm25Index,
    _compact_context,
    _load_archive,
)


DEFAULT_OUTPUT = ROOT / "evals" / "legalbench_rag_retrieval_100_ettin" / "retrieval-results.json"
DEFAULT_REPORT = ROOT / "evals" / "legalbench_rag_retrieval_100_ettin" / "retrieval-report.md"
DEFAULT_SUMMARY = ROOT / "evals" / "legalbench_rag_retrieval_100_ettin" / "retrieval-summary.json"
DEFAULT_CHECKPOINT_DIR = ROOT / "evals" / "legalbench_rag_retrieval_100_ettin" / "checkpoints"

PASSAGE_TOKENS = 384
PASSAGE_OVERLAP_TOKENS = 96
BM25_CANDIDATE_K = 32
FINAL_TOP_K = 5


def _interval_union_length(intervals: Sequence[tuple[int, int]]) -> int:
    merged: list[tuple[int, int]] = []
    for start, end in sorted((int(start), int(end)) for start, end in intervals if end > start):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return sum(end - start for start, end in merged)


def _overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> int:
    return max(0, min(left_end, right_end) - max(left_start, right_start))


def _span_metrics(
    contexts: Sequence[dict[str, Any]],
    snippets: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Compute exact-span metrics without double-counting passage overlap."""

    gold = [
        (
            str(snippet.get("file_path") or ""),
            int((snippet.get("span") or [0, 0])[0]),
            int((snippet.get("span") or [0, 0])[1]),
        )
        for snippet in snippets
    ]
    retrieved = [
        (
            str(context.get("file_path") or ""),
            int(context.get("start_char") or 0),
            int(context.get("end_char") or 0),
        )
        for context in contexts
    ]
    total_gold = sum(max(0, end - start) for _, start, end in gold)
    total_retrieved = sum(
        _interval_union_length(
            [(start, end) for file_path, start, end in retrieved if file_path == selected_file]
        )
        for selected_file in {file_path for file_path, _, _ in retrieved}
    )
    relevant_retrieved = 0
    gold_hit_flags: list[bool] = []
    for gold_file, gold_start, gold_end in gold:
        clipped = [
            (max(start, gold_start), min(end, gold_end))
            for file_path, start, end in retrieved
            if file_path == gold_file and _overlap(start, end, gold_start, gold_end) > 0
        ]
        relevant_retrieved += _interval_union_length(clipped)
        gold_hit_flags.append(bool(clipped))
    first_hit_rank = None
    for context in contexts:
        context_file = str(context.get("file_path") or "")
        context_start = int(context.get("start_char") or 0)
        context_end = int(context.get("end_char") or 0)
        if any(
            context_file == gold_file
            and _overlap(context_start, context_end, gold_start, gold_end) > 0
            for gold_file, gold_start, gold_end in gold
        ):
            first_hit_rank = int(context.get("rank") or 0) or None
            break
    first_context_hit = bool(
        contexts
        and any(
            str(contexts[0].get("file_path") or "") == gold_file
            and _overlap(
                int(contexts[0].get("start_char") or 0),
                int(contexts[0].get("end_char") or 0),
                gold_start,
                gold_end,
            )
            > 0
            for gold_file, gold_start, gold_end in gold
        )
    )
    return {
        "gold_span_count": len(gold),
        "gold_characters": total_gold,
        "retrieved_characters": total_retrieved,
        "relevant_retrieved_characters": relevant_retrieved,
        "char_recall": round(relevant_retrieved / total_gold, 8) if total_gold else 0.0,
        "char_precision": round(relevant_retrieved / total_retrieved, 8) if total_retrieved else 0.0,
        "any_gold_span_hit": bool(any(gold_hit_flags)),
        "complete_gold_span_hit": bool(gold_hit_flags and all(gold_hit_flags)),
        "rank1_gold_span_hit": first_context_hit,
        "first_gold_hit_rank": first_hit_rank,
    }


def _build_passages(
    corpus: dict[str, str],
    *,
    tokenizer: PreTrainedTokenizerBase,
    passage_tokens: int = PASSAGE_TOKENS,
    overlap_tokens: int = PASSAGE_OVERLAP_TOKENS,
) -> tuple[list[dict[str, Any]], int]:
    """Build tokenizer-aligned 384/96 passages with exact source offsets.

    The public model card defines passage size and overlap in model tokens.
    Using the Ettin tokenizer here avoids the character/word approximation
    used by the interactive ingestion path and keeps the Cross-Encoder input
    shape comparable with the published experiment.
    """

    passages: list[dict[str, Any]] = []
    mapping_failures = 0
    stride = max(1, int(passage_tokens) - int(overlap_tokens))
    for file_index, file_path in enumerate(sorted(corpus), start=1):
        raw_body = str(corpus[file_path] or "")
        if not raw_body.strip():
            continue
        encoded = tokenizer(
            raw_body,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        raw_offsets = encoded.get("offset_mapping") or []
        offsets = [
            (int(pair[0]), int(pair[1]))
            for pair in raw_offsets
            if len(pair) == 2 and int(pair[1]) > int(pair[0])
        ]
        if not offsets:
            mapping_failures += 1
            continue
        for passage_index, token_start in enumerate(range(0, len(offsets), stride), start=1):
            token_end = min(len(offsets), token_start + int(passage_tokens))
            char_start = max(0, offsets[token_start][0])
            char_end = min(len(raw_body), offsets[token_end - 1][1])
            if char_end <= char_start:
                mapping_failures += 1
                continue
            passages.append(
                {
                    "id": f"legalbench-passage-{file_index:04d}-{passage_index:04d}",
                    "title": file_path,
                    "page": file_path,
                    "source": file_path,
                    "source_id": file_path,
                    "file_path": file_path,
                    "content": raw_body[char_start:char_end],
                    "start_char": int(char_start),
                    "end_char": int(char_end),
                    "chunk_level": "passage",
                    "passage_tokens": int(passage_tokens),
                    "overlap_tokens": int(overlap_tokens),
                }
            )
    return passages, mapping_failures


def _contexts_from_hits(
    hits: Sequence[dict[str, Any]],
    passages: Sequence[dict[str, Any]],
    *,
    branch: str,
) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=1):
        passage = passages[int(hit["index"])]
        rerank_score = hit.get("rerank_score")
        score = rerank_score if rerank_score is not None else hit.get("bm25_score", 0.0)
        contexts.append(
            {
                "id": str(passage.get("id") or hit["index"]),
                "rank": rank,
                "title": str(passage.get("title") or "LegalBench-RAG"),
                "page": str(passage.get("page") or ""),
                "source": str(passage.get("source_id") or passage.get("source") or ""),
                "file_path": str(passage.get("file_path") or ""),
                "start_char": int(passage.get("start_char") or 0),
                "end_char": int(passage.get("end_char") or 0),
                "content": str(passage.get("content") or ""),
                "branch": branch,
                "score": round(float(score or 0.0), 8),
                "bm25Score": round(float(hit.get("bm25_score", 0.0)), 8),
                "rerankScore": round(float(rerank_score or 0.0), 8),
                "matchedTerms": list(hit.get("matched_terms") or []),
            }
        )
    return contexts


def _retrieve_bm25(
    question: str,
    passages: Sequence[dict[str, Any]],
    bm25: FastBm25Index,
    allowed_indices: Sequence[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = bm25.search(
        question,
        top_k=BM25_CANDIDATE_K,
        allowed_indices=allowed_indices,
    )
    selected = [
        {
            **candidate,
            "bm25_score": float(candidate.get("score", 0.0)),
        }
        for candidate in candidates[:FINAL_TOP_K]
    ]
    return _contexts_from_hits(selected, passages, branch="bm25-direct"), {
        "first_stage": "BM25",
        "candidate_pool": BM25_CANDIDATE_K,
        "candidate_count": len(candidates),
        "rerank_applied": False,
        "reranker": None,
        "final_top_k": FINAL_TOP_K,
    }


def _retrieve_ettin(
    question: str,
    passages: Sequence[dict[str, Any]],
    bm25: FastBm25Index,
    reranker: CrossEncoderReranker,
    allowed_indices: Sequence[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = bm25.search(
        question,
        top_k=BM25_CANDIDATE_K,
        allowed_indices=allowed_indices,
    )
    candidate_records = [
        {
            **candidate,
            "bm25_score": float(candidate.get("score", 0.0)),
        }
        for candidate in candidates
    ]
    documents = [str(passages[int(candidate["index"])].get("content") or "") for candidate in candidate_records]
    scores = reranker.score(question, documents) if documents else []
    if len(scores) != len(candidate_records):
        raise RuntimeError(
            "LegalBench-RAG Cross-Encoder returned a different number of scores: "
            f"{len(scores)} for {len(candidate_records)} candidates"
        )
    for candidate, score in zip(candidate_records, scores):
        candidate["rerank_score"] = float(score)
    candidate_records.sort(
        key=lambda candidate: (
            float(candidate.get("rerank_score", 0.0)),
            float(candidate.get("bm25_score", 0.0)),
        ),
        reverse=True,
    )
    return _contexts_from_hits(
        candidate_records[:FINAL_TOP_K],
        passages,
        branch="bm25-ettin-rerank",
    ), {
        "first_stage": "BM25",
        "candidate_pool": BM25_CANDIDATE_K,
        "candidate_count": len(candidates),
        "rerank_applied": True,
        "reranker": LEGALBENCH_RERANKER_MODEL,
        "final_top_k": FINAL_TOP_K,
    }


def _summarize(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        return {"question_count": 0}
    metrics = [case["metrics"] for case in cases]
    latencies = [float(case["timings_ms"]["question_stage"]) for case in cases]
    hit_ranks = [
        int(metric["first_gold_hit_rank"])
        for metric in metrics
        if metric.get("first_gold_hit_rank") is not None
    ]
    total_gold = sum(int(metric["gold_characters"]) for metric in metrics)
    total_retrieved = sum(int(metric["retrieved_characters"]) for metric in metrics)
    total_relevant = sum(int(metric["relevant_retrieved_characters"]) for metric in metrics)
    rerank_count = sum(bool(case["retrieval"].get("rerank_applied")) for case in cases)
    return {
        "question_count": len(cases),
        "macro_char_recall": round(statistics.mean(metric["char_recall"] for metric in metrics), 8),
        "macro_char_precision": round(statistics.mean(metric["char_precision"] for metric in metrics), 8),
        "micro_char_recall": round(total_relevant / total_gold, 8) if total_gold else 0.0,
        "micro_char_precision": round(total_relevant / total_retrieved, 8) if total_retrieved else 0.0,
        "any_gold_span_hit_rate": round(
            sum(bool(metric["any_gold_span_hit"]) for metric in metrics) / len(metrics),
            8,
        ),
        "complete_gold_span_hit_rate": round(
            sum(bool(metric["complete_gold_span_hit"]) for metric in metrics) / len(metrics),
            8,
        ),
        "rank1_gold_span_hit_rate": round(
            sum(bool(metric["rank1_gold_span_hit"]) for metric in metrics) / len(metrics),
            8,
        ),
        "first_hit_rate": round(len(hit_ranks) / len(metrics), 8),
        "mean_first_hit_rank": round(statistics.mean(hit_ranks), 4) if hit_ranks else None,
        "mean_context_count": round(statistics.mean(len(case["contexts"]) for case in cases), 4),
        "rerank_count": rerank_count,
        "rerank_rate": round(rerank_count / len(cases), 8),
        "average_question_stage_ms": round(statistics.mean(latencies), 3),
        "p50_question_stage_ms": round(float(__import__("numpy").percentile(latencies, 50)), 3),
        "p95_question_stage_ms": round(float(__import__("numpy").percentile(latencies, 95)), 3),
        "p99_question_stage_ms": round(float(__import__("numpy").percentile(latencies, 99)), 3),
        "overall_retrieval_ms_per_question": round(statistics.mean(latencies), 3),
    }


def _breakdown(cases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[str(case.get("subset") or "unknown")].append(case)
    return [
        {"subset": subset, **_summarize(group)}
        for subset, group in sorted(grouped.items())
    ]


def _run_version(
    *,
    version: str,
    label: str,
    items: Sequence[dict[str, Any]],
    passages: Sequence[dict[str, Any]],
    bm25: FastBm25Index,
    reranker: CrossEncoderReranker | None,
    checkpoint_path: Path,
    resume: bool,
    checkpoint_every: int,
    allowed_indices_by_item: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    if resume and checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("version") == version:
            cases = list(checkpoint.get("cases") or [])
            print(f"Resuming {label}: {len(cases)}/{len(items)}", flush=True)
    if len(cases) > len(items):
        raise ValueError(f"Checkpoint for {version} has more cases than the dataset")

    for position, item in enumerate(items[len(cases) :], start=len(cases) + 1):
        question = str(item.get("query") or "").strip()
        allowed_indices = (
            allowed_indices_by_item[position - 1]
            if allowed_indices_by_item is not None
            else None
        )
        started = time.perf_counter()
        if reranker is None:
            contexts, retrieval = _retrieve_bm25(
                question,
                passages,
                bm25,
                allowed_indices=allowed_indices,
            )
        else:
            contexts, retrieval = _retrieve_ettin(
                question,
                passages,
                bm25,
                reranker,
                allowed_indices=allowed_indices,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        metrics = _span_metrics(contexts, item.get("snippets") or [])
        cases.append(
            {
                "id": item["id"],
                "subset": item.get("subset"),
                "subset_index": item.get("subset_index"),
                "question": question,
                "retrieval": retrieval,
                "metrics": {**metrics, "question_stage_ms": round(elapsed_ms, 3)},
                "timings_ms": {"question_stage": round(elapsed_ms, 3)},
                "contexts": [_compact_context(context) for context in contexts],
            }
        )
        if position == 1 or position % 10 == 0 or position == len(items):
            print(
                f"[{label}] {position}/{len(items)} | {elapsed_ms:.1f} ms | "
                f"char recall {metrics['char_recall']:.3f} | any {metrics['any_gold_span_hit']}",
                flush=True,
            )
        if position % max(1, int(checkpoint_every)) == 0 or position == len(items):
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text(
                json.dumps(
                    {"version": version, "question_count": len(items), "cases": cases},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

    if reranker is None:
        configuration = "384-token tokenizer-aligned passages with 96-token overlap; BM25 Top 32; direct Top 5"
    else:
        configuration = "384-token tokenizer-aligned passages with 96-token overlap; BM25 Top 32; LegalBench-RAG Ettin Cross-Encoder Top 5"
    return {
        "name": label,
        "version": version,
        "configuration": configuration,
        "summary": _summarize(cases),
        "breakdown": _breakdown(cases),
        "cases": cases,
    }


def _render_report(payload: dict[str, Any]) -> str:
    versions = {version["version"]: version for version in payload["versions"]}
    baseline = versions["bm25_baseline"]
    optimized = versions["ettin_optimized"]
    bs = baseline["summary"]
    osummary = optimized["summary"]
    lines = [
        "# LegalBench-RAG 發表方法：100 題檢索重跑",
        "",
        "本次將優化版改為公開 LegalBench-RAG 方法：以 Ettin tokenizer 切 384-token 段落、96-token 重疊、BM25 Top 32，再以 LegalBench-RAG 微調的 Ettin 150M Cross-Encoder 重排並取 Top 5。",
        "",
        "參考：[LegalBenchRAG-Ettin-150M-Reranker 模型卡](https://huggingface.co/lxyuan/LegalBenchRAG-Ettin-150M-Reranker)｜[LegalBench-RAG 論文](https://arxiv.org/abs/2408.10343)",
        "",
        "## 評測範圍",
        "",
        f"- 論文對齊題集：{payload['full_evaluated_count']:,} 題；本次取壓縮檔順序前 {payload['evaluated_count']} 題。",
        f"- 本次 100 題皆屬 ContractNLI；另外 {payload['sample_excluded_count']:,} 題未納入本次樣本。",
        f"- Corpus：{payload['corpus_documents']:,} 份文件、{payload['corpus_characters']:,} 字元；建立 {payload['passage_count']:,} 個 passages。",
        "- 只執行證據檢索；沒有回答模型、對話記憶、外部搜尋或 Gold evidence 注入。",
        "",
        "## 主要結果",
        "",
        "| 版本 | Macro char recall | Micro char recall | 任一證據命中 | 完整證據命中 | Rank 1 命中 | 平均 ms/題 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {baseline['name']} | {bs['macro_char_recall']:.2%} | {bs['micro_char_recall']:.2%} | {bs['any_gold_span_hit_rate']:.2%} | {bs['complete_gold_span_hit_rate']:.2%} | {bs['rank1_gold_span_hit_rate']:.2%} | {bs['average_question_stage_ms']:.1f} |",
        f"| {optimized['name']} | {osummary['macro_char_recall']:.2%} | {osummary['micro_char_recall']:.2%} | {osummary['any_gold_span_hit_rate']:.2%} | {osummary['complete_gold_span_hit_rate']:.2%} | {osummary['rank1_gold_span_hit_rate']:.2%} | {osummary['average_question_stage_ms']:.1f} |",
        "",
        "Macro char recall 是逐題字元召回率平均；Micro char recall 是所有題目的 Gold 字元合併後計算。任一證據命中表示 Top 5 至少與一個 Gold span 重疊；完整證據命中表示該題每個 Gold span 都被命中；Rank 1 命中只檢查第一個證據。",
        "",
        "## 題型分組",
        "",
        "| 版本 | 題型 | 題數 | Macro recall | 任一命中 | 完整命中 | 平均 ms/題 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for version in (baseline, optimized):
        for row in version["breakdown"]:
            lines.append(
                f"| {version['name']} | `{row['subset']}` | {row['question_count']} | "
                f"{row['macro_char_recall']:.2%} | {row['any_gold_span_hit_rate']:.2%} | "
                f"{row['complete_gold_span_hit_rate']:.2%} | {row['average_question_stage_ms']:.1f} |"
            )
    lines.extend(
        [
            "",
            "## 可比性說明",
            "",
            "- 無優化版是同一段落切分與 BM25 Top 32 的直接 Top 5 基線；優化版只增加發表的 Ettin Cross-Encoder 重排，便於隔離重排器的影響。",
            "- 本次評測直接使用 Ettin tokenizer 的 offset mapping 建立段落，讓段落邊界與 Cross-Encoder 的 384-token 設定一致。",
            "- 模型卡的 86.76% Hit@5 與 80.41% Character recall@5 是 710 題 held-out 評測；本報告是本機 archive 前 100 題，不能直接視為同一統計結果。",
            "",
            "## 輸出",
            "",
            "- 逐題結果 JSON 同時保存排名、來源與字元區間，不保存 passage 全文。",
            "- checkpoint 可用 `--resume` 接續；本報告沒有呼叫生成模型。",
            "",
        ]
    )
    return "\n".join(lines)


def _summary_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: payload[key]
        for key in (
            "schema_version",
            "dataset",
            "published_question_count",
            "archive_question_count",
            "evaluated_count",
            "full_evaluated_count",
            "sample_excluded_count",
            "sample_limit",
            "excluded_count",
            "subset_counts",
            "corpus_documents",
            "corpus_characters",
            "passage_count",
            "mapping_failures",
            "passage_tokens",
            "passage_overlap_tokens",
            "bm25_candidate_k",
            "final_top_k",
            "run_at",
            "reranker_model",
            "reranker_device",
        )
    } | {
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    archive_path = args.archive.resolve()
    items, corpus, archive_metadata = _load_archive(archive_path)
    full_evaluated_count = len(items)
    full_counts = dict(archive_metadata.get("evaluated_counts") or {})
    sample_limit = int(args.limit) if int(args.limit or 0) > 0 else None
    if sample_limit is not None:
        items = items[:sample_limit]
        sample_counts = Counter(str(item.get("subset") or "unknown") for item in items)
        archive_metadata = {
            **archive_metadata,
            "full_evaluated_total": full_evaluated_count,
            "full_evaluated_counts": full_counts,
            "evaluated_total": len(items),
            "evaluated_counts": dict(sorted(sample_counts.items())),
            "sample_limit": len(items),
            "sample_excluded_count": full_evaluated_count - len(items),
        }
    else:
        archive_metadata = {
            **archive_metadata,
            "full_evaluated_total": full_evaluated_count,
            "full_evaluated_counts": full_counts,
            "sample_limit": None,
            "sample_excluded_count": 0,
        }
    print(
        f"Loaded {len(items)} paper-aligned questions and {len(corpus)} corpus documents from {archive_path}",
        flush=True,
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.reranker_model, use_fast=True)
    started = time.perf_counter()
    passages, mapping_failures = _build_passages(corpus, tokenizer=tokenizer)
    build_ms = (time.perf_counter() - started) * 1000.0
    print(
        f"Built {len(passages)} passages ({PASSAGE_TOKENS}/{PASSAGE_OVERLAP_TOKENS} tokens), "
        f"mapping_failures={mapping_failures}, {build_ms / 1000.0:.1f}s",
        flush=True,
    )
    bm25_started = time.perf_counter()
    bm25 = FastBm25Index(passages)
    bm25_build_ms = (time.perf_counter() - bm25_started) * 1000.0
    print(f"Built BM25 index in {bm25_build_ms / 1000.0:.1f}s", flush=True)

    try:
        batch_size = max(1, int(args.batch_size))
    except (TypeError, ValueError):
        batch_size = 64
    os.environ["RAG_CROSS_ENCODER_BATCH_SIZE"] = str(batch_size)
    reranker = CrossEncoderReranker(args.reranker_model)
    checkpoint_dir = args.checkpoint_dir.resolve()
    baseline = _run_version(
        version="bm25_baseline",
        label="無優化版（BM25）",
        items=items,
        passages=passages,
        bm25=bm25,
        reranker=None,
        checkpoint_path=checkpoint_dir / "bm25-baseline.json",
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
    )
    optimized = _run_version(
        version="ettin_optimized",
        label="LegalBench-RAG（Ettin）",
        items=items,
        passages=passages,
        bm25=bm25,
        reranker=reranker,
        checkpoint_path=checkpoint_dir / "ettin-optimized.json",
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
    )

    reranker_device = "unknown"
    model = getattr(reranker, "_model", None)
    if model is not None:
        reranker_device = str(getattr(model, "device", None) or "unknown")
    return {
        "schema_version": "legalbench-rag-ettin-retrieval-only-v1",
        "dataset": "ZeroEntropy-AI/legalbenchrag",
        "published_question_count": sum(PUBLISHED_COUNTS.values()),
        "archive_question_count": archive_metadata["downloaded_total"],
        "evaluated_count": len(items),
        "full_evaluated_count": archive_metadata["full_evaluated_total"],
        "sample_excluded_count": archive_metadata["sample_excluded_count"],
        "sample_limit": archive_metadata["sample_limit"],
        "excluded_count": archive_metadata["downloaded_total"] - archive_metadata["full_evaluated_total"],
        "subset_counts": archive_metadata["evaluated_counts"],
        "corpus_documents": len(corpus),
        "corpus_characters": sum(len(text) for text in corpus.values()),
        "passage_count": len(passages),
        "mapping_failures": mapping_failures,
        "passage_tokens": PASSAGE_TOKENS,
        "passage_overlap_tokens": PASSAGE_OVERLAP_TOKENS,
        "bm25_candidate_k": BM25_CANDIDATE_K,
        "final_top_k": FINAL_TOP_K,
        "build_ms": round(build_ms, 3),
        "bm25_build_ms": round(bm25_build_ms, 3),
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "reranker_model": args.reranker_model,
        "reranker_device": reranker_device,
        "batch_size": batch_size,
        "retrieval_only": True,
        "model_calls": 0,
        "versions": [baseline, optimized],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--reranker-model", default=LEGALBENCH_RERANKER_MODEL)
    args = parser.parse_args()

    payload = run(args)
    output_path = args.output.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    summary_path = args.summary.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(_render_report(payload), encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "report": str(report_path),
                "summary": str(summary_path),
                "dataset_count": payload["evaluated_count"],
                "model_calls": payload["model_calls"],
                "versions": {version["name"]: version["summary"] for version in payload["versions"]},
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
