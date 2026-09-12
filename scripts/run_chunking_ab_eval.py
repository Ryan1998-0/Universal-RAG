#!/usr/bin/env python3
"""Compare hard and sentence-aware chunking on the bundled 20-question set.

This is a retrieval upper-bound test: it scores whether the top retrieved
contexts contain each question's labelled focus fact. No answer model is
called, so the result isolates the impact of chunk boundaries.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from rag_demo.chunk_strategies import (
    CHUNK_STRATEGY_DYNAMIC,
    CHUNK_STRATEGY_HARD,
    estimate_token_count,
    split_text,
)
from rag_demo.config import RagConfig
from rag_demo.embeddings import DEFAULT_EMBEDDING_MODEL, embed_chunks
from rag_demo.hybrid_retrieval import HybridRetriever


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "evals" / "taiwan_law_versions" / "questions_20.json"
DEFAULT_SOURCE_2016 = ROOT / "evals" / "taiwan_law_versions" / "source" / "勞工請假規則_2016有效版.txt"
DEFAULT_SOURCE_2026 = ROOT / "evals" / "taiwan_law_versions" / "source" / "勞工請假規則_2026有效版.txt"
DEFAULT_OUTPUT = ROOT / "evals" / "chunking_ab" / "latest-results.json"
DEFAULT_REPORT = ROOT / "evals" / "chunking_ab" / "latest-report.md"


def normalize(text: object) -> str:
    return re.sub(r"\s+", "", str(text or "")).casefold()


def load_benchmark(path: Path, limit: int | None) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError("Benchmark must contain an items list.")
    return items[:limit] if limit is not None else items


def build_chunks(source_id: str, text: str, strategy: str) -> list[dict]:
    chunks = []
    pieces = list(
        split_text(
            text,
            chunk_size=600,
            chunk_stride=600,
            strategy=strategy,
            overlap_tokens=200,
        )
    )
    for index, piece in enumerate(pieces):
        chunks.append(
            {
                "id": f"{source_id}::{index}",
                "source_id": source_id,
                "source": source_id,
                "chunk_index": index,
                "parent_title": source_id,
                "title": f"{source_id} / chunk {index + 1}",
                "content": piece,
            }
        )
    return chunks


def score_contexts(contexts: list[dict], item: dict) -> dict:
    context_text = normalize(
        "\n".join(
            f"{context.get('title', '')}\n{context.get('content', '')}"
            for context in contexts
        )
    )
    facts = [str(fact) for fact in item.get("focus_facts", []) if str(fact).strip()]
    matched = [fact for fact in facts if normalize(fact) in context_text]
    missed = [fact for fact in facts if normalize(fact) not in context_text]
    return {
        "matched_facts": matched,
        "missed_facts": missed,
        "fact_score": len(matched),
        "fact_max": len(facts),
        "perfect": bool(facts) and not missed,
    }


def run_variant(
    name: str,
    label: str,
    strategy: str,
    source_texts: list[tuple[str, str]],
    questions: list[dict],
    embedding_model: str,
    top_k: int,
    candidate_k: int,
) -> dict:
    chunks = [
        chunk
        for source_id, text in source_texts
        for chunk in build_chunks(source_id, text, strategy)
    ]
    embedding_started = time.perf_counter()
    embeddings = embed_chunks(chunks, model_name=embedding_model)
    embedding_ms = (time.perf_counter() - embedding_started) * 1000.0
    settings = replace(
        RagConfig.from_env().normalized(),
        multi_query_enabled=False,
        # Keep this benchmark focused on chunk boundaries.  The production
        # query-complexity gate is evaluated separately and would cap simple
        # questions at three contexts, confounding the A/B comparison.
        complexity_routing_enabled=False,
        hybrid_top_k=top_k,
        hybrid_candidate_k=candidate_k,
    ).normalized()
    retriever = HybridRetriever(
        chunks=chunks,
        aliases=[],
        embeddings=embeddings,
        embedding_model=embedding_model,
        settings=settings,
    )

    results = []
    for item in questions:
        started = time.perf_counter()
        response = retriever.retrieve(
            question=str(item["question"]),
            top_k=top_k,
            candidate_k=candidate_k,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        contexts = response.get("contexts") or []
        score = score_contexts(contexts, item)
        results.append(
            {
                "id": item.get("id", ""),
                "question": item.get("question", ""),
                "latency_ms": round(latency_ms, 3),
                "retrieved_context_ids": [context.get("id", "") for context in contexts],
                "retrieved_sources": [context.get("source", "") for context in contexts],
                **score,
            }
        )

    total_facts = sum(result["fact_max"] for result in results)
    matched_facts = sum(result["fact_score"] for result in results)
    latencies = [result["latency_ms"] for result in results]
    avg_chunk_tokens = statistics.mean(estimate_token_count(chunk["content"]) for chunk in chunks)
    return {
        "name": name,
        "label": label,
        "strategy": strategy,
        "chunk_config": {
            "hard_chunk_size_chars": 600 if strategy == CHUNK_STRATEGY_HARD else None,
            "dynamic_chunk_budget_tokens": 600 if strategy == CHUNK_STRATEGY_DYNAMIC else None,
            "dynamic_overlap_tokens": 200 if strategy == CHUNK_STRATEGY_DYNAMIC else 0,
            "top_k": top_k,
            "candidate_k": candidate_k,
        },
        "chunk_count": len(chunks),
        "embedding_ms": round(embedding_ms, 3),
        "average_chunk_chars": round(statistics.mean(len(chunk["content"]) for chunk in chunks), 2),
        "average_chunk_tokens": round(avg_chunk_tokens, 2),
        "summary": {
            "question_count": len(results),
            "fact_accuracy": round(matched_facts / total_facts, 4) if total_facts else 0.0,
            "fact_score": f"{matched_facts}/{total_facts}",
            "perfect_questions": sum(result["perfect"] for result in results),
            "average_latency_ms": round(statistics.mean(latencies), 3) if latencies else 0.0,
            "p95_latency_ms": round(sorted(latencies)[int(round((len(latencies) - 1) * 0.95))], 3) if latencies else 0.0,
        },
        "results": results,
    }


def render_report(payload: dict) -> str:
    lines = [
        "# Chunking A/B 20 題檢索比較",
        "",
        "本測試固定使用同一批 20 題、同一個多語 embedding 模型與同一套 BM25 + Dense + LambdaMART score mapping + Rerank。為隔離 chunk 邊界影響，本測試暫停 production 的問題複雜度 Top-3 gate。評分只判斷 Top-K 證據是否包含題目標註的 focus facts，屬於檢索上限測試，不代表完整回答模型準確率。Token 使用相容的多語近似計數器；B 的 200 token overlap 優先保留完整句子，因此實際重疊量可能小於 200。",
        "",
        f"- 題庫：`{payload['benchmark']}`",
        f"- 執行時間：`{payload['run_at']}`",
        f"- Embedding：`{payload['embedding_model']}`",
        f"- 題數：`{payload['question_count']}`",
        "",
        "| 版本 | 分塊設定 | Chunks | Fact accuracy | 完整題數 | 平均檢索 ms | P95 ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in payload["variants"]:
        summary = variant["summary"]
        config = variant["chunk_config"]
        config_text = (
            "600 字元硬切、0 overlap"
            if variant["strategy"] == CHUNK_STRATEGY_HARD
            else "句子邊界、600 token budget、200 token overlap"
        )
        lines.append(
            f"| {variant['label']} | {config_text} | `{variant['chunk_count']}` | "
            f"`{summary['fact_score']} = {summary['fact_accuracy']:.1%}` | "
            f"`{summary['perfect_questions']}/{summary['question_count']}` | "
            f"`{summary['average_latency_ms']}` | `{summary['p95_latency_ms']}` |"
        )
    lines.extend(["", "## 題目明細", ""])
    lines.extend([
        "| 題目 | A 命中/缺漏 | B 命中/缺漏 |",
        "| --- | --- | --- |",
    ])
    grouped = {variant["name"]: {item["id"]: item for item in variant["results"]} for variant in payload["variants"]}
    for question in payload["questions"]:
        a = grouped["hard_600"][question["id"]]
        b = grouped["dynamic_600_overlap_200"][question["id"]]
        lines.append(
            f"| `{question['id']}` | {a['fact_score']}/{a['fact_max']} "
            f"({', '.join(a['missed_facts']) or '全命中'}) | {b['fact_score']}/{b['fact_max']} "
            f"({', '.join(b['missed_facts']) or '全命中'}) |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--source-2016", type=Path, default=DEFAULT_SOURCE_2016)
    parser.add_argument("--source-2026", type=Path, default=DEFAULT_SOURCE_2026)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--candidate-k", type=int, default=16)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    args = parser.parse_args()

    benchmark_path = args.benchmark.expanduser().resolve()
    source_2016_path = args.source_2016.expanduser().resolve()
    source_2026_path = args.source_2026.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    questions = load_benchmark(benchmark_path, args.limit)
    source_texts = [
        ("2016", source_2016_path.read_text(encoding="utf-8")),
        ("2026", source_2026_path.read_text(encoding="utf-8")),
    ]
    variants = [
        run_variant(
            "hard_600",
            "A 硬切分",
            CHUNK_STRATEGY_HARD,
            source_texts,
            questions,
            args.embedding_model,
            args.top_k,
            args.candidate_k,
        ),
        run_variant(
            "dynamic_600_overlap_200",
            "B 動態句界切分",
            CHUNK_STRATEGY_DYNAMIC,
            source_texts,
            questions,
            args.embedding_model,
            args.top_k,
            args.candidate_k,
        ),
    ]
    payload = {
        "benchmark": str(benchmark_path.relative_to(ROOT)),
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "embedding_model": args.embedding_model,
        "question_count": len(questions),
        "questions": questions,
        "variants": variants,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")
    for variant in variants:
        print(json.dumps({"variant": variant["label"], **variant["summary"]}, ensure_ascii=False))
    print(json.dumps({"output": str(output_path), "report": str(report_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
