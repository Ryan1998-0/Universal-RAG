#!/usr/bin/env python3
"""Run 20 grounded-answer questions with one independent model request each.

Retrieval is fixed to the current dynamic chunking strategy.  Each question
gets exactly one answer-generation call; the default model is the local
Ollama ``qwen2.5:7b``.  The request receives no conversation history or
long-term memory.  Scoring is deterministic and split into evidence presence
and hallucination indicators.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_demo.answer_quality import evaluate_answer_quality
from rag_demo.chunk_strategies import CHUNK_STRATEGY_DYNAMIC, estimate_token_count, split_text
from rag_demo.config import RagConfig
from rag_demo.embeddings import DEFAULT_EMBEDDING_MODEL, embed_chunks, embed_texts
from rag_demo.hybrid_retrieval import HybridRetriever
from rag_demo.model_providers import ask_model, parse_model_spec
from rag_demo.rag_pipeline import build_grounded_answer_request, enforce_grounded_answer_contract


DEFAULT_BENCHMARK = ROOT / "evals" / "taiwan_law_versions" / "questions_20.json"
DEFAULT_SOURCE_2016 = ROOT / "evals" / "taiwan_law_versions" / "source" / "勞工請假規則_2016有效版.txt"
DEFAULT_SOURCE_2026 = ROOT / "evals" / "taiwan_law_versions" / "source" / "勞工請假規則_2026有效版.txt"
DEFAULT_MODEL = "ollama:qwen2.5:7b"


def load_questions(path: Path, limit: int) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError("Benchmark must contain an items list.")
    return items[:limit] if limit > 0 else items


def build_chunks(source_id: str, text: str) -> list[dict]:
    pieces = split_text(
        text,
        chunk_size=600,
        chunk_stride=600,
        strategy=CHUNK_STRATEGY_DYNAMIC,
        overlap_tokens=200,
    )
    return [
        {
            "id": f"{source_id}::{index}",
            "source_id": source_id,
            "source": source_id,
            "chunk_index": index,
            "parent_title": source_id,
            "title": f"{source_id} / chunk {index + 1}",
            "content": piece,
        }
        for index, piece in enumerate(pieces)
    ]


def run(args: argparse.Namespace) -> dict:
    questions = load_questions(args.benchmark.resolve(), args.limit)
    chunks = [
        *build_chunks("2016", args.source_2016.resolve().read_text(encoding="utf-8")),
        *build_chunks("2026", args.source_2026.resolve().read_text(encoding="utf-8")),
    ]
    print(f"Embedding {len(chunks)} dynamic chunks...", flush=True)
    embedding_started = time.perf_counter()
    embeddings = embed_chunks(chunks, model_name=args.embedding_model)
    embedding_ms = (time.perf_counter() - embedding_started) * 1000.0
    settings = RagConfig.from_env().normalized()
    retriever = HybridRetriever(
        chunks=chunks,
        aliases=[],
        embeddings=embeddings,
        embedding_model=args.embedding_model,
        settings=settings,
    )

    results = []
    for index, item in enumerate(questions, start=1):
        question = str(item.get("question") or "").strip()
        print(f"[{index}/{len(questions)}] {item.get('id', index)} independent {args.model}", flush=True)
        retrieval_started = time.perf_counter()
        retrieval = retriever.retrieve(
            question=question,
            retrieval_query=question,
            query_variants=(question,),
            evidence_query=question,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
        )
        contexts = retrieval.get("contexts") or []
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000.0
        request = build_grounded_answer_request(
            question=question,
            contexts=contexts,
            history=[],
            memories=[],
        )
        generation_started = time.perf_counter()
        raw_answer = ask_model(
            request["prompt"],
            model=args.model,
            system=request["system"],
        )
        answer = enforce_grounded_answer_contract(raw_answer, contexts)
        generation_ms = (time.perf_counter() - generation_started) * 1000.0
        quality = evaluate_answer_quality(
            question=question,
            answer=answer,
            contexts=contexts,
            expected_facts=item.get("focus_facts") or (),
            embedding_fn=embed_texts,
            semantic_threshold=args.semantic_threshold,
        )
        results.append({
            "id": item.get("id", f"case-{index:02d}"),
            "question": question,
            "answer": answer,
            "raw_answer": raw_answer,
            "retrieved_context_ids": [context.get("id", "") for context in contexts],
            "retrieved_sources": [context.get("source", "") for context in contexts],
            "context_count": len(contexts),
            "timings_ms": {
                "retrieval": round(retrieval_ms, 2),
                "generation": round(generation_ms, 2),
                "total": round(retrieval_ms + generation_ms, 2),
            },
            "quality": quality,
        })

    evidence_passed = sum(
        bool(item["quality"]["answer_in_retrieved_chunks"]["passed"])
        for item in results
    )
    hallucination_free = sum(
        bool(item["quality"]["hallucination"]["hallucination_free"])
        for item in results
    )
    evidence_scores = [
        float(item["quality"]["answer_in_retrieved_chunks"]["score"])
        for item in results
    ]
    generation_latencies = [item["timings_ms"]["generation"] for item in results]
    return {
        "schema_version": "grounded-answer-quality-v2",
        "benchmark": str(args.benchmark.resolve().relative_to(ROOT)),
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": args.model,
        "answer_request_policy": {
            "independent_request_per_question": True,
            "session_mode": "ephemeral" if parse_model_spec(args.model).provider == "codex" else "ollama-request",
            "memory": False,
            "tools": False,
            "history_passed": False,
            "retrieval_mode": "fixed dynamic chunks; no query rewrite in this isolated answer test",
        },
        "chunking": {
            "strategy": CHUNK_STRATEGY_DYNAMIC,
            "chunk_size_tokens": 600,
            "overlap_tokens": 200,
            "chunk_count": len(chunks),
            "average_chunk_tokens": round(statistics.mean(estimate_token_count(chunk["content"]) for chunk in chunks), 2),
            "embedding_ms": round(embedding_ms, 2),
        },
        "configuration": {
            "top_k": args.top_k,
            "candidate_k": args.candidate_k,
            "embedding_model": args.embedding_model,
            "semantic_threshold": args.semantic_threshold,
        },
        "summary": {
            "question_count": len(results),
            "evidence_presence_passed": evidence_passed,
            "evidence_presence_rate": round(evidence_passed / len(results), 4) if results else 0.0,
            "average_evidence_coverage": round(statistics.mean(evidence_scores), 4) if evidence_scores else 0.0,
            "hallucination_free": hallucination_free,
            "hallucination_free_rate": round(hallucination_free / len(results), 4) if results else 0.0,
            "average_generation_ms": round(statistics.mean(generation_latencies), 2) if generation_latencies else 0.0,
            "p95_generation_ms": round(sorted(generation_latencies)[int(round((len(generation_latencies) - 1) * 0.95))], 2) if generation_latencies else 0.0,
        },
        "results": results,
    }


def render_report(payload: dict) -> str:
    summary = payload["summary"]
    model = str(payload.get("model") or "")
    provider = parse_model_spec(model).provider
    if provider == "codex":
        title = "GPT-5.5 一題一子代理回答品質測試"
        policy = "每題都建立新的 Codex ephemeral session；子代理不接收對話歷史或長期記憶，並停用工具。"
    else:
        title = f"{model} 一題一請求回答品質測試"
        policy = "每題都建立獨立的 Ollama 請求；模型不接收對話歷史或長期記憶，並不使用工具。"
    lines = [
        f"# {title}",
        "",
        f"{policy}檢索固定使用目前 dynamic 600 token／200 token overlap 分塊，評估分為答案是否可由檢索切片支持，以及是否出現幻覺。",
        "",
        f"- 題庫：`{payload['benchmark']}`",
        f"- 模型：`{payload['model']}`",
        f"- 執行時間：`{payload['run_at']}`",
        f"- 題數：`{summary['question_count']}`",
        f"- 檢索切片：`{payload['chunking']['chunk_count']}`",
        "",
        "## 總體指標",
        "",
        "| 指標 | 結果 |",
        "| --- | ---: |",
        f"| 答案在檢索切片中 | `{summary['evidence_presence_passed']}/{summary['question_count']}` = `{summary['evidence_presence_rate']:.1%}` |",
        f"| 平均答案證據覆蓋率 | `{summary['average_evidence_coverage']:.1%}` |",
        f"| 無幻覺回答 | `{summary['hallucination_free']}/{summary['question_count']}` = `{summary['hallucination_free_rate']:.1%}` |",
        f"| 平均生成時間 | `{summary['average_generation_ms']} ms` |",
        f"| P95 生成時間 | `{summary['p95_generation_ms']} ms` |",
        "",
        "## 題目明細",
        "",
        "| 題目 | 檢索切片支持 | 幻覺 | 證據覆蓋率 |",
        "| --- | --- | --- | ---: |",
    ]
    for item in payload["results"]:
        quality = item["quality"]
        evidence = quality["answer_in_retrieved_chunks"]
        hallucination = quality["hallucination"]
        lines.append(
            f"| `{item['id']}` | {'通過' if evidence['passed'] else '未通過'} "
            f"({evidence['supported_claim_count']}/{evidence['claim_count']}) | "
            f"{'有' if hallucination['detected'] else '無'} | `{evidence['score']:.1%}` |"
        )
    lines.extend(["", "## 判定說明", "", "- 答案在檢索切片中：每個實質回答 claim 都能在至少一個檢索 chunk 找到詞彙或 embedding 支持，且沒有無效 citation；題庫已標註的完整比較結論，在其必要 facts 與引用均被支持時視為可驗證推論。", "- 幻覺：回答 claim 無法由任何檢索 chunk 支持、引用不存在的 rank，或回答中的數值不在證據中時判定。拒答不算幻覺，但答案在切片中指標會判定未通過。", "- 這是可重現的 deterministic 評估，不使用另一個模型當裁判。", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--source-2016", type=Path, default=DEFAULT_SOURCE_2016)
    parser.add_argument("--source-2026", type=Path, default=DEFAULT_SOURCE_2026)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--candidate-k", type=int, default=32)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--semantic-threshold", type=float, default=0.50)
    args = parser.parse_args()
    if args.output is None or args.report is None:
        output_dir = "gpt55_subagent_quality" if parse_model_spec(args.model).provider == "codex" else "qwen_quality"
        args.output = args.output or (ROOT / "evals" / output_dir / "latest-results.json")
        args.report = args.report or (ROOT / "evals" / output_dir / "latest-report.md")
    payload = run(args)
    output_path = args.output.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "report": str(report_path), **payload["summary"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
