#!/usr/bin/env python3
"""Run two-pass retrieval independently against the 2016 and 2026 corpora."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from rag_demo.config import RagConfig
from rag_demo.fine_evidence import retrieve_fine_evidence
from rag_demo.hybrid_retrieval import evaluate_retrieval_evidence
from rag_demo.rag_pipeline import normalize_contexts


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "evals/taiwan_law_versions/questions_20.json"
OUTPUT_DIR = ROOT / "evals/taiwan_law_versions/runs"


def retrieve(endpoint: str, source_id: str, question: str) -> dict:
    payload = json.dumps(
        {
            "profile": "default",
            "question": question,
            "source_ids": [source_id],
            "top_k": 8,
            "candidate_k": 24,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=240) as response:
        body = json.loads(response.read().decode("utf-8"))
    body["client_wall_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return body


def run_version(endpoint: str, source_id: str, question: str, settings: RagConfig) -> dict:
    first = retrieve(endpoint, source_id, question)
    parent_contexts = normalize_contexts(first.get("contexts"), max_contexts=12)
    fine_contexts, fine_trace = retrieve_fine_evidence(
        question=question,
        contexts=parent_contexts,
        settings=settings,
        chunk_fraction=settings.fine_evidence_chunk_fraction,
    )
    evidence = evaluate_retrieval_evidence(
        fine_contexts,
        settings=settings,
        question=question,
    ) if fine_contexts else {"sufficient": False, "reason": "沒有可用細粒度證據"}
    return {
        "source_id": source_id,
        "pass_1_parent": {
            "contexts": parent_contexts,
            "raw_response": first,
            "context_count": len(parent_contexts),
            "context_chars": sum(len(str(item.get("content") or "")) for item in parent_contexts),
        },
        "pass_2_fine": {
            "contexts": fine_contexts,
            "trace": fine_trace,
            "evidence_evaluation": evidence,
            "context_count": len(fine_contexts),
            "context_chars": sum(len(str(item.get("content") or "")) for item in fine_contexts),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8765/api/retrieve")
    parser.add_argument("--source-2016", required=True)
    parser.add_argument("--source-2026", required=True)
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    settings = replace(
        RagConfig.from_env().normalized(),
        fine_evidence_enabled=True,
        fine_evidence_chunk_fraction=1 / 3,
        fine_evidence_top_k=6,
        fine_evidence_max_groups=3,
    ).normalized()
    results = []
    total = len(benchmark["items"])
    for index, item in enumerate(benchmark["items"], start=1):
        print(f"[{index}/{total}] {item['id']} 2016", flush=True)
        r2016 = run_version(args.endpoint, args.source_2016, item["question"], settings)
        print(f"[{index}/{total}] {item['id']} 2026", flush=True)
        r2026 = run_version(args.endpoint, args.source_2026, item["question"], settings)
        results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "comparison_articles": item["comparison_articles"],
                "oracle_2016": item["oracle_2016"],
                "oracle_2026": item["oracle_2026"],
                "retrieval_2016": r2016,
                "retrieval_2026": r2026,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = args.output_dir / f"{stamp}-two-version-two-pass-retrieval.json"
    payload = {
        "benchmark_id": benchmark["benchmark_id"],
        "mode": "two-pass-per-version-retrieval",
        "endpoint": args.endpoint,
        "source_ids": {"2016": args.source_2016, "2026": args.source_2026},
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "question_count": total,
        "two_pass_policy": {
            "pass_1": "hybrid parent recall top_k=8 candidate_k=24",
            "pass_2": "fine evidence re-embedding, parent-relative chunk_fraction=1/3, max 3 groups",
            "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        },
        "results": results,
        "summary": {
            "2016_parent_retrieved": sum(bool(r["retrieval_2016"]["pass_1_parent"]["contexts"]) for r in results),
            "2026_parent_retrieved": sum(bool(r["retrieval_2026"]["pass_1_parent"]["contexts"]) for r in results),
            "2016_fine_retrieved": sum(bool(r["retrieval_2016"]["pass_2_fine"]["contexts"]) for r in results),
            "2026_fine_retrieved": sum(bool(r["retrieval_2026"]["pass_2_fine"]["contexts"]) for r in results),
            "2016_fine_evidence_sufficient": sum(r["retrieval_2016"]["pass_2_fine"]["evidence_evaluation"].get("sufficient") is True for r in results),
            "2026_fine_evidence_sufficient": sum(r["retrieval_2026"]["pass_2_fine"]["evidence_evaluation"].get("sufficient") is True for r in results),
        },
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **payload["summary"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
