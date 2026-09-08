#!/usr/bin/env python3
"""Run the frozen ten-case leave-rule set through the real /api/ask pipeline."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from run_closed_book_oracle_eval import score_answer


ROOT = Path(__file__).resolve().parents[1]
TEST_SET = ROOT / "evals/leave_rules_ab/c-test-set.json"
BENCHMARK = ROOT / "evals/leave_rules_ab/benchmark.json"
OUTPUT_DIR = ROOT / "evals/leave_rules_ab/runs"


def ask_rag(endpoint: str, source_id: str, question: str, model: str) -> dict:
    payload = json.dumps(
        {
            "profile": "default",
            "model": {"provider": "ollama", "name": model},
            "question": question,
            "source_ids": [source_id],
            "top_k": 8,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8765/api/ask")
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK)
    parser.add_argument("--test-set", type=Path, default=TEST_SET)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model", default="qwen2.5:7b")
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    if args.test_set.exists():
        test_set = json.loads(args.test_set.read_text(encoding="utf-8"))
    else:
        test_set = {
            "test_set_id": f"{benchmark['benchmark_id']}-c",
            "cases": benchmark["items"],
        }
    benchmark_by_id = {item["id"]: item for item in benchmark["items"]}
    results = []
    for index, case in enumerate(test_set["cases"], start=1):
        print(f"[{index}/{len(test_set['cases'])}] {case['id']}", flush=True)
        response = ask_rag(args.endpoint, args.source_id, case["question"], args.model)
        evaluation = score_answer(str(response.get("answer") or ""), benchmark_by_id[case["id"]])
        retrieval = response.get("retrieval") or {}
        evidence = retrieval.get("evidence_evaluation") or {}
        contexts = retrieval.get("contexts") or []
        results.append(
            {
                "id": case["id"],
                "question": case["question"],
                "answer": response.get("answer"),
                "score": evaluation,
                "correct": evaluation["score"] >= 90,
                "retrieval_needed": retrieval.get("needed"),
                "retrieval_queries": retrieval.get("queries") or [],
                "context_count": len(contexts),
                "context_ranks": [context.get("rank") for context in contexts],
                "evidence_sufficient": evidence.get("sufficient"),
                "evidence_confidence": evidence.get("confidence"),
                "evidence_reason": evidence.get("reason"),
                "citations": response.get("citations") or [],
                "grounding_warnings": response.get("grounding_warnings") or [],
                "timings": response.get("timings") or {},
                "client_wall_ms": response.get("client_wall_ms"),
                "raw_response": response,
            }
        )

    correct = sum(item["correct"] for item in results)
    retrieved = sum(bool(item["context_count"]) for item in results)
    sufficient = sum(item["evidence_sufficient"] is True for item in results)
    payload = {
        "test_set_id": test_set["test_set_id"],
        "mode": "C End-to-end RAG",
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "endpoint": args.endpoint,
        "source_id": args.source_id,
        "model": f"ollama:{args.model}",
        "case_count": len(results),
        "correct_count": correct,
        "retrieved_context_count": retrieved,
        "evidence_sufficient_count": sufficient,
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = args.output_dir / f"{stem}-c-end-to-end-results.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "correct": correct, "total": len(results), "sufficient": sufficient}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
