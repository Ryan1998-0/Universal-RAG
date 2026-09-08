#!/usr/bin/env python3
"""Run the 50-question benchmark through the real retrieval endpoint."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "evals/leave_rules_corrupted/benchmark_50.json"
DEFAULT_OUTPUT_DIR = ROOT / "evals/leave_rules_corrupted/runs"


def retrieve(endpoint: str, source_id: str, question: str) -> dict:
    payload = json.dumps(
        {
            "profile": "default",
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
    parser.add_argument("--endpoint", default="http://127.0.0.1:8765/api/retrieve")
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    results = []
    for index, case in enumerate(benchmark["items"], start=1):
        print(f"[{index}/{len(benchmark['items'])}] {case['id']}", flush=True)
        response = retrieve(args.endpoint, args.source_id, case["question"])
        contexts = response.get("contexts") or []
        evidence = response.get("evidenceEvaluation") or {}
        results.append(
            {
                "id": case["id"],
                "question": case["question"],
                "article": case["article"],
                "oracle_context": case["oracle_context"],
                "contexts": contexts,
                "raw_contexts": response.get("raw_contexts") or [],
                "evidence_focus": response.get("evidence_focus"),
                "evidence_evaluation": evidence,
                "context_count": len(contexts),
                "context_chars": sum(len(str(c.get("content") or "")) for c in contexts),
                "timings": (response.get("timings") or response.get("retrieval", {}).get("timings") or {}),
                "client_wall_ms": response.get("client_wall_ms"),
            }
        )

    payload = {
        "benchmark_id": benchmark["benchmark_id"],
        "benchmark_type": benchmark["benchmark_type"],
        "mode": "two-pass-retrieval-parent-then-one-third-fine",
        "endpoint": args.endpoint,
        "source_id": args.source_id,
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "question_count": len(results),
        "fine_evidence_enabled_expected": True,
        "chunk_fraction_expected": 1 / 3,
        "results": results,
        "summary": {
            "retrieved_count": sum(bool(r["contexts"]) for r in results),
            "evidence_sufficient_count": sum(
                r["evidence_evaluation"].get("sufficient") is True for r in results
            ),
            "average_context_chars": round(
                sum(r["context_chars"] for r in results) / max(1, len(results)), 1
            ),
            "average_context_count": round(
                sum(r["context_count"] for r in results) / max(1, len(results)), 2
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = args.output_dir / f"{stamp}-50q-two-pass-retrieval-results.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **payload["summary"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
