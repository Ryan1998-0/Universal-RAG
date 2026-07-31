#!/usr/bin/env python3
"""Run a bounded authenticated ask-load check against staging."""

from __future__ import annotations

import asyncio
import json
import math
import os
import statistics
import sys
import time

import httpx


BASE_URL = os.environ.get("RAG_LOAD_BASE_URL", "").rstrip("/")
TOKEN = os.environ.get("RAG_LOAD_BEARER_TOKEN", "")
KNOWLEDGE_BASE_ID = os.environ.get("RAG_LOAD_KNOWLEDGE_BASE_ID", "")
MODEL = os.environ.get("RAG_LOAD_MODEL", "ollama:qwen2.5:7b")
QUESTION = os.environ.get("RAG_LOAD_QUESTION", "請用一句話說明這個知識庫的主題。")
SOURCE_IDS = [item.strip() for item in os.environ.get("RAG_LOAD_SOURCE_IDS", "").split(",") if item.strip()]
REQUESTS = int(os.environ.get("RAG_LOAD_REQUESTS", "10"))
CONCURRENCY = int(os.environ.get("RAG_LOAD_CONCURRENCY", "5"))
TIMEOUT_SECONDS = float(os.environ.get("RAG_LOAD_TIMEOUT_SECONDS", "180"))


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


async def main() -> int:
    if os.environ.get("RAG_LOAD_CONFIRM_STAGING") != "yes":
        print("Set RAG_LOAD_CONFIRM_STAGING=yes before running this load check.", file=sys.stderr)
        return 2
    if not BASE_URL.startswith(("http://", "https://")) or not TOKEN or not KNOWLEDGE_BASE_ID:
        print(
            "RAG_LOAD_BASE_URL, RAG_LOAD_BEARER_TOKEN, and RAG_LOAD_KNOWLEDGE_BASE_ID are required.",
            file=sys.stderr,
        )
        return 2
    if not 1 <= CONCURRENCY <= 20 or not 1 <= REQUESTS <= 200:
        print("Concurrency must be 1-20 and requests must be 1-200.", file=sys.stderr)
        return 2

    semaphore = asyncio.Semaphore(CONCURRENCY)
    outcomes: list[dict] = []
    headers = {"Authorization": f"Bearer {TOKEN}"}
    limits = httpx.Limits(max_connections=CONCURRENCY, max_keepalive_connections=CONCURRENCY)
    timeout = httpx.Timeout(TIMEOUT_SECONDS)

    async with httpx.AsyncClient(base_url=BASE_URL, headers=headers, limits=limits, timeout=timeout) as client:
        async def run_one(sequence: int) -> None:
            async with semaphore:
                started = time.perf_counter()
                try:
                    response = await client.post("/v1/ask", json={
                        "question": QUESTION,
                        "knowledge_base_id": KNOWLEDGE_BASE_ID,
                        "source_ids": SOURCE_IDS,
                        "model": MODEL,
                    })
                    elapsed = time.perf_counter() - started
                    body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
                    outcomes.append({
                        "sequence": sequence,
                        "ok": response.is_success and bool(body.get("run_id")),
                        "status": response.status_code,
                        "elapsed_seconds": elapsed,
                        "error_code": body.get("error", {}).get("code"),
                    })
                except Exception as exc:
                    outcomes.append({
                        "sequence": sequence,
                        "ok": False,
                        "status": 0,
                        "elapsed_seconds": time.perf_counter() - started,
                        "error_code": type(exc).__name__,
                    })

        wall_started = time.perf_counter()
        await asyncio.gather(*(run_one(index + 1) for index in range(REQUESTS)))
        wall_seconds = time.perf_counter() - wall_started

    latencies = [item["elapsed_seconds"] for item in outcomes]
    passed = sum(1 for item in outcomes if item["ok"])
    report = {
        "status": "passed" if passed == REQUESTS else "failed",
        "requests": REQUESTS,
        "concurrency": CONCURRENCY,
        "succeeded": passed,
        "failed": REQUESTS - passed,
        "wall_seconds": round(wall_seconds, 3),
        "requests_per_second": round(REQUESTS / wall_seconds, 3) if wall_seconds else 0,
        "latency_seconds": {
            "mean": round(statistics.fmean(latencies), 3),
            "p50": round(percentile(latencies, 0.50), 3),
            "p95": round(percentile(latencies, 0.95), 3),
            "max": round(max(latencies), 3),
        },
        "failures": [item for item in outcomes if not item["ok"]],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed == REQUESTS else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
