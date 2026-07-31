#!/usr/bin/env python3
"""Run a non-destructive smoke test against a deployed RAG service."""

from __future__ import annotations

import json
import os
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


BASE_URL = os.environ.get("RAG_SMOKE_BASE_URL", "").rstrip("/")
TOKEN = os.environ.get("RAG_SMOKE_BEARER_TOKEN", "")
KNOWLEDGE_BASE_ID = os.environ.get("RAG_SMOKE_KNOWLEDGE_BASE_ID", "")
QUESTION = os.environ.get("RAG_SMOKE_QUESTION", "今天是星期幾？")
TIMEOUT_SECONDS = float(os.environ.get("RAG_SMOKE_TIMEOUT_SECONDS", "30"))


def request_json(path: str, *, method: str = "GET", payload: dict | None = None) -> dict:
    headers = {"Accept": "application/json"}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")
    request = Request(
        urljoin(f"{BASE_URL}/", path.lstrip("/")),
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            raw = response.read()
            return json.loads(raw) if raw else {}
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"{path} returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"{path} is unreachable: {exc.reason}") from exc


def main() -> int:
    if not BASE_URL.startswith(("http://", "https://")):
        print("RAG_SMOKE_BASE_URL must be an absolute HTTP(S) URL", file=sys.stderr)
        return 2
    if not TOKEN:
        print("RAG_SMOKE_BEARER_TOKEN is required for protected checks", file=sys.stderr)
        return 2

    started = time.perf_counter()
    checks: list[dict] = []

    live = request_json("/health/live")
    checks.append({"name": "liveness", "status": live.get("status")})
    if live.get("status") != "live":
        raise RuntimeError("liveness check did not return live")

    ready = request_json("/health/ready")
    checks.append({"name": "readiness", "status": ready.get("status")})
    if ready.get("status") != "ready":
        raise RuntimeError(f"readiness check failed: {ready}")

    runtime = request_json("/v1/runtime")
    checks.append({"name": "authenticated-runtime", "model": runtime.get("default_model")})

    knowledge_bases = request_json("/v1/knowledge-bases").get("items", [])
    checks.append({"name": "knowledge-base-list", "count": len(knowledge_bases)})

    if KNOWLEDGE_BASE_ID:
        answer = request_json(
            "/v1/ask",
            method="POST",
            payload={
                "question": QUESTION,
                "knowledge_base_id": KNOWLEDGE_BASE_ID,
                "source_ids": [],
                "model": runtime.get("default_model"),
            },
        )
        if not answer.get("run_id") or not str(answer.get("answer") or "").strip():
            raise RuntimeError("ask check returned no run_id or answer")
        checks.append({
            "name": "ask",
            "run_id": answer["run_id"],
            "retrieval_needed": answer.get("retrieval", {}).get("needed"),
            "citation_count": len(answer.get("citations") or []),
        })

    print(json.dumps({
        "status": "passed",
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        "checks": checks,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1) from exc
