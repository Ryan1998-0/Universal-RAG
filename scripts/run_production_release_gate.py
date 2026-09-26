#!/usr/bin/env python3
"""Evaluate real /v1/ask responses from a seeded staging knowledge base.

The gate reads a frozen manifest and writes a detailed local artifact. It does
not upload files, modify the knowledge base or print the bearer token.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = PROJECT_ROOT / "evals" / "production_release_gate" / "fixtures"
PROVENANCE_FIELDS = (
    "deployment_image_digest",
    "generation_model_revision",
    "embedding_model_revision",
    "reranker_model_revision",
    "active_index_id",
    "parser_version",
    "chunk_schema_version",
    "hardware_profile",
)
REQUIRED_TAGS = {
    "traditional_chinese",
    "version_conflict",
    "no_answer",
    "ocr",
    "multi_hop",
    "citation_support",
}
SCHEMA_VERSION = "production-rag-release-gate-v1"


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"manifest schema_version must be {SCHEMA_VERSION}")
    if not _configured_id(manifest.get("knowledge_base_id")):
        raise ValueError("manifest needs a real staging knowledge_base_id")
    declared = manifest.get("declared_provenance")
    if not isinstance(declared, dict) or any(
        not _configured_id(declared.get(field)) for field in PROVENANCE_FIELDS
    ):
        raise ValueError("manifest needs complete declared_provenance")
    sources = manifest.get("fixture_sources")
    expected_files = {path.name for path in FIXTURE_DIR.iterdir() if path.is_file()}
    if not isinstance(sources, list) or {item.get("file") for item in sources if isinstance(item, dict)} != expected_files:
        raise ValueError("fixture_sources must map every frozen fixture file")
    if len(sources) != len(expected_files) or any(
        not isinstance(item, dict) or not _configured_id(item.get("source_id"))
        for item in sources
    ):
        raise ValueError("fixture_sources need unique files and real source IDs")
    fixture_source_ids = [item["source_id"] for item in sources]
    if len(set(fixture_source_ids)) != len(fixture_source_ids):
        raise ValueError("fixture source IDs must be unique")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases or any(not isinstance(case, dict) for case in cases):
        raise ValueError("manifest needs nonempty cases")
    case_ids = [str(case.get("id") or "") for case in cases if isinstance(case, dict)]
    if len(case_ids) != len(cases) or any(not value for value in case_ids):
        raise ValueError("each case needs an id")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("case ids must be unique")
    for case in cases:
        if not str(case.get("question") or "").strip():
            raise ValueError(f"case {case['id']} needs a question")
        if not isinstance(case.get("tags"), list):
            raise ValueError(f"case {case['id']} needs a list of tags")
        if "source_ids" in case and not isinstance(case["source_ids"], list):
            raise ValueError(f"case {case['id']} source_ids must be a list")
        if any(not _configured_id(source_id) for source_id in case.get("source_ids") or []):
            raise ValueError(f"case {case['id']} has an unconfigured source_id")
        if any(source_id not in fixture_source_ids for source_id in case.get("source_ids") or []):
            raise ValueError(f"case {case['id']} uses a source outside the frozen corpus")
        expect = case.get("expect") or {}
        if not isinstance(expect, dict) or not isinstance(expect.get("retrieval_needed"), bool):
            raise ValueError(f"case {case['id']} needs a boolean retrieval_needed expectation")
        if expect.get("status") not in {"passed", "threshold", "refused"}:
            raise ValueError(f"case {case['id']} needs an expected status")
        if expect["status"] != "refused" and not expect.get("required_fact_groups"):
            raise ValueError(f"case {case['id']} needs required_fact_groups")
        groups = expect.get("required_fact_groups") or []
        if not isinstance(groups, list) or any(
            not isinstance(group, list) or not group or any(not str(fact).strip() for fact in group)
            for group in groups
        ):
            raise ValueError(f"case {case['id']} has invalid required_fact_groups")
        if "multi_hop" in case.get("tags", []) and int(expect.get("min_citations", 0)) < 2:
            raise ValueError(f"multi_hop case {case['id']} needs at least two citations")
        if any(not _configured_id(source_id) for source_id in expect.get("allowed_citation_source_ids") or []):
            raise ValueError(f"case {case['id']} has an unconfigured citation source_id")
        if any(source_id not in fixture_source_ids for source_id in expect.get("allowed_citation_source_ids") or []):
            raise ValueError(f"case {case['id']} permits a citation outside the frozen corpus")
    tags = {tag for case in cases for tag in case["tags"]}
    missing_tags = REQUIRED_TAGS - tags
    if missing_tags:
        raise ValueError(f"manifest is missing required tags: {', '.join(sorted(missing_tags))}")
    if sum("version_conflict" in case["tags"] for case in cases) < 2:
        raise ValueError("version_conflict needs two cases, one for each version")
    if not isinstance(manifest.get("unauthorized_checks"), list) or not manifest["unauthorized_checks"]:
        raise ValueError("manifest needs at least one cross-tenant unauthorized check")
    for check in manifest["unauthorized_checks"]:
        if not isinstance(check, dict) or not _configured_id(check.get("knowledge_base_id")):
            raise ValueError("unauthorized check needs a real other-tenant knowledge_base_id")
        if check["knowledge_base_id"] == manifest["knowledge_base_id"]:
            raise ValueError("unauthorized check must use another tenant's knowledge base")
    if float(manifest.get("max_p95_wall_ms", 0)) <= 0:
        raise ValueError("max_p95_wall_ms must be positive")
    max_error_rate = float(manifest.get("max_error_rate", -1))
    if not 0 <= max_error_rate < 1:
        raise ValueError("max_error_rate must be between 0 and 1")
    return manifest


def _configured_id(value: object) -> bool:
    text = str(value or "").strip()
    return bool(text) and not text.startswith(("SET_", "REPLACE_", "<"))


class HttpClient:
    def __init__(self, base_url: str, token: str, timeout_seconds: float):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds

    def request(self, path: str, *, method: str = "GET", payload: dict | None = None) -> tuple[int, dict]:
        headers = {"Accept": "application/json", "Authorization": f"Bearer {self.token}"}
        body = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            urljoin(f"{self.base_url}/", path.lstrip("/")),
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
                return response.status, json.loads(raw) if raw else {}
        except HTTPError as exc:
            raw = exc.read()
            try:
                detail = json.loads(raw) if raw else {}
            except (ValueError, UnicodeDecodeError):
                detail = {"detail": raw.decode("utf-8", errors="replace")[:300]}
            return exc.code, detail
        except URLError as exc:
            raise RuntimeError(f"{path} is unreachable: {exc.reason}") from exc


def run_gate(manifest: dict, client: HttpClient) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    case_results = []
    request_errors = 0
    preflight = _preflight(manifest, client)
    if not preflight["passed"]:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "preflight": preflight,
            "summary": {
                "case_count": 0, "passed_cases": 0, "p95_wall_ms": None,
                "max_p95_wall_ms": manifest["max_p95_wall_ms"],
                "error_rate": 1.0, "max_error_rate": manifest["max_error_rate"],
                "unauthorized_checks_passed": False,
            },
            "cases": [],
            "unauthorized_checks": [],
        }

    for case in manifest["cases"]:
        request_payload = {
            "question": case["question"],
            "knowledge_base_id": manifest["knowledge_base_id"],
        }
        for key in ("model", "source_ids", "top_k"):
            value = case.get(key, manifest.get(key))
            if value is not None:
                request_payload[key] = value
        started = time.perf_counter()
        try:
            status, response = client.request("/v1/ask", method="POST", payload=request_payload)
            error = ""
        except Exception as exc:
            status, response, error = 0, {}, f"{type(exc).__name__}: {exc}"
        wall_ms = round((time.perf_counter() - started) * 1000, 2)
        if status != 200:
            request_errors += 1
        failures = _check_answer(case, status, response)
        if error:
            failures.append(error)
        case_results.append({
            "id": case["id"],
            "tags": case.get("tags", []),
            "passed": not failures,
            "failures": failures,
            "http_status": status,
            "wall_ms": wall_ms,
            "run_id": response.get("run_id", ""),
            "request_id": response.get("request_id", ""),
            "answer": response.get("answer", ""),
            "citations": response.get("citations", []),
            "evidence_validation": response.get("evidence_validation"),
            "timings": response.get("timings", {}),
        })

    unauthorized_results = []
    for check in manifest["unauthorized_checks"]:
        try:
            status, _ = client.request("/v1/ask", method="POST", payload={
                "question": check.get("question", "請列出這個知識庫的內容。"),
                "knowledge_base_id": check["knowledge_base_id"],
            })
            error = ""
        except Exception as exc:
            status, error = 0, f"{type(exc).__name__}: {exc}"
        unauthorized_results.append({
            "id": check.get("id", "cross-tenant"),
            "passed": status in {403, 404},
            "http_status": status,
            "error": error,
        })

    durations = sorted(item["wall_ms"] for item in case_results)
    p95_wall_ms = _percentile_nearest_rank(durations, 0.95)
    error_rate = request_errors / len(case_results)
    passed = (
        all(item["passed"] for item in case_results)
        and all(item["passed"] for item in unauthorized_results)
        and p95_wall_ms <= float(manifest["max_p95_wall_ms"])
        and error_rate <= float(manifest["max_error_rate"])
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "preflight": preflight,
        "summary": {
            "case_count": len(case_results),
            "passed_cases": sum(item["passed"] for item in case_results),
            "latency": {
                "unit": "ms",
                "method": "client wall time; nearest-rank percentiles; answer requests only",
                "sample_count": len(durations),
                "min": durations[0],
                "p50": _percentile_nearest_rank(durations, 0.50),
                "p95": p95_wall_ms,
                "p99": _percentile_nearest_rank(durations, 0.99),
                "max": durations[-1],
                "mean": round(statistics.mean(durations), 2),
            },
            "p95_wall_ms": p95_wall_ms,
            "max_p95_wall_ms": manifest["max_p95_wall_ms"],
            "error_rate": round(error_rate, 4),
            "max_error_rate": manifest["max_error_rate"],
            "unauthorized_checks_passed": all(item["passed"] for item in unauthorized_results),
        },
        "cases": case_results,
        "unauthorized_checks": unauthorized_results,
    }


def _preflight(manifest: dict, client: HttpClient) -> dict:
    try:
        ready_status, ready = client.request("/health/ready")
        ready_error = ""
    except Exception as exc:
        ready_status, ready = 0, {}
        ready_error = f"{type(exc).__name__}: {exc}"
    ready_ok = ready_status == 200 and ready.get("status") == "ready"
    snapshot = {"ready": ready_ok, "http_status": ready_status, "error": ready_error}
    if not ready_ok:
        snapshot["passed"] = False
        return snapshot
    try:
        runtime_status, runtime = client.request("/v1/runtime")
        models_status, models = client.request("/v1/models")
        kb_status, kb_payload = client.request("/v1/knowledge-bases")
    except Exception as exc:
        snapshot.update({"passed": False, "error": f"runtime provenance unavailable: {exc}"})
        return snapshot
    knowledge_base = next((item for item in kb_payload.get("items", [])
                           if item.get("id") == manifest["knowledge_base_id"]), None)
    active_index = (knowledge_base or {}).get("active_index_version_id")
    snapshot.update({
        "runtime_http_status": runtime_status,
        "models_http_status": models_status,
        "knowledge_bases_http_status": kb_status,
        "observed": {
            "default_model": runtime.get("default_model"),
            "allowed_models": runtime.get("allowed_models"),
            "embedding_model": (models.get("components") or {}).get("embedding", {}).get("model"),
            "sparse_embedding_model": (models.get("components") or {}).get("sparse_embedding", {}).get("model"),
            "reranker_model": (models.get("components") or {}).get("reranker", {}).get("model"),
            "active_index_id": active_index,
        },
    })
    snapshot["passed"] = (
        runtime_status == models_status == kb_status == 200
        and bool(knowledge_base)
        and active_index == manifest["declared_provenance"]["active_index_id"]
        and bool(snapshot["observed"]["embedding_model"])
        and bool(snapshot["observed"]["reranker_model"])
    )
    if not snapshot["passed"]:
        snapshot["error"] = "staging runtime or active index does not match the release manifest"
    return snapshot


def _percentile_nearest_rank(sorted_values: list[float], quantile: float) -> float:
    return sorted_values[math.ceil(quantile * len(sorted_values)) - 1]


def _check_answer(case: dict, status: int, response: dict) -> list[str]:
    if status != 200:
        return [f"/v1/ask returned HTTP {status}"]
    failures = []
    answer = str(response.get("answer") or "")
    citations = response.get("citations") or []
    validation = response.get("evidence_validation") or {}
    expected = case["expect"]
    if not response.get("run_id") or not answer.strip():
        failures.append("missing run_id or answer")
    if (response.get("retrieval") or {}).get("needed") is not expected.get("retrieval_needed"):
        failures.append("retrieval_needed differs from gold expectation")
    if expected["status"] == "refused":
        if not answer.startswith("根據目前檢索資料無法確認"):
            failures.append("answer did not refuse")
        if citations:
            failures.append("refusal included citations")
        if validation.get("status") not in (None, "", "refused"):
            failures.append("refusal validation status is inconsistent")
    else:
        if validation.get("status") != expected["status"] or not validation.get("sufficient"):
            failures.append("evidence validation did not pass")
        if validation.get("uncited_claims") or validation.get("unsupported_claims"):
            failures.append("answer contains ungrounded claims")
        if len(citations) < int(expected.get("min_citations", 1)):
            failures.append("too few citations")
        valid_ranks = set(validation.get("valid_citations") or [])
        if expected["status"] == "passed" and any(
            citation.get("rank") not in valid_ranks for citation in citations
        ):
            failures.append("citation rank was not validated")
        allowed_sources = set(expected.get("allowed_citation_source_ids") or [])
        if allowed_sources and any(citation.get("source") not in allowed_sources for citation in citations):
            failures.append("citation came from an unexpected source")
        for alternatives in expected.get("required_fact_groups") or []:
            if not any(str(fact) in answer for fact in alternatives):
                failures.append(f"missing expected fact group: {alternatives}")
    for phrase in expected.get("answer_excludes") or []:
        if str(phrase) in answer:
            failures.append(f"forbidden phrase in answer: {phrase}")
    return failures


def _git_provenance() -> dict:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True,
            check=True,
        )
        return result.stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def _corpus_provenance(manifest: dict) -> dict:
    files = []
    for item in sorted(manifest["fixture_sources"], key=lambda source: source["file"]):
        name = item["file"]
        files.append({
            "file": name,
            "source_id": item["source_id"],
            "sha256": hashlib.sha256((FIXTURE_DIR / name).read_bytes()).hexdigest(),
        })
    canonical = json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "fixture_sources": files,
        "fixture_manifest_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "scope": "staging knowledge base; case source_ids further restrict retrieval",
        "staging_content_verified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base_url = os.environ.get("RAG_RELEASE_BASE_URL", "").rstrip("/")
    token = os.environ.get("RAG_RELEASE_BEARER_TOKEN", "")
    if not base_url.startswith(("http://", "https://")) or not token:
        parser.error("set RAG_RELEASE_BASE_URL and RAG_RELEASE_BEARER_TOKEN")
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    client = HttpClient(base_url, token, float(os.environ.get("RAG_RELEASE_TIMEOUT_SECONDS", "180")))
    report = run_gate(manifest, client)
    report["provenance"] = {
        **_git_provenance(),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "base_url": base_url,
        "declared": manifest["declared_provenance"],
        "corpus": _corpus_provenance(manifest),
    }
    output = args.output or (
        PROJECT_ROOT / "evals" / "production_release_gate" / "runs"
        / f"gate-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "summary": report["summary"], "artifact": str(output)}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(f"release gate could not run: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
