#!/usr/bin/env python3
import argparse
import json
import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = PROJECT_ROOT / "evals" / "multimodal_ingestion"


def request_json(url, *, method="GET", payload=None, headers=None, timeout=600):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request_headers = dict(headers or {})
    if payload is not None:
        request_headers["Content-Type"] = "application/json"
    request = Request(url, data=data, method=method, headers=request_headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def upload_document(base_url: str, item: dict) -> dict:
    path = Path(item["path"])
    request = Request(
        f"{base_url}/api/documents/upload",
        data=path.read_bytes(),
        method="POST",
        headers={
            "Content-Type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            "X-File-Name": quote(path.name),
        },
    )
    try:
        with urlopen(request, timeout=900) as response:
            body = json.loads(response.read().decode("utf-8"))
            body["http_status"] = response.status
            return body
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Upload {path.name} failed with HTTP {exc.code}: {body}") from exc


def evaluate(base_url: str, manifest_path: Path, profile: str) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_url = base_url.rstrip("/")
    _, health = request_json(f"{base_url}/api/health")
    uploads = []
    cases = []

    for item in manifest["documents"]:
        item = dict(item)
        item_path = Path(item["path"])
        if not item_path.is_absolute():
            item_path = PROJECT_ROOT / item_path
        item["path"] = str(item_path)
        result = upload_document(base_url, item)
        document = result["document"]
        upload_record = {
            "filename": item["filename"],
            "marker": item["marker"],
            "source_id": document["source_id"],
            "source_type": document["source_type"],
            "chunk_count": document["chunk_count"],
            "selected_by_default": document.get("selected_by_default"),
            "status": document.get("status"),
            "extraction": document.get("extraction") or {},
            "pipeline": document.get("pipeline") or [],
            "duplicate": result.get("duplicate", False),
        }
        uploads.append(upload_record)

        _, retrieval = request_json(
            f"{base_url}/api/retrieve",
            method="POST",
            payload={
                "profile": profile,
                "question": item["question"],
                "retrieval_query": item["marker"],
                "source_ids": [document["source_id"]],
                "top_k": 3,
                "candidate_k": 8,
            },
        )
        contexts = retrieval.get("contexts") or []
        cases.append(
            {
                "case": f"selected:{item['filename']}",
                "passed": bool(contexts)
                and all(context.get("source") == document["source_id"] for context in contexts)
                and any(item["marker"] in context.get("content", "") for context in contexts),
                "context_count": len(contexts),
                "sources": sorted({context.get("source") for context in contexts}),
                "marker_found": any(
                    item["marker"] in context.get("content", "") for context in contexts
                ),
            }
        )

    _, empty_result = request_json(
        f"{base_url}/api/retrieve",
        method="POST",
        payload={
            "profile": profile,
            "question": "What is PDF-620?",
            "source_ids": [],
            "top_k": 3,
        },
    )
    cases.append(
        {
            "case": "explicit-empty-selection",
            "passed": empty_result.get("contexts") == [],
            "context_count": len(empty_result.get("contexts") or []),
        }
    )

    first, second = uploads[0], uploads[1]
    _, isolation = request_json(
        f"{base_url}/api/retrieve",
        method="POST",
        payload={
            "profile": profile,
            "question": first["marker"],
            "source_ids": [second["source_id"]],
            "top_k": 3,
        },
    )
    isolation_contexts = isolation.get("contexts") or []
    cases.append(
        {
            "case": "cross-document-isolation",
            "passed": all(
                context.get("source") == second["source_id"]
                and first["marker"] not in context.get("content", "")
                for context in isolation_contexts
            ),
            "context_count": len(isolation_contexts),
        }
    )

    _, listed = request_json(f"{base_url}/api/documents")
    listed_ids = {document.get("source_id") for document in listed.get("documents") or []}
    cases.append(
        {
            "case": "persisted-document-list",
            "passed": all(upload["source_id"] in listed_ids for upload in uploads),
            "listed_count": len(listed_ids),
        }
    )

    passed = sum(bool(case["passed"]) for case in cases)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "health": health,
        "summary": {"passed": passed, "total": len(cases)},
        "uploads": uploads,
        "cases": cases,
    }
    report = portable_artifact(report)
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    (EVAL_ROOT / "latest-results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (EVAL_ROOT / "latest-report.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def portable_artifact(value):
    if isinstance(value, dict):
        return {key: portable_artifact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_artifact(item) for item in value]
    if isinstance(value, str):
        return value.replace(f"{PROJECT_ROOT}/", "")
    return value


def render_markdown(report: dict) -> str:
    lines = [
        "# Multimodal ingestion evaluation",
        "",
        f"- Result: {report['summary']['passed']}/{report['summary']['total']} passed",
        f"- Generated: {report['generated_at']}",
        "",
        "## Uploads",
        "",
        "| File | Type | Method | Chunks | Default selected |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for upload in report["uploads"]:
        lines.append(
            f"| {upload['filename']} | {upload['source_type']} | "
            f"{upload['extraction'].get('method', '')} | {upload['chunk_count']} | "
            f"{upload['selected_by_default']} |"
        )
    lines.extend(["", "## Cases", ""])
    for case in report["cases"]:
        lines.append(f"- {'PASS' if case['passed'] else 'FAIL'}: `{case['case']}`")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8877")
    parser.add_argument("--manifest", type=Path, default=EVAL_ROOT / "manifest.json")
    parser.add_argument("--profile", default=os.getenv("RAG_PROFILE", "default"))
    args = parser.parse_args()
    result = evaluate(args.base_url, args.manifest, args.profile)
    print(json.dumps(result["summary"], ensure_ascii=False))
    raise SystemExit(0 if result["summary"]["passed"] == result["summary"]["total"] else 1)
