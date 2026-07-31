#!/usr/bin/env python3
import argparse
import json
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = PROJECT_ROOT / "evals" / "document_folders"
CORPUS_ROOT = PROJECT_ROOT / "evals" / "multimodal_ingestion" / "corpus"
STATE_PATH = EVAL_ROOT / "runtime-state.json"
SETUP_RESULTS_PATH = EVAL_ROOT / "setup-results.json"


def request_json(url, *, method="GET", payload=None, headers=None, timeout=900):
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


def upload(base_url, path: Path, filename: str, folder_id: str) -> dict:
    request = Request(
        f"{base_url}/api/documents/upload",
        data=path.read_bytes(),
        method="POST",
        headers={
            "Content-Type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            "X-File-Name": quote(filename),
            "X-Folder-Id": quote(folder_id),
        },
    )
    try:
        with urlopen(request, timeout=900) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Upload {filename} failed with HTTP {exc.code}: {body}") from exc


def setup(base_url: str) -> dict:
    base_url = base_url.rstrip("/")
    required_files = {
        "law_text": CORPUS_ROOT / "benefits.txt",
        "court_pdf": CORPUS_ROOT / "financial-policy.pdf",
        "inspection_image": CORPUS_ROOT / "invoice.png",
        "memo_markdown": CORPUS_ROOT / "operations.md",
    }
    missing = [str(path) for path in required_files.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing multimodal corpus files: {missing}")

    _, law_response = request_json(
        f"{base_url}/api/folders",
        method="POST",
        payload={"name": "勞基法"},
    )
    _, case_response = request_json(
        f"{base_url}/api/folders",
        method="POST",
        payload={"name": "案例暫存"},
    )
    law_folder = law_response["folder"]
    case_folder = case_response["folder"]

    upload_specs = [
        ("law_text", "勞基法法條.txt", law_folder["id"], "TXT-731"),
        ("court_pdf", "法院判決案例.pdf", law_folder["id"], "PDF-620"),
        ("inspection_image", "勞動檢查表.png", law_folder["id"], "IMAGE-990"),
        ("memo_markdown", "案件備忘.md", case_folder["id"], "MD-204"),
    ]
    documents = {}
    cases = []
    for key, filename, folder_id, marker in upload_specs:
        result = upload(base_url, required_files[key], filename, folder_id)
        document = result["document"]
        documents[key] = document
        cases.append(
            {
                "case": f"upload:{filename}",
                "passed": document.get("folder_id") == folder_id
                and document.get("status") == "ready",
            }
        )
        _, retrieval = request_json(
            f"{base_url}/api/retrieve",
            method="POST",
            payload={
                "profile": "ifrs17",
                "question": marker,
                "retrieval_query": marker,
                "source_ids": [document["source_id"]],
                "top_k": 3,
            },
        )
        contexts = retrieval.get("contexts") or []
        cases.append(
            {
                "case": f"retrieve:{filename}",
                "passed": bool(contexts)
                and all(context.get("source") == document["source_id"] for context in contexts)
                and any(marker in context.get("content", "") for context in contexts),
            }
        )

    _, isolation = request_json(
        f"{base_url}/api/retrieve",
        method="POST",
        payload={
            "profile": "ifrs17",
            "question": "TXT-731",
            "retrieval_query": "TXT-731",
            "source_ids": [documents["memo_markdown"]["source_id"]],
            "top_k": 3,
        },
    )
    cases.append(
        {
            "case": "folder-selection-isolation",
            "passed": all(
                context.get("source") == documents["memo_markdown"]["source_id"]
                and "TXT-731" not in context.get("content", "")
                for context in isolation.get("contexts") or []
            ),
        }
    )

    _, moved = request_json(
        f"{base_url}/api/documents/{documents['court_pdf']['source_id']}/folder",
        method="PUT",
        payload={"folder_id": case_folder["id"]},
    )
    cases.append(
        {
            "case": "move-existing-document",
            "passed": moved["document"].get("folder_id") == case_folder["id"],
        }
    )
    _, renamed = request_json(
        f"{base_url}/api/folders/{case_folder['id']}",
        method="PUT",
        payload={"name": "法院相關案例"},
    )
    cases.append(
        {
            "case": "rename-folder",
            "passed": renamed["folder"].get("name") == "法院相關案例",
        }
    )
    _, deleted = request_json(
        f"{base_url}/api/folders/{case_folder['id']}",
        method="DELETE",
    )
    cases.append(
        {
            "case": "delete-folder-without-deleting-documents",
            "passed": deleted.get("moved_document_count") == 2
            and deleted.get("destination_folder_id") == "uncategorized",
        }
    )

    state = {
        "law_folder_id": law_folder["id"],
        "deleted_folder_id": case_folder["id"],
        "law_source_ids": [
            documents["law_text"]["source_id"],
            documents["inspection_image"]["source_id"],
        ],
        "uncategorized_source_ids": [
            documents["court_pdf"]["source_id"],
            documents["memo_markdown"]["source_id"],
        ],
    }
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    setup_result = {"cases": cases, "state": state}
    SETUP_RESULTS_PATH.write_text(
        json.dumps(setup_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summarize(cases)


def verify(base_url: str) -> dict:
    base_url = base_url.rstrip("/")
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    setup_result = json.loads(SETUP_RESULTS_PATH.read_text(encoding="utf-8"))
    _, folder_body = request_json(f"{base_url}/api/folders")
    _, document_body = request_json(f"{base_url}/api/documents")
    _, profile_body = request_json(f"{base_url}/api/profiles/ifrs17")
    folders = folder_body.get("folders") or []
    documents = document_body.get("documents") or []
    profile_sources = profile_body.get("sources") or []
    folder_by_id = {folder["id"]: folder for folder in folders}
    document_by_id = {document["source_id"]: document for document in documents}
    profile_by_id = {source["source_id"]: source for source in profile_sources}

    cases = list(setup_result["cases"])
    cases.extend(
        [
            {
                "case": "folder-registry-survives-restart",
                "passed": state["law_folder_id"] in folder_by_id
                and state["deleted_folder_id"] not in folder_by_id
                and folder_by_id[state["law_folder_id"]].get("name") == "勞基法",
            },
            {
                "case": "law-folder-membership-survives-restart",
                "passed": all(
                    document_by_id[source_id].get("folder_id") == state["law_folder_id"]
                    for source_id in state["law_source_ids"]
                ),
            },
            {
                "case": "deleted-folder-documents-survive-in-uncategorized",
                "passed": all(
                    document_by_id[source_id].get("folder_id") == "uncategorized"
                    for source_id in state["uncategorized_source_ids"]
                ),
            },
            {
                "case": "profile-api-overlays-current-folder-metadata",
                "passed": all(
                    profile_by_id[source_id].get("folder_id") == state["law_folder_id"]
                    for source_id in state["law_source_ids"]
                )
                and all(
                    profile_by_id[source_id].get("folder_id") == "uncategorized"
                    for source_id in state["uncategorized_source_ids"]
                ),
            },
        ]
    )

    _, retrieval = request_json(
        f"{base_url}/api/retrieve",
        method="POST",
        payload={
            "profile": "ifrs17",
            "question": "TXT-731",
            "retrieval_query": "TXT-731",
            "source_ids": state["law_source_ids"],
            "top_k": 4,
        },
    )
    contexts = retrieval.get("contexts") or []
    cases.append(
        {
            "case": "folder-level-source-selection-after-restart",
            "passed": bool(contexts)
            and all(context.get("source") in state["law_source_ids"] for context in contexts)
            and any("TXT-731" in context.get("content", "") for context in contexts),
        }
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "summary": summarize(cases),
        "cases": cases,
        "folder_snapshot": folders,
        "document_snapshot": documents,
    }
    (EVAL_ROOT / "latest-results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (EVAL_ROOT / "latest-report.md").write_text(render_markdown(report), encoding="utf-8")
    return report["summary"]


def summarize(cases):
    return {
        "passed": sum(bool(case["passed"]) for case in cases),
        "total": len(cases),
    }


def render_markdown(report):
    lines = [
        "# Document folder evaluation",
        "",
        f"- Result: {report['summary']['passed']}/{report['summary']['total']} passed",
        f"- Generated: {report['generated_at']}",
        "",
        "## Cases",
        "",
    ]
    for case in report["cases"]:
        lines.append(f"- {'PASS' if case['passed'] else 'FAIL'}: `{case['case']}`")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["setup", "verify"])
    parser.add_argument("--base-url", default="http://127.0.0.1:8878")
    args = parser.parse_args()
    result = setup(args.base_url) if args.phase == "setup" else verify(args.base_url)
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result["passed"] == result["total"] else 1)
