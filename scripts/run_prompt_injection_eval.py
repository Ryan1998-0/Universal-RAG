#!/usr/bin/env python3
"""Measure the production ingestion scanner against a frozen attack corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_demo.production.file_security import scan_prompt_injection  # noqa: E402


def evaluate(corpus: dict) -> dict:
    if corpus.get("schema_version") != "prompt-injection-corpus-v1":
        raise ValueError("unsupported corpus schema")
    cases = corpus.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("corpus needs nonempty cases")
    results = []
    counts = {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0}
    ids = set()
    for case in cases:
        case_id = str(case.get("id") or "")
        if not case_id or case_id in ids or not isinstance(case.get("expected_quarantine"), bool):
            raise ValueError("cases need unique IDs and boolean expected_quarantine")
        ids.add(case_id)
        findings = scan_prompt_injection(str(case.get("text") or ""))
        predicted = bool(findings)
        expected = case["expected_quarantine"]
        bucket = (
            "true_positive" if expected and predicted else
            "false_negative" if expected else
            "false_positive" if predicted else "true_negative"
        )
        counts[bucket] += 1
        results.append({
            "id": case_id,
            "category": case.get("category", ""),
            "expected_quarantine": expected,
            "predicted_quarantine": predicted,
            "findings": findings,
            "passed": expected == predicted,
        })
    attack_count = counts["true_positive"] + counts["false_negative"]
    benign_count = counts["true_negative"] + counts["false_positive"]
    return {
        "schema_version": "prompt-injection-scan-eval-v1",
        "status": "passed" if all(item["passed"] for item in results) else "failed",
        "summary": {
            "case_count": len(results),
            **counts,
            "attack_recall": counts["true_positive"] / attack_count if attack_count else None,
            "benign_pass_rate": counts["true_negative"] / benign_count if benign_count else None,
        },
        "cases": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path,
        default=PROJECT_ROOT / "evals" / "prompt_injection" / "corpus.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    corpus_path = args.corpus.resolve()
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    report = evaluate(corpus)
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
        capture_output=True, text=True, check=False,
    )
    report["provenance"] = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "commit": git.stdout.strip() if git.returncode == 0 else "",
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "scope": "static scanner only; no OCR, ingestion, retrieval or model answer",
    }
    output = args.output or (
        PROJECT_ROOT / "evals" / "prompt_injection" / "runs"
        / f"scan-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "summary": report["summary"], "artifact": str(output)}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(f"prompt injection eval could not run: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
