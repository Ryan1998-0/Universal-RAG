#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_demo.rag_pipeline import RagPipeline, RagPipelineRequest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an evaluation set through the same canonical RAGPipeline used by /api/ask."
    )
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default="ollama:qwen2.5:7b")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--top-k", type=int)
    args = parser.parse_args()

    questions = load_questions(args.questions)
    if not questions:
        raise SystemExit("No evaluation questions were found.")

    output_path = args.output or default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline = RagPipeline()
    started_at = utc_now()
    results = []

    for index, item in enumerate(questions, start=1):
        question = str(item.get("question") or "").strip()
        if not question:
            raise ValueError(f"Question {index} is empty.")
        result = pipeline.run(RagPipelineRequest(
            question=question,
            model=str(item.get("model") or args.model),
            profile=str(item.get("profile") or args.profile),
            source_ids=item.get("source_ids"),
            top_k=item.get("top_k") or args.top_k,
            persist_conversation=False,
        ))
        results.append({
            "case_id": str(item.get("id") or f"case-{index:04d}"),
            "question": question,
            "expectations": dict(item.get("expectations") or {}),
            "result": result,
        })

    artifact = {
        "schema_version": "canonical-rag-eval-v1",
        "started_at": started_at,
        "finished_at": utc_now(),
        "provenance": repository_provenance(),
        "configuration": {
            "model": args.model,
            "profile": args.profile,
            "top_k": args.top_k,
            "question_file": portable_path(args.questions),
            "question_count": len(questions),
        },
        "results": results,
    }
    output_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(output_path)


def load_questions(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict) and isinstance(parsed.get("questions"), list):
        return parsed["questions"]
    raise ValueError("Question file must be a JSON array, JSONL, or an object with questions[].")


def repository_provenance() -> dict:
    return {
        "commit": git_output("rev-parse", "HEAD"),
        "dirty": bool(git_output("status", "--porcelain")),
    }


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def git_output(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def default_output_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return PROJECT_ROOT / "evals" / "canonical_pipeline" / "runs" / f"canonical-{stamp}.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    main()
