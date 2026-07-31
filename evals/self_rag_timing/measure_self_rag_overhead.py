from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("RAG_PROFILE", "ifrs17")

from rag_demo.knowledge_base import active_knowledge_base
from rag_demo.query import answer_question_self_rag, answer_question_v2


DEFAULT_QUESTIONS = [
    "你好",
    "IFRS17 的 CSM 是什麼？",
    "舊制 IFRS 4 和新制 IFRS 17 差在哪？",
    "risk adjustment for non-financial risk 是什麼？",
    "IFRS17 是否改變 insurance contract 的定義？",
]
TIMING_PATTERN = re.compile(r"^-\s+([A-Za-z0-9_]+):\s+([0-9.]+)s\s*$", re.MULTILINE)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or Path(__file__).resolve().parent / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = output_dir / f"self_rag_timing_{run_id}.json"
    report_path = output_dir / f"self_rag_timing_{run_id}.md"
    questions = load_questions(args.questions_file, limit=args.limit)
    kb_status = inspect_knowledge_base()

    records = []
    for index, question in enumerate(questions, start=1):
        print(f"[{index}/{len(questions)}] baseline v2: {question}", flush=True)
        baseline = run_pipeline(
            label="baseline_v2",
            fn=lambda: answer_question_v2(question, model=args.model, top_k=args.top_k),
        )
        print(f"[{index}/{len(questions)}] engineering self-rag: {question}", flush=True)
        self_rag = run_pipeline(
            label="engineering_self_rag",
            fn=lambda: answer_question_self_rag(
                question,
                model=args.model,
                top_k=args.top_k,
                max_attempts=args.max_attempts,
            ),
        )
        records.append(
            {
                "question": question,
                "baseline": baseline,
                "self_rag": self_rag,
                "summary": summarize_pair(question, baseline, self_rag),
            }
        )

    payload = {
        "run_id": run_id,
        "model": args.model,
        "rag_profile": os.getenv("RAG_PROFILE", ""),
        "top_k": args.top_k,
        "max_attempts": args.max_attempts,
        "knowledge_base_status": kb_status,
        "records": records,
        "aggregate": aggregate_summaries([record["summary"] for record in records]),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_markdown_report(payload, json_path), encoding="utf-8")
    print(f"JSON: {json_path}")
    print(f"Report: {report_path}")
    return 0


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Measure baseline RAG v2 vs engineering Self-RAG controller timing.")
    parser.add_argument("--model", default=os.getenv("RAG_MODEL", "ollama:qwen2.5:7b"))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--questions-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-id", default="")
    return parser.parse_args(argv)


def load_questions(path: Optional[Path], limit: Optional[int] = None) -> List[str]:
    if path is None:
        questions = list(DEFAULT_QUESTIONS)
    elif path.suffix.lower() == ".json":
        loaded = json.loads(path.read_text(encoding="utf-8"))
        questions = []
        for item in loaded:
            if isinstance(item, str):
                questions.append(item)
            elif isinstance(item, dict) and item.get("question"):
                questions.append(str(item["question"]))
    else:
        questions = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None:
        return questions[: max(0, int(limit))]
    return questions


def run_pipeline(label: str, fn: Callable[[], str]) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        output = fn()
        elapsed = time.perf_counter() - started
        timings = extract_timing_nodes(output)
        timings.setdefault("outer_wall_seconds", round(elapsed, 4))
        return {
            "label": label,
            "ok": True,
            "elapsed_seconds": round(elapsed, 4),
            "timings": timings,
            "output_preview": output[:1200],
        }
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return {
            "label": label,
            "ok": False,
            "elapsed_seconds": round(elapsed, 4),
            "timings": {"outer_wall_seconds": round(elapsed, 4)},
            "error": f"{type(exc).__name__}: {exc}",
        }


def extract_timing_nodes(output: str) -> Dict[str, float]:
    timings = {}
    for key, value in TIMING_PATTERN.findall(str(output or "")):
        timings[key] = float(value)
    return timings


def summarize_pair(question: str, baseline: Dict[str, Any], self_rag: Dict[str, Any]) -> Dict[str, Any]:
    baseline_total = _total_seconds(baseline)
    self_rag_total = _total_seconds(self_rag)
    overhead = None
    multiplier = None
    if baseline_total is not None and self_rag_total is not None:
        overhead = round(self_rag_total - baseline_total, 4)
        multiplier = round(self_rag_total / baseline_total, 4) if baseline_total > 0 else None
    return {
        "question": question,
        "ok": bool(baseline.get("ok")) and bool(self_rag.get("ok")),
        "baseline_total_seconds": baseline_total,
        "self_rag_total_seconds": self_rag_total,
        "overhead_seconds": overhead,
        "overhead_multiplier": multiplier,
        "baseline_error": baseline.get("error", ""),
        "self_rag_error": self_rag.get("error", ""),
    }


def aggregate_summaries(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok_summaries = [item for item in summaries if item.get("ok")]
    overheads = [item["overhead_seconds"] for item in ok_summaries if item.get("overhead_seconds") is not None]
    multipliers = [item["overhead_multiplier"] for item in ok_summaries if item.get("overhead_multiplier") is not None]
    return {
        "cases": len(summaries),
        "ok_cases": len(ok_summaries),
        "mean_overhead_seconds": round(statistics.mean(overheads), 4) if overheads else None,
        "median_overhead_seconds": round(statistics.median(overheads), 4) if overheads else None,
        "mean_overhead_multiplier": round(statistics.mean(multipliers), 4) if multipliers else None,
        "median_overhead_multiplier": round(statistics.median(multipliers), 4) if multipliers else None,
    }


def inspect_knowledge_base() -> Dict[str, Any]:
    kb = active_knowledge_base(project_root=ROOT)
    index_path = kb.index_dir / "chunks.json"
    embedding_path = kb.index_dir / "embeddings.npy"
    raw_files = [path for path in kb.raw_dir.glob("**/*") if path.is_file() and not path.name.startswith(".")]
    return {
        "profile": kb.name,
        "raw_dir": portable_path(kb.raw_dir),
        "index_dir": portable_path(kb.index_dir),
        "chunks_index_exists": index_path.exists(),
        "embeddings_exists": embedding_path.exists(),
        "raw_file_count": len(raw_files),
        "warning": "" if index_path.exists() else "No local chunks.json index found; retrieval timing may include empty-KB fallback behavior.",
    }


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def render_markdown_report(payload: Dict[str, Any], json_path: Path) -> str:
    aggregate = payload["aggregate"]
    kb_status = payload["knowledge_base_status"]
    lines = [
        "# Engineering Self-RAG Timing Report",
        "",
        f"- Run ID: `{payload['run_id']}`",
        f"- Model: `{payload['model']}`",
        f"- RAG profile: `{payload['rag_profile']}`",
        f"- Top K: `{payload['top_k']}`",
        f"- Max attempts: `{payload['max_attempts']}`",
        f"- Raw JSON: `{portable_path(json_path)}`",
        f"- KB chunks index exists: `{kb_status['chunks_index_exists']}`",
        f"- KB embeddings exists: `{kb_status['embeddings_exists']}`",
        f"- KB raw file count: `{kb_status['raw_file_count']}`",
    ]
    if kb_status.get("warning"):
        lines.append(f"- Warning: {kb_status['warning']}")
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- OK cases: `{aggregate['ok_cases']}/{aggregate['cases']}`",
            f"- Mean overhead seconds: `{aggregate['mean_overhead_seconds']}`",
            f"- Median overhead seconds: `{aggregate['median_overhead_seconds']}`",
            f"- Mean overhead multiplier: `{aggregate['mean_overhead_multiplier']}`",
            f"- Median overhead multiplier: `{aggregate['median_overhead_multiplier']}`",
            "",
            "## Cases",
            "",
            "| Question | Baseline total | Self-RAG total | Overhead | Multiplier | Status |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for record in payload["records"]:
        summary = record["summary"]
        status = "ok" if summary["ok"] else "error"
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(summary["question"]),
                    _format_number(summary["baseline_total_seconds"]),
                    _format_number(summary["self_rag_total_seconds"]),
                    _format_number(summary["overhead_seconds"]),
                    _format_number(summary["overhead_multiplier"]),
                    status,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _total_seconds(record: Dict[str, Any]) -> Optional[float]:
    timings = record.get("timings") or {}
    if "total" in timings:
        return float(timings["total"])
    if "outer_wall_seconds" in timings:
        return float(timings["outer_wall_seconds"])
    if "elapsed_seconds" in record:
        return float(record["elapsed_seconds"])
    return None


def _format_number(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def _md_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())
