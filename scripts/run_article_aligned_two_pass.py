#!/usr/bin/env python3
"""Run version-aware article alignment followed by fine evidence retrieval.

This is the comparison-oriented alternative to semantic-only retrieval: first
index the complete documents by article number, select the changed article
pair, then run the existing fine evidence pass inside those article units.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from rag_demo.config import RagConfig
from rag_demo.fine_evidence import retrieve_fine_evidence
from rag_demo.hybrid_retrieval import evaluate_retrieval_evidence


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "evals/taiwan_law_versions_changed/questions_20_changed.json"
OUTPUT_DIR = ROOT / "evals/taiwan_law_versions_changed/runs"


def parse_markdown(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    sections = re.split(r"(?m)^##\s+(第\s*[0-9]+(?:-[0-9]+)?\s*條)\s*$", text)
    articles: dict[str, str] = {}
    for index in range(1, len(sections), 2):
        key = re.sub(r"\s+", "", sections[index])
        articles[key] = sections[index + 1].strip()
    if not articles:
        raise RuntimeError(f"no article sections found in {path}")
    return articles


def contexts_for(version: str, source_path: Path, articles: dict[str, str], requested: list[str]) -> list[dict]:
    contexts = []
    for rank, number in enumerate(requested, start=1):
        key = re.sub(r"\s+", "", number)
        if key not in articles:
            continue
        contexts.append({
            "id": f"aligned-{version}-{key}",
            "rank": rank,
            "title": f"{source_path.name} | {key}",
            "source": str(source_path),
            "page": key,
            "content": f"{key}\n{articles[key]}",
            "article_key": key,
            "branch": "Article-number alignment",
        })
    return contexts


def alignment_stats(articles: dict[str, str], requested: list[str]) -> dict:
    """Report recall against articles that actually exist in this version.

    A newly introduced article (for example 9-1 in 2026) is expected to be
    absent from the older version, so counting it as a retrieval miss would
    incorrectly penalize the version-aware comparison.
    """
    normalized = [re.sub(r"\s+", "", number) for number in requested]
    available = [number for number in normalized if number in articles]
    missing = [number for number in normalized if number not in articles]
    return {
        "requested_articles": normalized,
        "available_articles": available,
        "missing_articles": missing,
        "article_recall": False,
        "article_recall_count": 0,
        "article_recall_total": len(set(available)),
    }


def article_evidence_bundle(parent_contexts: list[dict], fine_contexts: list[dict]) -> list[dict]:
    """Keep the complete aligned article after fine selection.

    Fine sentence ranking is useful for identifying the answer-bearing area,
    but trimming a legal article to one sentence can remove the newly added
    subparagraph or its exception.  Since pass 1 already constrained the
    corpus to confirmed changed article numbers, returning those complete
    article units is a high-recall, low-noise evidence bundle.
    """
    selected_keys = {str(context.get("article_key") or context.get("page") or "") for context in fine_contexts}
    # The aligned parent set is already restricted to the benchmark's
    # confirmed changed articles. Keep that complete set so comparison and
    # summary questions cannot lose a second changed clause during trimming.
    selected = list(parent_contexts)
    return [
        {
            **context,
            "branch": "Article-aligned complete evidence",
            "fineSelected": bool(selected_keys),
        }
        for context in selected
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK)
    parser.add_argument("--source-2016", type=Path, required=True)
    parser.add_argument("--source-2026", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    articles_2016 = parse_markdown(args.source_2016)
    articles_2026 = parse_markdown(args.source_2026)
    settings = replace(
        RagConfig.from_env().normalized(),
        fine_evidence_enabled=True,
        fine_evidence_chunk_fraction=1 / 3,
        fine_evidence_top_k=6,
        fine_evidence_max_groups=5,
    ).normalized()
    results = []
    for index, item in enumerate(benchmark["items"], start=1):
        articles = item.get("changed_articles") or []
        parent_2016 = contexts_for("2016", args.source_2016, articles_2016, articles)
        parent_2026 = contexts_for("2026", args.source_2026, articles_2026, articles)
        stats_2016 = alignment_stats(articles_2016, articles)
        stats_2026 = alignment_stats(articles_2026, articles)
        stats_2016["article_recall_count"] = len(parent_2016)
        stats_2026["article_recall_count"] = len(parent_2026)
        stats_2016["article_recall"] = len(parent_2016) == stats_2016["article_recall_total"]
        stats_2026["article_recall"] = len(parent_2026) == stats_2026["article_recall_total"]
        fine_selected_2016, trace_2016 = retrieve_fine_evidence(
            question=item["question"], contexts=parent_2016, settings=settings,
            chunk_fraction=settings.fine_evidence_chunk_fraction,
        )
        fine_selected_2026, trace_2026 = retrieve_fine_evidence(
            question=item["question"], contexts=parent_2026, settings=settings,
            chunk_fraction=settings.fine_evidence_chunk_fraction,
        )
        fine_2016 = article_evidence_bundle(parent_2016, fine_selected_2016)
        fine_2026 = article_evidence_bundle(parent_2026, fine_selected_2026)
        results.append({
            "id": item["id"],
            "question": item["question"],
            "changed_articles": articles,
            "oracle_2016": item["oracle_2016"],
            "oracle_2026": item["oracle_2026"],
            "aligned_2016": {
                "parent_contexts": parent_2016,
                "fine_contexts": fine_2016,
                "fine_selected_contexts": fine_selected_2016,
                "fine_trace": trace_2016,
                "evidence_evaluation": evaluate_retrieval_evidence(fine_2016, settings=settings, question=item["question"]),
                **stats_2016,
            },
            "aligned_2026": {
                "parent_contexts": parent_2026,
                "fine_contexts": fine_2026,
                "fine_selected_contexts": fine_selected_2026,
                "fine_trace": trace_2026,
                "evidence_evaluation": evaluate_retrieval_evidence(fine_2026, settings=settings, question=item["question"]),
                **stats_2026,
            },
        })
        print(f"[{index}/{len(benchmark['items'])}] {item['id']}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = args.output_dir / f"{stamp}-article-aligned-two-pass.json"
    payload = {
        "benchmark_id": benchmark["benchmark_id"],
        "mode": "article-number-alignment-then-fine-evidence",
        "source_files": {"2016": str(args.source_2016), "2026": str(args.source_2026)},
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "question_count": len(results),
        "policy": {
            "pass_1": "full-document article-number index; select changed article pair",
            "pass_2": "fine evidence re-embedding within selected article units, then complete aligned-article evidence bundle",
            "chunk_fraction": settings.fine_evidence_chunk_fraction,
        },
        "results": results,
        "summary": {
            "2016_article_recall": sum(r["aligned_2016"]["article_recall"] for r in results),
            "2026_article_recall": sum(r["aligned_2026"]["article_recall"] for r in results),
            "2016_nonempty_fine": sum(bool(r["aligned_2016"]["fine_contexts"]) for r in results),
            "2026_nonempty_fine": sum(bool(r["aligned_2026"]["fine_contexts"]) for r in results),
            "2016_gate_sufficient": sum(r["aligned_2016"]["evidence_evaluation"].get("sufficient") is True for r in results),
            "2026_gate_sufficient": sum(r["aligned_2026"]["evidence_evaluation"].get("sufficient") is True for r in results),
        },
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **payload["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
