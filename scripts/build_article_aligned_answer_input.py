#!/usr/bin/env python3
"""Build evidence-only inputs from the article-aligned two-pass run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("retrieval_json", type=Path)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    retrieval = json.loads(args.retrieval_json.read_text(encoding="utf-8"))
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    benchmark_by_id = {item["id"]: item for item in benchmark["items"]}
    items = []
    for result in retrieval["results"]:
        source = benchmark_by_id[result["id"]]
        items.append(
            {
                "id": result["id"],
                "question": result["question"],
                "changed_articles": source.get("changed_articles", []),
                "evidence_2016": result["aligned_2016"]["fine_contexts"],
                "evidence_2026": result["aligned_2026"]["fine_contexts"],
                "article_alignment": {
                    "2016": {
                        "available_articles": result["aligned_2016"]["available_articles"],
                        "missing_articles": result["aligned_2016"]["missing_articles"],
                        "article_recall": result["aligned_2016"]["article_recall"],
                    },
                    "2026": {
                        "available_articles": result["aligned_2026"]["available_articles"],
                        "missing_articles": result["aligned_2026"]["missing_articles"],
                        "article_recall": result["aligned_2026"]["article_recall"],
                    },
                },
                "retrieval_trace": {
                    "2016": result["aligned_2016"]["fine_trace"],
                    "2026": result["aligned_2026"]["fine_trace"],
                },
            }
        )
    output = {
        "benchmark_id": retrieval["benchmark_id"],
        "mode": "answer-input-article-aligned-evidence-only",
        "source_files": retrieval["source_files"],
        "two_pass_policy": retrieval["policy"],
        "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "question_count": len(items)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
