#!/usr/bin/env python3
"""Strip oracle answers and render only retrieved evidence for the answer model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("retrieval_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.retrieval_json.read_text(encoding="utf-8"))
    items = []
    for result in payload["results"]:
        items.append(
            {
                "id": result["id"],
                "question": result["question"],
                "comparison_articles": result["comparison_articles"],
                "evidence_2016": result["retrieval_2016"]["pass_2_fine"]["contexts"],
                "evidence_2026": result["retrieval_2026"]["pass_2_fine"]["contexts"],
                "retrieval_trace": {
                    "2016": result["retrieval_2016"]["pass_2_fine"]["trace"],
                    "2026": result["retrieval_2026"]["pass_2_fine"]["trace"],
                },
            }
        )
    output = {
        "benchmark_id": payload["benchmark_id"],
        "mode": "answer-input-evidence-only",
        "source_ids": payload["source_ids"],
        "two_pass_policy": payload["two_pass_policy"],
        "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "question_count": len(items)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
