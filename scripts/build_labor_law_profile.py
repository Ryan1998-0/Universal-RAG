#!/usr/bin/env python3
"""Build the reusable labor-law retrieval profile from the fetched source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import re

from rag_demo.parent_child import build_parent_child_index


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "profiles" / "labor_standards_act" / "raw" / "labor_standards_act.txt"
DEFAULT_PROFILE = ROOT / "profiles" / "labor_standards_act"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=ROOT / "profiles" / "labor_standards_act" / "raw" / "metadata.json",
    )
    args = parser.parse_args()
    source = args.source.resolve().read_text(encoding="utf-8")
    metadata = {}
    if args.metadata.exists():
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    revision = str(metadata.get("revision") or "官方頁面最新版本").strip()
    source_url = str(metadata.get("source_url") or "").strip()
    matches = re.findall(
        r"(?m)^(第\s*[^\n]+\s*條)\n(.*?)(?=^第\s*[^\n]+\s*條\n|\Z)",
        source,
        flags=re.DOTALL,
    )
    units = [
        {"title": article.strip(), "page": article.strip(), "content": f"{article.strip()}\n{content.strip()}"}
        for article, content in matches
    ]
    index = build_parent_child_index(
        units,
        source_id="labor-standards-act",
        filename="勞動基準法",
        source_type="official-law",
        extraction_method="official-text",
        parent_size_tokens=1024,
        child_size_tokens=256,
        parent_overlap_tokens=0,
        child_overlap_tokens=0,
    )
    chunks = index.children
    payload = {
        "meta": {
            "title": "勞動基準法",
            "revision": revision,
            "source_url": source_url,
        },
        "sources": [
            {
                "source_id": "labor-standards-act",
                "title": f"勞動基準法（{revision}）",
                "source_type": "official-law",
                "url": source_url,
            }
        ],
        "aliases": [],
        "graph": {"relations": []},
        "chunks": chunks,
    }
    output = args.profile.resolve() / "data" / "retrieval_data.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "parent_count": len(index.parents), "child_count": len(chunks)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
