#!/usr/bin/env python3
"""Fetch the current Labor Standards Act text from the Ministry of Labor.

The source page is intentionally kept in the generated metadata so benchmark
results can be traced back to the exact revision used by the corpus.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "https://laws.mol.gov.tw/FLAW/PrintFLAWDAT0201.aspx?id=FL014930"
DEFAULT_OUTPUT = ROOT / "profiles" / "labor_standards_act" / "raw" / "labor_standards_act.txt"
DEFAULT_METADATA = ROOT / "profiles" / "labor_standards_act" / "raw" / "metadata.json"


def _strip_tags(value: str) -> str:
    value = re.sub(r"<[^>]+>", "", value)
    value = html.unescape(value).replace("\r", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.splitlines()]
    return "\n".join(line for line in lines if line)


def parse_articles(page: str) -> list[dict[str, str]]:
    pattern = re.compile(
        r'<div class="itemName">.*?>(.*?)</a>.*?'
        r'<div class="itemContent">\s*<pre>(.*?)</pre>',
        re.IGNORECASE | re.DOTALL,
    )
    articles = []
    for name, content in pattern.findall(page):
        article = _strip_tags(name)
        text = _strip_tags(content)
        if article and text:
            articles.append({"article": article, "content": text})
    if not articles:
        raise RuntimeError("No law articles were found in the official page.")
    return articles


def extract_revision(page: str) -> str:
    """Return the revision shown by the official page, if present."""

    visible_text = html.unescape(re.sub(r"<[^>]+>", " ", page))
    match = re.search(
        r"民國\s*(\d+)\s*年\s*(\d+)\s*月\s*(\d+)\s*日\s*修正",
        visible_text,
    )
    if not match:
        return "官方頁面最新版本"
    year, month, day = match.groups()
    return f"民國 {year} 年 {month} 月 {day} 日修正"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    args = parser.parse_args()

    request = Request(
        args.url,
        headers={"User-Agent": "Universal-RAG research benchmark/1.0"},
    )
    with urlopen(request, timeout=60) as response:
        page = response.read().decode("utf-8", errors="replace")
    articles = parse_articles(page)
    revision = extract_revision(page)
    output = [
        "勞動基準法",
        "資料來源：勞動部勞動法令查詢系統",
        f"來源網址：{args.url}",
        f"版本：{revision}",
        "",
    ]
    for item in articles:
        output.extend([item["article"], item["content"], ""])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(output), encoding="utf-8")
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(
        json.dumps(
            {
                "title": "勞動基準法",
                "source_url": args.url,
                "source_authority": "勞動部勞動法令查詢系統",
                "revision": revision,
                "article_count": len(articles),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "article_count": len(articles)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
