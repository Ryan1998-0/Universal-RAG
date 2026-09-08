#!/usr/bin/env python3
"""Build an article-aligned benchmark containing only confirmed changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_labor_rule_version_benchmark import extract_articles, render_snapshot


ROOT = Path(__file__).resolve().parents[1]
VERSION_DIR = ROOT / "evals/taiwan_law_versions"
OUT = ROOT / "evals/taiwan_law_versions_changed"


QUESTIONS = [
    ("c01-article6-term", "第6條由2016版到2026版，職業災害致身體狀況的用語改了哪一個詞？", ["第 6 條"], "用語替換"),
    ("c02-article6-meaning", "比較第6條兩版，除了用語變更外，公傷病假的給假期間與觸發條件是否仍相同？", ["第 6 條"], "用語替換但核心規則保留"),
    ("c03-article7-wording", "第7條第一句在2016版與2026版的文字有何細微修改？請指出刪除或保留的語助詞。", ["第 7 條"], "文字修訂"),
    ("c04-article7-family-care", "2026版第7條新增了哪一種事假使用情境？請與2016版原有事由比較。", ["第 7 條"], "新增家庭照顧事由"),
    ("c05-article7-family-exception", "2026版家庭照顧事假之前的法律例外條件是什麼？這段在修法中增加了什麼限制語意？", ["第 7 條"], "新增法律例外"),
    ("c06-article7-hour-unit", "2026版第7條新增的請假單位選擇是什麼？它適用於哪一種新增情境？", ["第 7 條"], "新增小時單位"),
    ("c07-article7-limit-carryover", "家庭照顧事假依第7條前項規定辦理時，2026版如何承接原有事假一年14日規則？", ["第 7 條"], "新增事由承接既有上限"),
    ("c08-article9-opening", "第9條開頭從2016版的『不得扣發』改成2026版的什麼全勤獎金保護表述？", ["第 9 條"], "保護語意改寫"),
    ("c09-article9-miscarriage", "2026版第9條新增的流產與普通傷病假情境，具體保護了什麼？", ["第 9 條"], "新增流產情境"),
    ("c10-article9-family-attendance", "2026版第9條新增家庭照顧事假列舉後，雇主不得將該假如何處理？", ["第 9 條"], "新增家庭照顧全勤保護"),
    ("c11-article9-proportional", "2026版第9條對普通傷病假全勤獎金扣發新增了什麼計算方式？", ["第 9 條"], "新增按日數比例扣發"),
    ("c12-article9-better-agreement", "2026版第9條對勞雇雙方另有較優約定，新增了什麼保留條款？", ["第 9 條"], "新增較優約定例外"),
    ("c13-article9-other-law", "2026版第9條在全勤獎金保護列舉後新增哪一個法律衝突處理語句？", ["第 9 條"], "新增其他法律例外"),
    ("c14-article91-new", "2026版新增第9-1條的核心保護對象與普通傷病假日數門檻是什麼？", ["第 9-1 條"], "新增條文與十日門檻"),
    ("c15-article91-burden", "第9-1條新增的雇主舉證責任，要求證明哪兩者之間沒有關聯？", ["第 9-1 條"], "新增舉證責任"),
    ("c16-article91-appraisal", "第9-1條對普通傷病假超過門檻後的人事考核，新增哪些不得只看單一因素的要求？", ["第 9-1 條"], "新增考核限制"),
    ("c17-article12-effective", "2026版第12條新增的修正條文施行日期是哪一天？2016版第12條原本只保留什麼一般規定？", ["第 12 條"], "新增施行日期"),
    ("c18-article12-scope", "第12條從2016版到2026版，新增施行日期句子的法律效果是什麼？", ["第 12 條"], "施行範圍明確化"),
    ("c19-changed-articles", "依兩份全文的條號對照，確定有實質變更或新增內容的條文有哪些？請列出條號。", ["第 6 條", "第 7 條", "第 9 條", "第 9-1 條", "第 12 條"], "條號差異盤點"),
    ("c20-change-summary", "綜合比較第6、7、9、9-1、12條，請分別概括：用語變更、新增請假情境、全勤保護、舉證／考核限制、施行日期變更。", ["第 6 條", "第 7 條", "第 9 條", "第 9-1 條", "第 12 條"], "跨條文改動摘要"),
]


def article_context(articles: dict[str, str], numbers: list[str]) -> str:
    parts = []
    for number in numbers:
        key = number.replace(" ", "")
        parts.append(f"{number}\n{articles.get(key, '[此版本缺少此條文]')}")
    return "\n\n".join(parts)


def markdown_fulltext(label: str, effective: str, source_url: str, articles: dict[str, str]) -> str:
    lines = [
        f"# 勞工請假規則｜{label}",
        f"- 有效日期：{effective}",
        f"- 官方來源：{source_url}",
        "",
    ]
    for number, text in articles.items():
        lines.extend([f"## {number}", text, ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "source").mkdir(exist_ok=True)
    source_2016 = VERSION_DIR / "source/勞工請假規則_2016有效版.txt"
    source_2026 = VERSION_DIR / "source/勞工請假規則_2026有效版.txt"
    # The generated TXT files are already normalized; parse article blocks directly.
    import re
    def parse(path: Path) -> dict[str, str]:
        text = path.read_text(encoding="utf-8")
        blocks = re.split(r"(?m)^第([0-9]+(?:-[0-9]+)?)條\s*$", text)
        result = {}
        for index in range(1, len(blocks), 2):
            result[f"第{blocks[index]}條"] = blocks[index + 1].strip()
        return result
    articles_2016 = parse(source_2016)
    articles_2026 = parse(source_2026)
    url_2016 = "https://laws.mol.gov.tw/FLAW/FLAWDAT0801.aspx?id=FL014935&ldate=20111014"
    url_2026 = "https://laws.mol.gov.tw/FLAW/FLAWDAT0801.aspx?id=FL014935&ldate=20251209"
    (args.output_dir / "source/勞工請假規則_2016全文分條.md").write_text(
        markdown_fulltext("2016有效版", "2016-01-01 至 2016-12-31", url_2016, articles_2016), encoding="utf-8"
    )
    (args.output_dir / "source/勞工請假規則_2026全文分條.md").write_text(
        markdown_fulltext("2026有效版", "2026-01-01 起", url_2026, articles_2026), encoding="utf-8"
    )
    items = []
    for item_id, question, articles, change_type in QUESTIONS:
        items.append({
            "id": item_id,
            "question": question,
            "changed_articles": articles,
            "change_type": change_type,
            "oracle_2016": article_context(articles_2016, articles),
            "oracle_2026": article_context(articles_2026, articles),
        })
    payload = {
        "benchmark_id": "taiwan-labor-leave-confirmed-changes-v2",
        "benchmark_type": "article-aligned-confirmed-change-comparison",
        "source": "勞動部勞動法令查詢系統",
        "version_2016": {"source_file": "source/勞工請假規則_2016全文分條.md", "snapshot_url": url_2016},
        "version_2026": {"source_file": "source/勞工請假規則_2026全文分條.md", "snapshot_url": url_2026},
        "changed_articles": ["第 6 條", "第 7 條", "第 9 條", "第 9-1 條", "第 12 條"],
        "question_count": len(items),
        "items": items,
    }
    (args.output_dir / "questions_20_changed.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        "# 2016 vs 2026 已確認改動條文評測 v2\n\n"
        "兩份全文先依條號切成 Markdown section，再針對已確認有變更的第 6、7、9、9-1、12 條設計 20 題。\n"
        "這版避免用『某版本是否沒有規定』作為主要題型；每一題都要求模型比較實際改動內容。\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "question_count": len(items), "changed_articles": payload["changed_articles"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
