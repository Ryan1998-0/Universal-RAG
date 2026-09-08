#!/usr/bin/env python3
"""Download official labor-leave-rule snapshots and build a comparison set.

The 2016 snapshot is the last version effective before 2016 (2011-10-14);
the history page has no amendment in 2016.  The 2026 snapshot is the version
promulgated on 2025-12-09 and effective 2026-01-01.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import urllib.request
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "evals/taiwan_law_versions"
HISTORY_URL = "https://laws.mol.gov.tw/FLAW/FLAWDAT07.aspx?id=FL014935"
SNAPSHOT_URL = "https://laws.mol.gov.tw/FLAW/FLAWDAT0801.aspx?id=FL014935&ldate={}"


def fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Universal-RAG-eval/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def extract_articles(source_html: str) -> dict[str, str]:
    rows = re.findall(
        r'<div class="col-no">(.*?)</div>\s*<div class="col-data">\s*<pre>(.*?)</pre>',
        source_html,
        re.S,
    )
    articles: dict[str, str] = {}
    for raw_number, raw_text in rows:
        number = re.sub(r"<[^>]+>", " ", raw_number)
        number = " ".join(html.unescape(number).split())
        number = number.replace("第 ", "第").replace(" 條", "條")
        text = html.unescape(raw_text).strip().replace("\r\n", "\n")
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        articles[number] = text
    if not articles:
        raise RuntimeError("official page did not contain article rows")
    return articles


def render_snapshot(title: str, effective_date: str, amendment_date: str, source_url: str, articles: dict[str, str]) -> str:
    lines = [
        "法規名稱：勞工請假規則",
        f"版本標籤：{title}",
        f"有效日期：{effective_date}",
        f"最近修正日期：{amendment_date}",
        "資料來源：勞動部勞動法令查詢系統",
        f"官方全文：{source_url}",
        "",
    ]
    for number, text in articles.items():
        lines.extend([number, text, ""])
    return "\n".join(lines).rstrip() + "\n"


def build_questions(articles_2016: dict[str, str], articles_2026: dict[str, str]) -> list[dict]:
    def context(articles: dict[str, str], *numbers: str) -> str:
        return "\n\n".join(
            f"{number}\n{articles.get(number.replace(' ', ''), '[此版本無此條文]')}"
            for number in numbers
        )

    questions = [
        ("q01-article6-term", "第6條的職業災害用語，2016版與2026版分別使用什麼詞？", ["第 6 條"], ["殘廢", "失能"]),
        ("q02-article6-period", "兩個版本第6條對公傷病假的期間判定是否相同？請比較共同條件。", ["第 6 條"], ["治療、休養期間"]),
        ("q03-personal-limit", "2016版與2026版事假一年上限各是多少？是否改變？", ["第 7 條"], ["十四日"]),
        ("q04-family-care", "2026版相較2016版，事假新增了哪一種照顧情境？", ["第 7 條"], ["親自照顧家庭成員"]),
        ("q05-family-care-hour", "哪一個版本明文允許家庭照顧事假以小時為請假單位？", ["第 7 條"], ["小時"]),
        ("q06-family-care-condition", "2026版家庭照顧事假受到哪些法律例外限制？2016版是否有這段？", ["第 7 條"], ["本法或其他法律另有規定"]),
        ("q07-attendance-scope", "2016版第9條與2026版第9條，在全勤獎金保護的列舉方式上有何差異？", ["第 9 條"], ["婚假、喪假、公傷病假及公假"]),
        ("q08-miscarriage", "妊娠未滿三個月流產而請普通傷病假，哪一版開始明文保護全勤獎金？", ["第 9 條"], ["妊娠未滿三個月流產", "普通傷病假"]),
        ("q09-family-care-attendance", "家庭照顧事假不得影響全勤獎金的規定，兩版是否都有？請指出版本差異。", ["第 9 條"], ["親自照顧家庭成員", "不得視為缺勤"]),
        ("q10-sick-attendance-deduction", "2026版對普通傷病假全勤獎金扣發採取什麼計算方式？2016版有沒有相同規定？", ["第 9 條"], ["按請普通傷病假日數依比例計算"]),
        ("q11-better-agreement", "2026版第9條是否保留勞雇雙方較優約定的例外？2016版呢？", ["第 9 條"], ["勞雇雙方另有優於法令之約定"]),
        ("q12-article9-1", "第9-1條在2016版與2026版的存在情況為何？", ["第 9-1 條"], ["不利之處分"]),
        ("q13-adverse-threshold", "2026版普通傷病假未超過幾日，雇主不得因此為不利處分？2016版是否有同樣門檻？", ["第 9-1 條"], ["十日"]),
        ("q14-burden-proof", "2026版對普通傷病假造成不利處分時，雇主負擔什麼舉證責任？2016版有沒有？", ["第 9-1 條"], ["負舉證責任"]),
        ("q15-performance-review", "2026版對普通傷病假超過十日後的人事考核，有何限制？2016版有沒有？", ["第 9-1 條"], ["不得僅以請普通傷病假日數作為考量因素"]),
        ("q16-article12-effective", "2026版第12條新增的施行日期規定是什麼？2016版第12條有沒有？", ["第 12 條"], ["一百十四年十二月九日", "一百十五年一月一日"]),
        ("q17-marriage-unchanged", "婚假日數與工資照給的規定，2016版與2026版是否一致？", ["第 2 條"], ["婚假八日", "工資照給"]),
        ("q18-sick-limits-unchanged", "普通傷病假未住院、住院及合計上限，兩版是否有改變？請列出數值。", ["第 4 條"], ["三十日", "二年", "一年"]),
        ("q19-advance-notice-unchanged", "請假事前通知、緊急事故委託代辦及證明文件要求，兩版是否一致？", ["第 10 條"], ["事前", "委託他人", "證明文件"]),
        ("q20-overall-diff", "整體比較兩份文件：哪些條文有實質文字變更或新增？哪些核心給假數值維持不變？", ["第 2 條", "第 4 條", "第 6 條", "第 7 條", "第 9 條", "第 9-1 條", "第 12 條"], ["第 9-1 條", "十四日", "三十日"]),
    ]
    items = []
    for qid, question, articles, focus in questions:
        items.append(
            {
                "id": qid,
                "question": question,
                "comparison_articles": articles,
                "focus_facts": focus,
                "oracle_2016": context(articles_2016, *articles),
                "oracle_2026": context(articles_2026, *articles),
            }
        )
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    url_2016 = SNAPSHOT_URL.format("20111014")
    url_2026 = SNAPSHOT_URL.format("20251209")
    articles_2016 = extract_articles(fetch(url_2016))
    articles_2026 = extract_articles(fetch(url_2026))
    (out / "source").mkdir(exist_ok=True)
    (out / "source" / "勞工請假規則_2016有效版.txt").write_text(
        render_snapshot("2016有效版（最近修正：2011-10-14）", "2016-01-01 至 2016-12-31", "民國100年10月14日", url_2016, articles_2016),
        encoding="utf-8",
    )
    (out / "source" / "勞工請假規則_2026有效版.txt").write_text(
        render_snapshot("2026有效版（2026-01-01施行）", "2026-01-01 起", "民國114年12月09日", url_2026, articles_2026),
        encoding="utf-8",
    )
    history = fetch(HISTORY_URL)
    (out / "source" / "勞工請假規則_法規沿革.html").write_text(history, encoding="utf-8")
    benchmark = {
        "benchmark_id": "taiwan-labor-leave-version-comparison-v1",
        "benchmark_type": "two-version-comparison",
        "source": "勞動部勞動法令查詢系統",
        "history_url": HISTORY_URL,
        "version_2016": {"label": "2016 effective snapshot", "snapshot_url": url_2016, "source_file": "source/勞工請假規則_2016有效版.txt"},
        "version_2026": {"label": "2026 effective snapshot", "snapshot_url": url_2026, "source_file": "source/勞工請假規則_2026有效版.txt"},
        "question_count": 20,
        "items": build_questions(articles_2016, articles_2026),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (out / "questions_20.json").write_text(json.dumps(benchmark, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(
        "# 2016 vs 2026《勞工請假規則》版本比較\n\n"
        "2016 有效版使用官方沿革中 2011-10-14 修正版本（2016 年沒有新的修正版本）；2026 有效版使用 2025-12-09 修正、2026-01-01 施行版本。\n\n"
        "每題應對兩份文件各做一次 parent recall 與一次 fine evidence retrieval，再把兩個版本的證據交給回答模型。\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(out), "question_count": 20, "source_2016": url_2016, "source_2026": url_2026}, ensure_ascii=False))


if __name__ == "__main__":
    main()
