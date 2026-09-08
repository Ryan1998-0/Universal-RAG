#!/usr/bin/env python3
"""Deterministic audit for the changed-clause Luna answers.

This is a transparent coverage check against facts in the two official
snapshots. It is not a substitute for a human/model judge.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path


RULES = {
    "c01-article6-term": [("2016", ["殘廢"]), ("2026", ["失能"])],
    "c02-article6-meaning": [("combined", ["治療", "休養期間"]), ("combined", ["相同", "一致", "僅用語", "文字修正"])],
    "c03-article7-wording": [("2016", ["者"]), ("2026", ["刪除", "去掉", "移除", "不再有者"])],
    "c04-article7-family-care": [("2026", ["親自照顧家庭成員"])],
    "c05-article7-family-exception": [("2026", ["本法或其他法律另有規定"])],
    "c06-article7-hour-unit": [("2026", ["小時", "小時為請假單位"])],
    "c07-article7-limit-carryover": [("combined", ["十四日"]), ("combined", ["前項", "相同", "未改變", "沒有改變"])],
    "c08-article9-opening": [("2016", ["不得因", "扣發全勤獎金"]), ("2026", ["不得視為缺勤", "影響其全勤獎金"])],
    "c09-article9-miscarriage": [("2026", ["妊娠未滿三個月流產", "普通傷病假"])],
    "c10-article9-family-attendance": [("2026", ["親自照顧家庭成員", "第七條", "事假"])],
    "c11-article9-proportional": [("2026", ["按請普通傷病假日數依比例計算", "依比例", "比例計算"])],
    "c12-article9-better-agreement": [("2026", ["勞雇雙方另有優於法令之約定"])],
    "c13-article9-other-law": [("2026", ["其他法律另有規定者", "從其規定"])],
    "c14-article91-new": [("2026", ["第9-1條", "9-1"]), ("2026", ["十日"]), ("2026", ["不利之處分", "不利處分"])],
    "c15-article91-burden": [("2026", ["負舉證責任", "舉證責任", "須證明", "證明"])],
    "c16-article91-appraisal": [("2026", ["工作能力", "工作態度", "實際績效"]), ("2026", ["不得僅以", "請普通傷病假日數"])],
    "c17-article12-effective": [("2016", ["發布日施行"]), ("2026", ["一百十五年一月一日", "115年1月1日", "2026-01-01", "2026年1月1日"])],
    "c18-article12-scope": [("2026", ["修正發布之條文", "修正發布", "施行"])],
    "c19-changed-articles": [("combined", ["第6條", "第6"]), ("combined", ["第7條", "第7"]), ("combined", ["第9條", "第9"]), ("combined", ["第9-1條", "9-1"]), ("combined", ["第12條", "第12"])],
    "c20-change-summary": [("combined", ["殘廢", "失能"]), ("combined", ["家庭成員", "小時"]), ("combined", ["第9-1條", "9-1"]), ("combined", ["一百十五年一月一日", "115年1月1日", "2026-01-01", "2026年1月1日"])],
}


def compact(text: str) -> str:
    return re.sub(r"[\s，。；：、（）()／/\[\]【】\-]", "", str(text or "")).casefold()


def match_group(text: str, terms: list[str]) -> str:
    compacted = compact(text)
    for term in terms:
        if term.startswith("第") and term.endswith("條"):
            number = term[1:-1]
            if re.search(rf"第\s*{re.escape(number)}\s*(?:條|[、,，])", str(text)):
                return term
            if re.search(rf"[、,，]\s*{re.escape(number)}\s*(?:條|[、,，])", str(text)):
                return term
        if compact(term) in compacted:
            return term
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("answers", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    answer_payload = json.loads(args.answers.read_text(encoding="utf-8"))
    if isinstance(answer_payload, list):
        raw_answers = answer_payload
    else:
        raw_answers = answer_payload.get("results") or answer_payload.get("items") or []
    answers = {item["id"]: item for item in raw_answers if isinstance(item, dict) and item.get("id")}
    results = []
    for item in benchmark["items"]:
        answer_item = answers.get(item["id"], {})
        a16 = str(answer_item.get("answer_2016") or "")
        a26 = str(answer_item.get("answer_2026") or "")
        combined = " ".join([a16, a26, str(answer_item.get("notes") or "")])
        checks = []
        for target, terms in RULES[item["id"]]:
            text = {"2016": a16, "2026": a26, "combined": combined}[target]
            checks.append({"target": target, "accepted": terms, "matched": match_group(text, terms)})
        passed = sum(bool(check["matched"]) for check in checks)
        results.append({
            "id": item["id"], "question": item["question"],
            "answer_2016": a16, "answer_2026": a26,
            "citations_2016": answer_item.get("citations_2016", []),
            "citations_2026": answer_item.get("citations_2026", []),
            "abstained_2016": answer_item.get("abstained_2016"),
            "abstained_2026": answer_item.get("abstained_2026"),
            "checks": checks, "score": round(100 * passed / max(1, len(checks)), 1),
            "full_pass": passed == len(checks),
        })
    scores = [result["score"] for result in results]
    output = {
        "benchmark_id": benchmark["benchmark_id"],
        "answer_file": str(args.answers),
        "scoring": "changed-clause factual coverage; deterministic lexical audit, not a model judge",
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "summary": {
            "question_count": len(results),
            "answered_count": sum(bool(r["answer_2016"].strip() or r["answer_2026"].strip()) for r in results),
            "full_pass": sum(r["full_pass"] for r in results),
            "average_score": round(sum(scores) / max(1, len(scores)), 1),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **output["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
