#!/usr/bin/env python3
"""Score two-version answers for version facts and comparison completeness."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


RULES = {
    "q01-article6-term": [["殘廢"], ["失能"]],
    "q02-article6-period": [["治療", "休養期間"], ["相同", "一致", "都"]],
    "q03-personal-limit": [["十四日"], ["相同", "一致", "未改變", "沒有改變"]],
    "q04-family-care": [["親自照顧家庭成員"], ["2026", "新增"]],
    "q05-family-care-hour": [["小時"], ["2026", "新增"]],
    "q06-family-care-condition": [["本法或其他法律另有規定"], ["2016", "沒有", "未有"]],
    "q07-attendance-scope": [["婚假", "喪假", "公傷病假", "公假"], ["2026", "擴大", "列舉", "增加"]],
    "q08-miscarriage": [["妊娠未滿三個月流產", "普通傷病假"], ["2026", "新增", "2016", "沒有"]],
    "q09-family-care-attendance": [["親自照顧家庭成員", "不得視為缺勤"], ["2026", "新增", "2016", "沒有"]],
    "q10-sick-attendance-deduction": [["按請普通傷病假日數依比例計算", "依比例"], ["2026", "新增", "2016", "沒有"]],
    "q11-better-agreement": [["勞雇雙方另有優於法令之約定"], ["2026", "2016", "都", "相同"]],
    "q12-article9-1": [["第9-1條", "9-1"], ["2026", "新增", "2016", "沒有"]],
    "q13-adverse-threshold": [["十日"], ["2026", "2016", "沒有", "新增"]],
    "q14-burden-proof": [["負舉證責任"], ["2026", "2016", "沒有", "新增"]],
    "q15-performance-review": [["不得僅以請普通傷病假日數作為考量因素", "不得僅以", "工作能力", "工作態度", "實際績效"], ["2026", "2016", "沒有", "新增"]],
    "q16-article12-effective": [["一百十五年一月一日", "115年1月1日", "2026-01-01"], ["2016", "沒有", "新增"]],
    "q17-marriage-unchanged": [["婚假八日", "八日"], ["工資照給"], ["相同", "一致", "未改變"]],
    "q18-sick-limits-unchanged": [["三十日"], ["二年"], ["一年"], ["相同", "一致", "未改變"]],
    "q19-advance-notice-unchanged": [["事前"], ["委託他人"], ["證明文件"], ["相同", "一致", "未改變"]],
    "q20-overall-diff": [["第9-1條", "9-1"], ["第7條", "第7"], ["第9條", "第9"], ["第12條", "第12"], ["十四日"], ["三十日"]],
}


def compact(text: str) -> str:
    return re.sub(r"[\s，。；：、（）()／/\[\]【】]", "", str(text or "")).casefold()


def group_pass(answer: str, group: list[str]) -> str:
    text = compact(answer)
    for term in group:
        if compact(term) in text:
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
    answers = {item["id"]: item for item in answer_payload.get("results", [])}
    results = []
    for item in benchmark["items"]:
        answer_item = answers.get(item["id"], {})
        answer = str(answer_item.get("answer") or "")
        groups = RULES[item["id"]]
        checks = [{"accepted": group, "matched": group_pass(answer, group)} for group in groups]
        passed = sum(bool(check["matched"]) for check in checks)
        results.append({
            "id": item["id"],
            "question": item["question"],
            "answer": answer,
            "citations": answer_item.get("citations", []),
            "evidence_sufficient": answer_item.get("evidence_sufficient"),
            "checks": checks,
            "score": round(100 * passed / max(1, len(checks)), 1),
            "full_pass": passed == len(checks),
        })
    scores = [item["score"] for item in results]
    output = {
        "benchmark_id": benchmark["benchmark_id"],
        "answer_file": str(args.answers),
        "scoring": "comparison-fact coverage; deterministic lexical audit, not a model judge",
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "summary": {
            "question_count": len(results),
            "answered_count": sum(bool(item["answer"].strip()) for item in results),
            "full_pass": sum(item["full_pass"] for item in results),
            "average_score": round(sum(scores) / max(1, len(scores)), 1),
        },
        "results": results,
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **output["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
