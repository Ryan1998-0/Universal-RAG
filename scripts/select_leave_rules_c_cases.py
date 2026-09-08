#!/usr/bin/env python3
"""Freeze the manually reviewed A<B cases for the later end-to-end RAG run."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_RESULT = ROOT / "evals/leave_rules_ab/runs/20260908-105207-ab-results.json"
BENCHMARK = ROOT / "evals/leave_rules_ab/benchmark.json"
OUTPUT = ROOT / "evals/leave_rules_ab/c-test-set.json"

SELECTED = [
    ("family-care-hour-unit", "A 將新制小時請假錯答為不允許"),
    ("ordinary-sick-bonus-proportional", "A 錯答為通常不扣全勤獎金"),
    ("ten-day-adverse-action", "A 捏造第79條之2並錯答六日"),
    ("amendment-effective-date", "A 不知道施行日並猜測公布後30日"),
    ("personal-leave-limit-pay", "A 將十四日錯答為三十日"),
    ("cancer-outpatient-classification", "A 未辨識應併入住院傷病假"),
    ("pregnancy-bedrest-classification", "A 錯分為產前或預防性休養假"),
    ("official-leave-pay-duration", "A 錯答公假通常不給薪"),
    ("combined-sick-leave-limit", "A 捏造每年十二週上限"),
    ("spouse-adoptive-parent-bereavement", "A 錯答沒有法定喪假"),
]


def main() -> None:
    raw = json.loads(RAW_RESULT.read_text(encoding="utf-8"))
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    raw_by_id = {item["id"]: item for item in raw["results"]}
    benchmark_by_id = {item["id"]: item for item in benchmark["items"]}
    cases = []
    for order, (case_id, reason) in enumerate(SELECTED, start=1):
        observed = raw_by_id[case_id]
        source = benchmark_by_id[case_id]
        cases.append(
            {
                "order": order,
                "id": case_id,
                "question": source["question"],
                "article": source["article"],
                "oracle_context": source["oracle_context"],
                "expected_required": source["required"],
                "selection_reason": reason,
                "a_answer": observed["closed_book"]["answer"],
                "b_answer": observed["oracle_context"]["answer"],
                "manual_a_key_conclusion_correct": False,
                "manual_b_key_conclusion_correct": True,
            }
        )
    payload = {
        "test_set_id": "taiwan-leave-rules-c-v1",
        "derived_from": str(RAW_RESULT.relative_to(ROOT)),
        "source_url": benchmark["source_url"],
        "selection_method": "人工覆核：A 關鍵結論錯誤且 B 關鍵結論正確；固定十題",
        "case_count": len(cases),
        "cases": cases,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
