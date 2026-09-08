#!/usr/bin/env python3
"""Build a compact Markdown report from two-version retrieval and Luna answers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def gate(result: dict, version: str) -> str:
    evidence = result[f"retrieval_{version}"]["pass_2_fine"]["evidence_evaluation"]
    return "通過" if evidence.get("sufficient") is True else "不足"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("retrieval", type=Path)
    parser.add_argument("answers", type=Path)
    parser.add_argument("score", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    retrieval = json.loads(args.retrieval.read_text(encoding="utf-8"))
    answers = json.loads(args.answers.read_text(encoding="utf-8"))
    score = json.loads(args.score.read_text(encoding="utf-8"))
    answer_by_id = {item["id"]: item for item in answers["results"]}
    score_by_id = {item["id"]: item for item in score["results"]}
    lines = [
        "# 2016 vs 2026《勞工請假規則》雙版本兩次 RAG 測試",
        "",
        "## 測試設定",
        "",
        "- 題數：20 題，全部包含版本比較元素。",
        "- 每題：2016 文件與 2026 文件各執行一次 parent hybrid recall，再執行一次 fine evidence re-embedding（父 chunk 約 1/3，最多 3 組）。",
        "- 2016 source：`upload-7453306c9812bff0372b`（2016 有效版本；官方沿革最近修正 2011-10-14）。",
        "- 2026 source：`upload-7598ae1020595b115381`（2025-12-09 修正、2026-01-01 施行版本）。",
        "- 回答模型：`gpt-5.6-luna`，隔離輸入，只提供實際 fine evidence，不提供 oracle answer。",
        "",
        "## 總結",
        "",
        f"- Parent recall：2016 {retrieval['summary']['2016_parent_retrieved']}/20；2026 {retrieval['summary']['2026_parent_retrieved']}/20。",
        f"- Fine evidence 有結果：2016 {retrieval['summary']['2016_fine_retrieved']}/20；2026 {retrieval['summary']['2026_fine_retrieved']}/20。",
        f"- Evidence Gate sufficient：2016 {retrieval['summary']['2016_fine_evidence_sufficient']}/20；2026 {retrieval['summary']['2026_fine_evidence_sufficient']}/20。",
        f"- Luna 回答：{score['summary']['answered_count']}/20 題有回答；coverage audit 平均 {score['summary']['average_score']:.1f}；完整通過 {score['summary']['full_pass']}/20。",
        "- coverage audit 是可追溯的字詞覆蓋檢查，不是另一個模型裁判；對於證據不足而正確拒答的題目，需另外看 Gate 狀態。",
        "",
        "## 每題結果",
        "",
        "| 題號 | 2016 Gate | 2026 Gate | audit | Luna 回答 |",
        "|---|---|---|---:|---|",
    ]
    for result in retrieval["results"]:
        item_id = result["id"]
        answer = answer_by_id.get(item_id, {}).get("answer", "").replace("|", "／").replace("\n", " ")
        answer = answer[:320] + ("…" if len(answer) > 320 else "")
        lines.append(
            f"| `{item_id}` | {gate(result, '2016')} | {gate(result, '2026')} | "
            f"{score_by_id.get(item_id, {}).get('score', 0):.1f} | {answer} |"
        )
    lines.extend([
        "",
        "## 主要失敗模式",
        "",
        "- `q03-personal-limit`：兩個版本都沒有把第 7 條的事假上限保留在 fine evidence，模型因此拒答；這是第二階段 evidence selection 漏掉正確條文，不是原始文件沒有資料。",
        "- `q07-attendance-scope`：2026 版 fine evidence 只保留第 9 條開頭與扣發比例，沒有完整列舉三款，模型對完整比較採保守拒答。",
        "- `q11-better-agreement`：2026 版抓到較優約定，2016 版沒有同等語句，因此模型正確標示 2016 證據不足；若要求判斷全文差異，應補一個版本差異檢索器。",
        "- `q19-advance-notice-unchanged`：兩版 fine evidence 都只抓到緊急代辦與證明文件，沒有抓到第 10 條前半段的「事前」，模型只回答部分並標示不足。",
        "- `q20-overall-diff`：整體盤點題不適合只靠 top-3 fine groups；需要先依條號做版本對齊，再交給模型整合。",
        "",
        "## 可重現檔案",
        "",
        f"- Retrieval：`{args.retrieval}`",
        f"- Evidence-only input：`{answers.get('input', '20260908-163226-answer-input-evidence-only.json')}`",
        f"- Luna answers：`{args.answers}`",
        f"- Score：`{args.score}`",
    ])
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
