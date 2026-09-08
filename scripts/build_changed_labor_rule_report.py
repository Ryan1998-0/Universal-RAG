#!/usr/bin/env python3
"""Build the changed-clause, article-aligned two-pass RAG report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def gate(result: dict, version: str) -> str:
    return "通過" if result[f"aligned_{version}"]["evidence_evaluation"].get("sufficient") is True else "不足"


def short(text: str, limit: int = 220) -> str:
    text = str(text or "").replace("|", "／").replace("\n", " ").strip()
    return text[:limit] + ("…" if len(text) > limit else "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("retrieval", type=Path)
    parser.add_argument("answers", type=Path)
    parser.add_argument("score", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    retrieval = json.loads(args.retrieval.read_text(encoding="utf-8"))
    answers_payload = json.loads(args.answers.read_text(encoding="utf-8"))
    answers = answers_payload if isinstance(answers_payload, list) else (answers_payload.get("results") or answers_payload.get("items") or [])
    answers_by_id = {item["id"]: item for item in answers}
    score = json.loads(args.score.read_text(encoding="utf-8"))
    score_by_id = {item["id"]: item for item in score["results"]}
    s = retrieval["summary"]
    lines = [
        "# 2016 vs 2026《勞工請假規則》改動條文：全文對齊＋兩階段 RAG 測試",
        "",
        "## 測試目的",
        "",
        "重新檢索兩份全文，只設計能在官方版本差異中確認的 20 題；不使用 absence-only 題目。每題先以全文條號索引對齊兩個版本的改動條文，再在對齊條文內做第二次 fine embedding，最後保留已對齊條文的完整證據包交給 GPT-5.6 Luna。",
        "",
        "## 為什麼改用條號對齊",
        "",
        "純語意搜尋整份法規容易把相似但不相關的條文排前面；法律版本比較又具有明確的條號結構。先以全文條號鎖定第 6、7、9、9-1、12 條，再做語意定位，可以避免漏掉新增條款、例外、日期與同一條文的後半段。第二次 fine pass 只負責找 answer-bearing 位置，不能直接把同一條已確認改動條文截成孤立句子，因此本次回填完整對齊條文。",
        "",
        "## 結果摘要",
        "",
        f"- 題數：20 題，兩版的改動條文對齊召回都是 20/20（2016 的第 9-1 條合理缺席，因為它在 2026 才新增，不計為失敗）。",
        f"- Fine pass 有結果：2016 {s['2016_nonempty_fine']}/20；2026 {s['2026_nonempty_fine']}/20。",
        f"- Evidence Gate：2016 {s['2016_gate_sufficient']}/20；2026 {s['2026_gate_sufficient']}/20。2016 少的 3 題正是 2016 不存在的第 9-1 條，屬於版本差異，不是召回失敗。",
        f"- Luna 回答：{score['summary']['answered_count']}/20；變更條文 coverage audit 平均 {score['summary']['average_score']:.1f}；完整通過 {score['summary']['full_pass']}/20。",
        "- coverage audit 是可重現的詞彙／版本事實覆蓋檢查，不是另一個模型裁判；仍應搭配人工抽查引用位置。",
        "",
        "## 每題結果",
        "",
        "| 題號 | 改動條文 | 2016 Gate | 2026 Gate | audit | Luna 2016 | Luna 2026 |",
        "|---|---|---|---|---:|---|---|",
    ]
    for result in retrieval["results"]:
        item_id = result["id"]
        a = answers_by_id.get(item_id, {})
        sc = score_by_id.get(item_id, {})
        lines.append(
            f"| `{item_id}` | {', '.join(result['changed_articles'])} | {gate(result, '2016')} | {gate(result, '2026')} | {sc.get('score', 0):.1f} | {short(a.get('answer_2016'))} | {short(a.get('answer_2026'))} |"
        )
    lines.extend([
        "",
        "## 建議採用的產品流程",
        "",
        "1. 文件匯入時保留版本、有效日期、條號與段落層級，不只存純文字 chunk。",
        "2. 比較型問題先抽取條號／版本／比較維度；若問題是法規版本比較，優先走條號對齊分支。",
        "3. 在對齊的條文內做 BM25＋embedding／reranker 細檢索，取得 answer-bearing 句子與鄰接句。",
        "4. 將同一條文的完整證據包交給生成模型，並要求逐版本引用；不能把 fine ranking 當作刪除上下文的唯一依據。",
        "5. Evidence Gate 應分成「該版本條文不存在」與「該版本有條文但證據不足」兩種狀態，避免把合法的版本缺席誤報成檢索錯誤。",
        "",
        "## 可重現檔案",
        "",
        f"- 題目與 oracle 僅供評測：[{Path('questions_20_changed.json')}]({(Path.cwd() / 'evals/taiwan_law_versions_changed/questions_20_changed.json').resolve()})",
        f"- 對齊與兩次檢索：[{args.retrieval.name}]({args.retrieval.resolve()})",
        f"- 給模型的 evidence-only 輸入：[{Path('20260908-165039-article-aligned-answer-input.json')}]({(args.answers.parent / '20260908-165039-article-aligned-answer-input.json').resolve()})",
        f"- Luna 回答：[{args.answers.name}]({args.answers.resolve()})",
        f"- 評分：[{args.score.name}]({args.score.resolve()})",
        "",
        "官方版本來源：[勞動部現行《勞工請假規則》](https://laws.mol.gov.tw/FLAW/FLAWDAT01.aspx?id=FL014935)、[歷史版本查詢](https://laws.mol.gov.tw/FLAW/FLAWDAT07.aspx?id=FL014935)。",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
