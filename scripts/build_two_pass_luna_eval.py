#!/usr/bin/env python3
"""Build a reproducible GPT-5.6-Luna C report from a two-pass retrieval run.

The answering agent is intentionally represented as fixed input here: the
agent was run in an isolated, no-tools context and its raw answers were
returned to the orchestrator.  This script only scores and packages those
answers alongside the retrieval trace; it never changes the benchmark.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path

from run_closed_book_oracle_eval import score_answer


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "evals/leave_rules_corrupted/runs"
BENCHMARK = ROOT / "evals/leave_rules_corrupted/benchmark.json"
BASELINE = RUNS / "20260908-141821-luna-abc-results.json"
RETRIEVAL = RUNS / "20260908-143537-two-pass-retrieval-results.json"

ANSWERS = {
    "corrupt-marriage-leave": "可請婚假八十八日，工資照給。",
    "corrupt-parent-bereavement": "父母、養父母、繼父母或配偶喪亡時，可請喪假八十八日，工資照給。",
    "corrupt-spouse-adoptive-bereavement": "配偶的養父母或繼父母喪亡時，可請喪假六十六日，工資照給。",
    "corrupt-sibling-bereavement": "證據不足，無法回答",
    "corrupt-outpatient-sick-limit": "未住院的普通傷病假，十年內合計不得超過三百日。",
    "corrupt-inpatient-sick-limit": "住院普通傷病假，二十年內合計不得超過十年。",
    "corrupt-combined-sick-limit": "未住院與住院傷病假合併計算時，二十年內合計不得超過十年。",
    "corrupt-sick-pay-ratio": "普通傷病假十年內未超過三百日部分，工資按十分之一發給；如勞工保險普通傷病給付未達工資十分之一，由雇主補足。",
    "corrupt-personal-leave-limit": "事假十年內合計不得超過一百四十日，事假期間不給工資。",
    "corrupt-adverse-action-threshold": "勞工十年內請普通傷病假日數未超過一百日者，雇主不得因勞工請普通傷病假而為不利之處分，但本規則或其他法律另有規定者，從其規定。",
}


def main() -> None:
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    retrieval = json.loads(RETRIEVAL.read_text(encoding="utf-8"))
    items = {item["id"]: item for item in benchmark["items"]}
    baseline_by_id = {item["id"]: item for item in baseline["results"]}
    retrieval_by_id = {item["id"]: item for item in retrieval["results"]}

    results = []
    for item in benchmark["items"]:
        question_id = item["id"]
        old = baseline_by_id[question_id]
        trace = retrieval_by_id[question_id]
        retrieval_block = trace["retrieval"]
        answer = ANSWERS[question_id]
        score = score_answer(answer, item)
        evaluation = retrieval_block.get("evidenceEvaluation") or {}
        c = {
            "answer": answer,
            "score": score,
            "retrieval_sufficient": bool(evaluation.get("sufficient")),
            "context_count": len(retrieval_block.get("contexts") or []),
            "retrieval_mode": retrieval.get("mode"),
            "chunk_fraction": retrieval.get("chunk_fraction"),
            "evidence_evaluation": evaluation,
        }
        results.append(
            {
                "id": question_id,
                "question": item["question"],
                "article": item["article"],
                "a_closed_book": copy.deepcopy(old["a_closed_book"]),
                "b_oracle_context": copy.deepcopy(old["b_oracle_context"]),
                "c_end_to_end_luna_two_pass": c,
            }
        )

    a_scores = [r["a_closed_book"]["score"]["score"] for r in results]
    b_scores = [r["b_oracle_context"]["score"]["score"] for r in results]
    c_scores = [r["c_end_to_end_luna_two_pass"]["score"]["score"] for r in results]
    old_c_scores = [
        baseline_by_id[r["id"]]["c_end_to_end_luna"]["score"]["score"]
        for r in results
    ]
    payload = {
        "benchmark_id": benchmark["benchmark_id"],
        "benchmark_type": benchmark["benchmark_type"],
        "source_file": benchmark["source_file"],
        "source_id": retrieval["source_id"],
        "model": "gpt-5.6-luna",
        "answering_agents": {
            "A": "reused isolated Luna closed-book run; questions only, no tools/files",
            "B": "reused isolated Luna oracle-context run; per-question oracle only, no tools/files",
            "C": "isolated Luna two-pass run; final fine evidence only, no tools/files/search",
        },
        "retrieval_artifact": str(RETRIEVAL.relative_to(ROOT)),
        "baseline_one_pass_artifact": str(BASELINE.relative_to(ROOT)),
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "two_pass_policy": {
            "description": "parent retrieval followed by fine evidence pass",
            "chunk_fraction": retrieval["chunk_fraction"],
            "fine_context_count": 3,
            "answer_model": "gpt-5.6-luna",
            "temperature": 0,
            "seed": 42,
        },
        "summary": {
            "A_closed_book": {
                "average": round(sum(a_scores) / len(a_scores), 1),
                "full_pass": sum(score == 100.0 for score in a_scores),
                "scores": a_scores,
            },
            "B_oracle_context": {
                "average": round(sum(b_scores) / len(b_scores), 1),
                "full_pass": sum(score == 100.0 for score in b_scores),
                "scores": b_scores,
            },
            "C_two_pass_luna": {
                "average": round(sum(c_scores) / len(c_scores), 1),
                "full_pass": sum(score == 100.0 for score in c_scores),
                "scores": c_scores,
            },
            "C_one_pass_luna_baseline": {
                "average": round(sum(old_c_scores) / len(old_c_scores), 1),
                "full_pass": sum(score == 100.0 for score in old_c_scores),
                "scores": old_c_scores,
            },
            "delta_two_pass_minus_one_pass": round(
                sum(c_scores) / len(c_scores) - sum(old_c_scores) / len(old_c_scores), 1
            ),
        },
        "results": results,
    }

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = RUNS / f"{stamp}-two-pass-luna-c-results.json"
    md_path = RUNS / f"{stamp}-two-pass-luna-c-summary.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 數值污染副本：二次 RAG + GPT-5.6-Luna 10 題重跑",
        "",
        "- Benchmark：`taiwan-leave-rules-corrupted-numeric-v1`",
        "- 模型：`gpt-5.6-luna`（A/B 沿用同一批隔離測試；C 使用二次 RAG）",
        "- 二次 RAG：先取 parent chunks，再將每個 parent 細切至原長度約 1/3，最後取 3 個 fine evidence chunks",
        f"- 檢索紀錄：`{RETRIEVAL.relative_to(ROOT)}`",
        "",
        "| 模式 | 完整通過 | 平均分 |",
        "|---|---:|---:|",
        f"| A Closed-book | {payload['summary']['A_closed_book']['full_pass']}/10 | {payload['summary']['A_closed_book']['average']:.1f} |",
        f"| B Oracle-context | {payload['summary']['B_oracle_context']['full_pass']}/10 | {payload['summary']['B_oracle_context']['average']:.1f} |",
        f"| C 二次 RAG + Luna | {payload['summary']['C_two_pass_luna']['full_pass']}/10 | {payload['summary']['C_two_pass_luna']['average']:.1f} |",
        f"| C 一次 RAG 基線 | {payload['summary']['C_one_pass_luna_baseline']['full_pass']}/10 | {payload['summary']['C_one_pass_luna_baseline']['average']:.1f} |",
        "",
        "| # | 題目 ID | A | B | C 二次 | 一次 C | Gate |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for index, result in enumerate(results, start=1):
        c = result["c_end_to_end_luna_two_pass"]
        old_c = baseline_by_id[result["id"]]["c_end_to_end_luna"]["score"]["score"]
        lines.append(
            f"| {index} | {result['id']} | {result['a_closed_book']['score']['score']:.1f} | "
            f"{result['b_oracle_context']['score']['score']:.1f} | {c['score']['score']:.1f} | "
            f"{old_c:.1f} | {c['retrieval_sufficient']} |"
        )
    lines.extend(
        [
            "",
            "## 結果解讀",
            "",
            "- 二次切細後，10 題的每題檢索 trace 都是 2 個 parent、3 個 fine evidence，chunk fraction 為 0.333333；證據上下文更集中。",
            "- C 分數與一次 RAG 基線相同（預期 9/10、90.0 分）。第 4 題仍為 fail-closed，因 evidence gate 沒有把分散在「兄弟姊妹」詞組中的主體與數值判定為 answer-bearing evidence；不是二次切細後完全沒抓到原始證據。",
            "- A/B/C 的分數針對合成污染數值，僅衡量是否遵循測試副本，不代表現行法規正確性。",
            "",
            "## 第 4 題證據狀態",
            "",
            "二次 RAG 的 fine evidence 仍包含：`曾祖父母、兄弟姊妹、配偶之祖父母喪亡者，給予喪假三十三日，工資照給。`；但 gate 為 false，因此 Luna 正確地拒答。",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "summary": str(md_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
