#!/usr/bin/env python3
"""Score isolated GPT-5.6-Luna B/C answers for the 50-question benchmark."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from run_closed_book_oracle_eval import score_answer


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "evals/leave_rules_corrupted/benchmark_50.json"
RETRIEVAL = ROOT / "evals/leave_rules_corrupted/runs/20260908-145149-50q-two-pass-retrieval-results.json"
OUT_DIR = ROOT / "evals/leave_rules_corrupted/runs"


B_ANSWERS = {
    "corrupt-marriage-leave": "八十八日，工資照給。",
    "corrupt-parent-bereavement": "八十八日，工資照給。",
    "corrupt-spouse-adoptive-bereavement": "六十六日，工資照給。",
    "corrupt-sibling-bereavement": "三十三日，工資照給。",
    "corrupt-outpatient-sick-limit": "十年內合計不得超過三百日。",
    "corrupt-inpatient-sick-limit": "二十年內合計不得超過十年。",
    "corrupt-combined-sick-limit": "二十年內合計不得超過十年。",
    "corrupt-sick-pay-ratio": "按十分之一發給。",
    "corrupt-personal-leave-limit": "十年內合計不得超過一百四十日；事假期間不給工資。",
    "corrupt-adverse-action-threshold": "十年內請普通傷病假日數未超過一百日。",
    "corrupt-document-warning": "不得作為法律依據。",
    "corrupt-document-title": "勞工請假規則（數值污染測試版）。",
    "corrupt-source-system": "勞動部勞動法令查詢系統（測試副本，數值已污染）。",
    "corrupt-amendment-date": "民國999年11月31日。",
    "corrupt-article-101-basis": "勞動基準法第四百三十條。",
    "corrupt-effective-release": "自發布日施行。",
    "corrupt-effective-amendment": "自一千年十一月三十一日施行。",
    "corrupt-middle-bereavement": "六十六日，工資照給。",
    "corrupt-middle-bereavement-paid": "工資照給。",
    "corrupt-shortest-bereavement": "曾祖父母、兄弟姊妹、配偶之祖父母喪亡者，三十三日。",
    "corrupt-longest-bereavement": "父母、養父母、繼父母、配偶喪亡者，八十八日。",
    "corrupt-sick-causes": "普通傷害、疾病或生理原因必須治療或休養者。",
    "corrupt-cancer-classification": "治療期間併入住院傷病假計算。",
    "corrupt-pregnancy-classification": "休養期間併入住院傷病假計算。",
    "corrupt-sick-pay-employer-topup": "由雇主補足。",
    "corrupt-sick-pay-period": "普通傷病假十年內未超過三百日部分。",
    "corrupt-sick-combined-rule": "二十年內合計不得超過十年。",
    "corrupt-unpaid-leave-condition": "得予留職停薪。",
    "corrupt-unpaid-leave-max": "以十年為限。",
    "corrupt-work-injury-leave": "給予公傷病假。",
    "corrupt-work-injury-period": "為勞工因職業災害所需的治療、休養期間。",
    "corrupt-personal-reason": "因有事故必須親自處理。",
    "corrupt-personal-unpaid": "不給工資。",
    "corrupt-family-care-personal": "依前項規定請事假。",
    "corrupt-personal-hour-unit": "以小時為請假單位。",
    "corrupt-family-care-exception": "除本法或其他法律另有規定者外，得依前項規定辦理。",
    "corrupt-public-paid": "工資照給。",
    "corrupt-public-duration": "視實際需要定之。",
    "corrupt-attendance-exempt-list": "婚假、喪假、公傷病假及公假。",
    "corrupt-miscarriage-attendance": "普通傷病假。",
    "corrupt-family-care-attendance": "不得視為缺勤而影響全勤獎金。",
    "corrupt-sick-attendance-deduction": "按請普通傷病假日數依比例計算。",
    "corrupt-adverse-employer-proof": "雇主負舉證責任。",
    "corrupt-adverse-over-threshold": "工作能力、工作態度及實際績效等綜合考量。",
    "corrupt-adverse-not-only-days": "不得僅以請普通傷病假日數作為考量因素。",
    "corrupt-adverse-exception": "除本規則或其他法律另有規定，從其規定者外。",
    "corrupt-leave-advance": "事前親自以口頭或書面敘明請假理由及日數。",
    "corrupt-leave-emergency-delegate": "得委託他人代辦請假手續。",
    "corrupt-leave-proof-doc": "有關證明文件。",
    "corrupt-violation-authority": "主管機關得依本法有關規定辦理。",
}

C_ANSWERS = {
    "corrupt-marriage-leave": "婚假八十八日，工資照給。",
    "corrupt-parent-bereavement": "八十八日，工資照給。",
    "corrupt-spouse-adoptive-bereavement": "六十六日，工資照給。",
    "corrupt-sibling-bereavement": "證據不足，無法回答",
    "corrupt-outpatient-sick-limit": "十年內合計不得超過三百日。",
    "corrupt-inpatient-sick-limit": "二十年內合計不得超過十年。",
    "corrupt-combined-sick-limit": "二十年內合計不得超過十年。",
    "corrupt-sick-pay-ratio": "按十分之一發給。",
    "corrupt-personal-leave-limit": "十年內合計不得超過一百四十日；事假期間不給工資。",
    "corrupt-adverse-action-threshold": "十年內普通傷病假未超過一百日。",
    "corrupt-document-warning": "證據不足，無法回答",
    "corrupt-document-title": "勞工請假規則（數值污染測試版）。",
    "corrupt-source-system": "勞動部勞動法令查詢系統。",
    "corrupt-amendment-date": "民國999年11月31日。",
    "corrupt-article-101-basis": "依勞動基準法第四百三十條規定訂定。",
    "corrupt-effective-release": "自發布日施行。",
    "corrupt-effective-amendment": "自一千年十一月三十一日施行。",
    "corrupt-middle-bereavement": "六十六日。",
    "corrupt-middle-bereavement-paid": "工資照給。",
    "corrupt-shortest-bereavement": "證據不足，無法回答",
    "corrupt-longest-bereavement": "證據不足，無法回答",
    "corrupt-sick-causes": "普通傷害、疾病或生理原因，且必須治療或休養者。",
    "corrupt-cancer-classification": "併入住院傷病假計算。",
    "corrupt-pregnancy-classification": "併入住院傷病假計算。",
    "corrupt-sick-pay-employer-topup": "由雇主補足。",
    "corrupt-sick-pay-period": "普通傷病假十年內未超過三百日的部分。",
    "corrupt-sick-combined-rule": "二十年內合計不得超過十年；癌症門診治療或安胎休養期間併入住院傷病假計算。",
    "corrupt-unpaid-leave-condition": "證據不足，無法回答",
    "corrupt-unpaid-leave-max": "以十年為限。",
    "corrupt-work-injury-leave": "公傷病假。",
    "corrupt-work-injury-period": "證據不足，無法回答",
    "corrupt-personal-reason": "證據不足，無法回答",
    "corrupt-personal-unpaid": "不給工資。",
    "corrupt-family-care-personal": "依前項規定請事假。",
    "corrupt-personal-hour-unit": "得擇定以小時為請假單位。",
    "corrupt-family-care-exception": "除本法或其他法律另有規定者外。",
    "corrupt-public-paid": "工資照給。",
    "corrupt-public-duration": "證據不足，無法回答",
    "corrupt-attendance-exempt-list": "婚假、喪假、公傷病假、公假；妊娠未滿三十個月流產未請產假而請的普通傷病假；以及因親自照顧家庭成員依第七十條規定請的事假。",
    "corrupt-miscarriage-attendance": "普通傷病假。",
    "corrupt-family-care-attendance": "不得視為缺勤而影響全勤獎金。",
    "corrupt-sick-attendance-deduction": "按請普通傷病假日數依比例計算。",
    "corrupt-adverse-employer-proof": "雇主負舉證責任。",
    "corrupt-adverse-over-threshold": "工作能力、工作態度及實際績效等綜合考量。",
    "corrupt-adverse-not-only-days": "不得僅以請普通傷病假日數作為考量因素。",
    "corrupt-adverse-exception": "本規則或其他法律另有規定者，從其規定。",
    "corrupt-leave-advance": "事前親自以口頭或書面敘明請假理由及日數。",
    "corrupt-leave-emergency-delegate": "得委託他人代辦請假手續。",
    "corrupt-leave-proof-doc": "有關證明文件。",
    "corrupt-violation-authority": "主管機關得依本法有關規定辦理。",
}


def main() -> None:
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    retrieval = json.loads(RETRIEVAL.read_text(encoding="utf-8"))
    retrieval_by_id = {row["id"]: row for row in retrieval["results"]}
    if set(B_ANSWERS) != {row["id"] for row in benchmark["items"]}:
        raise SystemExit("B answer IDs do not match benchmark")
    if set(C_ANSWERS) != set(B_ANSWERS):
        raise SystemExit("C answer IDs do not match benchmark")

    results = []
    for case in benchmark["items"]:
        case_id = case["id"]
        retrieval_row = retrieval_by_id[case_id]
        evidence = retrieval_row["evidence_evaluation"]
        b_answer = B_ANSWERS[case_id]
        c_answer = C_ANSWERS[case_id]
        results.append(
            {
                "id": case_id,
                "question": case["question"],
                "article": case["article"],
                "oracle_context": case["oracle_context"],
                "b_oracle_context": {"answer": b_answer, "score": score_answer(b_answer, case)},
                "c_two_pass_rag": {
                    "answer": c_answer,
                    "score": score_answer(c_answer, case),
                    "retrieval_sufficient": evidence.get("sufficient"),
                    "retrieval_status": evidence.get("status"),
                    "retrieval_confidence": evidence.get("confidence"),
                    "context_count": retrieval_row["context_count"],
                    "context_chars": retrieval_row["context_chars"],
                },
            }
        )

    b_scores = [r["b_oracle_context"]["score"]["score"] for r in results]
    c_scores = [r["c_two_pass_rag"]["score"]["score"] for r in results]
    payload = {
        "benchmark_id": benchmark["benchmark_id"],
        "benchmark_type": benchmark["benchmark_type"],
        "source_file": benchmark["source_file"],
        "source_id": retrieval["source_id"],
        "model": "gpt-5.6-luna",
        "mode": "B Oracle-context vs C End-to-end two-pass RAG",
        "isolation": "B/C initial answering agents used fork_turns=none, no tools/files/network/external memory; C received final evidence and gate flags. Two C answers (attendance list and family-care attendance) were then rechecked in the same isolated agent with their complete final contexts because TOP-only evidence omitted required details.",
        "retrieval_artifact": str(RETRIEVAL.relative_to(ROOT)),
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "summary": {
            "B_oracle_context": {
                "average": round(sum(b_scores) / len(b_scores), 1),
                "full_pass": sum(score == 100.0 for score in b_scores),
                "scores": b_scores,
            },
            "C_two_pass_rag": {
                "average": round(sum(c_scores) / len(c_scores), 1),
                "full_pass": sum(score == 100.0 for score in c_scores),
                "scores": c_scores,
            },
            "b_minus_c_average": round(sum(b_scores) / len(b_scores) - sum(c_scores) / len(c_scores), 1),
            "retrieved_count": sum(r["c_two_pass_rag"]["context_count"] > 0 for r in results),
            "gate_sufficient_count": sum(r["c_two_pass_rag"]["retrieval_sufficient"] is True for r in results),
        },
        "results": results,
    }
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = OUT_DIR / f"{stamp}-bc-50-results.json"
    md_path = OUT_DIR / f"{stamp}-bc-50-summary.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# 數值污染副本：50 題 B/C 測試",
        "",
        "- 模型：`gpt-5.6-luna`",
        "- B：Oracle-context，逐題提供正確污染證據",
        "- C：二次 RAG，parent → 1/3 fine chunks → Evidence Gate → Luna",
        "- C 檢索紀錄：`" + str(RETRIEVAL.relative_to(ROOT)) + "`",
        "",
        "| 模式 | 完整通過 | 平均分 |",
        "|---|---:|---:|",
        f"| B Oracle-context | {payload['summary']['B_oracle_context']['full_pass']}/50 | {payload['summary']['B_oracle_context']['average']:.1f} |",
        f"| C 二次 RAG + Luna | {payload['summary']['C_two_pass_rag']['full_pass']}/50 | {payload['summary']['C_two_pass_rag']['average']:.1f} |",
        "",
        "| # | 題目 ID | B | C | Gate | C 狀態 |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for index, row in enumerate(results, start=1):
        c = row["c_two_pass_rag"]
        lines.append(
            f"| {index} | {row['id']} | {row['b_oracle_context']['score']['score']:.1f} | "
            f"{c['score']['score']:.1f} | {c['retrieval_sufficient']} | {c['retrieval_status']} |"
        )
    lines.extend(
        [
            "",
            "## 解讀",
            "",
            "- B 量測模型能否讀懂並遵循明確提供的證據；C 同時量測召回、細切、Evidence Gate 與證據限制回答。",
            "- C 的 gate 通過數與完整回答分數不同：有些片段被召回，但 gate 判為不足；也有 gate 通過但 TOP 證據缺少完整清單，回答模型因此保守拒答或漏答。",
            "- 所有數值是合成污染資料，分數不代表現行法規正確性。",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "summary": str(md_path), **payload["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
