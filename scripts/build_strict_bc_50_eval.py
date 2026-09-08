#!/usr/bin/env python3
"""Score the per-question, fresh-session GPT-5.6-Luna B/C run.

Every answer in this file was collected from one new `fork_turns=none`
subagent.  This script only scores and preserves those raw strings; it never
asks a model to judge another model.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from run_closed_book_oracle_eval import score_answer

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "evals/leave_rules_corrupted/benchmark_50.json"
RETRIEVAL = ROOT / "evals/leave_rules_corrupted/runs/20260908-145149-50q-two-pass-retrieval-results.json"
OUT_DIR = ROOT / "evals/leave_rules_corrupted/runs"

# Raw answers from one independent B session per question.
B_ANSWERS = {
"corrupt-marriage-leave":"勞工結婚可以請八十八日婚假，工資照給。","corrupt-parent-bereavement":"可請喪假八十八日，工資照給。","corrupt-spouse-adoptive-bereavement":"配偶的養父母或繼父母喪亡時，可請喪假六十六日，工資照給。","corrupt-sibling-bereavement":"依提供證據，兄弟姊妹喪亡可請「三十三日」喪假，且工資照給；但因屬數值污染副本，無法確認原始天數是否為三十三日。","corrupt-outpatient-sick-limit":"十年內合計不得超過三百日。","corrupt-inpatient-sick-limit":"十年。","corrupt-combined-sick-limit":"未住院傷病假與住院傷病假二十年內合計不得超過十年。","corrupt-sick-pay-ratio":"普通傷病假十年內未超過三百日部分，工資按十分之一發給。","corrupt-personal-leave-limit":"事假在十年內合計不得超過一百四十日；事假期間不給薪。","corrupt-adverse-action-threshold":"十年內，普通傷病假未超過一百日。","corrupt-document-warning":"不可以。本檔案的數字與數量均經刻意改寫，嚴重偏離真實法規，因此不得作為法律依據。","corrupt-document-title":"勞工請假規則（數值污染測試版）","corrupt-source-system":"勞動部勞動法令查詢系統。","corrupt-amendment-date":"民國999年11月31日。","corrupt-article-101-basis":"本規則依《勞動基準法》第四百三十條規定訂定。","corrupt-effective-release":"本規則自發布日施行。","corrupt-effective-amendment":"自一千年十一月三十一日施行。","corrupt-middle-bereavement":"六十六日","corrupt-middle-bereavement-paid":"是，給薪，工資照給。","corrupt-shortest-bereavement":"配偶之祖父母。","corrupt-longest-bereavement":"父母、養父母、繼父母、配偶喪亡者，喪假八十八日，為三組中最長。","corrupt-sick-causes":"勞工因普通傷害、疾病或生理原因，必須治療或休養者。","corrupt-cancer-classification":"經醫師診斷罹患癌症（含原位癌）採門診方式治療者，其治療期間併入住院傷病假計算。","corrupt-pregnancy-classification":"住院傷病假。","corrupt-sick-pay-employer-topup":"由雇主補足。","corrupt-sick-pay-period":"普通傷病假在十年內未超過三百日的部分，工資按十分之一發給。","corrupt-sick-combined-rule":"未住院與住院傷病假合併計算時，以二十年為期間，合計不得超過十年。","corrupt-unpaid-leave-condition":"得予留職停薪。","corrupt-unpaid-leave-max":"留職停薪期間以十年為限。","corrupt-work-injury-leave":"公傷病假。","corrupt-work-injury-period":"公傷病假給予期間為勞工因職業災害所需的治療、休養期間。","corrupt-personal-reason":"勞工因有事故必須親自處理，得請事假。","corrupt-personal-unpaid":"事假期間不給工資。","corrupt-family-care-personal":"得依前項規定請事假。","corrupt-personal-hour-unit":"小時。","corrupt-family-care-exception":"有。除本法或其他法律另有規定外，得依前項規定辦理。","corrupt-public-paid":"依法令規定應給予公假者，工資照給。","corrupt-public-duration":"視實際需要決定。","corrupt-attendance-exempt-list":"勞工請婚假、喪假、公傷病假及公假時，不得視為缺勤而影響全勤獎金。","corrupt-miscarriage-attendance":"普通傷病假","corrupt-family-care-attendance":"不影響全勤獎金。","corrupt-sick-attendance-deduction":"普通傷病假（妊娠流產例外除外）的全勤獎金扣發，應按請普通傷病假日數依比例計算。","corrupt-adverse-employer-proof":"雇主負舉證責任。","corrupt-adverse-over-threshold":"雇主仍應綜合考量員工的工作能力、工作態度及實際績效等事項。","corrupt-adverse-not-only-days":"不可以。不得僅以請普通傷病假日數作為人事考核因素。","corrupt-adverse-exception":"有。除本規則或其他法律另有規定外，從其規定。","corrupt-leave-advance":"一般請假時，應於事前親自以口頭或書面敘明請假理由及日數。","corrupt-leave-emergency-delegate":"遇有急病或緊急事故時，得委託他人代辦請假手續。","corrupt-leave-proof-doc":"有關證明文件。","corrupt-violation-authority":"主管機關得依本法有關規定辦理。"}

# Raw answers from one independent C session per question.  The C prompt
# included only that question's two-pass final evidence and its gate value.
C_ANSWERS = {
"corrupt-marriage-leave":"勞工結婚可請婚假八十八日，工資照給。","corrupt-parent-bereavement":"父母、養父母、繼父母或配偶喪亡時，可請喪假八十八日，工資照給。","corrupt-spouse-adoptive-bereavement":"配偶的養父母或繼父母喪亡時，可請喪假六十六日，工資照給。","corrupt-sibling-bereavement":"證據不足，無法回答。","corrupt-outpatient-sick-limit":"十年內合計不得超過三百日。","corrupt-inpatient-sick-limit":"二十年內合計不得超過十年。","corrupt-combined-sick-limit":"未住院傷病假與住院傷病假二十年內合計不得超過十年。","corrupt-sick-pay-ratio":"普通傷病假十年內未超過三百日部分，工資按十分之一發給。","corrupt-personal-leave-limit":"事假十年內合計不得超過一百四十日；事假期間不給工資。","corrupt-adverse-action-threshold":"十年內，普通傷病假未超過一百日。","corrupt-document-warning":"不能作為法律依據；這是數值污染壓力測試副本，數字與數量均已刻意改寫，嚴重偏離真實法規。","corrupt-document-title":"勞工請假規則（數值污染測試版）","corrupt-source-system":"勞動部勞動法令查詢系統。","corrupt-amendment-date":"民國999年11月31日","corrupt-article-101-basis":"本規則依《勞動基準法》第430條規定訂定。","corrupt-effective-release":"本規則自發布日施行。","corrupt-effective-amendment":"自一千年十一月三十一日施行。","corrupt-middle-bereavement":"六十六日，工資照給。","corrupt-middle-bereavement-paid":"是，喪假六十六日工資照給。","corrupt-shortest-bereavement":"證據不足，無法回答","corrupt-longest-bereavement":"證據不足，無法回答","corrupt-sick-causes":"普通傷害、疾病或生理原因，且必須治療或休養者。","corrupt-cancer-classification":"經醫師診斷罹患癌症（含原位癌）並採門診治療者，治療期間應併入住院傷病假計算。","corrupt-pregnancy-classification":"住院傷病假。","corrupt-sick-pay-employer-topup":"由雇主補足。","corrupt-sick-pay-period":"普通傷病假於十年內未超過三百日的部分，工資按十分之一發給。","corrupt-sick-combined-rule":"未住院與住院傷病假於二十年內合併計算，合計不得超過十年；癌症門診及安胎休養期間併入住院傷病假計算。","corrupt-unpaid-leave-condition":"證據不足，無法回答","corrupt-unpaid-leave-max":"留職停薪期間最長以十年為限。","corrupt-work-injury-leave":"公傷病假。","corrupt-work-injury-period":"證據不足，無法回答","corrupt-personal-reason":"證據不足，無法回答","corrupt-personal-unpaid":"事假期間不給工資。","corrupt-family-care-personal":"依前項規定請事假，並得以小時為請假單位。","corrupt-personal-hour-unit":"得以小時為請假單位。","corrupt-family-care-exception":"有。親自照顧家庭成員請事假時，除《本法》或其他法律另有規定外，得依前項規定辦理。","corrupt-public-paid":"依法令應給的公假，工資照給；假期依實際需要決定。","corrupt-public-duration":"證據不足，無法回答","corrupt-attendance-exempt-list":"婚假、喪假、公傷病假及公假，不得視為缺勤而影響全勤獎金。","corrupt-miscarriage-attendance":"普通傷病假。","corrupt-family-care-attendance":"不影響全勤獎金。親自照顧家庭成員依第七十條規定請事假，不得視為缺勤。","corrupt-sick-attendance-deduction":"普通傷病假的全勤獎金，應依請假日數按比例扣發。","corrupt-adverse-employer-proof":"雇主。","corrupt-adverse-over-threshold":"雇主應綜合考量員工的工作能力、工作態度及實際績效等，不得僅以請假日數作為考核依據。","corrupt-adverse-not-only-days":"不可以。不得僅以請普通傷病假日數作為人事考核因素，仍應綜合考量工作能力、工作態度與實際績效。","corrupt-adverse-exception":"有。除本規則另有規定，或其他法律另有規定外，雇主不得因請普通傷病假而為不利處分。","corrupt-leave-advance":"一般請假時，應於事前親自以口頭或書面說明請假理由及請假日數。","corrupt-leave-emergency-delegate":"遇有急病或緊急事故，可委託他人代辦請假手續；辦理時，雇主得要求提供證明文件。","corrupt-leave-proof-doc":"有關證明文件。","corrupt-violation-authority":"由主管機關依本法有關規定辦理。"}

def main() -> None:
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    retrieval = json.loads(RETRIEVAL.read_text(encoding="utf-8"))
    rows = {r["id"]: r for r in retrieval["results"]}
    ids = {x["id"] for x in benchmark["items"]}
    if set(B_ANSWERS) != ids or set(C_ANSWERS) != ids:
        raise SystemExit("strict answer IDs do not match benchmark")
    results = []
    for item in benchmark["items"]:
        row = rows[item["id"]]
        b = B_ANSWERS[item["id"]]
        c = C_ANSWERS[item["id"]]
        results.append({
            "id": item["id"], "question": item["question"], "article": item["article"],
            "oracle_context": item["oracle_context"],
            "b_oracle_context": {"answer": b, "score": score_answer(b, item)},
            "c_two_pass_rag": {"answer": c, "score": score_answer(c, item),
                "retrieval_sufficient": row["evidence_evaluation"].get("sufficient"),
                "retrieval_status": row["evidence_evaluation"].get("status"),
                "retrieval_confidence": row["evidence_evaluation"].get("confidence"),
                "context_count": row["context_count"], "context_chars": row["context_chars"],
                "contexts": row["contexts"]},
        })
    bs = [r["b_oracle_context"]["score"]["score"] for r in results]
    cs = [r["c_two_pass_rag"]["score"]["score"] for r in results]
    payload = {"benchmark_id": benchmark["benchmark_id"], "benchmark_type": benchmark["benchmark_type"],
        "source_file": benchmark["source_file"], "source_id": retrieval["source_id"],
        "model": "gpt-5.6-luna", "mode": "strict per-question B Oracle vs C two-pass RAG",
        "protocol": {"fresh_session_per_answer": True, "fork_turns": "none", "tools": False,
            "files": False, "network": False, "external_knowledge": False,
            "cross_question_history": False, "followups": False, "retries": False,
            "c_gate_false_policy": "證據不足，無法回答"},
        "retrieval_artifact": str(RETRIEVAL.relative_to(ROOT)),
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "summary": {"B": {"average": round(sum(bs)/len(bs),1), "full_pass": sum(x==100 for x in bs), "scores": bs},
                    "C": {"average": round(sum(cs)/len(cs),1), "full_pass": sum(x==100 for x in cs), "scores": cs},
                    "gate_true": sum(r["c_two_pass_rag"]["retrieval_sufficient"] is True for r in results),
                    "gate_false": sum(r["c_two_pass_rag"]["retrieval_sufficient"] is False for r in results)},
        "results": results}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_json = OUT_DIR / f"{stamp}-strict-bc-50-results.json"
    out_md = OUT_DIR / f"{stamp}-strict-bc-50-summary.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 嚴格隔離 B/C 測試結果", "", "- 每題 B/C 各自使用一個全新 `fork_turns=none` session。", "- 子代理未提供工具、檔案、網路、外部知識或其他題目歷史；沒有 follow-up/retry。", "- C 僅收到該題 two-pass final evidence 與 gate；gate=False 時固定拒答。", "", f"- B Oracle：{payload['summary']['B']['full_pass']}/50 full pass，平均 {payload['summary']['B']['average']:.1f}", f"- C RAG：{payload['summary']['C']['full_pass']}/50 full pass，平均 {payload['summary']['C']['average']:.1f}", f"- Retrieval gate：{payload['summary']['gate_true']} true / {payload['summary']['gate_false']} false", "", "| # | 題目 | B | C | Gate |", "|---:|---|---:|---:|:---:|"]
    for i, r in enumerate(results, 1):
        lines.append(f"| {i} | {r['question'].replace('|','／')} | {r['b_oracle_context']['score']['score']:.1f} | {r['c_two_pass_rag']['score']['score']:.1f} | {r['c_two_pass_rag']['retrieval_sufficient']} |")
    out_md.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(out_json); print(out_md); print(json.dumps(payload["summary"], ensure_ascii=False))

if __name__ == "__main__":
    main()
