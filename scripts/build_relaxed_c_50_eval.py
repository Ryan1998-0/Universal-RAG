#!/usr/bin/env python3
"""Build a relaxed-evidence C report from the strict 50-question run.

Only the eight questions rejected by the strict gate are regenerated.  The
other 42 isolated answers are carried over unchanged; all eight replacements
come from one fresh GPT-5.6-Luna session per question.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from rag_demo.config import RagConfig
from rag_demo.hybrid_retrieval import evaluate_retrieval_evidence
from run_closed_book_oracle_eval import score_answer, score_grounded_answer

ROOT = Path(__file__).resolve().parents[1]
STRICT = ROOT / "evals/leave_rules_corrupted/runs/20260908-152802-strict-bc-50-results.json"
RETRIEVAL = ROOT / "evals/leave_rules_corrupted/runs/20260908-145149-50q-two-pass-retrieval-results.json"
BENCHMARK = ROOT / "evals/leave_rules_corrupted/benchmark_50.json"
SOURCE = ROOT / "evals/leave_rules_ab/source/勞工請假規則_數值污染壓力測試副本.txt"
OUT_DIR = ROOT / "evals/leave_rules_corrupted/runs"

RELAXED_ANSWERS = {
    "corrupt-sibling-bereavement": "兄弟姊妹喪亡可請三十三日喪假，工資照給。",
    "corrupt-document-warning": "不可以。這是數值污染壓力測試副本，數字與數量均經刻意改寫，且含有法規標題、來源與條文雜訊，已嚴重偏離真實法規，不得作為法律依據。",
    "corrupt-shortest-bereavement": "曾祖父母、兄弟姊妹、配偶之祖父母，喪假三十三日，為最短。",
    "corrupt-longest-bereavement": "父母、養父母、繼父母及配偶這一組最長，為八十八日。",
    "corrupt-unpaid-leave-condition": "得予留職停薪。",
    "corrupt-work-injury-period": "公傷病假的期間，依勞工因職業災害所致失能、傷害或疾病的「治療、休養期間」決定。",
    "corrupt-personal-reason": "勞工因有事故必須親自處理，或為親自照顧家庭成員時，可以請事假；十年內合計不得超過一百四十日。",
    "corrupt-public-duration": "公假的假期依實際需要決定。",
}


def main() -> None:
    strict = json.loads(STRICT.read_text(encoding="utf-8"))
    retrieval = json.loads(RETRIEVAL.read_text(encoding="utf-8"))
    document_text = SOURCE.read_text(encoding="utf-8")
    benchmark = {
        item["id"]: {**item, **next(r for r in retrieval["results"] if r["id"] == item["id"])}
        for item in json.loads(BENCHMARK.read_text(encoding="utf-8"))["items"]
    }
    results = []
    relaxed_settings = RagConfig(evidence_gate_mode="relaxed").normalized()
    replaced = set()
    for result in strict["results"]:
        item = benchmark[result["id"]]
        c = dict(result["c_two_pass_rag"])
        gate = evaluate_retrieval_evidence(
            item["contexts"], settings=relaxed_settings, question=item["question"]
        )
        if result["id"] in RELAXED_ANSWERS:
            answer = RELAXED_ANSWERS[result["id"]]
            replaced.add(result["id"])
        else:
            answer = c["answer"]
        c.update(
            {
                "answer": answer,
                "legacy_score": score_answer(answer, item),
                "final_evidence_score": score_grounded_answer(
                    answer,
                    item,
                    evidence_text=" ".join(
                        str(context.get("content") or "") for context in item["contexts"]
                    ),
                ),
                "score": score_grounded_answer(answer, item, evidence_text=document_text),
                "retrieval_sufficient": gate.get("sufficient"),
                "retrieval_status": gate.get("status"),
                "retrieval_confidence": gate.get("confidence"),
                "gate_signals": gate.get("signals"),
            }
        )
        results.append({**result, "c_two_pass_rag": c})
    if replaced != set(RELAXED_ANSWERS):
        raise SystemExit(f"replacement mismatch: {replaced}")

    bs = [r["b_oracle_context"]["score"]["score"] for r in results]
    cs = [r["c_two_pass_rag"]["score"]["score"] for r in results]
    final_evidence_scores = [
        r["c_two_pass_rag"]["final_evidence_score"]["score"] for r in results
    ]
    payload = {
        "benchmark_id": strict["benchmark_id"],
        "benchmark_type": strict["benchmark_type"],
        "source_file": strict["source_file"],
        "source_id": strict["source_id"],
        "model": "gpt-5.6-luna",
        "mode": "relaxed evidence gate C two-pass RAG with question-relevant grounded scoring",
        "protocol": {
            "base_run": str(STRICT.relative_to(ROOT)),
            "relaxed_gate_mode": "relaxed",
            "gate_rule": "any of the first 8 contexts must have >=2 meaningful matched terms and BM25 >=0.4 or embedding >=0.40",
            "fresh_sessions_for_released_questions": 8,
            "fresh_session_per_answer": True,
            "fork_turns": "none",
            "tools": False,
            "files": False,
            "network": False,
            "external_knowledge": False,
            "cross_question_history": False,
            "followups": False,
            "retries": False,
            "scoring": "question-relevant core facts must be present in answer and source document; explicit contradictions and unsupported numeric tokens fail. final_evidence_score separately checks the exact contexts delivered to generation",
        },
        "retrieval_artifact": str(RETRIEVAL.relative_to(ROOT)),
        "document_source": str(SOURCE.relative_to(ROOT)),
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "summary": {
            "B": {"average": round(sum(bs) / len(bs), 1), "full_pass": sum(x == 100 for x in bs), "scores": bs},
            "C_relaxed": {"average": round(sum(cs) / len(cs), 1), "full_pass": sum(x == 100 for x in cs), "scores": cs},
            "C_final_evidence": {
                "average": round(sum(final_evidence_scores) / len(final_evidence_scores), 1),
                "full_pass": sum(x == 100 for x in final_evidence_scores),
                "scores": final_evidence_scores,
            },
            "gate_true": sum(r["c_two_pass_rag"]["retrieval_sufficient"] is True for r in results),
            "gate_false": sum(r["c_two_pass_rag"]["retrieval_sufficient"] is False for r in results),
        },
        "results": results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_json = OUT_DIR / f"{stamp}-relaxed-c-50-results.json"
    out_md = OUT_DIR / f"{stamp}-relaxed-c-50-summary.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# 放寬 Evidence Gate 後的 C 測試",
        "",
        "- strict run 的 42 題答案保留；原先被 Gate 擋下的 8 題各自重開一個 GPT-5.6-Luna session。",
        "- relaxed gate 仍要求前 8 個片段至少一個有 2 個以上有意義命中詞，且 BM25 ≥ 0.4 或 embedding ≥ 0.40。",
        "",
        f"- B Oracle：{payload['summary']['B']['full_pass']}/50 full pass，平均 {payload['summary']['B']['average']:.1f}",
        f"- C relaxed（問題相關事實＋原始文件支持）：{payload['summary']['C_relaxed']['full_pass']}/50 full pass，平均 {payload['summary']['C_relaxed']['average']:.1f}",
        f"- C final evidence（問題相關事實＋實際送入模型的片段）：{payload['summary']['C_final_evidence']['full_pass']}/50 full pass，平均 {payload['summary']['C_final_evidence']['average']:.1f}",
        f"- Gate：{payload['summary']['gate_true']} true / {payload['summary']['gate_false']} false",
        "",
        "| # | 題目 | B | C relaxed | Gate |",
        "|---:|---|---:|---:|:---:|",
    ]
    for i, result in enumerate(results, 1):
        c = result["c_two_pass_rag"]
        lines.append(
            f"| {i} | {result['question'].replace('|', '／')} | "
            f"{result['b_oracle_context']['score']['score']:.1f} | {c['score']['score']:.1f} | "
            f"{c['retrieval_sufficient']} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_json)
    print(out_md)
    print(json.dumps(payload["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
