#!/usr/bin/env python3
"""Run comparable closed-book and oracle-context evaluations via Ollama."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "evals/leave_rules_ab/benchmark.json"
DEFAULT_OUTPUT_DIR = ROOT / "evals/leave_rules_ab/runs"
OLLAMA_GENERATE_URL = "http://127.0.0.1:11434/api/generate"


def ask_ollama(model: str, system: str, prompt: str) -> dict:
    payload = json.dumps(
        {
            "model": model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "keep_alive": -1,
            "options": {"temperature": 0, "seed": 42, "num_predict": 180},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_GENERATE_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=180) as response:
        body = json.loads(response.read().decode("utf-8"))
    return {
        "answer": str(body.get("response") or "").strip(),
        "wall_ms": round((time.perf_counter() - started) * 1000, 2),
        "ollama_total_ms": round(float(body.get("total_duration") or 0) / 1_000_000, 2),
        "prompt_tokens": int(body.get("prompt_eval_count") or 0),
        "answer_tokens": int(body.get("eval_count") or 0),
        "done_reason": body.get("done_reason"),
    }


def normalized(text: str) -> str:
    return re.sub(r"[\s，。；：、（）()／/]", "", text).lower()


def score_answer(answer: str, item: dict) -> dict:
    compact = normalized(answer)
    checks = []
    for accepted in item["required"]:
        matched = next((term for term in accepted if normalized(term) in compact), "")
        checks.append({"accepted": accepted, "matched": matched, "pass": bool(matched)})
    forbidden_hits = [term for term in item.get("forbidden", []) if normalized(term) in compact]
    # Every benchmark item has a complete oracle answer. A refusal that happens
    # to quote the expected words is still a failed answer, not a true positive.
    if normalized("根據目前檢索資料無法確認") in compact:
        forbidden_hits.append("根據目前檢索資料無法確認")
    base = sum(check["pass"] for check in checks) / max(1, len(checks))
    score = 0.0 if forbidden_hits else base
    return {
        "score": round(score * 100, 1),
        "required_checks": checks,
        "forbidden_hits": forbidden_hits,
    }


# The original benchmark's `required` groups describe the complete oracle
# paragraph.  That is useful for a strict completeness metric, but it can
# penalize an answer that correctly answers the narrower question.  These
# groups describe only the facts the question asks for.  Evidence still has
# to contain each group; this is not a free-form semantic pass.
QUERY_REQUIRED = {
    "corrupt-middle-bereavement": [["六十六日", "66日"]],
    "corrupt-middle-bereavement-paid": [["工資照給", "給薪", "有薪"]],
    "corrupt-shortest-bereavement": [["三十三日", "33日"], ["曾祖父母", "兄弟姊妹", "配偶之祖父母"]],
    "corrupt-longest-bereavement": [["八十八日", "88日"], ["父母", "養父母", "繼父母", "配偶"]],
    "corrupt-cancer-classification": [["併入住院傷病假", "住院傷病假"]],
    "corrupt-pregnancy-classification": [["住院傷病假"]],
    "corrupt-sick-pay-employer-topup": [["雇主補足", "由雇主補足", "雇主"]],
    "corrupt-sick-combined-rule": [["未住院與住院傷病假", "未住院"], ["住院傷病假"], ["二十年", "20年"], ["十年", "10年"]],
    "corrupt-unpaid-leave-condition": [["留職停薪"]],
    "corrupt-unpaid-leave-max": [["十年", "10年"]],
    "corrupt-work-injury-leave": [["公傷病假"]],
    "corrupt-work-injury-period": [["治療"], ["休養期間", "休養"]],
    "corrupt-family-care-personal": [["請事假", "事假"]],
    "corrupt-personal-hour-unit": [["小時"]],
    "corrupt-public-paid": [["工資照給", "給薪", "有薪"]],
    "corrupt-public-duration": [["實際需要"]],
    "corrupt-miscarriage-attendance": [["普通傷病假"]],
    "corrupt-family-care-attendance": [["不得視為缺勤", "不影響全勤獎金"]],
    "corrupt-sick-attendance-deduction": [["扣發"], ["按請普通傷病假日數依比例計算", "按請假日數按比例扣發", "依請假日數按比例扣發"]],
    "corrupt-adverse-employer-proof": [["雇主"]],
    "corrupt-adverse-not-only-days": [["不得僅以請普通傷病假日數作為考量因素", "不得僅以請普通傷病假日數作為人事考核因素"]],
    "corrupt-adverse-exception": [["本規則或其他法律另有規定", "本規則另有規定或其他法律另有規定"]],
    "corrupt-leave-advance": [["事前"], ["口頭", "書面"], ["請假理由"], ["日數"]],
    "corrupt-leave-emergency-delegate": [["急病"], ["緊急事故"], ["委託他人"]],
    "corrupt-leave-proof-doc": [["證明文件"]],
    "corrupt-violation-authority": [["主管機關"], ["本法"]],
}


def _semantic_normalized(text: str) -> str:
    """Normalize punctuation and bracket variants for fact matching."""
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", str(text or "")).casefold()


def _contains_any_fact(text: str, alternatives: list[str]) -> str:
    compact = _semantic_normalized(text)
    return next((term for term in alternatives if _semantic_normalized(term) in compact), "")


def _answer_number_tokens(text: str) -> list[str]:
    return re.findall(
        r"[零〇一二兩三四五六七八九十百千万億\d]+\s*(?:日|天|年|月|小時|分鐘|%|分之一)",
        str(text or ""),
    )


def score_grounded_answer(answer: str, item: dict, evidence_text: str = "") -> dict:
    """Score question-relevant facts while requiring evidence support.

    Unlike ``score_answer``, this does not require the model to restate every
    sentence in the oracle paragraph.  It checks only the question's core
    fact groups, verifies those groups occur in the supplied evidence, rejects
    explicit contradictions/refusals, and rejects answer numbers absent from
    the evidence.  This is intentionally deterministic and auditable.
    """
    answer_compact = _semantic_normalized(answer)
    evidence_compact = _semantic_normalized(evidence_text or item.get("oracle_context", ""))
    required = QUERY_REQUIRED.get(item.get("id"), item.get("required", []))
    checks = []
    for alternatives in required:
        answer_match = _contains_any_fact(answer, alternatives)
        evidence_match = _contains_any_fact(evidence_text or item.get("oracle_context", ""), alternatives)
        checks.append({
            "accepted": alternatives,
            "answer_match": answer_match,
            "evidence_match": evidence_match,
            "pass": bool(answer_match and evidence_match),
        })

    forbidden_hits = []
    for term in item.get("forbidden", []):
        normalized_term = _semantic_normalized(term)
        # `給薪` is a valid positive alternative but is also a substring of
        # `不給薪`; treat the explicit negative phrase as the decisive fact.
        if normalized_term in {"給薪", "有薪"} and (
            "不給薪" in answer_compact or "無薪" in answer_compact
        ):
            continue
        if normalized_term and normalized_term in answer_compact:
            forbidden_hits.append(term)
    if "根據目前檢索資料無法確認" in answer_compact or "證據不足" in answer_compact:
        forbidden_hits.append("拒答")

    answer_numbers = Counter(_semantic_normalized(token) for token in _answer_number_tokens(answer))
    evidence_numbers = Counter(_semantic_normalized(token) for token in _answer_number_tokens(evidence_text or item.get("oracle_context", "")))
    unsupported_numbers = []
    for token, count in answer_numbers.items():
        missing = max(0, count - evidence_numbers.get(token, 0))
        unsupported_numbers.extend([token] * missing)
    if unsupported_numbers:
        forbidden_hits.extend([f"未在證據出現的數值:{token}" for token in unsupported_numbers])

    passed = all(check["pass"] for check in checks) and not forbidden_hits
    return {
        "score": 100.0 if passed else 0.0,
        "required_checks": checks,
        "forbidden_hits": forbidden_hits,
        "metric": "question-relevant-grounded-facts",
    }


def write_summary(path: Path, payload: dict) -> None:
    selected = payload["selected_for_c"]
    lines = [
        "# Closed-book / Oracle-context A-B 測試結果",
        "",
        f"- Benchmark：`{payload['benchmark_id']}`",
        f"- 模型：`{payload['model']}`",
        f"- 執行時間：{payload['run_at']}",
        f"- 候選題數：{len(payload['results'])}",
        f"- 選入 C 測試：{len(selected)} 題（B 分數高於 A，依差距排序）",
        "",
        "| # | 題目 | A | B | 差距 |",
        "|---:|---|---:|---:|---:|",
    ]
    for index, result in enumerate(payload["results"], start=1):
        question = result["question"].replace("|", "／")
        lines.append(
            f"| {index} | {question} | {result['closed_book']['score']['score']:.1f} | "
            f"{result['oracle_context']['score']['score']:.1f} | {result['delta']:+.1f} |"
        )
    lines.extend(["", "## 預定進入 C 的題目", ""])
    for index, result in enumerate(selected, start=1):
        lines.extend(
            [
                f"### {index}. {result['question']}",
                "",
                f"- A：{result['closed_book']['answer']}",
                f"- B：{result['oracle_context']['answer']}",
                f"- 分數：A {result['closed_book']['score']['score']:.1f} / B {result['oracle_context']['score']['score']:.1f}",
                f"- Oracle：{result['article']}，{result['oracle_context_text']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--select", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    items = benchmark["items"][: args.limit or None]
    results = []
    closed_system = (
        "你正在接受 Closed-book 測試。不可使用外部資料或假裝看到文件。"
        "請依自身知識以繁體中文直接回答；若不確定請明說。答案限100字。"
    )
    oracle_system = (
        "你正在接受 Oracle-context 測試。只能根據使用者提供的正確證據，以繁體中文回答。"
        "不得加入證據沒有的規則、數字或例外；若證據不足必須明說。答案限100字。"
    )

    for index, item in enumerate(items, start=1):
        print(f"[{index}/{len(items)}] {item['id']} A", flush=True)
        closed = ask_ollama(args.model, closed_system, item["question"])
        closed["score"] = score_answer(closed["answer"], item)
        oracle_prompt = f"正確證據（{item['article']}）：\n{item['oracle_context']}\n\n問題：{item['question']}"
        print(f"[{index}/{len(items)}] {item['id']} B", flush=True)
        oracle = ask_ollama(args.model, oracle_system, oracle_prompt)
        oracle["score"] = score_answer(oracle["answer"], item)
        results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "article": item["article"],
                "oracle_context_text": item["oracle_context"],
                "closed_book": closed,
                "oracle_context": oracle,
                "delta": round(oracle["score"]["score"] - closed["score"]["score"], 1),
            }
        )

    eligible = [result for result in results if result["delta"] > 0 and result["oracle_context"]["score"]["score"] >= 90]
    eligible.sort(key=lambda result: (-result["delta"], result["closed_book"]["score"]["score"], result["id"]))
    payload = {
        "benchmark_id": benchmark["benchmark_id"],
        "source_url": benchmark["source_url"],
        "model": args.model,
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "policy": {"temperature": 0, "seed": 42, "max_answer_tokens": 180},
        "selection_rule": "oracle score >= 90 and oracle score > closed-book score; top delta first",
        "results": results,
        "selected_for_c": eligible[: args.select],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = args.output_dir / f"{stem}-ab-results.json"
    md_path = args.output_dir / f"{stem}-ab-summary.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(md_path, payload)
    print(json.dumps({"json": str(json_path), "summary": str(md_path), "selected": len(payload["selected_for_c"])}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
