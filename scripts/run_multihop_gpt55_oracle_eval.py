#!/usr/bin/env python3
"""Run a 10-question GPT-5.5 answer test with gold evidence in the prompt.

This benchmark deliberately bypasses retrieval.  For every MultiHop-RAG
question, the question's annotated ``evidence_list`` is converted into the
trusted-evidence section of the normal grounded-answer prompt.  The test
therefore measures answer generation, citation compliance and hallucination
handling when the correct evidence is available.

The ``codex:gpt-5.5`` provider creates one fresh ephemeral subagent per
question.  No conversation history, persistent memory, files, network or
tools are passed to those subagents.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_demo.answer_quality import evaluate_answer_quality
from rag_demo.embeddings import DEFAULT_EMBEDDING_MODEL, embed_texts
from rag_demo.model_providers import ask_model, model_request_timeout, parse_model_spec
from rag_demo.rag_pipeline import (
    build_grounded_answer_request,
    enforce_grounded_answer_contract,
)


DEFAULT_QUERIES = ROOT / "RAG測試題庫" / "01_MultiHop-RAG" / "data" / "MultiHopRAG.json"
DEFAULT_SAMPLE_SOURCE = ROOT / "evals" / "multihop_rag_qwen" / "sample-100.json"
DEFAULT_SAMPLE = ROOT / "evals" / "multihop_gpt55_oracle" / "sample-10.json"
DEFAULT_OUTPUT = ROOT / "evals" / "multihop_gpt55_oracle" / "gpt55-results.json"
DEFAULT_REPORT = ROOT / "evals" / "multihop_gpt55_oracle" / "gpt55-report.md"
DEFAULT_MODEL = "codex:gpt-5.5"
DEFAULT_SEMANTIC_THRESHOLD = 0.40


def _compact(value: object) -> str:
    return re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", str(value or "").casefold())


def _answer_is_refusal(answer: str) -> bool:
    text = str(answer or "").strip().casefold()
    # A comparison answer can mention the refusal phrase while still giving a
    # clear conclusion, for example "不是；...無法確認...".  Only treat the
    # response as a refusal when the refusal is the leading conclusion.
    head = text[:240]
    explicit_binary = (
        re.match(r"^(?:答案|結論)?\s*(?:是|不是|否|對|不對)(?:[。！？；：:，,\s]|$)", head)
        or re.match(r"^(?:答案|結論)?\s*(?:為|是)?\s*[:：]?\s*(?:yes|no)\b", head)
    )
    if explicit_binary:
        return False
    markers = (
        "根據目前檢索資料無法確認",
        "目前檢索資料不足",
        "資料不足",
        "insufficient information",
        "cannot determine",
        "cannot be determined",
        "not enough information",
    )
    return any(marker in text for marker in markers)


def _answer_matches_gold(answer: str, gold: str) -> bool:
    if _answer_is_refusal(answer):
        return False
    expected = str(gold or "").strip()
    if not expected:
        return False
    if expected.casefold() in {"yes", "no"}:
        answer_text = str(answer or "").strip()
        head = re.split(r"[\r\n。！？；;]", answer_text, maxsplit=1)[0]
        match = re.search(r"\b(yes|no)\b", head.casefold())
        if match:
            return match.group(1) == expected.casefold()
        match = re.search(
            r"^(?:答案|結論)?\s*(?:為|是)?\s*[：:]?\s*(是|不是|否|對|不對)",
            head,
        )
        if not match:
            return False
        answer_yes = match.group(1) in {"是", "對"}
        return answer_yes == (expected.casefold() == "yes")
    expected_compact = _compact(expected)
    answer_compact = _compact(answer)
    if expected_compact and expected_compact in answer_compact:
        return True
    expected_tokens = set(re.findall(r"[a-z0-9]+", expected.casefold()))
    answer_tokens = set(re.findall(r"[a-z0-9]+", str(answer or "").casefold()))
    expected_tokens = {token for token in expected_tokens if len(token) > 1}
    return bool(expected_tokens) and expected_tokens.issubset(answer_tokens)


def _load_questions(
    queries_path: Path,
    sample_path: Path,
    sample_size: int,
    seed: int,
    sample_source: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = json.loads(queries_path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) != 2556:
        raise ValueError(
            "MultiHop-RAG query file must contain 2556 records, got "
            f"{len(data) if isinstance(data, list) else 'invalid'}"
        )
    if sample_size <= 0 or sample_size > len(data):
        raise ValueError("sample_size must be between 1 and the full dataset size")

    selected: list[dict[str, Any]] = []
    selected_indices: list[int] = []
    selection_source = "random_seed"
    if sample_source is not None and sample_source.is_file():
        source_payload = json.loads(sample_source.read_text(encoding="utf-8"))
        source_items = source_payload.get("items") if isinstance(source_payload, dict) else None
        if isinstance(source_items, list) and len(source_items) >= sample_size:
            for item in source_items[:sample_size]:
                dataset_index = int(item.get("dataset_index", -1))
                if 0 <= dataset_index < len(data):
                    selected_indices.append(dataset_index)
            if len(selected_indices) == sample_size:
                selection_source = f"first_{sample_size}_from_{sample_source.relative_to(ROOT)}"

    if len(selected_indices) != sample_size:
        import random

        selected_indices = sorted(random.Random(seed).sample(range(len(data)), sample_size))

    for index in selected_indices:
        item = dict(data[index])
        question_type = str(item.get("question_type") or "unknown")
        gold = str(item.get("answer") or "").strip()
        item.update(
            {
                "id": f"multihop-{index + 1:04d}",
                "dataset_index": index,
                "answerable": question_type != "null_query"
                and gold.casefold() != "insufficient information.",
            }
        )
        selected.append(item)

    payload = {
        "dataset": "yixuantt/MultiHopRAG",
        "dataset_total": len(data),
        "sample_size": len(selected),
        "sample_seed": seed,
        "selection_source": selection_source,
        "indices": selected_indices,
        "items": selected,
    }
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return selected, payload


def _gold_contexts(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert annotated evidence into the exact context shape used by RAG."""

    contexts: list[dict[str, Any]] = []
    for rank, evidence in enumerate(item.get("evidence_list") or (), start=1):
        if not isinstance(evidence, dict):
            continue
        fact = str(evidence.get("fact") or "").strip()
        if not fact:
            continue
        source = str(evidence.get("source") or "MultiHop-RAG").strip()
        title = str(evidence.get("title") or source).strip()
        published = str(evidence.get("published_at") or "").strip()
        url = str(evidence.get("url") or "").strip()
        page = " | ".join(value for value in (source, published, url) if value)
        contexts.append(
            {
                "id": f"{item.get('id', 'question')}-gold-{rank:02d}",
                "rank": len(contexts) + 1,
                "title": title,
                "page": page or source,
                "source": url or source,
                "content": fact,
                "branch": "gold-evidence",
                "gold_evidence": True,
                "author": str(evidence.get("author") or "").strip(),
                "category": str(evidence.get("category") or "").strip(),
            }
        )
    return contexts


def _case_quality(
    item: dict[str, Any],
    raw_answer: str,
    answer: str,
    contexts: Sequence[dict[str, Any]],
    model_error: str = "",
    semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
) -> dict[str, Any]:
    expected_facts = [
        str(evidence.get("fact") or "").strip()
        for evidence in (item.get("evidence_list") or ())
        if isinstance(evidence, dict) and str(evidence.get("fact") or "").strip()
    ]
    context_facts = [str(context.get("content") or "").strip() for context in contexts]
    evidence_coverage = (
        sum(fact in context_facts for fact in expected_facts) / len(expected_facts)
        if expected_facts
        else (1.0 if not contexts else 0.0)
    )
    answerable = bool(item.get("answerable"))
    gold = str(item.get("answer") or "").strip()
    answer_correct = _answer_matches_gold(answer, gold) if answerable else _answer_is_refusal(answer)
    grounding = evaluate_answer_quality(
        question=str(item.get("query") or ""),
        answer=answer,
        contexts=contexts,
        expected_facts=(),
        embedding_fn=embed_texts,
        semantic_threshold=semantic_threshold,
    )
    strict_grounding = grounding
    grounding = _allow_gold_binary_inference(
        grounding,
        answer=answer,
        answer_correct=answer_correct,
        gold_answer=gold,
    )
    return {
        "answer_correct": answer_correct,
        "raw_answer_correct": _answer_matches_gold(raw_answer, gold) if answerable else _answer_is_refusal(raw_answer),
        "answer_supported_by_gold_evidence": bool(
            grounding["answer_in_retrieved_chunks"]["passed"]
        ),
        "gold_evidence_prompt_coverage": round(evidence_coverage, 4),
        "gold_evidence_fact_count": len(expected_facts),
        "refused": _answer_is_refusal(answer),
        "model_error": model_error,
        "hallucination": grounding["hallucination"],
        "answer_grounding": grounding["answer_in_retrieved_chunks"],
        "strict_hallucination": strict_grounding["hallucination"],
        "strict_answer_grounding": strict_grounding["answer_in_retrieved_chunks"],
    }


def _is_binary_conclusion(claim: str) -> bool:
    compact = re.sub(r"[\s。！？；：:，,]", "", str(claim or ""))
    if compact in {
        "是",
        "不是",
        "否",
        "對",
        "不對",
        "答案是",
        "答案不是",
        "答案為是",
        "答案為否",
        "結論是",
        "結論不是",
        "結論為是",
        "結論為否",
    }:
        return True
    # Models often attach a short qualification to the direction, such as
    # "不是，不能照題述確認".  It remains a binary conclusion when it is
    # short and starts with the direction token.
    return len(compact) <= 24 and compact.startswith(("是", "不是", "否", "對", "不對"))


def _allow_gold_binary_inference(
    quality: dict[str, Any],
    *,
    answer: str,
    answer_correct: bool,
    gold_answer: str,
) -> dict[str, Any]:
    """Count a correctly cited Yes/No conclusion as an evidence inference.

    MultiHop-RAG comparison and temporal labels often require a conclusion
    such as "不是" that is not verbatim in any single fact.  The benchmark
    gold answer supplies the direction check; all explanatory claims and
    numbers must still pass the ordinary evidence validator.
    """

    if not answer_correct or _answer_is_refusal(answer):
        return quality
    is_binary_question = str(gold_answer or "").strip().casefold() in {"yes", "no"}
    evidence = quality["answer_in_retrieved_chunks"]
    validation = evidence.get("citation_validation") or {}
    if not validation.get("valid_citations") or validation.get("invalid_citations"):
        return quality
    details = [dict(item) for item in evidence.get("claim_details") or []]
    changed = False
    for item in details:
        claim = str(item.get("claim") or "")
        identity_inference = bool(
            not is_binary_question
            and _compact(gold_answer)
            and _compact(gold_answer) in _compact(claim)
        )
        uncertainty_inference = bool(
            is_binary_question
            and re.search(r"無法確認|不能.*確認|資料不足|沒有.*直接證據|未.*表示|未.*提到", claim)
        )
        if not item.get("supported") and (
            (is_binary_question and _is_binary_conclusion(claim))
            or identity_inference
            or uncertainty_inference
        ):
            item["supported"] = True
            item["support_basis"] = "gold_evidence_logical_inference"
            changed = True
    if not changed:
        return quality
    unsupported = [item for item in details if not item.get("supported")]
    unsupported_numbers = list(quality["hallucination"].get("unsupported_numbers") or [])
    passed = not unsupported and not unsupported_numbers
    score = sum(bool(item.get("supported")) for item in details) / max(1, len(details))
    adjusted_evidence = dict(evidence)
    adjusted_evidence.update(
        {
            "passed": passed,
            "score": round(score, 4),
            "supported_claim_count": len(details) - len(unsupported),
            "unsupported_claims": [item["claim"] for item in unsupported],
            "claim_details": details,
        }
    )
    adjusted_hallucination = dict(quality["hallucination"])
    adjusted_hallucination.update(
        {
            "detected": bool(unsupported or unsupported_numbers),
            "hallucination_free": not bool(unsupported or unsupported_numbers),
            "unsupported_claims": [item["claim"] for item in unsupported],
            "reasons": [f"unsupported claim: {item['claim']}" for item in unsupported]
            + [f"unsupported number: {number}" for number in unsupported_numbers],
        }
    )
    adjusted = dict(quality)
    adjusted["answer_in_retrieved_chunks"] = adjusted_evidence
    adjusted["hallucination"] = adjusted_hallucination
    return adjusted


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    return ordered[int(round((len(ordered) - 1) * 0.95))]


def run(args: argparse.Namespace) -> dict[str, Any]:
    questions, sample_payload = _load_questions(
        args.queries.resolve(),
        args.sample.resolve(),
        args.sample_size,
        args.seed,
        args.sample_source.resolve() if args.sample_source else None,
    )
    cases: list[dict[str, Any]] = []
    spec = parse_model_spec(args.model)
    if spec.provider != "codex":
        raise ValueError("This benchmark requires a codex model, for example codex:gpt-5.5")

    for position, item in enumerate(questions, start=1):
        question = str(item.get("query") or "").strip()
        contexts = _gold_contexts(item)
        request = build_grounded_answer_request(
            question=question,
            contexts=contexts,
            history=[],
            memories=[],
        )
        print(
            f"[{position}/{len(questions)}] {item.get('id')} {item.get('question_type')} -> {args.model} "
            f"(gold evidence: {len(contexts)})",
            flush=True,
        )
        started = time.perf_counter()
        raw_answer = ""
        answer = ""
        model_error = ""
        try:
            with model_request_timeout(args.timeout_seconds):
                raw_answer = ask_model(
                    request["prompt"],
                    model=args.model,
                    system=request["system"],
                )
            answer = enforce_grounded_answer_contract(raw_answer, contexts)
        except Exception as exc:
            model_error = f"{type(exc).__name__}: {str(exc)[:500]}"
            print(f"  model error: {model_error}", flush=True)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        cases.append(
            {
                "id": item.get("id"),
                "dataset_index": item.get("dataset_index"),
                "question_type": item.get("question_type"),
                "question": question,
                "gold_answer": item.get("answer"),
                "answerable": bool(item.get("answerable")),
                "gold_evidence": contexts,
                "answer_prompt": request["prompt"],
                "raw_answer": raw_answer,
                "answer": answer,
                "timings_ms": {"generation": round(elapsed_ms, 2)},
                "quality": _case_quality(
                    item,
                    raw_answer,
                    answer,
                    contexts,
                    model_error,
                    semantic_threshold=args.semantic_threshold,
                ),
            }
        )

    answerable = [case for case in cases if case["answerable"]]
    unanswerable = [case for case in cases if not case["answerable"]]
    hallucination_free = sum(
        bool(case["quality"]["hallucination"]["hallucination_free"]) for case in cases
    )
    strict_hallucination_free = sum(
        bool(case["quality"]["strict_hallucination"]["hallucination_free"])
        for case in cases
    )
    summary = {
        "question_count": len(cases),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "answer_accuracy": round(
            sum(bool(case["quality"]["answer_correct"]) for case in answerable)
            / max(1, len(answerable)),
            4,
        ),
        "raw_answer_accuracy": round(
            sum(bool(case["quality"]["raw_answer_correct"]) for case in answerable)
            / max(1, len(answerable)),
            4,
        ),
        "answer_supported_by_gold_evidence_rate": round(
            sum(bool(case["quality"]["answer_supported_by_gold_evidence"]) for case in cases)
            / max(1, len(cases)),
            4,
        ),
        "strict_answer_supported_rate": round(
            sum(
                bool(case["quality"]["strict_answer_grounding"]["passed"])
                for case in cases
            )
            / max(1, len(cases)),
            4,
        ),
        "gold_evidence_prompt_coverage": round(
            statistics.mean(case["quality"]["gold_evidence_prompt_coverage"] for case in cases),
            4,
        )
        if cases
        else 0.0,
        "safe_refusal_rate": round(
            sum(bool(case["quality"]["refused"]) for case in unanswerable)
            / max(1, len(unanswerable)),
            4,
        ),
        "hallucination_free": hallucination_free,
        "hallucination_free_rate": round(hallucination_free / max(1, len(cases)), 4),
        "strict_hallucination_free": strict_hallucination_free,
        "strict_hallucination_free_rate": round(
            strict_hallucination_free / max(1, len(cases)), 4
        ),
        "model_error_count": sum(bool(case["quality"]["model_error"]) for case in cases),
        "average_generation_ms": round(
            statistics.mean(case["timings_ms"]["generation"] for case in cases), 2
        )
        if cases
        else 0.0,
        "p95_generation_ms": round(_p95([case["timings_ms"]["generation"] for case in cases]), 2),
    }
    return {
        "schema_version": "multihop-rag-gpt55-oracle-v1",
        "dataset": sample_payload["dataset"],
        "dataset_total": sample_payload["dataset_total"],
        "sample_size": sample_payload["sample_size"],
        "sample_seed": sample_payload["sample_seed"],
        "selection_source": sample_payload["selection_source"],
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": args.model,
        "answer_prompt_policy": {
            "evidence_mode": "gold evidence_list injected into trusted_evidence",
            "retrieval_used": False,
            "correct_evidence_attached": True,
            "independent_request_per_question": True,
            "session_mode": "ephemeral",
            "history_passed": False,
            "memory": False,
            "tools": False,
            "semantic_threshold": args.semantic_threshold,
        },
        "summary": summary,
        "cases": cases,
    }


def render_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# MultiHop-RAG GPT-5.5 正確證據注入測試",
        "",
        "本次從完整 2,556 題題庫測試 10 題。每題把題庫標註的 evidence_list 直接放入 trusted_evidence prompt，因此本報告評估的是回答模型與證據遵循能力，不包含檢索召回率。",
        "",
        f"- 模型：`{payload['model']}`",
        f"- 題庫總數：`{payload['dataset_total']}`",
        f"- 測試題數：`{payload['sample_size']}`",
        f"- 抽樣方式：`{payload['selection_source']}`",
        f"- 執行時間：`{payload['run_at']}`",
        "- 每題獨立建立一個 ephemeral GPT-5.5 子代理；不傳入對話歷史、長期記憶、檔案、網路或工具。",
        "",
        "## 主要指標",
        "",
        "| 指標 | 結果 |",
        "| --- | ---: |",
        f"| 最終答案正確率（可回答題） | `{summary['answer_accuracy']:.1%}` ({sum(bool(case['quality']['answer_correct']) for case in payload['cases'] if case['answerable'])}/{summary['answerable_count']}) |",
        f"| 原始答案正確率（契約檢查前） | `{summary['raw_answer_accuracy']:.1%}` |",
        f"| 答案由附帶正確證據支持 | `{summary['answer_supported_by_gold_evidence_rate']:.1%}` |",
        f"| 嚴格逐句證據支持率 | `{summary['strict_answer_supported_rate']:.1%}` |",
        f"| 正確 evidence 放入 prompt 覆蓋率 | `{summary['gold_evidence_prompt_coverage']:.1%}` |",
        f"| 無幻覺率 | `{summary['hallucination_free_rate']:.1%}` ({summary['hallucination_free']}/{summary['question_count']}) |",
        f"| 嚴格無幻覺率 | `{summary['strict_hallucination_free_rate']:.1%}` ({summary['strict_hallucination_free']}/{summary['question_count']}) |",
        f"| null_query 安全拒答率 | `{summary['safe_refusal_rate']:.1%}` |",
        f"| 平均生成時間 | `{summary['average_generation_ms']} ms` |",
        f"| P95 生成時間 | `{summary['p95_generation_ms']} ms` |",
        f"| 模型呼叫錯誤 | `{summary['model_error_count']}` |",
        "",
        "## 題目明細",
        "",
        "| 題目 | 類型 | Gold 答案 | 最終正確 | 證據支持 | 幻覺 | Gold facts | 生成 ms |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for case in payload["cases"]:
        quality = case["quality"]
        lines.append(
            f"| `{case['id']}` | `{case['question_type']}` | `{case['gold_answer']}` | "
            f"{'是' if quality['answer_correct'] else '否'} | "
            f"{'是' if quality['answer_supported_by_gold_evidence'] else '否'} | "
            f"{'否' if quality['hallucination']['hallucination_free'] else '是'} | "
            f"{quality['gold_evidence_fact_count']} | {case['timings_ms']['generation']} |"
        )
    lines.extend(
        [
            "",
            "## 指標定義",
            "",
            "- 最終答案正確率：可回答題中，通過來源契約檢查後的答案是否符合 MultiHop-RAG gold answer；Yes／No 題保留方向判斷。",
            "- 答案由附帶正確證據支持：回答中的實質主張是否能由 prompt 中的 gold evidence 透過詞彙或 multilingual embedding 支持，且引用 rank 有效。",
            "- 嚴格逐句證據支持率保留未套用 gold evidence 邏輯推論前的原始 deterministic 判定；放寬判定只接受方向正確、引用有效的二元結論或包含 gold entity 的必要識別句。",
            "- 正確 evidence 放入 prompt 覆蓋率：題庫 evidence_list 的 facts 是否完整注入 trusted_evidence；null_query 沒有 evidence 時視為符合。",
            "- 無幻覺率：回答中的主張、數字與 citation 均通過 deterministic evidence validation；安全拒答視為無幻覺。",
            "- 嚴格無幻覺率是未套用上述邏輯推論放寬規則的對照值。",
            "- 這次沒有執行向量或 BM25 檢索，所以不能把本報告的 evidence coverage 解讀成 retrieval recall。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--sample-source", type=Path, default=DEFAULT_SAMPLE_SOURCE)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--semantic-threshold", type=float, default=DEFAULT_SEMANTIC_THRESHOLD)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    args = parser.parse_args()
    payload = run(args)
    output_path = args.output.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")
    print(
        json.dumps(
            {"output": str(output_path), "report": str(report_path), **payload["summary"]},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
