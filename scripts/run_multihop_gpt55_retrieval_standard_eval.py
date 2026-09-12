#!/usr/bin/env python3
"""Compare MultiHop-RAG retrieval variants with GPT-5.5 and four gates.

The two versions use the real retrieved evidence produced by the existing
MultiHop-RAG benchmark:

* 無優化版（粗糙版）：hard 600-character chunks, raw 50/50 score addition and
  the basic reranker.
* 全優化版：1024-token parents / 256-token children, RRF, complexity routing,
  optional Cross-Encoder reranking and parent evidence expansion.

Each version answers the same fixed ten questions with ``codex:gpt-5.5``.
Gold evidence is used only by the evaluator; it is never placed in the answer
prompt.  A case is 100% correct only when answer correctness, evidence
sufficiency, strict citation compliance and hallucination-free gates all pass.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_demo.answer_quality import evaluate_answer_quality
from rag_demo.chunk_strategies import CHUNK_STRATEGY_DYNAMIC, CHUNK_STRATEGY_HARD
from rag_demo.config import RagConfig
from rag_demo.cross_encoder import DEFAULT_CROSS_ENCODER_MODEL, CrossEncoderReranker
from rag_demo.embeddings import DEFAULT_EMBEDDING_MODEL, embed_chunks, embed_texts
from rag_demo.evidence_validation import validate_answer_evidence
from rag_demo.hybrid_retrieval import Bm25Index, tokenize_bm25
from rag_demo.model_providers import ask_model, model_request_timeout, parse_model_spec
from rag_demo.rag_pipeline import build_grounded_answer_request, enforce_grounded_answer_contract

from scripts.run_multihop_qwen_ab import (
    _dense_hits,
    _hard_chunks,
    _normalise_embeddings,
    _parent_child_index,
    _retrieve_optimized,
    _retrieve_unoptimized,
)


DEFAULT_QUERIES = ROOT / "RAG測試題庫" / "01_MultiHop-RAG" / "data" / "MultiHopRAG.json"
DEFAULT_CORPUS = ROOT / "RAG測試題庫" / "01_MultiHop-RAG" / "data" / "corpus.json"
DEFAULT_SAMPLE = ROOT / "evals" / "multihop_gpt55_oracle" / "sample-10.json"
DEFAULT_OUTPUT = ROOT / "evals" / "multihop_gpt55_retrieval_standard" / "gpt55-ab-results.json"
DEFAULT_REPORT = ROOT / "evals" / "multihop_gpt55_retrieval_standard" / "gpt55-ab-report.md"
DEFAULT_SAMPLE_COPY = ROOT / "evals" / "multihop_gpt55_retrieval_standard" / "sample-10.json"
DEFAULT_CACHE_DIR = ROOT / "evals" / "multihop_gpt55_retrieval_standard" / "embedding-cache"
DEFAULT_MODEL = "codex:gpt-5.5"
DEFAULT_SEMANTIC_THRESHOLD = 0.40


def _compact(value: object) -> str:
    return re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", str(value or "").casefold())


def _answer_is_refusal(answer: str) -> bool:
    text = str(answer or "").strip().casefold()
    head = text[:320]
    # A response may give a clear No and then qualify a separate premise with
    # an evidence limitation.  That is not a total refusal.
    explicit_binary = (
        re.match(r"^(?:答案|結論)?\s*(?:是|不是|否|對|不對|並非|不完全正確|不能說)(?:[。！？；：:，,\s]|$)", head)
        or re.match(r"^(?:答案|結論)?\s*(?:為|是)?\s*[:：]?\s*(?:yes|no)\b", head)
        or re.match(r"^(?:答案|結論)?\s*(?:並非如此|不能說沒有變化|不完全正確)", head)
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
    return any(marker in head for marker in markers)


def _first_sentence(text: str) -> str:
    return re.split(r"[\r\n。！？；;]", str(text or ""), maxsplit=1)[0].strip()


def _answer_matches_gold(answer: str, gold: str) -> bool:
    expected = str(gold or "").strip()
    text = str(answer or "").strip()
    if not expected:
        return False
    if expected.casefold() == "insufficient information.":
        return _answer_is_refusal(text)
    if expected.casefold() in {"yes", "no"}:
        head = _first_sentence(text)[:320]
        no_patterns = (
            r"^(?:答案|結論)?\s*(?:為|是)?\s*[:：]?\s*(?:no|否|不是|並非|不完全正確|不能說|不成立|不對)",
            r"^(?:答案|結論)?\s*[:：]?\s*(?:no|否|不是|並非|不完全正確|不能說|不成立|不對)",
        )
        yes_patterns = (
            r"^(?:答案|結論)?\s*(?:為|是)?\s*[:：]?\s*(?:yes|是|對|正確|符合|同意|一致)",
            r"^(?:答案|結論)?\s*[:：]?\s*(?:yes|是|對|正確|符合|同意|一致)",
        )
        if expected.casefold() == "no" and any(re.search(pattern, head, re.IGNORECASE) for pattern in no_patterns):
            return True
        if expected.casefold() == "yes" and any(re.search(pattern, head, re.IGNORECASE) for pattern in yes_patterns):
            return True
        # Handle a qualified conclusion such as "不能說沒有變化" that is
        # not the first token but is still the answer to a binary question.
        leading = text[:500]
        if expected.casefold() == "no" and re.search(
            r"(?:並非如此|不完全正確|不能說沒有變化|答案\s*(?:為|是)?\s*(?:否|no)|結論\s*(?:為|是)?\s*(?:否|no))",
            leading,
            re.IGNORECASE,
        ):
            return True
        if expected.casefold() == "yes" and re.search(
            r"(?:答案\s*(?:為|是)?\s*(?:是|yes)|結論\s*(?:為|是)?\s*(?:是|yes))",
            leading,
            re.IGNORECASE,
        ):
            return True
        return False
    expected_compact = _compact(expected)
    answer_compact = _compact(text)
    if expected_compact and expected_compact in answer_compact:
        return True
    expected_tokens = {token for token in re.findall(r"[a-z0-9]+", expected.casefold()) if len(token) > 1}
    answer_tokens = set(re.findall(r"[a-z0-9]+", text.casefold()))
    return bool(expected_tokens) and expected_tokens.issubset(answer_tokens)


def _load_sample(path: Path, limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list) or len(items) < limit:
        raise ValueError(f"sample must contain at least {limit} items")
    selected = [dict(item) for item in items[:limit]]
    for item in selected:
        item["answerable"] = bool(
            item.get("answerable", str(item.get("question_type") or "") != "null_query")
        )
    sample_payload = {
        "dataset": str(payload.get("dataset") or "yixuantt/MultiHopRAG") if isinstance(payload, dict) else "yixuantt/MultiHopRAG",
        "dataset_total": int(payload.get("dataset_total") or 2556) if isinstance(payload, dict) else 2556,
        "sample_size": len(selected),
        "sample_seed": payload.get("sample_seed") if isinstance(payload, dict) else None,
        "source": str(path.resolve().relative_to(ROOT)),
        "items": selected,
    }
    return selected, sample_payload


def _gold_fact_recall(contexts: Sequence[dict[str, Any]], item: dict[str, Any]) -> tuple[int, int, list[str]]:
    context_text = "\n".join(str(context.get("content") or "") for context in contexts)
    context_tokens = set(tokenize_bm25(context_text))
    matched: list[str] = []
    total = 0
    for evidence in item.get("evidence_list") or ():
        if not isinstance(evidence, dict):
            continue
        fact = str(evidence.get("fact") or "").strip()
        if not fact:
            continue
        total += 1
        fact_tokens = set(tokenize_bm25(fact))
        lexical = len(fact_tokens & context_tokens) / max(1, len(fact_tokens))
        if _compact(fact) in _compact(context_text) or lexical >= 0.45:
            matched.append(fact)
    return len(matched), total, matched


def _segment_is_meta(segment: str) -> bool:
    clean = re.sub(r"\[(\d+)\]", "", str(segment or "")).strip()
    if not clean:
        return True
    if re.fullmatch(r"(?:來源|資料來源|引用|參考資料)\s*[：:]?.*", clean):
        return True
    if re.fullmatch(r"(?:[-•]|\d+[.)])?\s*(?:缺少|不足|無法確認|資料不足|沒有提供).{0,260}", clean):
        return True
    if re.search(r"根據目前檢索資料無法確認|目前檢索資料不足|資料不足|缺少(?:的)?證據", clean):
        # A sentence that contains only an evidence-boundary statement is a
        # permitted refusal/meta statement.  Factual clauses with numbers or
        # named entities still need a citation.
        factual_terms = re.sub(r"根據目前檢索資料無法確認|目前檢索資料不足|資料不足|缺少(?:的)?證據", "", clean).strip(" ：:，,。；")
        if not factual_terms or len(factual_terms) < 8:
            return True
    return False


def _strict_citation_gate(
    answer: str,
    contexts: Sequence[dict[str, Any]],
    *,
    safe_empty_evidence_refusal: bool = False,
) -> tuple[bool, list[str]]:
    text = str(answer or "").strip()
    validation = validate_answer_evidence(text, contexts)
    issues: list[str] = []
    invalid = list(validation.get("invalid_citations") or [])
    if invalid:
        issues.append("invalid citation rank: " + ",".join(str(rank) for rank in invalid))
    if safe_empty_evidence_refusal and _answer_is_refusal(text):
        # The standard permits a null_query refusal to explain missing input
        # without inventing a citation.  Retrieval may still return unrelated
        # candidates, but they are not evidence for the null_query itself.
        return (not invalid), issues
    # Keep a citation attached to the sentence that ends immediately before
    # it.  Newlines and whitespace after punctuation delimit sentences, while
    # a citation directly after punctuation stays in the same segment.
    segments = [
        segment.strip()
        for segment in re.split(r"(?<=[。！？；!?])(?=\s+|$)|\r?\n+", text)
        if segment.strip()
    ]
    substantive_count = 0
    for segment in segments:
        clean = re.sub(r"^\s*[-•*#]+\s*", "", segment).strip()
        if _segment_is_meta(clean):
            continue
        if re.fullmatch(r"\[(?:\d+\s*,?\s*)+\]", clean):
            continue
        substantive_count += 1
        ranks = [int(rank) for rank in re.findall(r"\[(\d+)\]", clean)]
        if not ranks:
            issues.append("uncited substantive sentence: " + clean[:160])
            continue
        if not re.search(r"(?:\[\d+\]\s*)+$", clean.rstrip("。！？；!?")):
            issues.append("citation is not at sentence end: " + clean[:160])
    if substantive_count == 0:
        # Empty-evidence safe refusals are explicitly allowed by the standard.
        return (not invalid), issues
    if not validation.get("valid_citations"):
        issues.append("no valid citation")
    return (not issues), issues


def _standard_case_quality(
    item: dict[str, Any],
    answer: str,
    raw_answer: str,
    contexts: Sequence[dict[str, Any]],
    semantic_threshold: float,
    model_error: str = "",
) -> dict[str, Any]:
    gold = str(item.get("answer") or "").strip()
    answerable = bool(item.get("answerable"))
    answer_correct = _answer_matches_gold(answer, gold) if answerable else _answer_is_refusal(answer)
    raw_answer_correct = _answer_matches_gold(raw_answer, gold) if answerable else _answer_is_refusal(raw_answer)
    diagnostics = evaluate_answer_quality(
        question=str(item.get("query") or ""),
        answer=answer,
        contexts=contexts,
        expected_facts=(),
        embedding_fn=embed_texts,
        semantic_threshold=semantic_threshold,
    )
    grounding = diagnostics["answer_in_retrieved_chunks"]
    hallucination = diagnostics["hallucination"]
    unsupported_claims = list(grounding.get("unsupported_claims") or [])
    unsupported_numbers = list(hallucination.get("unsupported_numbers") or [])
    invalid_citations = list((grounding.get("citation_validation") or {}).get("invalid_citations") or [])
    # Gate B allows a safe evidence-boundary refusal and otherwise requires
    # every extracted substantive claim to be supported.  It does not require
    # the answer to be correct; that is gate A.
    evidence_sufficient = not unsupported_claims and not unsupported_numbers and not invalid_citations
    if _answer_is_refusal(answer) and not unsupported_numbers and not invalid_citations:
        evidence_sufficient = True
    safe_empty_evidence_refusal = not any(
        isinstance(evidence, dict) and str(evidence.get("fact") or "").strip()
        for evidence in (item.get("evidence_list") or ())
    )
    citation_compliant, citation_issues = _strict_citation_gate(
        answer,
        contexts,
        safe_empty_evidence_refusal=safe_empty_evidence_refusal,
    )
    # Gate D is independent of citation placement.  Invalid or missing ranks
    # are reported by gate C; unsupported claims and numbers remain hallucination
    # failures.  A leading safe refusal is treated as hallucination-free.
    hallucination_free = (
        True
        if _answer_is_refusal(answer) and not unsupported_numbers
        else not unsupported_claims and not unsupported_numbers
    )
    return {
        "answer_correct": answer_correct,
        "raw_answer_correct": raw_answer_correct,
        "evidence_sufficient": evidence_sufficient,
        "citation_compliant": citation_compliant,
        "hallucination_free": hallucination_free,
        "standard_100_pass": bool(answer_correct and evidence_sufficient and citation_compliant and hallucination_free),
        "refused": _answer_is_refusal(answer),
        "model_error": model_error,
        "citation_issues": citation_issues,
        "unsupported_claims": unsupported_claims,
        "unsupported_numbers": unsupported_numbers,
        "diagnostics": diagnostics,
    }


def _summary(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    answerable = [case for case in cases if case["answerable"]]
    unanswerable = [case for case in cases if not case["answerable"]]
    total = len(cases)
    fact_numerator = sum(case["retrieval"]["gold_fact_matched"] for case in cases)
    fact_denominator = sum(case["retrieval"]["gold_fact_total"] for case in cases)
    generations = [float(case["timings_ms"]["generation"]) for case in cases]
    totals = [float(case["timings_ms"]["total"]) for case in cases]
    return {
        "question_count": total,
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "answer_correct_count": sum(bool(case["quality"]["answer_correct"]) for case in cases),
        "answer_accuracy_answerable": round(
            sum(bool(case["quality"]["answer_correct"]) for case in answerable) / max(1, len(answerable)),
            4,
        ),
        "answer_accuracy_all": round(
            sum(bool(case["quality"]["answer_correct"]) for case in cases) / max(1, total),
            4,
        ),
        "evidence_sufficient_rate": round(
            sum(bool(case["quality"]["evidence_sufficient"]) for case in cases) / max(1, total),
            4,
        ),
        "citation_compliant_rate": round(
            sum(bool(case["quality"]["citation_compliant"]) for case in cases) / max(1, total),
            4,
        ),
        "hallucination_free_rate": round(
            sum(bool(case["quality"]["hallucination_free"]) for case in cases) / max(1, total),
            4,
        ),
        "standard_100_pass_count": sum(bool(case["quality"]["standard_100_pass"]) for case in cases),
        "standard_100_rate": round(
            sum(bool(case["quality"]["standard_100_pass"]) for case in cases) / max(1, total),
            4,
        ),
        "gold_evidence_fact_recall": round(fact_numerator / max(1, fact_denominator), 4),
        "safe_refusal_rate": round(
            sum(bool(case["quality"]["refused"]) for case in unanswerable) / max(1, len(unanswerable)),
            4,
        ),
        "model_error_count": sum(bool(case["quality"]["model_error"]) for case in cases),
        "average_generation_ms": round(statistics.mean(generations), 2) if generations else 0.0,
        "p95_generation_ms": round(_p95(generations), 2),
        "average_total_ms": round(statistics.mean(totals), 2) if totals else 0.0,
        "p95_total_ms": round(_p95(totals), 2),
    }


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    return ordered[int(round((len(ordered) - 1) * 0.95))]


def _load_or_build_embeddings(
    chunks: Sequence[dict[str, Any]],
    cache_path: Path,
    *,
    model_name: str,
) -> np.ndarray:
    expected_count = len(chunks)
    if cache_path.is_file():
        try:
            matrix = np.load(cache_path).astype(np.float32)
            if matrix.ndim == 2 and matrix.shape[0] == expected_count:
                print(f"Loaded embedding cache {cache_path.name} ({matrix.shape[0]} rows)", flush=True)
                return _normalise_embeddings(matrix)
        except Exception:
            pass
    print(f"Embedding {expected_count} chunks for {cache_path.name}...", flush=True)
    matrix = _normalise_embeddings(embed_chunks(chunks, model_name=model_name))
    np.save(cache_path, matrix)
    return matrix


def _run_version(
    *,
    label: str,
    version: str,
    items: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
    matrix: np.ndarray,
    bm25: Bm25Index,
    embedding_model: str,
    model: str,
    settings: RagConfig,
    optimized: bool,
    parent_index=None,
    cross_encoder=None,
    semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    timeout_seconds: float = 240.0,
) -> dict[str, Any]:
    query_cache: dict[str, np.ndarray] = {}
    cases: list[dict[str, Any]] = []
    for position, item in enumerate(items, start=1):
        question = str(item.get("query") or "").strip()
        print(f"[{label} {position}/{len(items)}] {item.get('id')} -> {model}", flush=True)
        retrieval_started = time.perf_counter()
        if optimized:
            contexts, retrieval = _retrieve_optimized(
                question,
                parent_index,
                matrix,
                bm25,
                query_cache,
                embedding_model,
                cross_encoder,
                settings,
            )
        else:
            contexts, retrieval = _retrieve_unoptimized(
                question,
                chunks,
                matrix,
                bm25,
                query_cache,
                embedding_model,
                settings,
            )
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000.0
        matched, total, matched_facts = _gold_fact_recall(contexts, item)
        request = build_grounded_answer_request(
            question=question,
            contexts=contexts,
            history=[],
            memories=[],
        )
        generation_started = time.perf_counter()
        raw_answer = ""
        answer = ""
        model_error = ""
        try:
            with model_request_timeout(timeout_seconds):
                raw_answer = ask_model(request["prompt"], model=model, system=request["system"])
            answer = enforce_grounded_answer_contract(raw_answer, contexts)
        except Exception as exc:
            model_error = f"{type(exc).__name__}: {str(exc)[:500]}"
            print(f"  model error: {model_error}", flush=True)
        generation_ms = (time.perf_counter() - generation_started) * 1000.0
        quality = _standard_case_quality(
            item,
            answer,
            raw_answer,
            contexts,
            semantic_threshold=semantic_threshold,
            model_error=model_error,
        )
        cases.append(
            {
                "id": item.get("id"),
                "dataset_index": item.get("dataset_index"),
                "question_type": item.get("question_type"),
                "question": question,
                "gold_answer": item.get("answer"),
                "answerable": bool(item.get("answerable")),
                "answer": answer,
                "raw_answer": raw_answer,
                "contexts": contexts,
                "retrieval": {
                    **retrieval,
                    "gold_fact_matched": matched,
                    "gold_fact_total": total,
                    "matched_gold_facts": matched_facts,
                },
                "timings_ms": {
                    "retrieval": round(retrieval_ms, 2),
                    "generation": round(generation_ms, 2),
                    "total": round(retrieval_ms + generation_ms, 2),
                },
                "quality": quality,
            }
        )
    return {
        "name": label,
        "version": version,
        "model": model,
        "configuration": {
            "retrieval": (
                "hard 600-character chunks, raw 50/50 addition, basic reranker top 5"
                if not optimized
                else "1024-token parent / 256-token child, RRF, 100 candidates, complexity routing, Cross-Encoder for complex questions, parent expansion"
            ),
            "answer_prompt": "retrieved contexts only; no gold evidence, history or memory",
            "memory": "none",
        },
        "summary": _summary(cases),
        "cases": cases,
    }


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# MultiHop-RAG GPT-5.5 無優化版與全優化版四 Gate 評測",
        "",
        "本次使用同一組固定 10 題與同一份 609 文件 corpus。無優化版與全優化版各自執行實際檢索，再把檢索結果交給獨立的 GPT-5.5 ephemeral 子代理回答。Gold evidence 只在評分階段使用，沒有放進回答 prompt。",
        "",
        f"- 模型：`{payload['model']}`",
        f"- 題庫總數：`{payload['dataset_total']}`",
        f"- 測試題數：`{payload['sample_size']}`（可回答 `{payload['answerable_count']}` 題、null_query `{payload['unanswerable_count']}` 題）",
        f"- 抽樣來源：`{payload['sample_source']}`",
        f"- 執行時間：`{payload['run_at']}`",
        "",
        "## 100% 正確標準",
        "",
        "每題必須同時通過：A 答案語意正確、B 每個實質主張都有檢索證據、C 每個事實或結論句都有緊接的有效 rank、D 沒有未支持主張或數字。四項全部通過才算 100% 正確；null_query 的安全拒答可通過 A、B、D。",
        "",
        "## 版本比較",
        "",
        "| 版本 | A 答案正確率（可回答題） | B 證據充分 | C 引用合規 | D 無幻覺 | 四 Gate 全通過 | Gold fact recall | 安全拒答 | 平均總 ms | P95 總 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for version in payload["versions"]:
        summary = version["summary"]
        lines.append(
            f"| {version['name']} | {summary['answer_accuracy_answerable']:.1%} | {summary['evidence_sufficient_rate']:.1%} | "
            f"{summary['citation_compliant_rate']:.1%} | {summary['hallucination_free_rate']:.1%} | "
            f"{summary['standard_100_rate']:.1%} ({summary['standard_100_pass_count']}/{summary['question_count']}) | "
            f"{summary['gold_evidence_fact_recall']:.1%} | {summary['safe_refusal_rate']:.1%} | "
            f"{summary['average_total_ms']} | {summary['p95_total_ms']} |"
        )
    lines.extend(
        [
            "",
            "## 題目明細",
            "",
            "| 題目 | 類型 | 無優化版四 Gate | 全優化版四 Gate | 無優化版答案 | 全優化版答案 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    by_id = {
        version["name"]: {case["id"]: case for case in version["cases"]}
        for version in payload["versions"]
    }
    for item in payload["versions"][0]["cases"]:
        unoptimized = by_id["無優化版"][item["id"]]
        optimized = by_id["全優化版"][item["id"]]
        uq = unoptimized["quality"]
        oq = optimized["quality"]
        uflags = "".join("通過 " if uq[gate] else "失敗 " for gate in ("answer_correct", "evidence_sufficient", "citation_compliant", "hallucination_free")).strip()
        oflags = "".join("通過 " if oq[gate] else "失敗 " for gate in ("answer_correct", "evidence_sufficient", "citation_compliant", "hallucination_free")).strip()
        lines.append(
            f"| `{item['id']}` | `{item['question_type']}` | {uflags} | {oflags} | "
            f"{str(unoptimized['answer']).replace('|', '／').replace(chr(10), ' ')[:180]} | "
            f"{str(optimized['answer']).replace('|', '／').replace(chr(10), ' ')[:180]} |"
        )
    lines.extend(
        [
            "",
            "## 指標與限制",
            "",
            "- A 答案語意正確：Yes／No 會辨識同義否定與限定語；entity 題要求答案包含正確人物或項目；null_query 以安全拒答為正確。",
            "- B 證據充分：以檢索 contexts 做 lexical + multilingual embedding 支持檢查；拒答只要沒有提出未支持主張，可通過 B，但仍可能因 A 失敗而不是 100% 正確。",
            "- C 引用合規：沿用 accuracy-standard 的嚴格逐句規則，引用 rank 必須存在且在同一事實／結論句末端；格式錯誤不會自動算成幻覺。",
            "- D 無幻覺：檢查未支持的主張與數字；安全拒答視為無幻覺。",
            "- Gold fact recall 只用於比較檢索召回，不是四 Gate 其中一項；gold evidence 沒有注入 prompt。",
            "- GPT-5.5 每次呼叫都使用新的 ephemeral 子代理，不傳入對話歷史、長期記憶、檔案、網路或工具。",
            "- 這是 deterministic / heuristic 評分對 100% 標準的自動化實作；若要正式宣稱 100%，仍應人工覆核報告中的逐題主張與引用。",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    spec = parse_model_spec(args.model)
    if spec.provider != "codex" or spec.model != "gpt-5.5":
        raise ValueError("Both versions must use codex:gpt-5.5")
    items, sample_payload = _load_sample(args.sample.resolve(), args.sample_size)
    corpus = json.loads(args.corpus.resolve().read_text(encoding="utf-8"))
    if not isinstance(corpus, list) or len(corpus) != 609:
        raise ValueError(f"MultiHop-RAG corpus must contain 609 documents, got {len(corpus) if isinstance(corpus, list) else 'invalid'}")
    print(f"Loaded {len(items)} questions and {len(corpus)} corpus documents", flush=True)
    base = RagConfig.from_env().normalized()
    settings = replace(
        base,
        hybrid_candidate_k=100,
        hybrid_max_candidate_k=200,
        hybrid_top_k=5,
        rerank_top_k=5,
        simple_query_top_k=5,
        hybrid_rrf_k=60,
        query_complexity_threshold=2.0,
        rerank_fusion_weight=0.30,
        rerank_bm25_weight=0.32,
        rerank_embedding_weight=0.26,
        rerank_coverage_weight=0.10,
        rerank_phrase_weight=0.02,
    ).normalized()
    hard_chunks = _hard_chunks(corpus)
    parent_child = _parent_child_index(corpus)
    print(
        f"Built {len(hard_chunks)} hard chunks, {len(parent_child.parents)} parents, {len(parent_child.children)} children",
        flush=True,
    )
    args.cache_dir.resolve().mkdir(parents=True, exist_ok=True)
    hard_cache = args.cache_dir.resolve() / "hard-embeddings.npy"
    child_cache = args.cache_dir.resolve() / "child-embeddings.npy"
    hard_embeddings = _load_or_build_embeddings(
        hard_chunks,
        hard_cache,
        model_name=args.embedding_model,
    )
    child_embeddings = _load_or_build_embeddings(
        parent_child.children,
        child_cache,
        model_name=args.embedding_model,
    )
    hard_bm25 = Bm25Index(hard_chunks, k1=settings.hybrid_bm25_k1, b=settings.hybrid_bm25_b)
    child_bm25 = Bm25Index(parent_child.children, k1=settings.hybrid_bm25_k1, b=settings.hybrid_bm25_b)
    cross_encoder = CrossEncoderReranker(args.cross_encoder_model)
    unoptimized = _run_version(
        label="無優化版",
        version="unoptimized",
        items=items,
        chunks=hard_chunks,
        matrix=hard_embeddings,
        bm25=hard_bm25,
        embedding_model=args.embedding_model,
        model=args.model,
        settings=settings,
        optimized=False,
        semantic_threshold=args.semantic_threshold,
        timeout_seconds=args.timeout_seconds,
    )
    optimized = _run_version(
        label="全優化版",
        version="optimized",
        items=items,
        chunks=parent_child.children,
        matrix=child_embeddings,
        bm25=child_bm25,
        embedding_model=args.embedding_model,
        model=args.model,
        settings=settings,
        optimized=True,
        parent_index=parent_child,
        cross_encoder=cross_encoder,
        semantic_threshold=args.semantic_threshold,
        timeout_seconds=args.timeout_seconds,
    )
    return {
        "schema_version": "multihop-rag-gpt55-retrieval-standard-v1",
        "dataset": sample_payload["dataset"],
        "dataset_total": sample_payload["dataset_total"],
        "sample_size": sample_payload["sample_size"],
        "sample_source": sample_payload["source"],
        "sample_seed": sample_payload.get("sample_seed"),
        "answerable_count": sum(bool(item.get("answerable")) for item in items),
        "unanswerable_count": sum(not bool(item.get("answerable")) for item in items),
        "corpus_documents": len(corpus),
        "hard_chunk_count": len(hard_chunks),
        "parent_chunk_count": len(parent_child.parents),
        "child_chunk_count": len(parent_child.children),
        "embedding_model": args.embedding_model,
        "cross_encoder_model": args.cross_encoder_model,
        "model": args.model,
        "semantic_threshold": args.semantic_threshold,
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "versions": [unoptimized, optimized],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--sample-copy", type=Path, default=DEFAULT_SAMPLE_COPY)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--cross-encoder-model", default=DEFAULT_CROSS_ENCODER_MODEL)
    parser.add_argument("--semantic-threshold", type=float, default=DEFAULT_SEMANTIC_THRESHOLD)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    args = parser.parse_args()
    payload = run(args)
    output_path = args.output.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    sample_copy_path = args.sample_copy.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    sample_copy_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(_render_report(payload), encoding="utf-8")
    source_payload = json.loads(args.sample.resolve().read_text(encoding="utf-8"))
    source_items = source_payload.get("items") if isinstance(source_payload, dict) else source_payload
    sample_copy_payload = {
        "dataset": payload["dataset"],
        "dataset_total": payload["dataset_total"],
        "sample_size": payload["sample_size"],
        "sample_source": payload["sample_source"],
        "items": list(source_items[: args.sample_size]),
    }
    sample_copy_path.write_text(json.dumps(sample_copy_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for version in payload["versions"]:
        version["summary"] = _summary(version["cases"])
    # Re-write after summary fields are attached so the JSON and report have
    # exactly the same aggregate values.
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(_render_report(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "report": str(report_path),
                "versions": {version["name"]: version["summary"] for version in payload["versions"]},
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
