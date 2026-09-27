"""Auditable answer grounding and hallucination metrics.

The evaluator compares answer claims with the server-retrieved chunks.  It is
deliberately deterministic: no evaluator model, web search or conversation
memory is used.  Embedding similarity is optional; lexical evidence remains a
fallback for minimal installations and test doubles.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional, Sequence

from rag_demo.evidence_validation import validate_answer_evidence
from rag_demo.hybrid_retrieval import tokenize_bm25


_NUMBER_PATTERN = re.compile(
    r"[零〇一二兩三四五六七八九十百千万億\d]+\s*(?:日|天|年|月|小時|分鐘|秒|%|分之一)"
)
_META_CLAIM_PATTERN = re.compile(
    r"^(?:(?:兩者|二者|兩個版本)?(?:相同|一樣|一致|未改變|沒有改變|無異動)|"
    r"相較之下|綜上|總結|因此|可見)[了。！!？?：:]*$"
)
_CHINESE_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "兩": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CHINESE_UNITS = {"十": 10, "百": 100, "千": 1000, "萬": 10000, "億": 100000000}


def evaluate_answer_quality(
    question: str,
    answer: str,
    contexts: Sequence[dict[str, Any]],
    expected_facts: Sequence[str] = (),
    embedding_fn: Optional[Callable[..., Sequence[Sequence[float]]]] = None,
    semantic_threshold: float = 0.50,
) -> dict[str, Any]:
    """Return the two requested quality indicators for one answer.

    ``answer_in_retrieved_chunks`` asks whether every substantive answer claim
    has lexical or semantic support in at least one retrieved chunk.
    ``hallucination`` flags unsupported claims, invalid citation ranks and
    answer numbers that do not occur in the retrieved evidence.
    """

    del question  # retained in the public signature for future report labels
    text = str(answer or "").strip()
    evidence = [
        context
        for context in contexts
        if isinstance(context, dict) and str(context.get("content") or "").strip()
    ]
    citation_validation = validate_answer_evidence(text, evidence)
    refusal = citation_validation["status"] == "refused"
    claims = _extract_claims(text)
    support = _claim_support(claims, evidence, embedding_fn, semantic_threshold)
    expected = [str(fact).strip() for fact in expected_facts if str(fact).strip()]
    all_evidence_text = "\n".join(str(context.get("content") or "") for context in evidence)
    answer_fact_matches = [
        fact for fact in expected if _contains_normalized(text, fact)
    ]
    evidence_fact_matches = [
        fact for fact in expected if _contains_normalized(all_evidence_text, fact)
    ]
    if (
        citation_validation.get("valid_citations")
        and expected
        and len(answer_fact_matches) == len(expected)
        and len(evidence_fact_matches) == len(expected)
    ):
        # Comparative conclusions such as "兩版相同" or "2016 版未保留
        # 例外" are valid inferences when the answer's question facts and
        # citations are both supported, even if that exact conclusion sentence
        # does not occur verbatim in a chunk.
        for item in support:
            if not item["supported"] and _is_evidence_inference(item["claim"]):
                item["supported"] = True
                item["support_basis"] = "cited_inference_from_expected_facts"
    unsupported = [item for item in support if not item["supported"]]
    unsupported_numbers = _unsupported_numbers(text, evidence)
    invalid_citations = list(citation_validation.get("invalid_citations") or [])
    hallucination_reasons = [
        *[f"unsupported claim: {item['claim']}" for item in unsupported],
        *[f"unsupported number: {item}" for item in unsupported_numbers],
        *[f"invalid citation rank: {rank}" for rank in invalid_citations],
    ]

    answer_claim_count = len(claims)
    supported_claim_count = len(support) - len(unsupported)
    evidence_score = (
        supported_claim_count / answer_claim_count if answer_claim_count else 0.0
    )
    evidence_passed = (
        bool(answer_claim_count)
        and not unsupported
        and not unsupported_numbers
        and not invalid_citations
    )
    if refusal:
        evidence_passed = False
        evidence_score = 0.0
        hallucination_reasons = []

    return {
        "answer_in_retrieved_chunks": {
            "passed": evidence_passed,
            "score": round(evidence_score, 4),
            "claim_count": answer_claim_count,
            "supported_claim_count": supported_claim_count,
            "unsupported_claims": [item["claim"] for item in unsupported],
            "claim_details": support,
            "citation_validation": citation_validation,
        },
        "hallucination": {
            "detected": bool(hallucination_reasons) and not refusal,
            "hallucination_free": not bool(hallucination_reasons) or refusal,
            "unsupported_claims": [item["claim"] for item in unsupported],
            "unsupported_numbers": unsupported_numbers,
            "invalid_citations": invalid_citations,
            "reasons": hallucination_reasons,
        },
        "expected_fact_coverage": {
            "answer_matches": answer_fact_matches,
            "evidence_matches": evidence_fact_matches,
            "answer_score": round(len(answer_fact_matches) / len(expected), 4) if expected else None,
            "evidence_score": round(len(evidence_fact_matches) / len(expected), 4) if expected else None,
        },
        "refused": refusal,
    }


def _extract_claims(answer: str) -> list[dict[str, Any]]:
    claims = []
    for segment in re.split(r"[\r\n]+|(?<=[。！？；])", str(answer or "")):
        raw = str(segment or "").strip()
        if not raw:
            continue
        if re.fullmatch(
            r"(?:來源|資料來源|引用|參考資料)\s*[：:]?\s*(?:\[\d+\]\s*[,，、]?\s*)+",
            raw,
        ):
            continue
        clean = re.sub(r"^\s*[-•*#]+\s*", "", raw).strip()
        if re.fullmatch(r"[|\-: ]+", clean):
            continue
        if re.fullmatch(r"\*{1,3}.+\*{1,3}", raw):
            continue
        clean = re.sub(r"[*_`]+", "", clean).strip()
        clean = re.sub(r"\[(\d+)\]", "", clean).strip()
        if not clean or len(clean) < 3:
            continue
        if clean in {"回答：", "答案：", "依據：", "參考資料："}:
            continue
        if _META_CLAIM_PATTERN.fullmatch(clean):
            continue
        if re.search(r"無法確認|資料不足|無法判斷|未能確認", clean):
            # An explicit uncertainty statement is a safe abstention, not an
            # unsupported factual claim to count as hallucination.
            continue
        claims.append({
            "claim": clean[:800],
            "citation_ranks": [int(rank) for rank in re.findall(r"\[(\d+)\]", raw)],
        })
    return claims[:40]


def _claim_support(
    claims: Sequence[dict[str, Any]],
    contexts: Sequence[dict[str, Any]],
    embedding_fn: Optional[Callable[..., Sequence[Sequence[float]]]],
    semantic_threshold: float,
) -> list[dict[str, Any]]:
    if not claims:
        return []
    context_texts = [
        f"{context.get('title', '')}\n{context.get('content', '')}"
        for context in contexts
    ]
    vectors = None
    if embedding_fn and context_texts:
        try:
            texts = [item["claim"] for item in claims] + context_texts
            try:
                vectors = embedding_fn(texts)
            except TypeError:
                vectors = embedding_fn(texts, model_name=None)
        except Exception:
            vectors = None

    context_tokens = [set(tokenize_bm25(text)) for text in context_texts]
    results = []
    for claim_index, item in enumerate(claims):
        claim = item["claim"]
        claim_tokens = set(tokenize_bm25(claim))
        lexical_scores = [
            len(claim_tokens & tokens) / max(1, len(claim_tokens))
            for tokens in context_tokens
        ]
        semantic_scores = []
        if vectors is not None:
            claim_vector = vectors[claim_index]
            for context_vector in vectors[len(claims):]:
                semantic_scores.append(_cosine(claim_vector, context_vector))
        max_lexical = max(lexical_scores, default=0.0)
        max_semantic = max(semantic_scores, default=0.0)
        supported = bool(
            _contains_normalized("\n".join(context_texts), claim)
            or max_lexical >= 0.12
            or max_semantic >= float(semantic_threshold)
        )
        results.append({
            "claim": claim,
            "citation_ranks": list(item.get("citation_ranks") or []),
            "supported": supported,
            "max_lexical_overlap": round(max_lexical, 4),
            "max_semantic_similarity": round(max_semantic, 4),
        })
    return results


def _unsupported_numbers(answer: str, contexts: Sequence[dict[str, Any]]) -> list[str]:
    evidence_text = "\n".join(str(context.get("content") or "") for context in contexts)
    evidence_number_keys = {
        _normalize_number_token(token)
        for token in _NUMBER_PATTERN.findall(evidence_text)
    }
    unsupported = []
    answer_without_citations = re.sub(r"\[\d+\]", "", str(answer or ""))
    for token in _NUMBER_PATTERN.findall(answer_without_citations):
        clean = re.sub(r"\s+", "", token)
        if clean and _normalize_number_token(clean) not in evidence_number_keys:
            unsupported.append(clean)
    return list(dict.fromkeys(unsupported))


def _is_evidence_inference(claim: str) -> bool:
    clean = str(claim or "").strip()
    return bool(re.search(
        r"^(?:相同|一樣|一致|兩者|二者|兩版|兩個版本|因此|所以|綜上|總結|可見)|"
        r"(?:未見|未保留|沒有|並無|不具|不存在|差異|異同|相較)",
        clean,
    ))


def _normalize_number_token(token: str) -> tuple[int, str]:
    clean = re.sub(r"\s+", "", str(token or ""))
    unit_match = re.search(r"(日|天|年|月|小時|分鐘|秒|%|分之一)$", clean)
    unit = unit_match.group(1) if unit_match else ""
    number = clean[: -len(unit)] if unit else clean
    if number.isdigit():
        return int(number), unit
    if not number or not all(character in _CHINESE_DIGITS or character in _CHINESE_UNITS for character in number):
        return -1, unit
    if not any(character in _CHINESE_UNITS for character in number):
        return int("".join(str(_CHINESE_DIGITS[character]) for character in number)), unit
    total = 0
    section = 0
    current = 0
    for character in number:
        if character in _CHINESE_DIGITS:
            current = _CHINESE_DIGITS[character]
            continue
        multiplier = _CHINESE_UNITS[character]
        if multiplier >= 10_000:
            section += current
            total += section * multiplier
            section = 0
        else:
            section += (current or 1) * multiplier
        current = 0
    return total + section + current, unit


def _contains_normalized(haystack: str, needle: str) -> bool:
    return _normalize(needle) in _normalize(haystack)


def _normalize(value: object) -> str:
    return re.sub(r"[\s\u3000，。；：、！？（）()「」『』\[\],.!?]", "", str(value or "")).casefold()


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left or not right:
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = sum(float(value) * float(value) for value in left) ** 0.5
    right_norm = sum(float(value) * float(value) for value in right) ** 0.5
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
