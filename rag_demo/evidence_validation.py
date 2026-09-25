"""Deterministic post-generation evidence validation."""

from __future__ import annotations

import re
from typing import Any, Sequence


REFUSAL_MARKERS = (
    "根據目前檢索資料無法確認",
    "目前檢索資料不足",
    "資料不足",
)
_SOURCE_LINE = re.compile(
    r"(?:來源|資料來源|引用|參考資料)\s*[：:]\s*"
    r"(?:\[\d+\]\s*[,，、]?\s*)+"
)
_CITATIONS_AFTER_PUNCTUATION = re.compile(
    r"([。！？；])\s*((?:\[\d+\]\s*)+)"
)


def validate_answer_evidence(
    answer: str,
    contexts: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Validate answer citations against server-generated contexts.

    This deterministic post-generation check complements the upstream
    retrieval Evidence Gate. It rejects mixed refusals, fabricated citation
    ranks and factual sentences without their own citation. Citation presence
    does not by itself prove semantic support from the cited passage.
    """

    text = str(answer or "").strip()
    if any(re.fullmatch(re.escape(marker) + r"[。.!！]?", text) for marker in REFUSAL_MARKERS):
        return {
            "sufficient": True,
            "status": "refused",
            "valid_citations": [],
            "invalid_citations": [],
            "uncited_claims": [],
            "reason": "模型依證據不足規則拒答。",
        }

    context_by_rank = {
        int(context.get("rank") or index): context
        for index, context in enumerate(contexts, start=1)
        if isinstance(context, dict) and str(context.get("content") or "").strip()
    }
    referenced = list(
        dict.fromkeys(int(rank) for rank in re.findall(r"\[(\d+)\]", text))
    )
    valid = [rank for rank in referenced if rank in context_by_rank]
    invalid = [rank for rank in referenced if rank not in context_by_rank]

    claim_segments, footer_citations = _answer_segments(text)
    uncited_claims = _uncited_claims(claim_segments, footer_citations)
    if any(marker in text for marker in REFUSAL_MARKERS):
        reason = "拒答語句混有其他內容；只接受完整拒答。"
    elif not claim_segments:
        reason = "回答沒有可驗證的內容。"
    elif not referenced:
        reason = "回答沒有提供來源標記。"
    elif invalid:
        reason = "回答包含不在本次檢索結果中的來源標記。"
    elif uncited_claims:
        reason = "回答有未逐句引用的主張。"
    else:
        reason = "通過逐句來源標記與證據範圍檢查。"

    sufficient = bool(claim_segments) and bool(valid) and not invalid and not uncited_claims and not any(
        marker in text for marker in REFUSAL_MARKERS
    )

    return {
        "sufficient": sufficient,
        "status": "passed" if sufficient else "failed",
        "valid_citations": valid,
        "invalid_citations": invalid,
        "uncited_claims": uncited_claims,
        "reason": reason,
    }


def _answer_segments(text: str) -> tuple[list[str], list[str]]:
    """Separate answer sentences from a source footer."""
    normalized = _CITATIONS_AFTER_PUNCTUATION.sub(r"\2\1", text)
    segments = []
    footer_citations = []
    for line in re.split(r"[\r\n]+|(?<=[。！？；])", normalized):
        clean = re.sub(r"^\s*[-•*#]+\s*", "", line).strip()
        if not clean:
            continue
        if _SOURCE_LINE.fullmatch(clean):
            footer_citations.extend(re.findall(r"\[\d+\]", clean))
            continue
        if re.fullmatch(r"\[\d+\](?:\s*[,，、]\s*\[\d+\])*", clean):
            footer_citations.extend(re.findall(r"\[\d+\]", clean))
            continue
        if clean in {"回答：", "答案：", "結論：", "依據："}:
            continue
        segments.append(clean)
    return segments, footer_citations


def _uncited_claims(segments: list[str], footer_citations: list[str]) -> list[str]:
    """Return factual sentences lacking a local citation for diagnostics."""

    # Keep the common one-sentence answer + source footer format, while a
    # footer cannot excuse missing inline citations across multiple claims.
    if len(segments) == 1 and footer_citations:
        return []

    claims = []
    for clean in segments:
        if not re.search(r"\[\d+\]", clean):
            claims.append(clean[:500])
    return claims[:20]
