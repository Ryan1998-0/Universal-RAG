"""Deterministic post-generation evidence validation."""

from __future__ import annotations

import re
from typing import Any, Sequence


REFUSAL_MARKERS = (
    "根據目前檢索資料無法確認",
    "目前檢索資料不足",
    "資料不足",
)


def validate_answer_evidence(
    answer: str,
    contexts: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Validate answer citations against server-generated contexts.

    This deterministic post-generation check complements the upstream
    retrieval Evidence Gate. It prevents fabricated citation ranks and returns
    structured diagnostics for the API/UI.
    """

    text = str(answer or "").strip()
    if any(marker in text for marker in REFUSAL_MARKERS):
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

    if not referenced:
        reason = "回答沒有提供來源標記。"
    elif invalid:
        reason = "回答包含不在本次檢索結果中的來源標記。"
    else:
        reason = "通過來源標記與證據範圍檢查。"

    return {
        "sufficient": bool(valid) and not invalid,
        "status": "passed" if bool(valid) and not invalid else "failed",
        "valid_citations": valid,
        "invalid_citations": invalid,
        "uncited_claims": _uncited_claims(text),
        "reason": reason,
    }


def _uncited_claims(text: str) -> list[str]:
    """Return likely factual lines without a citation for diagnostics."""

    claims = []
    for line in re.split(r"[\r\n]+|(?<=[。！？；])", text):
        clean = re.sub(r"^\s*[-•*#]+\s*", "", line).strip()
        if not clean or "來源：" in clean:
            continue
        if re.fullmatch(r"\[\d+\](?:\s*[,，、]\s*\[\d+\])*", clean):
            continue
        if "[" not in clean:
            claims.append(clean[:500])
    return claims[:20]
