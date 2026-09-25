"""Deterministic post-generation evidence validation."""

from __future__ import annotations

import re
from decimal import Decimal
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
_REFUSAL_SOURCE_FOOTER = re.compile(
    r"\s+(?:來源|資料來源|引用|參考資料)\s*[：:]\s*"
    r"(?:\[\d+\]\s*[,，、]?\s*)+$"
)
_NUMBER_PATTERN = re.compile(
    r"(?:\d+(?:\.\d+)?|[零〇一二兩三四五六七八九十百千萬億]+?)\s*"
    r"(?:萬元|日|天|年|月|小時|分鐘|秒|%|％|元|分之一)"
)
_CHINESE_DIGITS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "兩": 2, "三": 3,
    "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_CHINESE_UNITS = {"十": 10, "百": 100, "千": 1000, "萬": 10000, "億": 100000000}
_MINIMUM_CLAIM_OVERLAP = 0.25
_POLARITY_CLAUSE_SPLIT = re.compile(
    r"[。！？；，、,\r\n]+|(?<=[.!?;])\s+|"
    r"但是|然而|不過|但|惟|(?i:\b(?:but|however|except)\b)"
)
_NEGATIVE_PERMISSION = re.compile(
    r"(?i:\b(?:may\s+not|may\s+be\s+(?:prohibited|forbidden)|"
    r"can\s+not|cannot|can't|must\s+not|shall\s+not|"
    r"not\s+(?:be\s+)?allowed|not\s+(?:be\s+)?permitted|"
    r"prohibited|forbidden|prohibits?|forbids?)\b)"
    r"|不得以|不可以|不允許|不准許|不得|不能|不可|不准|禁止|無權"
)
_POSITIVE_PERMISSION = re.compile(
    r"(?i:\b(?:may|can|allowed|permitted|authorized|authorised|entitled)\b)"
    r"|可以|准許|允許|准予|有權|可"
)
_MINIMUM_POLARITY_TAIL_OVERLAP = 0.60


def validate_answer_evidence(
    answer: str,
    contexts: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Validate answer citations against server-generated contexts.

    This deterministic post-generation check complements the upstream
    retrieval Evidence Gate. It rejects mixed refusals, fabricated citation
    ranks and factual sentences without their own citation. It also checks
    numerical details, minimum lexical support, and clear permission-polarity
    conflicts against the cited passages. These deterministic checks do not
    prove semantic entailment.
    """

    text = str(answer or "").strip()
    refusal_text = normalized_refusal_answer(text)
    if refusal_text is not None:
        return {
            "sufficient": True,
            "status": "refused",
            "valid_citations": [],
            "invalid_citations": [],
            "uncited_claims": [],
            "unsupported_claims": [],
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
    unsupported_claims = _unsupported_claims(
        claim_segments, footer_citations, context_by_rank
    )
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
    elif unsupported_claims:
        reason = "回答的主張與所引用片段缺少可檢查的直接支持。"
    else:
        reason = "通過逐句來源、數值與詞彙支持檢查。"

    sufficient = bool(claim_segments) and bool(valid) and not invalid and not uncited_claims and not unsupported_claims and not any(
        marker in text for marker in REFUSAL_MARKERS
    )

    return {
        "sufficient": sufficient,
        "status": "passed" if sufficient else "failed",
        "valid_citations": valid,
        "invalid_citations": invalid,
        "uncited_claims": uncited_claims,
        "unsupported_claims": unsupported_claims,
        "reason": reason,
    }


def normalized_refusal_answer(text: str) -> str | None:
    """Accept a refusal plus one missing-evidence explanation, without citations."""

    clean = _REFUSAL_SOURCE_FOOTER.sub("", str(text or "").strip()).strip()
    for marker in REFUSAL_MARKERS:
        match = re.fullmatch(
            re.escape(marker)
            + r"[。.!！]?"
            + r"(?:\s*(缺少(?:的(?:是|證據|資料|資訊))?[：:]?\s*[^。！？；\r\n]{1,300}[。.!！]?))?",
            clean,
        )
        if match:
            return f"{marker}。" + (match.group(1) or "")
    return None


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


def _unsupported_claims(
    segments: list[str],
    footer_citations: list[str],
    context_by_rank: dict[int, dict[str, Any]],
) -> list[str]:
    unsupported = []
    for segment in segments:
        ranks = [int(rank) for rank in re.findall(r"\[(\d+)\]", segment)]
        if not ranks and len(segments) == 1:
            ranks = [int(rank[1:-1]) for rank in footer_citations]
        if not ranks:
            continue  # Reported separately as an uncited claim.
        cited_text = "\n".join(
            str(context_by_rank[rank].get("content") or "")
            for rank in ranks if rank in context_by_rank
        )
        claim = re.sub(r"\[\d+\]", "", segment).strip()
        if (
            not cited_text
            or not _has_minimum_evidence_overlap(claim, cited_text)
            or _has_clear_polarity_conflict(claim, cited_text)
        ):
            unsupported.append(claim[:500])
    return unsupported[:20]


def _has_minimum_evidence_overlap(claim: str, cited_text: str) -> bool:
    claim_numbers = {_normalize_number_token(value) for value in _NUMBER_PATTERN.findall(claim)}
    source_numbers = {_normalize_number_token(value) for value in _NUMBER_PATTERN.findall(cited_text)}
    if not claim_numbers.issubset(source_numbers):
        return False
    normalized_claim = _normalize_text(claim)
    normalized_source = _normalize_text(cited_text)
    if normalized_claim and normalized_claim in normalized_source:
        return True
    claim_grams = _character_grams(normalized_claim)
    source_grams = _character_grams(normalized_source)
    return bool(claim_grams) and len(claim_grams & source_grams) / len(claim_grams) >= _MINIMUM_CLAIM_OVERLAP


def _has_clear_polarity_conflict(claim: str, cited_text: str) -> bool:
    """Reject a clearly matching permission claim with the opposite source polarity.

    This catches direct allow/deny inversions; it is not a general semantic
    entailment check. Clause splitting keeps separate permission and limit
    statements from being compared as if they described the same action.
    """

    claim_phrases = _permission_phrases(claim)
    source_phrases = _permission_phrases(cited_text)
    for claim_polarity, claim_tail in claim_phrases:
        claim_grams = _character_grams(claim_tail)
        if not claim_grams:
            continue
        matching_polarities = set()
        for source_polarity, source_tail in source_phrases:
            source_grams = _character_grams(source_tail)
            if not source_grams:
                continue
            overlap = len(claim_grams & source_grams) / len(claim_grams)
            if overlap >= _MINIMUM_POLARITY_TAIL_OVERLAP:
                matching_polarities.add(source_polarity)
        opposite = "negative" if claim_polarity == "positive" else "positive"
        if opposite in matching_polarities and claim_polarity not in matching_polarities:
            return True
    return False


def _permission_phrases(text: str) -> list[tuple[str, str]]:
    phrases = []
    for clause in _POLARITY_CLAUSE_SPLIT.split(str(text or "")):
        if not clause.strip():
            continue
        masked = list(clause)
        for match in _NEGATIVE_PERMISSION.finditer(clause):
            tail = _normalize_text(clause[match.end():])
            if tail and _character_grams(tail):
                phrases.append(("negative", tail))
            masked[match.start():match.end()] = " " * (match.end() - match.start())
        positive_source = "".join(masked)
        for match in _POSITIVE_PERMISSION.finditer(positive_source):
            tail = _normalize_text(clause[match.end():])
            if tail and _character_grams(tail):
                phrases.append(("positive", tail))
    return phrases


def _normalize_text(value: str) -> str:
    return re.sub(r"[^\w]+", "", value).casefold()


def _character_grams(text: str) -> set[str]:
    return {
        text[index:index + size]
        for size in (2, 3)
        for index in range(max(0, len(text) - size + 1))
    }


def _normalize_number_token(token: str) -> tuple[str, str]:
    clean = re.sub(r"\s+", "", token)
    unit_match = re.search(r"(萬元|日|天|年|月|小時|分鐘|秒|%|％|元|分之一)$", clean)
    unit = unit_match.group(1) if unit_match else ""
    number = clean[:-len(unit)] if unit else clean
    if re.fullmatch(r"\d+(?:\.\d+)?", number):
        value = Decimal(number)
        if unit == "萬元":
            value *= 10_000
        return format(value.normalize(), "f"), "元" if unit == "萬元" else unit.replace("％", "%")
    if not number or not all(char in _CHINESE_DIGITS or char in _CHINESE_UNITS for char in number):
        return clean, unit
    if not any(char in _CHINESE_UNITS for char in number):
        value = int("".join(str(_CHINESE_DIGITS[char]) for char in number))
        return str(value * (10_000 if unit == "萬元" else 1)), "元" if unit == "萬元" else unit
    total = section = current = 0
    for char in number:
        if char in _CHINESE_DIGITS:
            current = _CHINESE_DIGITS[char]
        else:
            multiplier = _CHINESE_UNITS[char]
            if multiplier >= 10_000:
                total += (section + current) * multiplier
                section = current = 0
            else:
                section += (current or 1) * multiplier
                current = 0
    value = total + section + current
    return str(value * (10_000 if unit == "萬元" else 1)), "元" if unit == "萬元" else unit
