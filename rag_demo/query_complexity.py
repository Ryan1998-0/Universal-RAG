"""Low-latency query complexity routing for retrieval reranking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class QueryComplexityDecision:
    label: str
    score: float
    reasons: tuple[str, ...]
    signals: dict

    @property
    def is_complex(self) -> bool:
        return self.label == "complex"

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "score": round(float(self.score), 3),
            "reasons": list(self.reasons),
            "signals": dict(self.signals),
        }


def classify_query_complexity(
    question: str,
    query_variants: Sequence[str] = (),
    sub_questions: Sequence[str] = (),
    threshold: float = 2.0,
) -> QueryComplexityDecision:
    """Classify a query before deciding whether expensive reranking is needed.

    This classifier intentionally uses only the current question and the
    already-built retrieval plan.  It adds no model call, memory lookup, or
    network latency.  A single factual question normally stays simple; multi-
    condition, comparative, multi-part, and acronym-definition questions are
    routed through the expensive reranker.
    """

    text = re.sub(r"\s+", " ", str(question or "")).strip()
    compact = re.sub(r"\s+", "", text).casefold()
    reasons: list[str] = []
    score = 0.0

    character_count = len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", text))
    if character_count >= 80:
        score += 2.0
        reasons.append("問題文字較長")
    elif character_count >= 45:
        score += 1.0
        reasons.append("問題文字偏長")

    clause_markers = (
        "以及", "並且", "同時", "分別", "另外", "此外", "還有",
        "且", "及", "與", "和", "或", "、", ",", "；", ";",
    )
    clause_hits = [marker for marker in clause_markers if marker in compact]
    if clause_hits:
        score += 1.0
        reasons.append("包含多條件或並列語句")

    complex_markers = (
        "比較", "差異", "不同", "各自", "為什麼", "為何", "原因",
        "如何影響", "如果", "條件", "例外", "流程", "步驟", "先", "再",
    )
    complex_hits = [marker for marker in complex_markers if marker in compact]
    if complex_hits:
        score += 1.0
        reasons.append("需要比較、因果、條件或流程推理")

    question_count = len(re.findall(r"[？?]", text))
    if question_count >= 2:
        score += 1.0
        reasons.append("包含多個問句")

    normalized_sub_questions = [
        str(value).strip() for value in (sub_questions or ()) if str(value).strip()
    ]
    if len(normalized_sub_questions) >= 2:
        score += 2.0
        reasons.append("檢索規劃拆出多個子問題")

    normalized_variants = [
        str(value).strip() for value in (query_variants or ()) if str(value).strip()
    ]
    if len(normalized_variants) >= 3:
        score += 1.0
        reasons.append("需要三個以上互補檢索查詢")

    # A domain acronym definition often needs precise evidence and benefits
    # from a cross-encoder/reranker even when the sentence is short.
    asks_definition = bool(
        re.search(r"(?:是什麼|什麼意思|代表什麼|定義|全名|meaning|define)", compact, re.I)
    )
    acronym_count = len(re.findall(r"\b[A-Z][A-Z0-9-]{2,12}\b", text))
    if asks_definition and acronym_count:
        score += 2.0
        reasons.append("專有縮寫定義需要精確比對")

    decision_threshold = max(1.0, float(threshold))
    is_complex = score >= decision_threshold
    if not reasons:
        reasons.append("單一、短句、無多條件的問題")
    return QueryComplexityDecision(
        label="complex" if is_complex else "simple",
        score=score,
        reasons=tuple(dict.fromkeys(reasons)),
        signals={
            "characterCount": character_count,
            "clauseMarkers": clause_hits,
            "complexMarkers": complex_hits,
            "questionCount": question_count,
            "subQuestionCount": len(normalized_sub_questions),
            "queryVariantCount": len(normalized_variants),
            "acronymCount": acronym_count,
            "threshold": decision_threshold,
        },
    )
