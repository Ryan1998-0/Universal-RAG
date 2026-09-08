"""Second-pass evidence focusing for noisy retrieved chunks.

The first retrieval pass is optimized for recall.  This module deliberately
uses a different objective for the small candidate set: preserve answer-bearing
sentences, favor lexical/phrase matches, and keep dense similarity as a
low-weight tie breaker.  The original contexts remain available in the raw
retrieval trace; only the focused snippets are sent to generation.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Sequence, Tuple

from rag_demo.config import RagConfig
from rag_demo.embeddings import embed_query, embed_texts
from rag_demo.hybrid_retrieval import tokenize_bm25
from rag_demo.retrieval_planner import extract_focus_terms


def focus_retrieved_evidence(
    question: str,
    contexts: Sequence[dict],
    settings: RagConfig | None = None,
    embed_query_fn: Callable[[str], Sequence[float]] = embed_query,
    embed_texts_fn: Callable[[Sequence[str]], Sequence[Sequence[float]]] = embed_texts,
) -> Tuple[List[dict], Dict[str, object]]:
    """Re-rank sentence-sized evidence units from the first-pass contexts."""

    settings = (settings or RagConfig.from_env()).normalized()
    units = _candidate_units(contexts, max_chars=settings.evidence_focus_max_chars)
    if not units:
        return [dict(context) for context in contexts], {
            "enabled": True,
            "candidateCount": 0,
            "selectedCount": len(contexts),
            "keywordWeight": settings.evidence_focus_keyword_weight,
            "embeddingWeight": settings.evidence_focus_embedding_weight,
            "fallback": True,
        }

    focus_terms = list(dict.fromkeys([*extract_focus_terms(question), *tokenize_bm25(question)]))
    scored: List[dict] = []
    query_vector = None
    dense_scores: List[float] = []
    try:
        query_vector = list(embed_query_fn(question))
        dense_vectors = embed_texts_fn([unit["content"] for unit in units])
        dense_scores = [_cosine(query_vector, vector) for vector in dense_vectors]
    except Exception:
        dense_scores = [0.0] * len(units)

    lexical_scores = [_lexical_focus_score(unit["content"], focus_terms) for unit in units]
    max_lexical = max(lexical_scores, default=1.0) or 1.0
    min_dense = min(dense_scores, default=0.0)
    max_dense = max(dense_scores, default=1.0)
    dense_span = max(max_dense - min_dense, 1e-9)
    for index, unit in enumerate(units):
        lexical = lexical_scores[index] / max_lexical
        dense = (dense_scores[index] - min_dense) / dense_span if dense_scores else 0.0
        score = (
            settings.evidence_focus_keyword_weight * lexical
            + settings.evidence_focus_embedding_weight * dense
        )
        scored.append(
            {
                **unit,
                "focus_score": score,
                "focus_keyword_score": lexical,
                "focus_embedding_score": dense,
            }
        )

    ranked = sorted(scored, key=lambda item: (-item["focus_score"], item["unit_index"]))
    selected = _deduplicate_units(ranked, limit=settings.evidence_focus_top_k)
    focused = []
    for rank, unit in enumerate(sorted(selected, key=lambda item: (-item["focus_score"], item["unit_index"])), start=1):
        focused.append(
            {
                "id": f"{unit['id']}::focus-{rank}",
                "rank": rank,
                "title": f"{unit['title']} / evidence focus",
                "source": unit["source"],
                "page": unit["page"],
                "content": unit["content"],
                "branch": "Evidence focus reranker",
                "score": round(float(unit["focus_score"]), 6),
                "bm25Score": round(float(unit["focus_keyword_score"]), 6),
                "embeddingScore": round(float(unit["focus_embedding_score"]), 6),
                "rerankScore": round(float(unit["focus_score"]), 6),
                "matchedTerms": [term for term in focus_terms if _contains(unit["content"], term)][:20],
                "originalRank": unit["original_rank"],
                "focus": True,
            }
        )
    return focused or [dict(context) for context in contexts], {
        "enabled": True,
        "candidateCount": len(units),
        "selectedCount": len(focused),
        "keywordWeight": settings.evidence_focus_keyword_weight,
        "embeddingWeight": settings.evidence_focus_embedding_weight,
        "fallback": not bool(focused),
    }


def _candidate_units(contexts: Sequence[dict], max_chars: int) -> List[dict]:
    units = []
    unit_index = 0
    for context in contexts:
        content = re.sub(r"\s+", " ", str(context.get("content") or "")).strip()
        if not content:
            continue
        pieces = [piece.strip() for piece in re.split(r"(?<=[。！？；;])\s*", content) if piece.strip()]
        fragments = []
        for piece in pieces:
            if len(piece) <= max_chars:
                fragments.append(piece)
            else:
                fragments.extend(
                    piece[start : start + max_chars].strip()
                    for start in range(0, len(piece), max_chars)
                )
        # Adjacent legal sentences often carry complementary facts, such as a
        # limit followed by its pay rule. Keep a two-sentence window as a
        # candidate without returning the entire noisy chunk.
        windows = list(fragments)
        for index in range(len(fragments) - 1):
            combined = f"{fragments[index]} {fragments[index + 1]}".strip()
            if len(combined) <= max_chars:
                windows.append(combined)
        for fragment in windows:
            if len(fragment) < 8:
                continue
            units.append(
                {
                    "id": str(context.get("id") or "context"),
                    "title": str(context.get("title") or "Untitled"),
                    "source": str(context.get("source") or ""),
                    "page": str(context.get("page") or ""),
                    "content": fragment,
                    "original_rank": int(context.get("rank") or 0),
                    "unit_index": unit_index,
                }
            )
            unit_index += 1
    return units


def _lexical_focus_score(content: str, terms: Sequence[str]) -> float:
    compact_content = _compact(content)
    score = 0.0
    for term in terms:
        compact_term = _compact(term)
        if compact_term and compact_term in compact_content:
            score += min(8.0, max(1.0, len(compact_term)))
    if re.search(r"(?:第\s*[0-9０-９一二三四五六七八九十]+\s*條|不得|應|得|合計|上限|期間)", content):
        score += 1.5
    if re.search(r"[零〇一二兩三四五六七八九十百千萬億\d]+\s*(?:日|天|年|月|小時|%|％)", content):
        score += 2.0
    if re.search(r"(?:事假|病假|傷病假|喪假|婚假)", content) and re.search(
        r"(?:不給工資|工資照給|給薪|全勤獎金|依比例)", content
    ):
        score += 3.0
    if "婚假" in content and "事假" in content:
        score -= 2.0
    return score


def _deduplicate_units(ranked: Sequence[dict], limit: int) -> List[dict]:
    selected = []
    for unit in ranked:
        compact = _compact(unit["content"])
        if any(compact in _compact(other["content"]) or _overlap_ratio(compact, _compact(other["content"])) >= 0.8 for other in selected):
            continue
        selected.append(unit)
        if len(selected) >= max(1, int(limit)):
            break
    return selected


def _overlap_ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    shorter, longer = sorted((left, right), key=len)
    return len(shorter) / len(longer) if shorter in longer else 0.0


def _contains(content: str, term: str) -> bool:
    return bool(term) and _compact(term) in _compact(content)


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).lower()


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = sum(float(value) ** 2 for value in left) ** 0.5
    right_norm = sum(float(value) ** 2 for value in right) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)
