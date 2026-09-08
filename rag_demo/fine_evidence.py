"""Fine-grained semantic evidence retrieval and grouping.

This is an optional second retrieval pass for noisy parent chunks.  It does
not call a generative model: parent chunks are split into sentence-sized
units, the units are embedded again, and only semantically and lexically
supported units are grouped into small evidence packets.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Sequence, Tuple

from rag_demo.config import RagConfig
from rag_demo.embeddings import embed_query, embed_texts
from rag_demo.hybrid_retrieval import tokenize_bm25
from rag_demo.retrieval_planner import extract_focus_terms


def retrieve_fine_evidence(
    question: str,
    contexts: Sequence[dict],
    settings: RagConfig | None = None,
    embed_query_fn: Callable[[str], Sequence[float]] = embed_query,
    embed_texts_fn: Callable[[Sequence[str]], Sequence[Sequence[float]]] = embed_texts,
    chunk_fraction: float | None = None,
) -> Tuple[List[dict], Dict[str, object]]:
    """Split, re-embed, score, and group answer-bearing evidence units.

    ``chunk_fraction`` is used by the two-pass pipeline to size each second
    pass unit relative to the parent chunk that was returned by pass one.  A
    ``None`` value preserves the fixed ``fine_evidence_chunk_chars`` behavior
    used by older callers and tests.
    """

    settings = (settings or RagConfig.from_env()).normalized()
    units = _fine_units(
        contexts,
        max_chars=settings.fine_evidence_chunk_chars,
        overlap_chars=settings.fine_evidence_overlap_chars,
        chunk_fraction=chunk_fraction,
    )
    base_trace = {
        "enabled": True,
        "mode": "fine-embedding-group",
        "candidateCount": len(units),
        "selectedUnitCount": 0,
        "groupCount": 0,
        "chunkChars": settings.fine_evidence_chunk_chars,
        "chunkFraction": chunk_fraction,
        "overlapChars": settings.fine_evidence_overlap_chars,
        "keywordWeight": settings.fine_evidence_keyword_weight,
        "embeddingWeight": settings.fine_evidence_embedding_weight,
        "minEmbeddingScore": settings.fine_evidence_min_embedding_score,
        "fallback": False,
    }
    if not units:
        return [dict(context) for context in contexts], {
            **base_trace,
            "groupCount": len(contexts),
            "fallback": True,
        }

    focus_terms = list(
        dict.fromkeys([*extract_focus_terms(question), *tokenize_bm25(question)])
    )
    query_tokens = set(tokenize_bm25(question))
    lexical_scores = [
        _lexical_score(unit["content"], focus_terms, query_tokens)
        for unit in units
    ]
    dense_scores = _dense_scores(question, units, embed_query_fn, embed_texts_fn)

    scored = []
    for index, unit in enumerate(units):
        lexical = lexical_scores[index]
        dense = dense_scores[index]
        score = (
            settings.fine_evidence_keyword_weight * lexical
            + settings.fine_evidence_embedding_weight * dense
        )
        scored.append(
            {
                **unit,
                "fine_score": score,
                "fine_keyword_score": lexical,
                "fine_embedding_score": dense,
            }
        )

    ranked = sorted(
        scored,
        key=lambda item: (-item["fine_score"], item["original_rank"], item["unit_index"]),
    )
    selected = _select_units(
        ranked,
        top_k=settings.fine_evidence_top_k,
        min_embedding_score=settings.fine_evidence_min_embedding_score,
    )
    groups = _group_units(selected, max_groups=settings.fine_evidence_max_groups)
    focused = _render_groups(groups)
    if not focused:
        # Keep a safe fallback rather than turning a retrieval hit into an
        # empty context when a very strict threshold rejects every unit.
        fallback = _render_groups(
            _group_units(ranked[:1], max_groups=1)
        )
        focused = fallback or [dict(context) for context in contexts]
        base_trace["fallback"] = True

    base_trace.update(
        {
            "selectedUnitCount": len(selected),
            "groupCount": len(focused),
        }
    )
    return focused, base_trace


def _fine_units(
    contexts: Sequence[dict],
    max_chars: int,
    overlap_chars: int,
    chunk_fraction: float | None = None,
) -> List[dict]:
    units: List[dict] = []
    unit_index = 0
    for context_index, context in enumerate(contexts):
        content = re.sub(r"\s+", " ", str(context.get("content") or "")).strip()
        if not content:
            continue
        context_max_chars = max_chars
        context_overlap_chars = overlap_chars
        if chunk_fraction is not None:
            fraction = min(1.0, max(0.05, float(chunk_fraction)))
            context_max_chars = max(80, round(len(content) * fraction))
            context_overlap_chars = min(
                context_overlap_chars,
                max(0, context_max_chars // 4),
            )
        article_key = ""
        pieces = _sentences(content)
        for sentence_index, sentence in enumerate(pieces):
            article_match = re.search(
                r"第\s*[0-9０-９一二三四五六七八九十百千]+(?:[-－][0-9０-９]+)?\s*[條条章節節]",
                sentence,
            )
            if article_match:
                article_key = article_match.group(0).replace(" ", "")
            windows = _split_sentence(
                sentence,
                context_max_chars,
                context_overlap_chars,
            )
            for window_index, window in enumerate(windows):
                if len(window) < 8:
                    continue
                units.append(
                    {
                        "id": str(context.get("id") or f"context-{context_index + 1}"),
                        "title": str(context.get("title") or "Untitled"),
                        "source": str(context.get("source") or ""),
                        "page": str(context.get("page") or ""),
                        "content": window,
                        "original_rank": int(context.get("rank") or context_index + 1),
                        "context_index": context_index,
                        "sentence_index": sentence_index,
                        "window_index": window_index,
                        "article_key": article_key,
                        "unit_index": unit_index,
                    }
                )
                unit_index += 1
    return units


def _sentences(content: str) -> List[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[。！？；;])\s*", content)
        if sentence.strip()
    ]


def _split_sentence(sentence: str, max_chars: int, overlap_chars: int) -> List[str]:
    if len(sentence) <= max_chars:
        return [sentence]
    step = max(1, max_chars - min(overlap_chars, max_chars - 1))
    pieces = []
    for start in range(0, len(sentence), step):
        piece = sentence[start : start + max_chars].strip()
        if piece:
            pieces.append(piece)
        if start + max_chars >= len(sentence):
            break
    return pieces


def _lexical_score(content: str, focus_terms: Sequence[str], query_tokens: set[str]) -> float:
    compact_content = _compact(content)
    content_tokens = set(tokenize_bm25(content))
    if query_tokens:
        token_coverage = len(query_tokens & content_tokens) / len(query_tokens)
    else:
        token_coverage = 0.0
    phrase_hits = sum(
        1
        for term in focus_terms
        if len(_compact(term)) >= 2 and _compact(term) in compact_content
    )
    score = min(1.0, token_coverage * 1.4) + min(0.35, phrase_hits * 0.08)
    if re.search(r"(?:不得|應|得|合計|上限|期間|施行|照給|不給)", content):
        score += 0.12
    if re.search(r"[零〇一二兩三四五六七八九十百千萬億\d]+\s*(?:日|天|年|月|小時|%|％)", content):
        score += 0.15
    return min(1.0, score)


def _dense_scores(
    question: str,
    units: Sequence[dict],
    embed_query_fn: Callable[[str], Sequence[float]],
    embed_texts_fn: Callable[[Sequence[str]], Sequence[Sequence[float]]],
) -> List[float]:
    try:
        query_vector = list(embed_query_fn(question))
        vectors = embed_texts_fn([unit["content"] for unit in units])
        return [_cosine(query_vector, vector) for vector in vectors]
    except Exception:
        return [0.0] * len(units)


def _select_units(
    ranked: Sequence[dict],
    top_k: int,
    min_embedding_score: float,
) -> List[dict]:
    selected = []
    for unit in ranked:
        lexical = float(unit["fine_keyword_score"])
        dense = float(unit["fine_embedding_score"])
        # A dense hit alone is not enough: at least some lexical coverage is
        # required unless it is the strongest candidate. This reduces the
        # common multilingual-embedding false positive problem.
        if selected and dense < min_embedding_score and lexical < 0.35:
            continue
        if selected and lexical < 0.12 and dense < min_embedding_score + 0.08:
            continue
        if any(
            _compact(unit["content"]) in _compact(other["content"])
            or _overlap_ratio(_compact(unit["content"]), _compact(other["content"])) >= 0.85
            for other in selected
        ):
            continue
        selected.append(unit)
        if len(selected) >= max(1, int(top_k)):
            break
    return selected


def _group_units(units: Sequence[dict], max_groups: int) -> List[List[dict]]:
    ordered = sorted(units, key=lambda item: (item["context_index"], item["sentence_index"], item["window_index"]))
    groups: List[List[dict]] = []
    for unit in ordered:
        if groups:
            previous = groups[-1][-1]
            same_section = (
                unit["context_index"] == previous["context_index"]
                and unit["article_key"] == previous["article_key"]
                and unit["sentence_index"] <= previous["sentence_index"] + 1
            )
            if same_section:
                groups[-1].append(unit)
                continue
        groups.append([unit])
    groups.sort(
        key=lambda group: (
            -max(float(unit["fine_score"]) for unit in group),
            min(int(unit["original_rank"]) for unit in group),
            min(int(unit["unit_index"]) for unit in group),
        )
    )
    return groups[: max(1, int(max_groups))]


def _render_groups(groups: Sequence[Sequence[dict]]) -> List[dict]:
    rendered = []
    for rank, group in enumerate(groups, start=1):
        first = group[0]
        content = " ".join(dict.fromkeys(unit["content"] for unit in group)).strip()
        if not content:
            continue
        rendered.append(
            {
                "id": f"{first['id']}::fine-evidence-{rank}",
                "rank": rank,
                "title": f"{first['title']} / fine evidence",
                "source": first["source"],
                "page": first["page"],
                "content": content,
                "branch": "Fine-grained embedding evidence",
                "score": round(max(float(unit["fine_score"]) for unit in group), 6),
                "bm25Score": round(max(float(unit["fine_keyword_score"]) for unit in group), 6),
                "embeddingScore": round(max(float(unit["fine_embedding_score"]) for unit in group), 6),
                "rerankScore": round(max(float(unit["fine_score"]) for unit in group), 6),
                "matchedTerms": list(
                    dict.fromkeys(
                        term
                        for unit in group
                        for term in tokenize_bm25(unit["content"])
                    )
                )[:20],
                "originalRank": first["original_rank"],
                "fineUnitCount": len(group),
                "fineArticleKey": first["article_key"],
                "fineGrouped": len(group) > 1,
                "focus": True,
            }
        )
    return rendered


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).lower()


def _overlap_ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    shorter, longer = sorted((left, right), key=len)
    return len(shorter) / len(longer) if shorter in longer else 0.0


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = sum(float(value) ** 2 for value in left) ** 0.5
    right_norm = sum(float(value) ** 2 for value in right) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)
