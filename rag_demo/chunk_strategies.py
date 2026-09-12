"""Deterministic chunking strategies used by every ingestion path.

The project exposes two benchmarkable strategies:

``hard``
    Fixed character windows with no boundary correction and no overlap.

``dynamic``
    Sentence-aware windows. Sentences are packed up to a token budget and
    complete trailing sentences are carried into the next window up to the
    requested token overlap.

The token counter is deliberately dependency-free. It is a stable
multilingual approximation (CJK characters and punctuation count as one
token; Latin/digit runs count as one token) rather than a claim that every
embedding or answer model uses the same tokenizer.
"""

from __future__ import annotations

import re
from typing import Iterator, List


CHUNK_STRATEGY_BOUNDARY = "boundary"
CHUNK_STRATEGY_HARD = "hard"
CHUNK_STRATEGY_DYNAMIC = "dynamic"

_VALID_STRATEGIES = {
    CHUNK_STRATEGY_BOUNDARY,
    CHUNK_STRATEGY_HARD,
    CHUNK_STRATEGY_DYNAMIC,
}

# Keep sentence-ending punctuation in the preceding unit. Newlines are split
# only when they separate non-empty lines, so legal/article line wrapping does
# not manufacture a sentence boundary on every physical line.
_SENTENCE_BOUNDARY_RE = re.compile(
    r"(?<=[。！？!?；;])\s*|(?<=[.!?])\s+(?=[A-Z0-9\u3400-\u9fff])"
)
_TOKEN_RE = re.compile(
    r"[\u3400-\u9fff]|[A-Za-z]+(?:['’][A-Za-z]+)?|\d+(?:[.,:/-]\d+)*|[^\s]"
)


def normalize_chunk_strategy(value: object, default: str = CHUNK_STRATEGY_DYNAMIC) -> str:
    candidate = str(value or "").strip().lower()
    return candidate if candidate in _VALID_STRATEGIES else default


def estimate_token_count(text: str) -> int:
    """Return a stable approximation used for dynamic chunk budgets."""

    return len(_TOKEN_RE.findall(str(text or "")))


def split_sentence_units(text: str) -> List[str]:
    """Split text at sentence boundaries while preserving punctuation."""

    normalized = str(text or "").strip()
    if not normalized:
        return []
    units: List[str] = []
    for paragraph in re.split(r"\n\s*\n+", normalized):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        pieces = _SENTENCE_BOUNDARY_RE.split(paragraph)
        for piece in pieces:
            piece = re.sub(r"\s+", " ", piece).strip()
            if piece:
                units.append(piece)
    return units


def split_text(
    text: str,
    chunk_size: int,
    chunk_stride: int,
    strategy: str = CHUNK_STRATEGY_BOUNDARY,
    overlap_tokens: int = 200,
) -> Iterator[str]:
    """Yield chunks according to the requested strategy.

    ``chunk_size`` is a character budget for ``boundary`` and ``hard``. For
    ``dynamic`` it is the approximate token budget. ``chunk_stride`` remains
    part of the API for compatibility; dynamic and hard modes intentionally do
    not use it because their overlap policies are explicit.
    """

    clean_text = str(text or "").strip()
    size = max(1, int(chunk_size or 1))
    if not clean_text:
        return
    selected = normalize_chunk_strategy(strategy, default=CHUNK_STRATEGY_BOUNDARY)
    if selected == CHUNK_STRATEGY_HARD:
        yield from _split_hard(clean_text, size)
        return
    if selected == CHUNK_STRATEGY_DYNAMIC:
        yield from _split_dynamic(
            clean_text,
            max_tokens=size,
            overlap_tokens=max(0, int(overlap_tokens or 0)),
        )
        return
    yield from _split_boundary(clean_text, size, max(1, min(int(chunk_stride or size), size)))


def _split_hard(text: str, chunk_size: int) -> Iterator[str]:
    for start in range(0, len(text), chunk_size):
        piece = text[start : start + chunk_size].strip()
        if piece:
            yield piece


def _split_boundary(text: str, chunk_size: int, chunk_stride: int) -> Iterator[str]:
    """Legacy character windows with a best-effort sentence/line boundary."""

    if len(text) <= chunk_size:
        yield text
        return
    start = 0
    while start < len(text):
        target_end = min(len(text), start + chunk_size)
        end = target_end
        if target_end < len(text):
            minimum_boundary = start + int(chunk_size * 0.65)
            boundary = max(
                text.rfind("\n", minimum_boundary, target_end),
                text.rfind("。", minimum_boundary, target_end),
                text.rfind(". ", minimum_boundary, target_end),
            )
            if boundary >= minimum_boundary:
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            yield piece
        if end >= len(text):
            break
        start = max(start + 1, min(start + chunk_stride, end))


def _split_dynamic(text: str, max_tokens: int, overlap_tokens: int) -> Iterator[str]:
    sentences = split_sentence_units(text)
    if not sentences:
        return

    # Long legal table rows or a malformed OCR sentence still need a bounded
    # fallback. Those pieces are hard-split only when one sentence exceeds the
    # dynamic budget; normal chunks always begin/end on sentence boundaries.
    expanded: List[str] = []
    for sentence in sentences:
        if estimate_token_count(sentence) <= max_tokens:
            expanded.append(sentence)
            continue
        expanded.extend(_split_long_sentence(sentence, max_tokens))

    index = 0
    while index < len(expanded):
        end = index
        budget = 0
        while end < len(expanded):
            sentence_tokens = estimate_token_count(expanded[end])
            if end > index and budget + sentence_tokens > max_tokens:
                break
            budget += sentence_tokens
            end += 1
            if budget >= max_tokens:
                break
        if end == index:  # Defensive; _split_long_sentence should prevent this.
            end = index + 1
        yield "\n".join(expanded[index:end]).strip()
        if end >= len(expanded):
            break

        next_index = _overlap_start(expanded, index, end, overlap_tokens)
        if next_index >= end:
            index = end
        else:
            index = max(index + 1, min(next_index, end - 1))


def _split_long_sentence(sentence: str, max_tokens: int) -> List[str]:
    tokens = _TOKEN_RE.findall(sentence)
    if not tokens:
        return [sentence]
    pieces = []
    for start in range(0, len(tokens), max_tokens):
        piece = "".join(tokens[start : start + max_tokens]).strip()
        if piece:
            pieces.append(piece)
    return pieces or [sentence]


def _overlap_start(sentences: List[str], start: int, end: int, overlap_tokens: int) -> int:
    if overlap_tokens <= 0:
        return end
    running = 0
    overlap_start = end
    for index in range(end - 1, start - 1, -1):
        sentence_tokens = estimate_token_count(sentences[index])
        if running and running + sentence_tokens > overlap_tokens:
            break
        if not running and sentence_tokens > overlap_tokens:
            # Keep complete sentences whenever possible. A very long sentence
            # cannot be carried into the next chunk without exceeding budget.
            break
        running += sentence_tokens
        overlap_start = index
    return overlap_start
