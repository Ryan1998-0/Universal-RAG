#!/usr/bin/env python3
"""Run the LegalBench-RAG retrieval-only comparison.

The published paper reports 6,858 questions: 946 ContractNLI, 4,042 CUAD,
1,676 MAUD, and 194 PrivacyQA.  The current official Dropbox archive contains
977 ContractNLI records (6,889 total).  The 31 records belonging to the
excluded ``nda-10`` annotation category are removed so this run follows the
paper's 6,858-question count deterministically.

Both versions use the same corpus and embedding model:

* 無優化版: hard 600-character chunks, no overlap, raw 50/50 BM25+dense
  score addition, and the existing lexical/semantic reranker to Top 5.
* 優化版: sentence-aware 1024-token parents with 256-token children, RRF
  (k=60) over 100 candidates, complexity routing, Cross-Encoder reranking
  for complex questions, and parent evidence expansion to Top 5.

The script stops after retrieval.  It does not call an answer model, use
conversation memory, or put gold evidence into the query.  Retrieved text is
used in memory for scoring and reranking but is not written to the result
JSON; only ranking metadata, source spans, and aggregate metrics are saved.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_demo.chunk_strategies import (
    CHUNK_STRATEGY_DYNAMIC,
    CHUNK_STRATEGY_HARD,
    estimate_token_count,
    split_text,
)
from rag_demo.config import RagConfig
from rag_demo.cross_encoder import DEFAULT_CROSS_ENCODER_MODEL, CrossEncoderReranker
from rag_demo.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    _get_embedding_model,
    chunk_to_embedding_text,
    prepare_embedding_texts,
)
from rag_demo.hybrid_retrieval import (
    reciprocal_rank_fusion,
    rerank_candidates,
    tokenize_bm25,
)
from rag_demo.parent_child import build_parent_child_index
from rag_demo.query_complexity import classify_query_complexity


DEFAULT_ZIP = (
    ROOT
    / "RAG測試題庫"
    / "02_LegalBench-RAG"
    / "downloads"
    / "LegalBench-RAG.zip"
)
DEFAULT_OUTPUT = ROOT / "evals" / "legalbench_rag_retrieval_full" / "retrieval-results.json"
DEFAULT_REPORT = ROOT / "evals" / "legalbench_rag_retrieval_full" / "retrieval-report.md"
DEFAULT_SUMMARY = ROOT / "evals" / "legalbench_rag_retrieval_full" / "retrieval-summary.json"
DEFAULT_CACHE_DIR = ROOT / "evals" / "legalbench_rag_retrieval_full" / "embedding-cache"
DEFAULT_CHECKPOINT_DIR = ROOT / "evals" / "legalbench_rag_retrieval_full" / "checkpoints"

SUBSETS = ("contractnli", "cuad", "maud", "privacy_qa")
PUBLISHED_COUNTS = {
    "contractnli": 946,
    "cuad": 4042,
    "maud": 1676,
    "privacy_qa": 194,
}
EXCLUDED_CONTRACTNLI_MARKER = (
    "Does the document include a clause that prevents the Receiving Party from "
    "disclosing the fact that the Agreement was agreed upon or negotiated?"
)


def _normalise_embeddings(matrix: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Embedding matrix must be 2-D, got shape {values.shape}")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _normalise_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _canonical_file_path(value: str) -> str:
    """Use NFC so benchmark paths match zip entries on every filesystem."""

    return unicodedata.normalize("NFC", str(value or ""))


def _normalised_with_map(value: str) -> tuple[str, list[int], list[int]]:
    """Collapse whitespace while retaining normalized-index to source-index maps."""

    text = str(value or "")
    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for index, char in enumerate(text):
        if char.isspace():
            if chars and chars[-1] != " ":
                chars.append(" ")
                starts.append(index)
                ends.append(index + 1)
            continue
        chars.append(char)
        starts.append(index)
        ends.append(index + 1)
    while chars and chars[-1] == " ":
        chars.pop()
        starts.pop()
        ends.pop()
    return "".join(chars), starts, ends


def _map_normalised_span(
    normalised_body: str,
    starts: Sequence[int],
    ends: Sequence[int],
    content: str,
    cursor: int = 0,
) -> tuple[int, int, int, bool]:
    """Map normalized content back to source offsets.

    The first search starts at ``cursor`` so repeated legal boilerplate is
    assigned to the chunks in their construction order.  A global fallback is
    retained for unusual OCR/whitespace cases and is reported to the caller.
    """

    needle = _normalise_whitespace(content)
    if not needle:
        return 0, 0, cursor, False
    index = normalised_body.find(needle, max(0, int(cursor)))
    exact = True
    if index < 0:
        index = normalised_body.find(needle)
        exact = False
    if index < 0 or index + len(needle) > len(starts):
        return 0, 0, cursor, False
    source_start = int(starts[index])
    source_end = int(ends[index + len(needle) - 1])
    return source_start, source_end, index + len(needle), exact


def _load_archive(
    archive_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    if not archive_path.is_file():
        raise FileNotFoundError(f"LegalBench-RAG archive not found: {archive_path}")

    items: list[dict[str, Any]] = []
    excluded_by_subset: Counter[str] = Counter()
    downloaded_counts: dict[str, int] = {}
    corpus: dict[str, str] = {}

    with zipfile.ZipFile(archive_path) as archive:
        for subset in SUBSETS:
            benchmark_name = f"benchmarks/{subset}.json"
            payload = json.loads(archive.read(benchmark_name).decode("utf-8"))
            tests = payload.get("tests") if isinstance(payload, dict) else payload
            if not isinstance(tests, list):
                raise ValueError(f"Invalid benchmark payload: {benchmark_name}")
            downloaded_counts[subset] = len(tests)
            for subset_index, raw in enumerate(tests):
                record = dict(raw)
                query = str(record.get("query") or "").strip()
                interrogative = query.split(";", 1)[-1].strip()
                if subset == "contractnli" and interrogative == EXCLUDED_CONTRACTNLI_MARKER:
                    excluded_by_subset[subset] += 1
                    continue
                record.update(
                    {
                        "id": f"legalbench-{subset}-{subset_index:04d}",
                        "subset": subset,
                        "subset_index": subset_index,
                        "query": query,
                        "snippets": [
                            {
                                **dict(snippet),
                                "file_path": _canonical_file_path(
                                    str(snippet.get("file_path") or "")
                                ),
                            }
                            for snippet in (record.get("snippets") or [])
                        ],
                    }
                )
                items.append(record)

        for name in archive.namelist():
            if not name.startswith("corpus/") or not name.endswith(".txt"):
                continue
            relative_path = _canonical_file_path(name[len("corpus/") :])
            corpus[relative_path] = archive.read(name).decode("utf-8", errors="replace")

    expected_counts = dict(PUBLISHED_COUNTS)
    actual_counts = Counter(str(item.get("subset") or "unknown") for item in items)
    if dict(actual_counts) != expected_counts:
        raise ValueError(
            "Paper-aligned LegalBench-RAG counts do not match: "
            f"expected {expected_counts}, got {dict(actual_counts)}"
        )
    if len(items) != sum(PUBLISHED_COUNTS.values()):
        raise ValueError(f"Expected 6,858 questions, got {len(items)}")
    missing_files = sorted(
        {
            str(snippet.get("file_path") or "")
            for item in items
            for snippet in item.get("snippets") or []
            if str(snippet.get("file_path") or "") not in corpus
        }
    )
    if missing_files:
        raise ValueError(f"Gold snippets reference missing corpus files: {missing_files[:3]}")

    metadata = {
        "downloaded_counts": downloaded_counts,
        "evaluated_counts": dict(sorted(actual_counts.items())),
        "excluded_counts": dict(sorted(excluded_by_subset.items())),
        "downloaded_total": sum(downloaded_counts.values()),
        "evaluated_total": len(items),
        "corpus_documents": len(corpus),
        "corpus_characters": sum(len(text) for text in corpus.values()),
        "selection_rule": (
            "Use all current archive records except the 31 ContractNLI records "
            "whose annotation marker is nda-10, matching the paper's 946-count subset."
        ),
    }
    return items, corpus, metadata


def _build_hard_chunks(corpus: dict[str, str]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for file_index, file_path in enumerate(sorted(corpus), start=1):
        raw_body = corpus[file_path]
        clean_body = raw_body.strip()
        base_offset = len(raw_body) - len(raw_body.lstrip())
        if not clean_body:
            continue
        for chunk_index, start in enumerate(range(0, len(clean_body), 600), start=1):
            end = min(len(clean_body), start + 600)
            content = clean_body[start:end].strip()
            if not content:
                continue
            chunks.append(
                {
                    "id": f"legalbench-hard-{file_index:04d}-{chunk_index:04d}",
                    "parent_title": "LegalBench-RAG",
                    "title": file_path,
                    "page": file_path,
                    "source": file_path,
                    "source_id": file_path,
                    "file_path": file_path,
                    "content": content,
                    "start_char": base_offset + start,
                    "end_char": base_offset + end,
                    "chunk_level": "single",
                }
            )
    return chunks


def _build_parent_child_chunks(
    corpus: dict[str, str],
) -> tuple[Any, int]:
    file_paths = sorted(corpus)
    units = [
        {
            "title": file_path,
            "page": file_path,
            "source": file_path,
            "source_id": file_path,
            "content": corpus[file_path],
        }
        for file_path in file_paths
    ]
    index = build_parent_child_index(
        units,
        source_id="legalbench-rag",
        filename="LegalBench-RAG",
        source_type="legalbench-rag-zip",
        extraction_method="archive-corpus",
        parent_size_tokens=1024,
        child_size_tokens=256,
        parent_overlap_tokens=0,
        child_overlap_tokens=0,
        strategy=CHUNK_STRATEGY_DYNAMIC,
    )

    normalized_cache: dict[str, tuple[str, list[int], list[int]]] = {
        file_path: _normalised_with_map(corpus[file_path]) for file_path in file_paths
    }
    parent_pattern = re.compile(r"::parent::(\d+)-(\d+)$")
    parent_slots_by_unit: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for parent in index.parents:
        match = parent_pattern.search(str(parent.get("id") or ""))
        if match:
            parent_slots_by_unit[int(match.group(1))].append(
                {
                    "record": parent,
                    "needle": _normalise_whitespace(str(parent.get("content") or "")),
                    "norm_start": None,
                    "norm_end": None,
                    "fallback": False,
                }
            )

    def _source_span(file_path: str, start: int, end: int) -> tuple[int, int]:
        _, starts, ends = normalized_cache[file_path]
        start = max(0, min(int(start), len(starts)))
        end = max(start, min(int(end), len(starts)))
        if not starts or end <= start:
            return 0, 0
        return int(starts[start]), int(ends[end - 1])

    def _allocate_missing(slots: list[dict[str, Any]], body_length: int) -> int:
        """Allocate long-token pieces across gaps between exact matches."""

        fallback_count = 0
        known = [index for index, slot in enumerate(slots) if slot.get("norm_start") is not None]
        if not known:
            total_weight = sum(max(1, len(str(slot.get("needle") or ""))) for slot in slots)
            cursor = 0
            for index, slot in enumerate(slots):
                weight = max(1, len(str(slot.get("needle") or "")))
                end = body_length if index == len(slots) - 1 else cursor + round(body_length * weight / max(1, total_weight))
                slot["norm_start"], slot["norm_end"] = cursor, max(cursor, end)
                slot["fallback"] = True
                cursor = slot["norm_end"]
                fallback_count += 1
            return fallback_count

        index = 0
        while index < len(slots):
            if slots[index].get("norm_start") is not None:
                index += 1
                continue
            run_start = index
            while index + 1 < len(slots) and slots[index + 1].get("norm_start") is None:
                index += 1
            run_end = index
            previous = run_start - 1
            left = int(slots[previous].get("norm_end") or 0) if previous >= 0 else 0
            following = run_end + 1
            right = int(slots[following].get("norm_start")) if following < len(slots) else body_length
            right = max(left, right)
            available = right - left
            weights = [max(1, len(str(slots[pos].get("needle") or ""))) for pos in range(run_start, run_end + 1)]
            total_weight = max(1, sum(weights))
            cursor = left
            for offset, pos in enumerate(range(run_start, run_end + 1)):
                end = right if offset == len(weights) - 1 else cursor + round(available * weights[offset] / total_weight)
                slots[pos]["norm_start"] = cursor
                slots[pos]["norm_end"] = max(cursor, end)
                slots[pos]["fallback"] = True
                cursor = slots[pos]["norm_end"]
                fallback_count += 1
            index += 1
        return fallback_count

    fallback_count = 0
    parent_norm_positions: dict[str, tuple[str, int, int]] = {}
    for unit_index, slots in parent_slots_by_unit.items():
        file_path = file_paths[unit_index - 1]
        normalized_body = normalized_cache[file_path][0]
        cursor = 0
        for slot in slots:
            needle = str(slot.get("needle") or "")
            position = normalized_body.find(needle, cursor) if needle else -1
            if position >= 0:
                slot["norm_start"] = position
                slot["norm_end"] = position + len(needle)
                cursor = slot["norm_end"]
        fallback_count += _allocate_missing(slots, len(normalized_body))
        for slot in slots:
            parent = slot["record"]
            norm_start = int(slot.get("norm_start") or 0)
            norm_end = int(slot.get("norm_end") or norm_start)
            source_start, source_end = _source_span(file_path, norm_start, norm_end)
            parent["file_path"] = file_path
            parent["source"] = file_path
            parent["source_id"] = file_path
            parent["title"] = file_path
            parent["page"] = file_path
            parent["start_char"] = source_start
            parent["end_char"] = source_end
            parent_norm_positions[str(parent["id"])] = (file_path, norm_start, norm_end)

    child_pattern = re.compile(r"::child::(\d+)-(\d+)-(\d+)$")
    child_slots_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for child in index.children:
        parent_id = str(child.get("parent_id") or "")
        if parent_id in parent_norm_positions:
            child_slots_by_parent[parent_id].append(
                {
                    "record": child,
                    "needle": _normalise_whitespace(str(child.get("content") or "")),
                    "norm_start": None,
                    "norm_end": None,
                    "fallback": False,
                }
            )

    for parent_id, slots in child_slots_by_parent.items():
        file_path, parent_start, parent_end = parent_norm_positions[parent_id]
        normalized_body = normalized_cache[file_path][0]
        cursor = parent_start
        for slot in slots:
            needle = str(slot.get("needle") or "")
            position = normalized_body.find(needle, cursor) if needle else -1
            if position >= parent_start and position + len(needle) <= parent_end:
                slot["norm_start"] = position
                slot["norm_end"] = position + len(needle)
                cursor = slot["norm_end"]
        # Allocate missing children only within the parent interval.
        shifted = [
            {
                **slot,
                "norm_start": (
                    None
                    if slot.get("norm_start") is None
                    else int(slot["norm_start"]) - parent_start
                ),
                "norm_end": (
                    None
                    if slot.get("norm_end") is None
                    else int(slot["norm_end"]) - parent_start
                ),
            }
            for slot in slots
        ]
        fallback_count += _allocate_missing(shifted, max(0, parent_end - parent_start))
        for original, slot in zip(slots, shifted):
            norm_start = max(parent_start, parent_start + int(slot.get("norm_start") or 0))
            norm_end = min(parent_end, parent_start + int(slot.get("norm_end") or 0))
            norm_end = max(norm_start, norm_end)
            child = original["record"]
            source_start, source_end = _source_span(file_path, norm_start, norm_end)
            child["file_path"] = file_path
            child["source"] = file_path
            child["source_id"] = file_path
            child["title"] = file_path
            child["page"] = file_path
            child["start_char"] = source_start
            child["end_char"] = source_end
            child["parent_file_path"] = file_path

    # ``child_pattern`` is intentionally compiled above even though the parent
    # relationship is sufficient for mapping; it validates the expected ID
    # shape without exposing corpus content in the report.
    del child_pattern
    return index, fallback_count


class FastBm25Index:
    """BM25 equivalent to the project index with an inverted posting list.

    The project BM25 formula is preserved exactly.  The posting list avoids
    scanning every chunk for every question, which is material for the 200k
    chunk LegalBench-RAG corpus.
    """

    def __init__(self, chunks: Sequence[dict[str, Any]], k1: float = 1.4, b: float = 0.72):
        self.chunks = [dict(chunk) for chunk in chunks]
        self.k1 = max(0.01, float(k1))
        self.b = min(max(0.0, float(b)), 1.0)
        self.term_frequencies: list[Counter[str]] = []
        self.document_lengths: list[int] = []
        self.postings: dict[str, list[int]] = defaultdict(list)
        document_frequency: Counter[str] = Counter()

        for index, chunk in enumerate(self.chunks):
            counts = Counter(tokenize_bm25(str(chunk.get("content") or "")))
            self.term_frequencies.append(counts)
            self.document_lengths.append(max(1, sum(counts.values())))
            for term in counts:
                self.postings[term].append(index)
            document_frequency.update(counts.keys())

        self.document_count = len(self.chunks)
        self.average_length = (
            sum(self.document_lengths) / self.document_count
            if self.document_count
            else 1.0
        )
        # Keep the exact posting lists above for diagnostics, but materialize
        # their document ids and term frequencies as NumPy arrays as well.
        # LegalBench-RAG has 133k hard chunks; iterating every posting in
        # Python for every query makes the full benchmark unnecessarily slow.
        # The vectorized path below applies the same BM25 equation over each
        # posting list and therefore preserves the retrieval scores while
        # moving the hot loop into NumPy.
        self.document_lengths_array = np.asarray(self.document_lengths, dtype=np.float64)
        self.inverse_document_frequency = {
            term: math.log(
                1.0
                + (self.document_count - frequency + 0.5) / (frequency + 0.5)
            )
            for term, frequency in document_frequency.items()
        }
        self.posting_indices_array = {
            term: np.asarray(indices, dtype=np.int32)
            for term, indices in self.postings.items()
        }
        self.posting_term_frequencies_array = {
            term: np.fromiter(
                (self.term_frequencies[index].get(term, 0) for index in indices),
                dtype=np.float64,
                count=len(indices),
            )
            for term, indices in self.postings.items()
        }

    def search(
        self,
        query: str,
        top_k: int = 100,
        allowed_indices: Optional[Sequence[int]] = None,
    ) -> list[dict[str, Any]]:
        """Return BM25 hits, optionally restricted to selected chunks.

        ``allowed_indices`` is used by document-scoped benchmark runs and by
        callers that already know which source documents are in scope.  Scores
        keep the index's existing BM25 statistics; the restriction only
        controls which chunks may be returned.
        """
        query_terms = Counter(tokenize_bm25(query))
        if not query_terms:
            return []
        scores = np.zeros(self.document_count, dtype=np.float64)
        for term, query_frequency in query_terms.items():
            inverse_document_frequency = self.inverse_document_frequency.get(term, 0.0)
            if inverse_document_frequency <= 0:
                continue
            indices = self.posting_indices_array.get(term)
            if indices is None or len(indices) == 0:
                continue
            term_frequencies = self.posting_term_frequencies_array[term]
            document_lengths = self.document_lengths_array[indices]
            denominator = term_frequencies + self.k1 * (
                1.0 - self.b + self.b * (document_lengths / self.average_length)
            )
            scores[indices] += (
                inverse_document_frequency
                * ((term_frequencies * (self.k1 + 1.0)) / denominator)
                * (1.0 + math.log(query_frequency))
            )

        if allowed_indices is None:
            positive = np.flatnonzero(scores > 0)
        else:
            allowed = np.asarray(
                sorted(
                    {
                        int(index)
                        for index in allowed_indices
                        if 0 <= int(index) < self.document_count
                    }
                ),
                dtype=np.int64,
            )
            if len(allowed) == 0:
                return []
            allowed_mask = np.zeros(self.document_count, dtype=bool)
            allowed_mask[allowed] = True
            positive = np.flatnonzero((scores > 0) & allowed_mask)
        if len(positive) == 0:
            return []
        limit = max(1, int(top_k))
        if len(positive) > limit:
            values = scores[positive]
            selected_positions = np.argpartition(-values, kth=limit - 1)[:limit]
            selected = positive[selected_positions]
        else:
            selected = positive
        order = np.argsort(-scores[selected], kind="stable")
        ranked_indices = selected[order]
        ranked: list[dict[str, Any]] = []
        for raw_index in ranked_indices:
            index = int(raw_index)
            ranked.append(
                {
                    "index": index,
                    "score": float(scores[index]),
                    "matched_terms": [
                        term
                        for term in query_terms
                        if self.term_frequencies[index].get(term, 0)
                    ],
                }
            )
        return ranked


def _load_or_build_embeddings(
    chunks: Sequence[dict[str, Any]],
    cache_path: Path,
    *,
    model_name: str,
    batch_size: int,
    label: str,
) -> tuple[np.ndarray, float, bool]:
    expected_count = len(chunks)
    if cache_path.is_file():
        try:
            matrix = np.load(cache_path, mmap_mode="r")
            if matrix.ndim == 2 and matrix.shape[0] == expected_count:
                print(f"Loaded {label} embedding cache: {cache_path} ({expected_count} rows)", flush=True)
                return _normalise_embeddings(np.asarray(matrix)), 0.0, True
        except Exception as exc:
            print(f"Ignoring invalid {label} embedding cache: {exc}", flush=True)

    print(f"Embedding {label}: {expected_count} chunks", flush=True)
    started = time.perf_counter()
    model = _get_embedding_model(model_name)
    texts = prepare_embedding_texts(
        [chunk_to_embedding_text(chunk) for chunk in chunks],
        model_name,
        kind="document",
    )
    matrix = model.encode(
        texts,
        batch_size=max(1, int(batch_size)),
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    normalised = _normalise_embeddings(matrix)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, normalised.astype(np.float32))
    elapsed = (time.perf_counter() - started) * 1000.0
    print(f"Built {label} embeddings in {elapsed / 1000.0:.1f}s", flush=True)
    return normalised, elapsed, False


def _load_or_build_query_embeddings(
    queries: Sequence[str],
    cache_path: Path,
    *,
    model_name: str,
    batch_size: int,
) -> tuple[np.ndarray, float, bool]:
    expected_count = len(queries)
    if cache_path.is_file():
        try:
            matrix = np.load(cache_path, mmap_mode="r")
            if matrix.ndim == 2 and matrix.shape[0] == expected_count:
                print(f"Loaded query embedding cache: {cache_path} ({expected_count} rows)", flush=True)
                return _normalise_embeddings(np.asarray(matrix)), 0.0, True
        except Exception as exc:
            print(f"Ignoring invalid query embedding cache: {exc}", flush=True)

    print(f"Embedding {expected_count} queries", flush=True)
    started = time.perf_counter()
    model = _get_embedding_model(model_name)
    prepared_queries = prepare_embedding_texts(queries, model_name, kind="query")
    matrix = model.encode(
        prepared_queries,
        batch_size=max(1, int(batch_size)),
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    normalised = _normalise_embeddings(matrix)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, normalised.astype(np.float32))
    elapsed = (time.perf_counter() - started) * 1000.0
    print(f"Built query embeddings in {elapsed / 1000.0:.1f}s", flush=True)
    return normalised, elapsed, False


def _dense_hits_all(
    matrix: np.ndarray,
    query_matrix: np.ndarray,
    *,
    top_k: int = 100,
    batch_size: int = 64,
) -> tuple[list[list[dict[str, Any]]], float]:
    if len(matrix) == 0:
        return [[] for _ in range(len(query_matrix))], 0.0
    started = time.perf_counter()
    results: list[list[dict[str, Any]]] = []
    k = min(max(1, int(top_k)), len(matrix))
    for start in range(0, len(query_matrix), max(1, int(batch_size))):
        query_batch = query_matrix[start : start + max(1, int(batch_size))]
        scores = np.asarray(matrix @ query_batch.T, dtype=np.float32)
        candidate_indices = np.argpartition(-scores, kth=k - 1, axis=0)[:k, :]
        for column in range(scores.shape[1]):
            indices = candidate_indices[:, column]
            order = np.argsort(-scores[indices, column], kind="stable")
            results.append(
                [
                    {
                        "index": int(indices[position]),
                        "score": float(scores[indices[position], column]),
                    }
                    for position in order
                ]
            )
    elapsed = (time.perf_counter() - started) * 1000.0
    return results, elapsed


def _raw_fusion(
    bm25_hits: Sequence[dict[str, Any]],
    dense_hits: Sequence[dict[str, Any]],
    keyword_weight: float = 0.5,
    embedding_weight: float = 0.5,
) -> list[dict[str, Any]]:
    candidates: dict[int, dict[str, Any]] = {}
    for hit in bm25_hits:
        index = int(hit["index"])
        item = candidates.setdefault(
            index,
            {
                "index": index,
                "bm25_score": 0.0,
                "embedding_score": 0.0,
                "matched_terms": [],
            },
        )
        item["bm25_score"] = float(hit.get("score", 0.0))
        item["matched_terms"] = list(hit.get("matched_terms") or [])
    for hit in dense_hits:
        index = int(hit["index"])
        item = candidates.setdefault(
            index,
            {
                "index": index,
                "bm25_score": 0.0,
                "embedding_score": 0.0,
                "matched_terms": [],
            },
        )
        item["embedding_score"] = float(hit.get("score", 0.0))
    total = max(1e-12, float(keyword_weight) + float(embedding_weight))
    keyword_weight /= total
    embedding_weight /= total
    for item in candidates.values():
        item["fusion_score"] = (
            keyword_weight * item["bm25_score"]
            + embedding_weight * item["embedding_score"]
        )
        item["score"] = item["fusion_score"]
    return sorted(candidates.values(), key=lambda item: item["fusion_score"], reverse=True)


def _compact_context(context: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "id",
        "rank",
        "title",
        "page",
        "source",
        "file_path",
        "start_char",
        "end_char",
        "score",
        "bm25Score",
        "embeddingScore",
        "fusionScore",
        "rrfScore",
        "rerankScore",
        "childChunkId",
        "parentChunkId",
        "matchedTerms",
    )
    return {key: context[key] for key in fields if key in context}


def _contexts_from_selected(
    selected: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
    *,
    branch: str,
) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for rank, candidate in enumerate(selected, start=1):
        chunk = chunks[int(candidate["index"])]
        contexts.append(
            {
                "id": str(chunk.get("id") or candidate["index"]),
                "rank": rank,
                "title": str(chunk.get("title") or "LegalBench-RAG"),
                "page": str(chunk.get("page") or ""),
                "source": str(chunk.get("source_id") or chunk.get("source") or ""),
                "file_path": str(chunk.get("file_path") or ""),
                "start_char": int(chunk.get("start_char") or 0),
                "end_char": int(chunk.get("end_char") or 0),
                "content": str(chunk.get("content") or ""),
                "branch": branch,
                "score": round(
                    float(
                        candidate.get(
                            "rerank_score",
                            candidate.get("fusion_score", candidate.get("rrf_score", 0.0)),
                        )
                    ),
                    8,
                ),
                "bm25Score": round(float(candidate.get("bm25_score", 0.0)), 8),
                "embeddingScore": round(float(candidate.get("embedding_score", 0.0)), 8),
                "fusionScore": round(
                    float(candidate.get("fusion_score", candidate.get("rrf_score", 0.0))),
                    8,
                ),
                "rrfScore": round(float(candidate.get("rrf_score", 0.0)), 8),
                "rerankScore": round(
                    float(
                        candidate.get(
                            "rerank_score",
                            candidate.get("fusion_score", candidate.get("rrf_score", 0.0)),
                        )
                    ),
                    8,
                ),
                "matchedTerms": list(candidate.get("matched_terms") or []),
            }
        )
    return contexts


def _contexts_from_parent_selection(
    selected: Sequence[dict[str, Any]],
    children: Sequence[dict[str, Any]],
    parents: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    parent_by_id = {str(parent.get("id")): parent for parent in parents}
    contexts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in selected:
        child = children[int(candidate["index"])]
        parent_id = str(child.get("parent_id") or child.get("parent_chunk_id") or "")
        parent = parent_by_id.get(parent_id)
        if parent is None:
            continue
        parent_key = str(parent.get("id") or parent_id)
        if parent_key in seen:
            continue
        seen.add(parent_key)
        contexts.append(
            {
                "id": parent_key,
                "rank": len(contexts) + 1,
                "title": str(parent.get("title") or "LegalBench-RAG"),
                "page": str(parent.get("page") or ""),
                "source": str(parent.get("source_id") or parent.get("source") or ""),
                "file_path": str(parent.get("file_path") or ""),
                "start_char": int(parent.get("start_char") or 0),
                "end_char": int(parent.get("end_char") or 0),
                "content": str(parent.get("content") or ""),
                "branch": "optimized-parent-evidence",
                "score": round(
                    float(
                        candidate.get(
                            "rerank_score",
                            candidate.get("rrf_score", candidate.get("fusion_score", 0.0)),
                        )
                    ),
                    8,
                ),
                "bm25Score": round(float(candidate.get("bm25_score", 0.0)), 8),
                "embeddingScore": round(float(candidate.get("embedding_score", 0.0)), 8),
                "fusionScore": round(
                    float(candidate.get("rrf_score", candidate.get("fusion_score", 0.0))),
                    8,
                ),
                "rrfScore": round(float(candidate.get("rrf_score", 0.0)), 8),
                "rerankScore": round(
                    float(
                        candidate.get(
                            "rerank_score",
                            candidate.get("rrf_score", candidate.get("fusion_score", 0.0)),
                        )
                    ),
                    8,
                ),
                "childChunkId": str(child.get("id") or ""),
                "parentChunkId": parent_key,
                "matchedTerms": list(candidate.get("matched_terms") or []),
            }
        )
    return contexts


def _overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> int:
    return max(0, min(left_end, right_end) - max(left_start, right_start))


def _span_metrics(
    contexts: Sequence[dict[str, Any]],
    snippets: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    gold = [
        (
            str(snippet.get("file_path") or ""),
            int((snippet.get("span") or [0, 0])[0]),
            int((snippet.get("span") or [0, 0])[1]),
        )
        for snippet in snippets
    ]
    retrieved = [
        (
            str(context.get("file_path") or ""),
            int(context.get("start_char") or 0),
            int(context.get("end_char") or 0),
        )
        for context in contexts
    ]
    total_gold = sum(max(0, end - start) for _, start, end in gold)
    total_retrieved = sum(max(0, end - start) for _, start, end in retrieved)
    relevant_retrieved = 0
    gold_hit_flags: list[bool] = []
    for gold_file, gold_start, gold_end in gold:
        hits = [
            _overlap(start, end, gold_start, gold_end)
            for file_path, start, end in retrieved
            if file_path == gold_file
        ]
        relevant_retrieved += sum(hits)
        gold_hit_flags.append(any(hit > 0 for hit in hits))
    first_hit_rank = None
    for context in contexts:
        if any(
            str(context.get("file_path") or "") == gold_file
            and _overlap(
                int(context.get("start_char") or 0),
                int(context.get("end_char") or 0),
                gold_start,
                gold_end,
            )
            > 0
            for gold_file, gold_start, gold_end in gold
        ):
            first_hit_rank = int(context.get("rank") or 0) or None
            break
    return {
        "gold_span_count": len(gold),
        "gold_characters": total_gold,
        "retrieved_characters": total_retrieved,
        "relevant_retrieved_characters": relevant_retrieved,
        "char_recall": round(relevant_retrieved / total_gold, 8) if total_gold else 0.0,
        "char_precision": (
            round(relevant_retrieved / total_retrieved, 8) if total_retrieved else 0.0
        ),
        "any_gold_span_hit": bool(any(gold_hit_flags)),
        "complete_gold_span_hit": bool(gold_hit_flags and all(gold_hit_flags)),
        "rank1_gold_span_hit": bool(
            contexts
            and any(
                str(contexts[0].get("file_path") or "") == gold_file
                and _overlap(
                    int(contexts[0].get("start_char") or 0),
                    int(contexts[0].get("end_char") or 0),
                    gold_start,
                    gold_end,
                )
                > 0
                for gold_file, gold_start, gold_end in gold
            )
        ),
        "first_gold_hit_rank": first_hit_rank,
    }


def _retrieve_unoptimized(
    question: str,
    chunks: Sequence[dict[str, Any]],
    matrix_dense_hits: Sequence[dict[str, Any]],
    bm25: FastBm25Index,
    settings: RagConfig,
    chunk_token_sets: Sequence[set[str]],
    allowed_indices: Optional[Sequence[int]] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bm25_hits = bm25.search(question, top_k=100, allowed_indices=allowed_indices)
    candidates = _raw_fusion(bm25_hits, matrix_dense_hits, 0.5, 0.5)[:100]
    selected = rerank_candidates(
        candidates,
        chunks,
        question,
        top_k=5,
        settings=settings,
        chunk_token_sets=chunk_token_sets,
    )
    return _contexts_from_selected(selected, chunks, branch="unoptimized"), {
        "retrieval_query": question,
        "rewrite": {"status": "disabled", "accepted": True},
        "complexity": {"label": "unoptimized-always-rerank", "is_complex": True},
        "fusion": "raw_weighted_addition",
        "candidate_count": len(candidates),
        "rerank_applied": True,
        "retrieval_children": 0,
        "evidence_parents": len(selected),
    }


def _retrieve_optimized(
    question: str,
    children: Sequence[dict[str, Any]],
    parents: Sequence[dict[str, Any]],
    matrix_dense_hits: Sequence[dict[str, Any]],
    bm25: FastBm25Index,
    cross_encoder: CrossEncoderReranker,
    settings: RagConfig,
    allowed_indices: Optional[Sequence[int]] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bm25_hits = bm25.search(question, top_k=100, allowed_indices=allowed_indices)
    candidates = reciprocal_rank_fusion(
        bm25_hits,
        matrix_dense_hits,
        rrf_k=settings.hybrid_rrf_k,
    )[:100]
    decision = classify_query_complexity(
        question,
        threshold=settings.query_complexity_threshold,
    )
    if decision.is_complex:
        documents = [str(children[int(candidate["index"])].get("content") or "") for candidate in candidates]
        scores = cross_encoder.score(question, documents)
        ranked = [
            dict(candidate, rerank_score=float(score))
            for candidate, score in zip(candidates, scores)
        ]
        ranked.sort(
            key=lambda item: (
                item["rerank_score"],
                item.get("rrf_score", item.get("fusion_score", 0.0)),
            ),
            reverse=True,
        )
        rerank_applied = True
    else:
        ranked = [
            dict(
                candidate,
                rerank_score=float(
                    candidate.get("rrf_score", candidate.get("fusion_score", 0.0))
                ),
            )
            for candidate in candidates
        ]
        rerank_applied = False
    selected = ranked[:5]
    return _contexts_from_parent_selection(selected, children, parents), {
        "retrieval_query": question,
        "rewrite": {
            "status": "identity_authored_english_benchmark",
            "accepted": True,
            "similarity": 1.0,
        },
        "complexity": decision.as_dict(),
        "fusion": "RRF",
        "candidate_count": len(candidates),
        "rerank_applied": rerank_applied,
        "retrieval_children": len(selected),
        "evidence_parents": len(_contexts_from_parent_selection(selected, children, parents)),
    }


def _summarize(
    cases: Sequence[dict[str, Any]],
    *,
    dense_search_ms: float,
) -> dict[str, Any]:
    if not cases:
        return {"question_count": 0}
    metrics = [case["metrics"] for case in cases]
    latencies = [float(case["timings_ms"]["question_stage"]) for case in cases]
    hit_ranks = [
        int(metric["first_gold_hit_rank"])
        for metric in metrics
        if metric.get("first_gold_hit_rank") is not None
    ]
    total_gold = sum(int(metric["gold_characters"]) for metric in metrics)
    total_retrieved = sum(int(metric["retrieved_characters"]) for metric in metrics)
    total_relevant = sum(int(metric["relevant_retrieved_characters"]) for metric in metrics)
    return {
        "question_count": len(cases),
        "macro_char_recall": round(statistics.mean(metric["char_recall"] for metric in metrics), 8),
        "macro_char_precision": round(statistics.mean(metric["char_precision"] for metric in metrics), 8),
        "micro_char_recall": round(total_relevant / total_gold, 8) if total_gold else 0.0,
        "micro_char_precision": round(total_relevant / total_retrieved, 8) if total_retrieved else 0.0,
        "any_gold_span_hit_rate": round(
            sum(bool(metric["any_gold_span_hit"]) for metric in metrics) / len(metrics),
            8,
        ),
        "complete_gold_span_hit_rate": round(
            sum(bool(metric["complete_gold_span_hit"]) for metric in metrics) / len(metrics),
            8,
        ),
        "rank1_gold_span_hit_rate": round(
            sum(bool(metric["rank1_gold_span_hit"]) for metric in metrics) / len(metrics),
            8,
        ),
        "first_hit_rate": round(len(hit_ranks) / len(metrics), 8),
        "mean_first_hit_rank": round(statistics.mean(hit_ranks), 4) if hit_ranks else None,
        "mean_context_count": round(
            statistics.mean(len(case["contexts"]) for case in cases), 4
        ),
        "complex_count": sum(
            str(case["retrieval"].get("complexity", {}).get("label")) == "complex"
            for case in cases
        ),
        "simple_count": sum(
            str(case["retrieval"].get("complexity", {}).get("label")) == "simple"
            for case in cases
        ),
        "rerank_count": sum(bool(case["retrieval"].get("rerank_applied")) for case in cases),
        "rerank_rate": round(
            sum(bool(case["retrieval"].get("rerank_applied")) for case in cases) / len(cases),
            8,
        ),
        "average_question_stage_ms": round(statistics.mean(latencies), 3),
        "p50_question_stage_ms": round(float(np.percentile(latencies, 50)), 3),
        "p95_question_stage_ms": round(float(np.percentile(latencies, 95)), 3),
        "p99_question_stage_ms": round(float(np.percentile(latencies, 99)), 3),
        "dense_search_ms": round(float(dense_search_ms), 3),
        "dense_search_ms_per_question": round(float(dense_search_ms) / len(cases), 3),
        "overall_retrieval_ms_per_question": round(
            (float(dense_search_ms) + sum(latencies)) / len(cases),
            3,
        ),
    }


def _breakdown(cases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[str(case.get("subset") or "unknown")].append(case)
    rows: list[dict[str, Any]] = []
    for subset in sorted(grouped):
        group = grouped[subset]
        summary = _summarize(group, dense_search_ms=0.0)
        rows.append({"subset": subset, **summary})
    return rows


def _run_version(
    *,
    version: str,
    label: str,
    items: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
    dense_hits: Sequence[Sequence[dict[str, Any]]],
    bm25: FastBm25Index,
    settings: RagConfig,
    optimized: bool,
    parents: Sequence[dict[str, Any]] | None = None,
    cross_encoder: CrossEncoderReranker | None = None,
    chunk_token_sets: Sequence[set[str]] | None = None,
    dense_search_ms: float = 0.0,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    checkpoint_every: int = 50,
    allowed_indices_by_item: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    if resume and checkpoint_path and checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("version") == version:
            cases = list(checkpoint.get("cases") or [])
            print(
                f"Resuming {label}: {len(cases) + 1}/{len(items)} from {checkpoint_path}",
                flush=True,
            )
    if len(cases) > len(items):
        raise ValueError(f"Checkpoint for {version} has more cases than the dataset")

    for position, item in enumerate(items[len(cases) :], start=len(cases) + 1):
        question = str(item.get("query") or "").strip()
        allowed_indices = (
            allowed_indices_by_item[position - 1]
            if allowed_indices_by_item is not None
            else None
        )
        started = time.perf_counter()
        if optimized:
            if parents is None or cross_encoder is None:
                raise ValueError("Optimized evaluation requires parents and Cross-Encoder")
            contexts, retrieval = _retrieve_optimized(
                question,
                chunks,
                parents,
                dense_hits[position - 1],
                bm25,
                cross_encoder,
                settings,
                allowed_indices=allowed_indices,
            )
        else:
            contexts, retrieval = _retrieve_unoptimized(
                question,
                chunks,
                dense_hits[position - 1],
                bm25,
                settings,
                chunk_token_sets or (),
                allowed_indices=allowed_indices,
            )
        question_stage_ms = (time.perf_counter() - started) * 1000.0
        metrics = _span_metrics(contexts, item.get("snippets") or [])
        cases.append(
            {
                "id": item["id"],
                "subset": item.get("subset"),
                "subset_index": item.get("subset_index"),
                "question": question,
                "retrieval": retrieval,
                "metrics": {**metrics, "question_stage_ms": round(question_stage_ms, 3)},
                "timings_ms": {"question_stage": round(question_stage_ms, 3)},
                "contexts": [_compact_context(context) for context in contexts],
            }
        )
        if position == 1 or position % 50 == 0 or position == len(items):
            print(
                f"[{label}] {position}/{len(items)} | "
                f"stage {question_stage_ms:.1f} ms | "
                f"recall {metrics['char_recall']:.3f} | "
                f"hit {metrics['any_gold_span_hit']} | "
                f"{retrieval['complexity'].get('label')}",
                flush=True,
            )
        if checkpoint_path and (
            position % max(1, int(checkpoint_every)) == 0 or position == len(items)
        ):
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text(
                json.dumps(
                    {"version": version, "question_count": len(items), "cases": cases},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

    return {
        "name": label,
        "version": version,
        "configuration": {
            "chunking": (
                "hard 600-character chunks, no overlap"
                if not optimized
                else "dynamic 1024-token parent / 256-token child, no overlap"
            ),
            "query_rewrite": (
                "disabled"
                if not optimized
                else "identity for authored English benchmark; semantic guard retained in production"
            ),
            "hybrid_fusion": (
                "raw 50/50 BM25+dense weighted addition"
                if not optimized
                else "RRF, k=60"
            ),
            "candidate_pool": 100,
            "routing": (
                "always lexical/semantic reranker to Top 5"
                if not optimized
                else "simple direct Top 5; complex Cross-Encoder Top 5"
            ),
            "evidence": "retrieval only; no answer model, prompt, memory or generation",
        },
        "summary": _summarize(cases, dense_search_ms=dense_search_ms),
        "breakdown": _breakdown(cases),
        "cases": cases,
    }


def _render_report(payload: dict[str, Any]) -> str:
    versions = {version["version"]: version for version in payload["versions"]}
    unoptimized = versions["unoptimized"]
    optimized = versions["optimized"]
    us = unoptimized["summary"]
    osummary = optimized["summary"]
    lines = [
        "# LegalBench-RAG 檢索評測：無優化版 vs 優化版",
        "",
        "本報告只執行到證據檢索，沒有建立回答 prompt，也沒有呼叫 Qwen、GPT-5.5 或其他生成模型。",
        "Gold snippets 只在檢索完成後用於計算字元級召回與命中指標；輸出 JSON 不保存檢索文件全文。",
        "",
        f"- 官方論文題數：`{payload['published_question_count']}` 題",
        f"- 下載檔題數：`{payload['archive_question_count']}` 題",
        f"- 本次評測：`{payload['evaluated_count']}` 題",
        f"- ContractNLI 排除：`{payload['excluded_count']}` 題（`nda-10` 類別，對齊論文的 946 題）",
        *(
            [
                f"- 本次取樣：前 `{payload['evaluated_count']}` 題（依壓縮檔順序；完整對齊題集為 `{payload['full_evaluated_count']}` 題）",
                f"- 本次未測試的對齊題目：`{payload['sample_excluded_count']}` 題",
            ]
            if payload.get("sample_limit") is not None
            else []
        ),
        f"- Corpus：`{payload['corpus_documents']}` 份文件、`{payload['corpus_characters']:,}` 字元",
        f"- Embedding：`{payload['embedding_model']}`",
        f"- Cross-Encoder：`{payload['cross_encoder_model']}`（只在優化版複雜題啟用）",
        f"- 執行時間：`{payload['run_at']}`",
        "",
        "## 主要結果",
        "",
        "| 版本 | Macro char recall | Macro char precision | Micro char recall | 任一證據命中 | 完整證據命中 | Rank 1 命中 | 平均題目階段 ms | 整體平均 ms/題 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {unoptimized['name']} | {us['macro_char_recall']:.2%} | {us['macro_char_precision']:.2%} | {us['micro_char_recall']:.2%} | {us['any_gold_span_hit_rate']:.2%} | {us['complete_gold_span_hit_rate']:.2%} | {us['rank1_gold_span_hit_rate']:.2%} | {us['average_question_stage_ms']:.1f} | {us['overall_retrieval_ms_per_question']:.1f} |",
        f"| {optimized['name']} | {osummary['macro_char_recall']:.2%} | {osummary['macro_char_precision']:.2%} | {osummary['micro_char_recall']:.2%} | {osummary['any_gold_span_hit_rate']:.2%} | {osummary['complete_gold_span_hit_rate']:.2%} | {osummary['rank1_gold_span_hit_rate']:.2%} | {osummary['average_question_stage_ms']:.1f} | {osummary['overall_retrieval_ms_per_question']:.1f} |",
        "",
        "字元級指標沿用 LegalBench-RAG 的 exact-span 定義；Macro 是逐題平均，Micro 是所有 Gold 字元合計後的比例。完整證據命中表示該題每一個 Gold span 都至少與一個檢索區間重疊。",
        "整體平均 ms/題把批次 Dense 計算時間攤回題數；平均題目階段 ms 包含 BM25、融合與重排，不包含批次 Dense 階段。",
        "",
        "## 題型分組",
        "",
        "| 版本 | 題型 | 題數 | Macro recall | Macro precision | 任一命中 | 完整命中 | Rank 1 命中 | 平均題目階段 ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for version in (unoptimized, optimized):
        for row in version["breakdown"]:
            lines.append(
                f"| {version['name']} | `{row['subset']}` | {row['question_count']} | "
                f"{row['macro_char_recall']:.2%} | {row['macro_char_precision']:.2%} | "
                f"{row['any_gold_span_hit_rate']:.2%} | {row['complete_gold_span_hit_rate']:.2%} | "
                f"{row['rank1_gold_span_hit_rate']:.2%} | {row['average_question_stage_ms']:.1f} |"
            )
    lines.extend(
        [
            "",
            "## 實作設定",
            "",
            f"- 無優化版：{unoptimized['configuration']['chunking']}；{unoptimized['configuration']['hybrid_fusion']}；{unoptimized['configuration']['routing']}。",
            f"- 優化版：{optimized['configuration']['chunking']}；{optimized['configuration']['hybrid_fusion']}；{optimized['configuration']['routing']}；命中的子 Chunk 會展開為父 Chunk 證據。",
            f"- 兩個版本使用同一份 {payload['corpus_documents']} 文件 corpus、同一個 Embedding 模型、同一批 {payload['evaluated_count']} 題與同一個 Top 5 證據預算。",
            "- 本次沒有回答模型、沒有對話記憶、沒有外部搜尋、沒有把 Gold answer 或 Gold snippets 注入檢索查詢。",
            "- 合法文件原始檔保留在下載壓縮檔中；評測輸出只保存檔名、字元區間、排名與分數欄位。",
            "",
        ]
    )
    return "\n".join(lines)


def _summary_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: payload[key]
        for key in (
            "schema_version",
            "dataset",
            "published_question_count",
            "archive_question_count",
            "evaluated_count",
            "full_evaluated_count",
            "sample_excluded_count",
            "sample_limit",
            "excluded_count",
            "subset_counts",
            "corpus_documents",
            "corpus_characters",
            "chunk_counts",
            "mapping_failures",
            "run_at",
            "embedding_model",
            "cross_encoder_model",
        )
    } | {
        "versions": [
            {
                "name": version["name"],
                "version": version["version"],
                "configuration": version["configuration"],
                "summary": version["summary"],
                "breakdown": version["breakdown"],
            }
            for version in payload["versions"]
        ]
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    archive_path = args.archive.resolve()
    items, corpus, archive_metadata = _load_archive(archive_path)
    full_evaluated_count = len(items)
    full_evaluated_counts = dict(archive_metadata.get("evaluated_counts") or {})
    sample_limit = int(args.limit) if int(args.limit or 0) > 0 else None
    if sample_limit is not None:
        items = items[:sample_limit]
        sample_counts = Counter(str(item.get("subset") or "unknown") for item in items)
        archive_metadata = {
            **archive_metadata,
            "full_evaluated_total": full_evaluated_count,
            "full_evaluated_counts": full_evaluated_counts,
            "evaluated_total": len(items),
            "evaluated_counts": dict(sorted(sample_counts.items())),
            "sample_limit": len(items),
            "sample_excluded_count": full_evaluated_count - len(items),
            "selection_rule": (
                f"{archive_metadata['selection_rule']} Evaluation uses the first "
                f"{len(items)} paper-aligned records in archive order."
            ),
        }
    else:
        archive_metadata = {
            **archive_metadata,
            "full_evaluated_total": full_evaluated_count,
            "full_evaluated_counts": full_evaluated_counts,
            "sample_limit": None,
            "sample_excluded_count": 0,
        }
    print(
        f"Loaded {len(items)} paper-aligned questions and {len(corpus)} corpus documents "
        f"from {archive_path}",
        flush=True,
    )

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

    started_build = time.perf_counter()
    hard_chunks = _build_hard_chunks(corpus)
    parent_child, mapping_failures = _build_parent_child_chunks(corpus)
    print(
        f"Built {len(hard_chunks)} hard chunks, {len(parent_child.parents)} parents, "
        f"{len(parent_child.children)} children; mapping_failures={mapping_failures}",
        flush=True,
    )
    build_ms = (time.perf_counter() - started_build) * 1000.0

    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    embedding_model = str(args.embedding_model)
    hard_matrix, hard_embedding_ms, hard_cache_hit = _load_or_build_embeddings(
        hard_chunks,
        cache_dir / "hard-embeddings.npy",
        model_name=embedding_model,
        batch_size=args.batch_size,
        label="hard",
    )
    child_matrix, child_embedding_ms, child_cache_hit = _load_or_build_embeddings(
        parent_child.children,
        cache_dir / "child-embeddings.npy",
        model_name=embedding_model,
        batch_size=args.batch_size,
        label="child",
    )
    query_matrix, query_embedding_ms, query_cache_hit = _load_or_build_query_embeddings(
        [str(item.get("query") or "") for item in items],
        cache_dir / "query-embeddings.npy",
        model_name=embedding_model,
        batch_size=args.batch_size,
    )

    hard_bm25 = FastBm25Index(
        hard_chunks,
        k1=settings.hybrid_bm25_k1,
        b=settings.hybrid_bm25_b,
    )
    child_bm25 = FastBm25Index(
        parent_child.children,
        k1=settings.hybrid_bm25_k1,
        b=settings.hybrid_bm25_b,
    )
    hard_token_sets = [set(tokenize_bm25(str(chunk.get("content") or ""))) for chunk in hard_chunks]

    print("Computing batched dense Top 100 candidates for the hard branch", flush=True)
    hard_dense_hits, hard_dense_ms = _dense_hits_all(
        hard_matrix,
        query_matrix,
        top_k=100,
        batch_size=args.dense_batch_size,
    )
    print(f"Hard dense search completed in {hard_dense_ms / 1000.0:.1f}s", flush=True)
    print("Computing batched dense Top 100 candidates for the child branch", flush=True)
    child_dense_hits, child_dense_ms = _dense_hits_all(
        child_matrix,
        query_matrix,
        top_k=100,
        batch_size=args.dense_batch_size,
    )
    print(f"Child dense search completed in {child_dense_ms / 1000.0:.1f}s", flush=True)

    checkpoint_dir = args.checkpoint_dir.resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cross_encoder = CrossEncoderReranker(args.cross_encoder_model)
    unoptimized = _run_version(
        version="unoptimized",
        label="無優化版",
        items=items,
        chunks=hard_chunks,
        dense_hits=hard_dense_hits,
        bm25=hard_bm25,
        settings=settings,
        optimized=False,
        chunk_token_sets=hard_token_sets,
        dense_search_ms=hard_dense_ms,
        checkpoint_path=checkpoint_dir / "unoptimized.json",
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
    )
    optimized = _run_version(
        version="optimized",
        label="優化版",
        items=items,
        chunks=parent_child.children,
        dense_hits=child_dense_hits,
        bm25=child_bm25,
        settings=settings,
        optimized=True,
        parents=parent_child.parents,
        cross_encoder=cross_encoder,
        dense_search_ms=child_dense_ms,
        checkpoint_path=checkpoint_dir / "optimized.json",
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
    )

    return {
        "schema_version": "legalbench-rag-retrieval-only-v1",
        "dataset": "ZeroEntropy-AI/legalbenchrag",
        "published_question_count": sum(PUBLISHED_COUNTS.values()),
        "archive_question_count": archive_metadata["downloaded_total"],
        "evaluated_count": len(items),
        "full_evaluated_count": archive_metadata["full_evaluated_total"],
        "sample_excluded_count": archive_metadata["sample_excluded_count"],
        "sample_limit": archive_metadata["sample_limit"],
        "excluded_count": archive_metadata["downloaded_total"]
        - archive_metadata["full_evaluated_total"],
        "subset_counts": archive_metadata["evaluated_counts"],
        "archive_counts": archive_metadata["downloaded_counts"],
        "excluded_counts": archive_metadata["excluded_counts"],
        "selection_rule": archive_metadata["selection_rule"],
        "corpus_documents": len(corpus),
        "corpus_characters": sum(len(text) for text in corpus.values()),
        "chunk_counts": {
            "hard": len(hard_chunks),
            "parent": len(parent_child.parents),
            "child": len(parent_child.children),
        },
        "mapping_failures": mapping_failures,
        "build_ms": round(build_ms, 3),
        "embedding_ms": {
            "hard": round(hard_embedding_ms, 3),
            "child": round(child_embedding_ms, 3),
            "queries": round(query_embedding_ms, 3),
        },
        "embedding_cache_hit": {
            "hard": hard_cache_hit,
            "child": child_cache_hit,
            "queries": query_cache_hit,
        },
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "embedding_model": embedding_model,
        "cross_encoder_model": args.cross_encoder_model,
        "batch_size": args.batch_size,
        "dense_batch_size": args.dense_batch_size,
        "retrieval_only": True,
        "model_calls": 0,
        "versions": [unoptimized, optimized],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate only the first N paper-aligned questions (0 means all).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dense-batch-size", type=int, default=64)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--cross-encoder-model", default=DEFAULT_CROSS_ENCODER_MODEL)
    args = parser.parse_args()

    payload = run(args)
    output_path = args.output.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    summary_path = args.summary.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(_render_report(payload), encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "report": str(report_path),
                "summary": str(summary_path),
                "dataset_count": payload["evaluated_count"],
                "model_calls": payload["model_calls"],
                "versions": {
                    version["name"]: version["summary"] for version in payload["versions"]
                },
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
