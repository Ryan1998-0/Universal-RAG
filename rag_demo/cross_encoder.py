"""Lazy Cross-Encoder adapter shared by benchmarks and local retrieval."""

from __future__ import annotations

import os
import threading
from typing import Iterable, Sequence


DEFAULT_CROSS_ENCODER_MODEL = os.getenv(
    "RAG_RERANKER_MODEL",
    "BAAI/bge-reranker-base",
)


class CrossEncoderReranker:
    """Score query/document pairs with fastembed's ONNX Cross-Encoder."""

    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER_MODEL, cache_dir: str | None = None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None
        self._lock = threading.Lock()

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        model = self._load()
        # A larger batch reduces tokenizer/ONNX session overhead for the
        # retrieval benchmark's fixed 100-candidate rerank pool.  Keep the
        # default conservative for interactive requests and allow deployments
        # to tune it without changing the public adapter API.
        try:
            batch_size = max(1, int(os.getenv("RAG_CROSS_ENCODER_BATCH_SIZE", "64")))
        except ValueError:
            batch_size = 64
        return [
            float(value)
            for value in model.rerank(
                str(query),
                list(documents),
                batch_size=batch_size,
            )
        ]

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from fastembed.rerank.cross_encoder import TextCrossEncoder
                    except ImportError as exc:
                        raise RuntimeError(
                            "fastembed is required for Cross-Encoder reranking; install fastembed."
                        ) from exc
                    self._model = TextCrossEncoder(
                        model_name=self.model_name,
                        cache_dir=self.cache_dir,
                    )
        return self._model


def cross_encoder_scores(
    query: str,
    documents: Iterable[str],
    *,
    model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
    cache_dir: str | None = None,
) -> list[float]:
    """Convenience function with a process-local lazy model cache."""

    key = (str(model_name), str(cache_dir or ""))
    reranker = _RERANKER_CACHE.setdefault(key, CrossEncoderReranker(*key))
    return reranker.score(query, list(documents))


_RERANKER_CACHE: dict[tuple[str, str], CrossEncoderReranker] = {}
