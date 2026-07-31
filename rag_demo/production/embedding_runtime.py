from __future__ import annotations

import threading
from typing import Iterable, Sequence


class FastEmbedRuntime:
    """Lazy ONNX runtime for dense, BM25 sparse, and cross-encoder scoring."""

    def __init__(
        self,
        *,
        embedding_model: str,
        sparse_model: str,
        reranker_model: str,
        cache_dir: str | None = None,
    ):
        self.embedding_model = embedding_model
        self.sparse_model = sparse_model
        self.reranker_model = reranker_model
        self.cache_dir = cache_dir
        self._dense = None
        self._sparse = None
        self._reranker = None
        self._load_lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings) -> "FastEmbedRuntime":
        return cls(
            embedding_model=settings.embedding_model,
            sparse_model=settings.sparse_embedding_model,
            reranker_model=settings.reranker_model,
        )

    def embedding_dimensions(self) -> int:
        return int(self._dense_model().embedding_size)

    def embed_query(self, query: str) -> list[float]:
        vector = next(iter(self._dense_model().query_embed([str(query)])))
        return [float(value) for value in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [
            [float(value) for value in vector]
            for vector in self._dense_model().passage_embed(list(texts))
        ]

    def sparse_query(self, query: str):
        return next(iter(self._sparse_model().query_embed([str(query)])))

    def sparse_documents(self, texts: Sequence[str]) -> list[object]:
        return list(self._sparse_model().embed(list(texts)))

    def rerank(self, query: str, documents: Iterable[str]) -> list[float]:
        return [
            float(score)
            for score in self._reranker_model().rerank(str(query), list(documents))
        ]

    def _dense_model(self):
        if self._dense is None:
            with self._load_lock:
                if self._dense is None:
                    TextEmbedding, _, _ = _fastembed_types()
                    self._dense = TextEmbedding(
                        model_name=self.embedding_model,
                        cache_dir=self.cache_dir,
                    )
        return self._dense

    def _sparse_model(self):
        if self._sparse is None:
            with self._load_lock:
                if self._sparse is None:
                    _, SparseTextEmbedding, _ = _fastembed_types()
                    self._sparse = SparseTextEmbedding(
                        model_name=self.sparse_model,
                        cache_dir=self.cache_dir,
                    )
        return self._sparse

    def _reranker_model(self):
        if self._reranker is None:
            with self._load_lock:
                if self._reranker is None:
                    _, _, TextCrossEncoder = _fastembed_types()
                    self._reranker = TextCrossEncoder(
                        model_name=self.reranker_model,
                        cache_dir=self.cache_dir,
                    )
        return self._reranker


def _fastembed_types():
    try:
        from fastembed import SparseTextEmbedding, TextEmbedding
        from fastembed.rerank.cross_encoder import TextCrossEncoder
    except ImportError as exc:
        raise RuntimeError("fastembed is required for the production retrieval runtime") from exc
    return TextEmbedding, SparseTextEmbedding, TextCrossEncoder
