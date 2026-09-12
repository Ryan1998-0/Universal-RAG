"""Rank-aware score calibration for hybrid retrieval.

LambdaMART is a learning-to-rank algorithm.  A trained LambdaMART model is
normally fitted with query/document relevance labels, while this project must
also work when no labelled ranking model has been supplied.  This module keeps
the fusion contract independent from the model implementation:

* branch scores are calibrated independently into the common ``[0, 1]`` space;
* calibrated keyword and dense scores are then combined with configured
  weights; and
* an optional future/production LambdaMART model can replace the fallback
  calibrator without changing the retriever API.

The default calibrator is an empirical rank map.  It is monotonic, robust to
BM25/cosine scale differences, and has no training or network latency.  It is
deliberately labelled as a LambdaMART-compatible fallback rather than
pretending that an untrained ranker is a learned model.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Iterable, Mapping, Sequence


LAMBDA_MART_FUSION_METHOD = "lambdamart_score_mapping_v1"


@dataclass(frozen=True)
class LambdaMARTFusionConfig:
    """Configuration used by the score mapping layer."""

    keyword_weight: float = 0.50
    embedding_weight: float = 0.50
    model_path: str = ""


class LambdaMARTScoreMapper:
    """Map one retrieval branch into a comparable relevance dimension.

    ``model`` is an optional adapter for a trained LambdaMART-compatible
    object.  It must expose ``predict(features)`` and return one score per
    feature row.  The default rank map is used when no model is configured or
    a model cannot be loaded, so an unavailable optional ranker never disables
    retrieval.
    """

    def __init__(self, model: Any = None):
        self.model = model

    def map_scores(self, scores: Sequence[float], branch: str) -> list[float]:
        values = [_finite_score(score) for score in scores]
        if not values:
            return []

        if self.model is not None:
            try:
                mapped = self._predict_model(values, branch)
                if len(mapped) == len(values):
                    return _min_max_scale(mapped)
            except Exception:
                # The optional learned model is an enhancement.  Retrieval
                # remains available through the deterministic fallback.
                pass

        return _rank_map(values)

    def _predict_model(self, values: Sequence[float], branch: str) -> list[float]:
        order = _rank_positions(values)
        branch_flag = 1.0 if branch == "bm25" else 0.0
        features = [
            [float(raw), float(rank), branch_flag]
            for raw, rank in zip(values, order)
        ]
        predictions = self.model.predict(features)
        return [float(value) for value in predictions]


def fuse_candidates_with_lambdamart(
    candidates: Sequence[Mapping[str, Any]],
    keyword_weight: float = 0.50,
    embedding_weight: float = 0.50,
    mapper: LambdaMARTScoreMapper | None = None,
) -> list[dict]:
    """Calibrate sparse/dense scores and add them in the common space.

    The returned dictionaries preserve the original branch scores and add
    ``mapped_bm25_score``, ``mapped_embedding_score`` and
    ``lambda_mart_fusion_score``.  The old ``fusion_score``/``rrf_score``
    values remain available as rank-union diagnostics and tie-break signals.
    """

    items = [dict(candidate) for candidate in candidates]
    if not items:
        return []

    keyword_weight, embedding_weight = _normalize_weights(
        keyword_weight,
        embedding_weight,
    )
    mapper = mapper or LambdaMARTScoreMapper()
    keyword_scores = [
        item.get(
            "bm25_score",
            item.get("sparse_score", item.get("keyword_score", 0.0)),
        )
        for item in items
    ]
    embedding_scores = [
        item.get("embedding_score", item.get("dense_score", 0.0))
        for item in items
    ]
    mapped_keyword = mapper.map_scores(keyword_scores, branch="bm25")
    mapped_embedding = mapper.map_scores(embedding_scores, branch="embedding")

    for item, keyword, embedding in zip(items, mapped_keyword, mapped_embedding):
        item["mapped_bm25_score"] = round(float(keyword), 8)
        item["mapped_embedding_score"] = round(float(embedding), 8)
        item["lambda_mart_fusion_score"] = round(
            keyword_weight * float(keyword) + embedding_weight * float(embedding),
            8,
        )
        # ``score_fusion`` is an explicit neutral name for consumers that do
        # not need to know whether a learned adapter or fallback was used.
        item["score_fusion"] = item["lambda_mart_fusion_score"]

    return sorted(
        items,
        key=lambda item: (
            float(item.get("lambda_mart_fusion_score", 0.0)),
            float(item.get("fusion_score", item.get("rrf_score", 0.0))),
        ),
        reverse=True,
    )


def _rank_map(values: Sequence[float]) -> list[float]:
    """Map scores to ``[0, 1]`` using average ranks with stable ties."""

    if not values:
        return []
    positive = [value for value in values if value > 0.0]
    if not positive:
        return [0.0 for _ in values]
    if len(values) == 1:
        return [1.0 if values[0] > 0.0 else 0.0]

    sorted_indices = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(sorted_indices):
        end = index + 1
        value = values[sorted_indices[index]]
        while end < len(sorted_indices) and values[sorted_indices[end]] == value:
            end += 1
        # Zero scores represent a branch miss and must remain the floor.
        rank = 0.0 if value <= 0.0 else ((index + 1) + end) / 2.0
        normalized = 0.0 if rank == 0.0 else (rank - 1.0) / (len(values) - 1.0)
        for sorted_index in sorted_indices[index:end]:
            ranks[sorted_index] = normalized
        index = end
    return ranks


def _rank_positions(values: Sequence[float]) -> list[float]:
    mapped = _rank_map(values)
    return mapped


def _min_max_scale(values: Iterable[float]) -> list[float]:
    numbers = [_finite_score(value) for value in values]
    if not numbers:
        return []
    minimum = min(numbers)
    maximum = max(numbers)
    span = maximum - minimum
    if span <= 1e-12:
        return [1.0 if number > 0.0 else 0.0 for number in numbers]
    return [max(0.0, min(1.0, (number - minimum) / span)) for number in numbers]


def _finite_score(value: Any) -> float:
    try:
        score = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return score if isfinite(score) else 0.0


def _normalize_weights(keyword_weight: float, embedding_weight: float) -> tuple[float, float]:
    try:
        keyword = max(0.0, float(keyword_weight))
    except (TypeError, ValueError):
        keyword = 0.50
    try:
        embedding = max(0.0, float(embedding_weight))
    except (TypeError, ValueError):
        embedding = 0.50
    total = keyword + embedding
    if total <= 0.0:
        return 0.50, 0.50
    return keyword / total, embedding / total
