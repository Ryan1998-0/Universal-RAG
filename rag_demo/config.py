import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional


@dataclass(frozen=True)
class RagConfig:
    top_k: int = 5
    chunk_size: int = 1200
    chunk_stride: int = 600
    retrieval_candidate_k: int = 50
    keyword_weight: float = 0.3
    embedding_weight: float = 0.7
    metadata_boost_max: float = 0.18
    verifier_auto_accept_enabled: bool = False
    verifier_auto_accept_score: float = 0.52
    verifier_auto_reject_score: float = 0.15
    verifier_min_keyword_score: float = 8.0
    verifier_min_embedding_score: float = 0.35
    verifier_context_chars: int = 600
    hybrid_top_k: int = 8
    hybrid_candidate_k: int = 24
    hybrid_max_top_k: int = 12
    hybrid_max_candidate_k: int = 80
    hybrid_rrf_k: int = 60
    hybrid_bm25_k1: float = 1.4
    hybrid_bm25_b: float = 0.72
    hybrid_min_bm25_score: float = 1.0
    hybrid_min_embedding_score: float = 0.32
    hybrid_high_embedding_score: float = 0.48
    rerank_fusion_weight: float = 0.30
    rerank_bm25_weight: float = 0.32
    rerank_embedding_weight: float = 0.26
    rerank_coverage_weight: float = 0.10
    rerank_phrase_weight: float = 0.02

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "RagConfig":
        values = env or os.environ
        return cls(
            top_k=_int_value(values, "RAG_TOP_K", cls.top_k),
            chunk_size=_int_value(values, "RAG_CHUNK_SIZE", cls.chunk_size),
            chunk_stride=_int_value(values, "RAG_CHUNK_STRIDE", cls.chunk_stride),
            retrieval_candidate_k=_int_value(
                values,
                "RAG_RETRIEVAL_CANDIDATE_K",
                cls.retrieval_candidate_k,
            ),
            keyword_weight=_float_value(values, "RAG_KEYWORD_WEIGHT", cls.keyword_weight),
            embedding_weight=_float_value(values, "RAG_EMBEDDING_WEIGHT", cls.embedding_weight),
            metadata_boost_max=_float_value(values, "RAG_METADATA_BOOST_MAX", cls.metadata_boost_max),
            verifier_auto_accept_enabled=_bool_value(
                values,
                "RAG_VERIFIER_AUTO_ACCEPT_ENABLED",
                cls.verifier_auto_accept_enabled,
            ),
            verifier_auto_accept_score=_float_value(
                values,
                "RAG_VERIFIER_AUTO_ACCEPT_SCORE",
                cls.verifier_auto_accept_score,
            ),
            verifier_auto_reject_score=_float_value(
                values,
                "RAG_VERIFIER_AUTO_REJECT_SCORE",
                cls.verifier_auto_reject_score,
            ),
            verifier_min_keyword_score=_float_value(
                values,
                "RAG_VERIFIER_MIN_KEYWORD_SCORE",
                cls.verifier_min_keyword_score,
            ),
            verifier_min_embedding_score=_float_value(
                values,
                "RAG_VERIFIER_MIN_EMBEDDING_SCORE",
                cls.verifier_min_embedding_score,
            ),
            verifier_context_chars=_int_value(
                values,
                "RAG_VERIFIER_CONTEXT_CHARS",
                cls.verifier_context_chars,
            ),
            hybrid_top_k=_int_value(values, "RAG_HYBRID_TOP_K", cls.hybrid_top_k),
            hybrid_candidate_k=_int_value(
                values,
                "RAG_HYBRID_CANDIDATE_K",
                cls.hybrid_candidate_k,
            ),
            hybrid_max_top_k=_int_value(
                values,
                "RAG_HYBRID_MAX_TOP_K",
                cls.hybrid_max_top_k,
            ),
            hybrid_max_candidate_k=_int_value(
                values,
                "RAG_HYBRID_MAX_CANDIDATE_K",
                cls.hybrid_max_candidate_k,
            ),
            hybrid_rrf_k=_int_value(values, "RAG_HYBRID_RRF_K", cls.hybrid_rrf_k),
            hybrid_bm25_k1=_float_value(
                values,
                "RAG_HYBRID_BM25_K1",
                cls.hybrid_bm25_k1,
            ),
            hybrid_bm25_b=_float_value(
                values,
                "RAG_HYBRID_BM25_B",
                cls.hybrid_bm25_b,
            ),
            hybrid_min_bm25_score=_float_value(
                values,
                "RAG_HYBRID_MIN_BM25_SCORE",
                cls.hybrid_min_bm25_score,
            ),
            hybrid_min_embedding_score=_float_value(
                values,
                "RAG_HYBRID_MIN_EMBEDDING_SCORE",
                cls.hybrid_min_embedding_score,
            ),
            hybrid_high_embedding_score=_float_value(
                values,
                "RAG_HYBRID_HIGH_EMBEDDING_SCORE",
                cls.hybrid_high_embedding_score,
            ),
            rerank_fusion_weight=_float_value(
                values,
                "RAG_RERANK_FUSION_WEIGHT",
                cls.rerank_fusion_weight,
            ),
            rerank_bm25_weight=_float_value(
                values,
                "RAG_RERANK_BM25_WEIGHT",
                cls.rerank_bm25_weight,
            ),
            rerank_embedding_weight=_float_value(
                values,
                "RAG_RERANK_EMBEDDING_WEIGHT",
                cls.rerank_embedding_weight,
            ),
            rerank_coverage_weight=_float_value(
                values,
                "RAG_RERANK_COVERAGE_WEIGHT",
                cls.rerank_coverage_weight,
            ),
            rerank_phrase_weight=_float_value(
                values,
                "RAG_RERANK_PHRASE_WEIGHT",
                cls.rerank_phrase_weight,
            ),
        ).normalized()

    def normalized(self) -> "RagConfig":
        chunk_size = max(100, int(self.chunk_size))
        chunk_stride = min(max(1, int(self.chunk_stride)), chunk_size)
        keyword_weight = max(0.0, float(self.keyword_weight))
        embedding_weight = max(0.0, float(self.embedding_weight))
        total = keyword_weight + embedding_weight
        if total <= 0:
            keyword_weight, embedding_weight = 0.3, 0.7
        else:
            keyword_weight = keyword_weight / total
            embedding_weight = embedding_weight / total
        rerank_weights = [
            max(0.0, float(self.rerank_fusion_weight)),
            max(0.0, float(self.rerank_bm25_weight)),
            max(0.0, float(self.rerank_embedding_weight)),
            max(0.0, float(self.rerank_coverage_weight)),
            max(0.0, float(self.rerank_phrase_weight)),
        ]
        rerank_total = sum(rerank_weights)
        if rerank_total <= 0:
            rerank_weights = [0.30, 0.32, 0.26, 0.10, 0.02]
            rerank_total = 1.0
        rerank_weights = [weight / rerank_total for weight in rerank_weights]
        hybrid_max_top_k = max(1, int(self.hybrid_max_top_k))
        hybrid_max_candidate_k = max(hybrid_max_top_k, int(self.hybrid_max_candidate_k))
        return RagConfig(
            top_k=max(1, int(self.top_k)),
            chunk_size=chunk_size,
            chunk_stride=chunk_stride,
            retrieval_candidate_k=max(1, int(self.retrieval_candidate_k)),
            keyword_weight=keyword_weight,
            embedding_weight=embedding_weight,
            metadata_boost_max=max(0.0, float(self.metadata_boost_max)),
            verifier_auto_accept_enabled=bool(self.verifier_auto_accept_enabled),
            verifier_auto_accept_score=max(0.0, float(self.verifier_auto_accept_score)),
            verifier_auto_reject_score=max(0.0, float(self.verifier_auto_reject_score)),
            verifier_min_keyword_score=max(0.0, float(self.verifier_min_keyword_score)),
            verifier_min_embedding_score=max(0.0, float(self.verifier_min_embedding_score)),
            verifier_context_chars=max(100, int(self.verifier_context_chars)),
            hybrid_top_k=min(max(1, int(self.hybrid_top_k)), hybrid_max_top_k),
            hybrid_candidate_k=min(
                max(1, int(self.hybrid_candidate_k)),
                hybrid_max_candidate_k,
            ),
            hybrid_max_top_k=hybrid_max_top_k,
            hybrid_max_candidate_k=hybrid_max_candidate_k,
            hybrid_rrf_k=max(1, int(self.hybrid_rrf_k)),
            hybrid_bm25_k1=max(0.01, float(self.hybrid_bm25_k1)),
            hybrid_bm25_b=min(max(0.0, float(self.hybrid_bm25_b)), 1.0),
            hybrid_min_bm25_score=max(0.0, float(self.hybrid_min_bm25_score)),
            hybrid_min_embedding_score=min(
                max(-1.0, float(self.hybrid_min_embedding_score)),
                1.0,
            ),
            hybrid_high_embedding_score=min(
                max(float(self.hybrid_min_embedding_score), float(self.hybrid_high_embedding_score)),
                1.0,
            ),
            rerank_fusion_weight=rerank_weights[0],
            rerank_bm25_weight=rerank_weights[1],
            rerank_embedding_weight=rerank_weights[2],
            rerank_coverage_weight=rerank_weights[3],
            rerank_phrase_weight=rerank_weights[4],
        )


def _int_value(values: Mapping[str, str], key: str, default: int) -> int:
    try:
        return int(values.get(key, str(default)))
    except (TypeError, ValueError):
        return default


def _float_value(values: Mapping[str, str], key: str, default: float) -> float:
    try:
        return float(values.get(key, str(default)))
    except (TypeError, ValueError):
        return default


def _bool_value(values: Mapping[str, str], key: str, default: bool) -> bool:
    value = values.get(key)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def resolve_project_path(
    env_key: str,
    default_relative_path: str,
    env: Optional[Mapping[str, str]] = None,
    project_root: Optional[Path] = None,
) -> Path:
    values = env or os.environ
    configured = values.get(env_key)
    if configured:
        return Path(configured).expanduser()

    root = project_root or Path(__file__).resolve().parents[1]
    return root / default_relative_path


def resolve_profile_path(
    env_key: str,
    default_relative_path: str,
    profile_relative_path: str,
    env: Optional[Mapping[str, str]] = None,
    project_root: Optional[Path] = None,
) -> Path:
    values = env or os.environ
    configured = values.get(env_key)
    if configured:
        return Path(configured).expanduser()

    root = project_root or Path(__file__).resolve().parents[1]
    profile = values.get("RAG_PROFILE")
    if profile:
        return root / "profiles" / profile / profile_relative_path

    return root / default_relative_path
