import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional


@dataclass(frozen=True)
class RagConfig:
    top_k: int = 5
    chunk_size: int = 600
    chunk_stride: int = 600
    # `dynamic` packs complete sentence units up to chunk_size tokens and
    # carries trailing sentences into the next chunk. `hard` is the A/B
    # baseline: fixed character windows with no boundary correction/overlap.
    chunk_strategy: str = "dynamic"
    chunk_overlap_tokens: int = 200
    # Parent-child indexing retrieves precise child windows and expands them
    # back to coherent parent evidence before generation.
    parent_child_enabled: bool = True
    parent_chunk_size_tokens: int = 1024
    child_chunk_size_tokens: int = 256
    parent_chunk_overlap_tokens: int = 0
    child_chunk_overlap_tokens: int = 0
    # Guard model-generated retrieval rewrites by comparing them with the
    # user's original question in embedding space.
    query_rewrite_semantic_enabled: bool = True
    query_rewrite_min_similarity: float = 0.60
    retrieval_candidate_k: int = 100
    keyword_weight: float = 0.6
    embedding_weight: float = 0.4
    metadata_boost_max: float = 0.18
    verifier_auto_accept_enabled: bool = False
    verifier_auto_accept_score: float = 0.52
    verifier_auto_reject_score: float = 0.15
    verifier_min_keyword_score: float = 8.0
    verifier_min_embedding_score: float = 0.35
    verifier_context_chars: int = 600
    hybrid_top_k: int = 5
    # Number of hybrid candidates sent to the second-stage reranker.  The
    # final answer still respects the request's top_k value.
    rerank_top_k: int = 5
    # 32 preserves more answer-bearing candidates than 24 at nearly identical
    # latency on the bundled IFRS 17 retrieval benchmark.
    hybrid_candidate_k: int = 100
    hybrid_max_top_k: int = 12
    hybrid_max_candidate_k: int = 200
    hybrid_rrf_k: int = 60
    # RRF combines BM25 and dense result ranks without assuming their raw
    # scores share a numeric scale. LambdaMART remains available for legacy
    # profiles that explicitly select it.
    hybrid_fusion_method: str = "rrf"
    complexity_routing_enabled: bool = True
    query_complexity_threshold: float = 2.0
    simple_query_top_k: int = 5
    multi_query_enabled: bool = True
    # Two complementary queries preserve the strongest quality/latency balance
    # on the bundled IFRS retrieval benchmark; callers can raise this to 3-4
    # for unusually ambiguous or multi-part questions.
    multi_query_max_variants: int = 2
    hybrid_bm25_k1: float = 1.4
    hybrid_bm25_b: float = 0.72
    hybrid_min_bm25_score: float = 1.0
    hybrid_min_embedding_score: float = 0.32
    hybrid_high_embedding_score: float = 0.48
    # `strict` preserves the original fail-closed gate.  `relaxed` permits a
    # high-signal context outside rank 1 to pass, while still requiring raw
    # lexical/semantic evidence (it is not an evidence-free bypass).
    evidence_gate_mode: str = "strict"
    rerank_fusion_weight: float = 0.30
    rerank_bm25_weight: float = 0.32
    rerank_embedding_weight: float = 0.26
    rerank_coverage_weight: float = 0.10
    rerank_phrase_weight: float = 0.02
    evidence_focus_enabled: bool = False
    evidence_focus_top_k: int = 4
    evidence_focus_max_chars: int = 480
    evidence_focus_keyword_weight: float = 0.80
    evidence_focus_embedding_weight: float = 0.20
    fine_evidence_enabled: bool = False
    fine_evidence_chunk_chars: int = 240
    fine_evidence_chunk_fraction: float = 1.0 / 3.0
    fine_evidence_overlap_chars: int = 32
    fine_evidence_top_k: int = 6
    fine_evidence_max_groups: int = 3
    fine_evidence_keyword_weight: float = 0.55
    fine_evidence_embedding_weight: float = 0.45
    fine_evidence_min_embedding_score: float = 0.45

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "RagConfig":
        values = env or os.environ
        return cls(
            top_k=_int_value(values, "RAG_TOP_K", cls.top_k),
            chunk_size=_int_value(values, "RAG_CHUNK_SIZE", cls.chunk_size),
            chunk_stride=_int_value(values, "RAG_CHUNK_STRIDE", cls.chunk_stride),
            chunk_strategy=str(
                values.get("RAG_CHUNK_STRATEGY", cls.chunk_strategy)
            ).strip().lower(),
            chunk_overlap_tokens=_int_value(
                values,
                "RAG_CHUNK_OVERLAP_TOKENS",
                cls.chunk_overlap_tokens,
            ),
            parent_child_enabled=_bool_value(
                values,
                "RAG_PARENT_CHILD_ENABLED",
                cls.parent_child_enabled,
            ),
            parent_chunk_size_tokens=_int_value(
                values,
                "RAG_PARENT_CHUNK_SIZE_TOKENS",
                cls.parent_chunk_size_tokens,
            ),
            child_chunk_size_tokens=_int_value(
                values,
                "RAG_CHILD_CHUNK_SIZE_TOKENS",
                cls.child_chunk_size_tokens,
            ),
            parent_chunk_overlap_tokens=_int_value(
                values,
                "RAG_PARENT_CHUNK_OVERLAP_TOKENS",
                cls.parent_chunk_overlap_tokens,
            ),
            child_chunk_overlap_tokens=_int_value(
                values,
                "RAG_CHILD_CHUNK_OVERLAP_TOKENS",
                cls.child_chunk_overlap_tokens,
            ),
            query_rewrite_semantic_enabled=_bool_value(
                values,
                "RAG_QUERY_REWRITE_SEMANTIC_ENABLED",
                cls.query_rewrite_semantic_enabled,
            ),
            query_rewrite_min_similarity=_float_value(
                values,
                "RAG_QUERY_REWRITE_MIN_SIMILARITY",
                cls.query_rewrite_min_similarity,
            ),
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
            rerank_top_k=_int_value(values, "RAG_RERANK_TOP_K", cls.rerank_top_k),
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
            hybrid_fusion_method=str(
                values.get("RAG_HYBRID_FUSION_METHOD", cls.hybrid_fusion_method)
            ).strip().lower(),
            complexity_routing_enabled=_bool_value(
                values,
                "RAG_COMPLEXITY_ROUTING_ENABLED",
                cls.complexity_routing_enabled,
            ),
            query_complexity_threshold=_float_value(
                values,
                "RAG_QUERY_COMPLEXITY_THRESHOLD",
                cls.query_complexity_threshold,
            ),
            simple_query_top_k=_int_value(
                values,
                "RAG_SIMPLE_QUERY_TOP_K",
                cls.simple_query_top_k,
            ),
            multi_query_enabled=_bool_value(
                values,
                "RAG_MULTI_QUERY_ENABLED",
                cls.multi_query_enabled,
            ),
            multi_query_max_variants=_int_value(
                values,
                "RAG_MULTI_QUERY_MAX_VARIANTS",
                cls.multi_query_max_variants,
            ),
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
            evidence_gate_mode=str(
                values.get("RAG_EVIDENCE_GATE_MODE", cls.evidence_gate_mode)
            ).strip().lower(),
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
            evidence_focus_enabled=_bool_value(
                values,
                "RAG_EVIDENCE_FOCUS_ENABLED",
                cls.evidence_focus_enabled,
            ),
            evidence_focus_top_k=_int_value(
                values,
                "RAG_EVIDENCE_FOCUS_TOP_K",
                cls.evidence_focus_top_k,
            ),
            evidence_focus_max_chars=_int_value(
                values,
                "RAG_EVIDENCE_FOCUS_MAX_CHARS",
                cls.evidence_focus_max_chars,
            ),
            evidence_focus_keyword_weight=_float_value(
                values,
                "RAG_EVIDENCE_FOCUS_KEYWORD_WEIGHT",
                cls.evidence_focus_keyword_weight,
            ),
            evidence_focus_embedding_weight=_float_value(
                values,
                "RAG_EVIDENCE_FOCUS_EMBEDDING_WEIGHT",
                cls.evidence_focus_embedding_weight,
            ),
            fine_evidence_enabled=_bool_value(
                values,
                "RAG_FINE_EVIDENCE_ENABLED",
                cls.fine_evidence_enabled,
            ),
            fine_evidence_chunk_chars=_int_value(
                values,
                "RAG_FINE_EVIDENCE_CHUNK_CHARS",
                cls.fine_evidence_chunk_chars,
            ),
            fine_evidence_chunk_fraction=_float_value(
                values,
                "RAG_FINE_EVIDENCE_CHUNK_FRACTION",
                cls.fine_evidence_chunk_fraction,
            ),
            fine_evidence_overlap_chars=_int_value(
                values,
                "RAG_FINE_EVIDENCE_OVERLAP_CHARS",
                cls.fine_evidence_overlap_chars,
            ),
            fine_evidence_top_k=_int_value(
                values,
                "RAG_FINE_EVIDENCE_TOP_K",
                cls.fine_evidence_top_k,
            ),
            fine_evidence_max_groups=_int_value(
                values,
                "RAG_FINE_EVIDENCE_MAX_GROUPS",
                cls.fine_evidence_max_groups,
            ),
            fine_evidence_keyword_weight=_float_value(
                values,
                "RAG_FINE_EVIDENCE_KEYWORD_WEIGHT",
                cls.fine_evidence_keyword_weight,
            ),
            fine_evidence_embedding_weight=_float_value(
                values,
                "RAG_FINE_EVIDENCE_EMBEDDING_WEIGHT",
                cls.fine_evidence_embedding_weight,
            ),
            fine_evidence_min_embedding_score=_float_value(
                values,
                "RAG_FINE_EVIDENCE_MIN_EMBEDDING_SCORE",
                cls.fine_evidence_min_embedding_score,
            ),
        ).normalized()

    def normalized(self) -> "RagConfig":
        chunk_size = max(100, int(self.chunk_size))
        chunk_stride = min(max(1, int(self.chunk_stride)), chunk_size)
        keyword_weight = max(0.0, float(self.keyword_weight))
        embedding_weight = max(0.0, float(self.embedding_weight))
        total = keyword_weight + embedding_weight
        if total <= 0:
            keyword_weight, embedding_weight = 0.5, 0.5
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
        evidence_focus_keyword_weight = max(0.0, float(self.evidence_focus_keyword_weight))
        evidence_focus_embedding_weight = max(0.0, float(self.evidence_focus_embedding_weight))
        evidence_focus_total = evidence_focus_keyword_weight + evidence_focus_embedding_weight
        if evidence_focus_total <= 0:
            evidence_focus_keyword_weight, evidence_focus_embedding_weight = 0.80, 0.20
        else:
            evidence_focus_keyword_weight /= evidence_focus_total
            evidence_focus_embedding_weight /= evidence_focus_total
        fine_evidence_keyword_weight = max(0.0, float(self.fine_evidence_keyword_weight))
        fine_evidence_embedding_weight = max(0.0, float(self.fine_evidence_embedding_weight))
        fine_evidence_total = fine_evidence_keyword_weight + fine_evidence_embedding_weight
        if fine_evidence_total <= 0:
            fine_evidence_keyword_weight, fine_evidence_embedding_weight = 0.55, 0.45
        else:
            fine_evidence_keyword_weight /= fine_evidence_total
            fine_evidence_embedding_weight /= fine_evidence_total
        hybrid_max_top_k = max(1, int(self.hybrid_max_top_k))
        hybrid_max_candidate_k = max(hybrid_max_top_k, int(self.hybrid_max_candidate_k))
        parent_chunk_size_tokens = max(128, int(self.parent_chunk_size_tokens))
        child_chunk_size_tokens = max(32, int(self.child_chunk_size_tokens))
        hybrid_fusion_method = str(self.hybrid_fusion_method).strip().lower()
        if hybrid_fusion_method not in {"lambdamart", "rrf"}:
            hybrid_fusion_method = "rrf"
        simple_query_top_k = max(1, int(self.simple_query_top_k))
        hybrid_top_k = min(max(1, int(self.hybrid_top_k)), hybrid_max_top_k)
        rerank_top_k = min(max(1, int(self.rerank_top_k)), hybrid_max_candidate_k)
        if hybrid_fusion_method == "rrf":
            # The RRF + Cross-Encoder route uses the requested five-item
            # answer budget unless a caller explicitly supplies another value.
            simple_query_top_k = max(5, simple_query_top_k)
            if hybrid_top_k == 8:
                hybrid_top_k = 5
        chunk_strategy = str(self.chunk_strategy).strip().lower()
        if chunk_strategy not in {"boundary", "hard", "dynamic"}:
            chunk_strategy = "dynamic"
        return RagConfig(
            top_k=max(1, int(self.top_k)),
            chunk_size=chunk_size,
            chunk_stride=chunk_stride,
            chunk_strategy=chunk_strategy,
            chunk_overlap_tokens=max(0, int(self.chunk_overlap_tokens)),
            parent_child_enabled=bool(self.parent_child_enabled),
            parent_chunk_size_tokens=parent_chunk_size_tokens,
            child_chunk_size_tokens=child_chunk_size_tokens,
            parent_chunk_overlap_tokens=min(
                max(0, int(self.parent_chunk_overlap_tokens)),
                parent_chunk_size_tokens - 1,
            ),
            child_chunk_overlap_tokens=min(
                max(0, int(self.child_chunk_overlap_tokens)),
                child_chunk_size_tokens - 1,
            ),
            query_rewrite_semantic_enabled=bool(self.query_rewrite_semantic_enabled),
            query_rewrite_min_similarity=min(
                max(-1.0, float(self.query_rewrite_min_similarity)),
                1.0,
            ),
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
            hybrid_top_k=hybrid_top_k,
            rerank_top_k=rerank_top_k,
            hybrid_candidate_k=min(
                max(1, int(self.hybrid_candidate_k)),
                hybrid_max_candidate_k,
            ),
            hybrid_max_top_k=hybrid_max_top_k,
            hybrid_max_candidate_k=hybrid_max_candidate_k,
            hybrid_rrf_k=max(1, int(self.hybrid_rrf_k)),
            hybrid_fusion_method=hybrid_fusion_method,
            complexity_routing_enabled=bool(self.complexity_routing_enabled),
            query_complexity_threshold=max(1.0, float(self.query_complexity_threshold)),
            simple_query_top_k=simple_query_top_k,
            multi_query_enabled=bool(self.multi_query_enabled),
            multi_query_max_variants=min(
                max(2, int(self.multi_query_max_variants)),
                8,
            ),
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
            evidence_gate_mode=(
                str(self.evidence_gate_mode).strip().lower()
                if str(self.evidence_gate_mode).strip().lower() in {"strict", "relaxed"}
                else "strict"
            ),
            rerank_fusion_weight=rerank_weights[0],
            rerank_bm25_weight=rerank_weights[1],
            rerank_embedding_weight=rerank_weights[2],
            rerank_coverage_weight=rerank_weights[3],
            rerank_phrase_weight=rerank_weights[4],
            evidence_focus_enabled=bool(self.evidence_focus_enabled),
            evidence_focus_top_k=max(1, int(self.evidence_focus_top_k)),
            evidence_focus_max_chars=max(120, int(self.evidence_focus_max_chars)),
            evidence_focus_keyword_weight=evidence_focus_keyword_weight,
            evidence_focus_embedding_weight=evidence_focus_embedding_weight,
            fine_evidence_enabled=bool(self.fine_evidence_enabled),
            fine_evidence_chunk_chars=max(80, int(self.fine_evidence_chunk_chars)),
            fine_evidence_chunk_fraction=min(
                1.0,
                max(0.05, float(self.fine_evidence_chunk_fraction)),
            ),
            fine_evidence_overlap_chars=max(0, int(self.fine_evidence_overlap_chars)),
            fine_evidence_top_k=max(1, int(self.fine_evidence_top_k)),
            fine_evidence_max_groups=max(1, int(self.fine_evidence_max_groups)),
            fine_evidence_keyword_weight=fine_evidence_keyword_weight,
            fine_evidence_embedding_weight=fine_evidence_embedding_weight,
            fine_evidence_min_embedding_score=min(
                max(-1.0, float(self.fine_evidence_min_embedding_score)),
                1.0,
            ),
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
    values = os.environ if env is None else env
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
    values = os.environ if env is None else env
    configured = values.get(env_key)
    if configured:
        return Path(configured).expanduser()

    root = project_root or Path(__file__).resolve().parents[1]
    profile = str(values.get("RAG_PROFILE") or "").strip() or "default"
    profile_root = root / "profiles" / profile
    if profile_root.is_dir():
        return profile_root / profile_relative_path

    return root / default_relative_path
