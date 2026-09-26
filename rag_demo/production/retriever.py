from __future__ import annotations

import math
import re
from time import perf_counter
from typing import Optional, Sequence

from rag_demo.config import RagConfig
from rag_demo.hybrid_retrieval import tokenize_bm25
from rag_demo.lambdamart_fusion import (
    LAMBDA_MART_FUSION_METHOD,
    fuse_candidates_with_lambdamart,
)
from rag_demo.hybrid_retrieval import RRF_FUSION_METHOD
from rag_demo.parent_child import expand_child_contexts
from rag_demo.observability import TimingTrace, elapsed_ms
from rag_demo.query_complexity import classify_query_complexity
from rag_demo.retrieval_scope import RetrievalScope


class ProductionHybridRetriever:
    """Server-backed sparse/dense retrieval with gated second-stage reranking."""

    def __init__(self, vector_repository, embedding_runtime, settings: RagConfig | None = None):
        self.vector_repository = vector_repository
        self.embedding_runtime = embedding_runtime
        self.settings = (settings or RagConfig.from_env()).normalized()

    def retrieve(
        self,
        question: str,
        retrieval_query: str = "",
        source_ids: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
        candidate_k: Optional[int] = None,
        retrieval_scope: Optional[RetrievalScope] = None,
        query_variants: Optional[Sequence[str]] = None,
        evidence_query: str = "",
    ) -> dict:
        started_at = perf_counter()
        trace = TimingTrace(component="production_retriever")
        complexity_started = perf_counter()
        complexity_decision = classify_query_complexity(
            question,
            query_variants=query_variants or (),
            threshold=self.settings.query_complexity_threshold,
        )
        trace.record("query.complexity", elapsed_ms(complexity_started))
        top_k = max(1, min(int(top_k or self.settings.hybrid_top_k), 50))
        candidate_k = max(top_k, min(int(candidate_k or self.settings.hybrid_candidate_k), 200))
        combined_query = " ".join(dict.fromkeys(
            item
            for item in (
                str(question or "").strip(),
                str(retrieval_query or "").strip(),
            )
            if item
        ))
        retrieval_queries = _retrieval_queries(
            combined_query,
            query_variants or (),
            evidence_query,
            enabled=self.settings.multi_query_enabled,
            max_variants=self.settings.multi_query_max_variants,
        )
        if retrieval_scope is None or not retrieval_scope.index_version_id:
            return _empty_result(
                question=question,
                retrieval_query=combined_query,
                reason="No active immutable index is available for this knowledge base.",
                started_at=started_at,
                trace=trace,
            )

        per_query_k = max(top_k, math.ceil(candidate_k / len(retrieval_queries)))
        query_hits = []
        dense_ms = sparse_ms = 0.0
        dense_hit_count = sparse_hit_count = 0
        for query in retrieval_queries:
            dense_started = perf_counter()
            query_vector = self.embedding_runtime.embed_query(query)
            dense_hits = self.vector_repository.search(
                scope=retrieval_scope,
                query_vector=query_vector,
                top_k=per_query_k,
                source_ids=source_ids,
            )
            dense_ms += _elapsed_ms(dense_started)
            dense_hit_count += len(dense_hits)

            sparse_started = perf_counter()
            sparse_vector = self.embedding_runtime.sparse_query(query)
            sparse_hits = self.vector_repository.search_sparse(
                scope=retrieval_scope,
                query_sparse_vector=sparse_vector,
                top_k=per_query_k,
                source_ids=source_ids,
            )
            sparse_ms += _elapsed_ms(sparse_started)
            sparse_hit_count += len(sparse_hits)
            query_hits.append((dense_hits, sparse_hits))
        trace.record("retrieval.embedding", dense_ms, candidates=dense_hit_count, query_count=len(retrieval_queries))
        trace.record("retrieval.bm25", sparse_ms, candidates=sparse_hit_count, query_count=len(retrieval_queries))

        fusion_started = perf_counter()
        candidates = _reciprocal_rank_fusion_many(
            query_hits,
            rrf_k=self.settings.hybrid_rrf_k,
            bm25_weight=self.settings.keyword_weight,
            embedding_weight=self.settings.embedding_weight,
        )
        candidates = candidates[: min(100, max(top_k, candidate_k))]
        legacy_single_level = not any(
            str(item.get("payload", {}).get("chunk_level") or "").lower() == "child"
            for item in candidates
        )
        simple_query_top_k = (
            3
            if legacy_single_level and self.settings.hybrid_fusion_method == "rrf"
            else self.settings.simple_query_top_k
        )
        if self.settings.hybrid_fusion_method == "rrf":
            for candidate in candidates:
                candidate.setdefault("lambda_mart_fusion_score", 0.0)
                candidate.setdefault("mapped_bm25_score", 0.0)
                candidate.setdefault("mapped_embedding_score", 0.0)
        else:
            candidates = fuse_candidates_with_lambdamart(
                candidates,
                keyword_weight=self.settings.keyword_weight,
                embedding_weight=self.settings.embedding_weight,
            )
        fusion_ms = _elapsed_ms(fusion_started)
        trace.record("retrieval.fusion", fusion_ms, candidates=len(candidates))

        rerank_applied = bool(
            not self.settings.complexity_routing_enabled
            or complexity_decision.is_complex
        )
        if rerank_applied:
            rerank_started = perf_counter()
            # The first stage deliberately produces a broad pool. Cross
            # encoder work is limited to that pool and the final answer is
            # always capped at top_k.
            rerank_pool_k = min(len(candidates), max(100, top_k, self.settings.rerank_top_k))
            rerank_candidates = candidates[:rerank_pool_k]
            documents = [str(item["payload"].get("content") or "") for item in rerank_candidates]
            scores = self.embedding_runtime.rerank(combined_query, documents) if documents else []
            if len(scores) != len(rerank_candidates):
                raise RuntimeError("reranker returned a different number of scores")
            for candidate, score in zip(rerank_candidates, scores):
                candidate["rerank_score"] = float(score)
            reranked = sorted(
                rerank_candidates,
                key=lambda item: (
                    item["rerank_score"],
                    item.get("lambda_mart_fusion_score", 0.0),
                    item["rrf_score"],
                ),
                reverse=True,
            )
            rerank_ms = _elapsed_ms(rerank_started)
            trace.record("retrieval.rerank", rerank_ms, candidates=len(rerank_candidates))
        else:
            rerank_ms = 0.0
            trace.record("retrieval.rerank", 0.0, status="skipped", reason="simple_query")
            reranked = []
            for candidate in candidates[: min(top_k, simple_query_top_k)]:
                candidate["rerank_score"] = float(
                    candidate.get("lambda_mart_fusion_score", candidate.get("rrf_score", 0.0))
                )
                reranked.append(candidate)
            reranked.sort(
                key=lambda item: (
                    item["rerank_score"],
                    item["rrf_score"],
                ),
                reverse=True,
            )

        query_terms = set(tokenize_bm25(combined_query))
        output_limit = top_k if rerank_applied else min(top_k, simple_query_top_k)
        selected_candidates = reranked[:output_limit]
        evidence_candidates = []
        for candidate in selected_candidates:
            payload = candidate["payload"]
            evidence_candidates.append({
                **payload,
                "id": str(payload.get("chunk_id") or candidate["point_id"]),
                "bm25_score": candidate.get("sparse_score", 0.0),
                "embedding_score": candidate.get("dense_score", 0.0),
                "fusion_score": candidate.get("rrf_score", 0.0),
                "rrf_score": candidate.get("rrf_score", 0.0),
                "rerank_score": candidate.get("rerank_score", candidate.get("rrf_score", 0.0)),
            })
        evidence_chunks = expand_child_contexts(evidence_candidates, limit=output_limit)
        candidate_by_child_id = {
            str(candidate["payload"].get("chunk_id") or candidate["point_id"]): candidate
            for candidate in selected_candidates
        }
        contexts = []
        for rank, evidence in enumerate(evidence_chunks, start=1):
            candidate = candidate_by_child_id.get(str(evidence.get("retrieval_child_id") or ""), {})
            payload = candidate.get("payload") or evidence
            content = str(evidence.get("content") or payload.get("content") or "")
            matched_terms = sorted(query_terms.intersection(tokenize_bm25(content)))[:20]
            contexts.append({
                "id": str(evidence.get("id") or payload.get("chunk_id") or candidate.get("point_id") or ""),
                "childChunkId": str(payload.get("chunk_id") or candidate.get("point_id") or ""),
                "parentChunkId": str(evidence.get("parent_chunk_id") or evidence.get("id") or ""),
                "rank": rank,
                "title": str(evidence.get("title") or payload.get("title") or payload.get("chunk_id") or "Untitled"),
                "source": str(evidence.get("source_id") or payload.get("source_id") or ""),
                "page": str(evidence.get("page") or payload.get("page") or ""),
                "content": content,
                "branch": (
                    "Qdrant hybrid + RRF fusion (direct top-k)"
                    if not rerank_applied
                    else "Qdrant hybrid + RRF fusion + cross encoder"
                ),
                "score": round(float(candidate.get("rerank_score", candidate.get("rrf_score", 0.0))), 6),
                "bm25Score": round(float(candidate["sparse_score"]), 6),
                "embeddingScore": round(float(candidate["dense_score"]), 6),
                "fusionScore": round(
                    float(candidate.get("lambda_mart_fusion_score", candidate.get("rrf_score", 0.0)))
                    if self.settings.hybrid_fusion_method != "rrf"
                    else float(candidate.get("rrf_score", 0.0)),
                    6,
                ),
                "rrfScore": round(float(candidate["rrf_score"]), 6),
                "mappedBm25Score": round(float(candidate.get("mapped_bm25_score", 0.0)), 6),
                "mappedEmbeddingScore": round(
                    float(candidate.get("mapped_embedding_score", 0.0)),
                    6,
                ),
                "lambdaMARTScore": round(
                    float(candidate.get("lambda_mart_fusion_score", 0.0)),
                    6,
                ),
                "rerankScore": round(float(candidate.get("rerank_score", candidate.get("rrf_score", 0.0))), 6),
                "matchedTerms": matched_terms,
                "documentVersionId": str(evidence.get("document_version_id") or payload.get("document_version_id") or ""),
                "chunkRecordId": str(evidence.get("chunk_record_id") or payload.get("chunk_record_id") or ""),
                "indexVersionId": str(evidence.get("index_version_id") or payload.get("index_version_id") or ""),
            })

        total_ms = _elapsed_ms(started_at)
        trace.record("retrieval.total", total_ms, contexts=len(contexts))
        timing_trace = trace.as_dict()
        return {
            "variant": "qdrant_bm25_dense_rrf_cross_encoder",
            "query": question,
            "retrievalQuery": combined_query,
            "retrievalQueries": retrieval_queries,
            "contexts": contexts,
            "pipeline": [
                {"name": "BM25 sparse", "detail": f"Qdrant sparse candidates: {sparse_hit_count} across {len(retrieval_queries)} queries."},
                {"name": "Dense embedding", "detail": f"Qdrant dense candidates: {dense_hit_count} across {len(retrieval_queries)} queries."},
                {
                    "name": "Candidate Merge",
                    "detail": f"Union {len(candidates)} candidates for rank diagnostics.",
                },
                {
                    "name": "RRF Fusion",
                    "detail": f"Fuse sparse and dense ranks with weighted reciprocal-rank k={self.settings.hybrid_rrf_k} (BM25 {self.settings.keyword_weight:.2f}, Embedding {self.settings.embedding_weight:.2f}); keep the first {min(100, len(candidates))} candidates.",
                },
                {
                    "name": "Cross encoder",
                    "detail": (
                        f"{'Reranked' if rerank_applied else 'Skipped for simple query; direct selected'} "
                        f"top {len(contexts)} contexts."
                    ),
                },
            ],
            "timings": {
                "bm25Ms": sparse_ms,
                "embeddingMs": dense_ms,
                "fusionMs": fusion_ms,
                "rerankMs": rerank_ms,
                "totalMs": total_ms,
                "stages": timing_trace["stages"],
            },
            "timing_trace": timing_trace,
            "diagnostics": {
                "tenantScopeApplied": True,
                "indexVersionId": retrieval_scope.index_version_id,
                "candidateCount": len(candidates),
                "queryCount": len(retrieval_queries),
                "perQueryCandidateK": per_query_k,
                "rerankApplied": rerank_applied,
                "queryComplexity": complexity_decision.as_dict(),
                "fusionMethod": RRF_FUSION_METHOD if self.settings.hybrid_fusion_method == "rrf" else LAMBDA_MART_FUSION_METHOD,
                "fusionDimension": "reciprocal-rank" if self.settings.hybrid_fusion_method == "rrf" else "[0, 1]",
                "fusionKeywordWeight": self.settings.keyword_weight,
                "fusionEmbeddingWeight": self.settings.embedding_weight,
                "simpleQueryTopK": simple_query_top_k,
                "rerankTopK": self.settings.rerank_top_k,
                "retrievalChunkLevel": "child" if any(str(item.get("payload", {}).get("chunk_level") or "").lower() == "child" for item in candidates) else "single",
                "evidenceChunkLevel": "parent" if any(str(item.get("payload", {}).get("chunk_level") or "").lower() == "child" for item in candidates) else "same",
            },
        }


def _retrieval_queries(
    combined_query: str,
    query_variants: Sequence[str],
    evidence_query: str,
    *,
    enabled: bool,
    max_variants: int,
) -> list[str]:
    """Keep the original query and a bounded set of distinct planned queries."""

    if not enabled:
        return [combined_query]
    limit = max(1, min(int(max_variants), 8))
    queries = []
    seen = set()
    for value in (combined_query, *query_variants):
        query = re.sub(r"\s+", " ", str(value or "")).strip()[:512].rstrip()
        if query and query.casefold() not in seen:
            queries.append(query)
            seen.add(query.casefold())
    focused = re.sub(r"\s+", " ", str(evidence_query or "")).strip()[:512].rstrip()
    if focused and focused.casefold() not in seen and limit > 1:
        queries = queries[:limit - 1] + [focused]
    return (queries or [combined_query])[:limit]


def _reciprocal_rank_fusion_many(
    query_hits,
    *,
    rrf_k: int,
    bm25_weight: float,
    embedding_weight: float,
) -> list[dict]:
    """Fuse both retrieval branches across queries without multiplying scores."""

    if len(query_hits) == 1:
        dense_hits, sparse_hits = query_hits[0]
        return _reciprocal_rank_fusion(
            dense_hits,
            sparse_hits,
            rrf_k=rrf_k,
            bm25_weight=bm25_weight,
            embedding_weight=embedding_weight,
        )
    candidates = {}
    for dense_hits, sparse_hits in query_hits:
        per_query = _reciprocal_rank_fusion(
            dense_hits,
            sparse_hits,
            rrf_k=rrf_k,
            bm25_weight=bm25_weight,
            embedding_weight=embedding_weight,
        )
        for hit in per_query:
            item = candidates.setdefault(hit["point_id"], {
                "point_id": hit["point_id"],
                "payload": hit["payload"],
                "dense_score": 0.0,
                "sparse_score": 0.0,
                "rrf_score": 0.0,
            })
            item["dense_score"] = max(item["dense_score"], hit["dense_score"])
            item["sparse_score"] = max(item["sparse_score"], hit["sparse_score"])
            item["rrf_score"] += hit["rrf_score"] / len(query_hits)
    return sorted(candidates.values(), key=lambda item: item["rrf_score"], reverse=True)


def _reciprocal_rank_fusion(
    dense_hits,
    sparse_hits,
    rrf_k: int = 60,
    bm25_weight: float = 0.6,
    embedding_weight: float = 0.4,
) -> list[dict]:
    bm25_weight, embedding_weight = _normalize_fusion_weights(
        bm25_weight,
        embedding_weight,
    )
    candidates = {}
    for branch, hits in (("dense", dense_hits), ("sparse", sparse_hits)):
        for rank, hit in enumerate(hits, start=1):
            item = candidates.setdefault(hit.point_id, {
                "point_id": hit.point_id,
                "payload": dict(hit.payload),
                "dense_score": 0.0,
                "sparse_score": 0.0,
                "rrf_score": 0.0,
            })
            item[f"{branch}_score"] = float(hit.score)
            branch_weight = embedding_weight if branch == "dense" else bm25_weight
            item["rrf_score"] += branch_weight / (rrf_k + rank)
    return sorted(candidates.values(), key=lambda item: item["rrf_score"], reverse=True)


def _normalize_fusion_weights(bm25_weight: float, embedding_weight: float) -> tuple[float, float]:
    bm25 = max(0.0, float(bm25_weight))
    embedding = max(0.0, float(embedding_weight))
    total = bm25 + embedding
    if total <= 0.0:
        return 0.5, 0.5
    return bm25 / total, embedding / total


def _empty_result(
    question: str,
    retrieval_query: str,
    reason: str,
    started_at: float,
    trace: Optional[TimingTrace] = None,
) -> dict:
    timing_trace = trace.as_dict() if trace is not None else None
    total_ms = _elapsed_ms(started_at)
    if trace is not None:
        trace.record("retrieval.total", total_ms, status="skipped", reason=reason)
        timing_trace = trace.as_dict()
    return {
        "variant": "qdrant_bm25_dense_rrf_cross_encoder",
        "query": question,
        "retrievalQuery": retrieval_query,
        "contexts": [],
        "pipeline": [{"name": "Index gate", "detail": reason}],
        "timings": {
            "totalMs": total_ms,
            "stages": timing_trace["stages"] if timing_trace else [],
        },
        "timing_trace": timing_trace or {},
        "diagnostics": {"tenantScopeApplied": True, "reason": reason},
    }


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000.0, 3)
