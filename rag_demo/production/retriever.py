from __future__ import annotations

from time import perf_counter
from typing import Optional, Sequence

from rag_demo.config import RagConfig
from rag_demo.hybrid_retrieval import tokenize_bm25
from rag_demo.retrieval_scope import RetrievalScope


class ProductionHybridRetriever:
    """Server-backed BM25 + dense retrieval followed by true cross-encoder rerank."""

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
    ) -> dict:
        started_at = perf_counter()
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
        if retrieval_scope is None or not retrieval_scope.index_version_id:
            return _empty_result(
                question=question,
                retrieval_query=combined_query,
                reason="No active immutable index is available for this knowledge base.",
                started_at=started_at,
            )

        dense_started = perf_counter()
        query_vector = self.embedding_runtime.embed_query(combined_query)
        dense_hits = self.vector_repository.search(
            scope=retrieval_scope,
            query_vector=query_vector,
            top_k=candidate_k,
            source_ids=source_ids,
        )
        dense_ms = _elapsed_ms(dense_started)

        sparse_started = perf_counter()
        sparse_vector = self.embedding_runtime.sparse_query(combined_query)
        sparse_hits = self.vector_repository.search_sparse(
            scope=retrieval_scope,
            query_sparse_vector=sparse_vector,
            top_k=candidate_k,
            source_ids=source_ids,
        )
        sparse_ms = _elapsed_ms(sparse_started)

        fusion_started = perf_counter()
        candidates = _reciprocal_rank_fusion(dense_hits, sparse_hits)
        fusion_ms = _elapsed_ms(fusion_started)

        rerank_started = perf_counter()
        documents = [str(item["payload"].get("content") or "") for item in candidates]
        scores = self.embedding_runtime.rerank(combined_query, documents) if documents else []
        if len(scores) != len(candidates):
            raise RuntimeError("reranker returned a different number of scores")
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)
        candidates.sort(
            key=lambda item: (item["rerank_score"], item["rrf_score"]),
            reverse=True,
        )
        rerank_ms = _elapsed_ms(rerank_started)

        query_terms = set(tokenize_bm25(combined_query))
        contexts = []
        for rank, candidate in enumerate(candidates[:top_k], start=1):
            payload = candidate["payload"]
            content = str(payload.get("content") or "")
            matched_terms = sorted(query_terms.intersection(tokenize_bm25(content)))[:20]
            contexts.append({
                "id": str(payload.get("chunk_id") or candidate["point_id"]),
                "rank": rank,
                "title": str(payload.get("title") or payload.get("chunk_id") or "Untitled"),
                "source": str(payload.get("source_id") or ""),
                "page": str(payload.get("page") or ""),
                "content": content,
                "branch": "Qdrant hybrid + cross encoder",
                "score": round(float(candidate["rerank_score"]), 6),
                "bm25Score": round(float(candidate["sparse_score"]), 6),
                "embeddingScore": round(float(candidate["dense_score"]), 6),
                "fusionScore": round(float(candidate["rrf_score"]), 6),
                "rerankScore": round(float(candidate["rerank_score"]), 6),
                "matchedTerms": matched_terms,
                "documentVersionId": str(payload.get("document_version_id") or ""),
                "chunkRecordId": str(payload.get("chunk_record_id") or ""),
                "indexVersionId": str(payload.get("index_version_id") or ""),
            })

        return {
            "variant": "qdrant_bm25_dense_rrf_cross_encoder",
            "query": question,
            "retrievalQuery": combined_query,
            "contexts": contexts,
            "pipeline": [
                {"name": "BM25 sparse", "detail": f"Qdrant sparse candidates: {len(sparse_hits)}."},
                {"name": "Dense embedding", "detail": f"Qdrant dense candidates: {len(dense_hits)}."},
                {"name": "RRF", "detail": f"Fused unique candidates: {len(candidates)}."},
                {"name": "Cross encoder", "detail": f"Reranked top {len(contexts)} contexts."},
            ],
            "timings": {
                "bm25Ms": sparse_ms,
                "embeddingMs": dense_ms,
                "fusionMs": fusion_ms,
                "rerankMs": rerank_ms,
                "totalMs": _elapsed_ms(started_at),
            },
            "diagnostics": {
                "tenantScopeApplied": True,
                "indexVersionId": retrieval_scope.index_version_id,
                "candidateCount": len(candidates),
            },
        }


def _reciprocal_rank_fusion(dense_hits, sparse_hits, rrf_k: int = 60) -> list[dict]:
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
            item["rrf_score"] += 1.0 / (rrf_k + rank)
    return sorted(candidates.values(), key=lambda item: item["rrf_score"], reverse=True)


def _empty_result(question: str, retrieval_query: str, reason: str, started_at: float) -> dict:
    return {
        "variant": "qdrant_bm25_dense_rrf_cross_encoder",
        "query": question,
        "retrievalQuery": retrieval_query,
        "contexts": [],
        "pipeline": [{"name": "Index gate", "detail": reason}],
        "timings": {"totalMs": _elapsed_ms(started_at)},
        "diagnostics": {"tenantScopeApplied": True, "reason": reason},
    }


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000.0, 3)
