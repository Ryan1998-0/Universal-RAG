import hashlib
import json
import math
import os
import re
import threading
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, List, Optional, Sequence, Set

import numpy as np

from rag_demo.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    embed_chunks,
    embed_query,
    embed_texts,
    load_embedding_matrix,
    save_embedding_matrix,
)
from rag_demo.config import RagConfig
from rag_demo.document_pipeline import DocumentStore
from rag_demo.lambdamart_fusion import (
    LAMBDA_MART_FUSION_METHOD,
    fuse_candidates_with_lambdamart,
)
from rag_demo.parent_child import expand_child_contexts
from rag_demo.cross_encoder import CrossEncoderReranker
from rag_demo.query_complexity import classify_query_complexity
from rag_demo.retrieval_planner import extract_focus_terms
from rag_demo.retrieval_scope import RetrievalScope


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_RETRIEVAL_CONFIG = "retrieval.json"
DEFAULT_PROFILE = "default"
RRF_FUSION_METHOD = "rrf_v1"
_PROFILE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "under",
    "what",
    "when",
    "which",
    "why",
    "with",
}

_RETRIEVERS: Dict[str, "HybridRetriever"] = {}
_RETRIEVER_LOCK = threading.Lock()


class Bm25Index:
    def __init__(self, chunks: Sequence[dict], k1: float = 1.4, b: float = 0.72):
        self.chunks = [dict(chunk) for chunk in chunks]
        self.k1 = max(0.01, float(k1))
        self.b = min(max(0.0, float(b)), 1.0)
        self.term_frequencies = []
        self.document_lengths = []
        self.document_frequency = Counter()

        for chunk in self.chunks:
            counts = Counter(tokenize_bm25(_chunk_text(chunk)))
            self.term_frequencies.append(counts)
            self.document_lengths.append(max(1, sum(counts.values())))
            self.document_frequency.update(counts.keys())

        self.document_count = len(self.chunks)
        self.average_length = (
            sum(self.document_lengths) / self.document_count if self.document_count else 1.0
        )
        self.inverse_document_frequency = {
            term: math.log(
                1.0
                + (self.document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            for term, document_frequency in self.document_frequency.items()
        }

    def search(
        self,
        query: str,
        top_k: int,
        allowed_indices: Optional[Sequence[int]] = None,
    ) -> List[dict]:
        query_terms = Counter(tokenize_bm25(query))
        if not query_terms:
            return []

        indices = allowed_indices if allowed_indices is not None else range(self.document_count)
        scored = []
        for index in indices:
            counts = self.term_frequencies[index]
            document_length = self.document_lengths[index]
            score = 0.0
            matched_terms = []
            for term, query_frequency in query_terms.items():
                term_frequency = counts.get(term, 0)
                if not term_frequency:
                    continue
                inverse_document_frequency = self.inverse_document_frequency.get(term, 0.0)
                denominator = term_frequency + self.k1 * (
                    1.0 - self.b + self.b * (document_length / self.average_length)
                )
                score += (
                    inverse_document_frequency
                    * ((term_frequency * (self.k1 + 1.0)) / denominator)
                    * (1.0 + math.log(query_frequency))
                )
                matched_terms.append(term)

            if score <= 0:
                continue
            scored.append(
                {
                    "index": index,
                    "score": score,
                    "matched_terms": matched_terms,
                }
            )

        return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]


class HybridRetriever:
    def __init__(
        self,
        chunks: Sequence[dict],
        aliases: Sequence[dict],
        embeddings: Sequence[Sequence[float]],
        embed_query_fn: Callable[[str], Sequence[float]] = embed_query,
        embed_queries_fn: Optional[Callable[[Sequence[str]], Sequence[Sequence[float]]]] = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        sources: Optional[Sequence[dict]] = None,
        profile: str = DEFAULT_PROFILE,
        query_expansions: Optional[Sequence[dict]] = None,
        metadata: Optional[dict] = None,
        settings: Optional[RagConfig] = None,
    ):
        matrix = np.asarray(embeddings, dtype=np.float32)
        if len(chunks) != len(matrix):
            raise ValueError("Chunk and embedding counts must match.")
        if len(chunks) and matrix.ndim != 2:
            raise ValueError("Embeddings must be a two-dimensional matrix.")
        if not len(chunks):
            matrix = np.empty((0, 0), dtype=np.float32)

        resolved_settings = settings or RagConfig.from_env()
        # Keep legacy one-level fixtures and old profile snapshots on their
        # explicit LambdaMART-compatible path. New parent-child indexes use
        # RRF by default.
        if (
            settings is None
            and resolved_settings.hybrid_fusion_method == "rrf"
            and not any(str(chunk.get("chunk_level") or "").lower() == "child" for chunk in chunks)
        ):
            resolved_settings = RagConfig(
                **{
                    **resolved_settings.__dict__,
                    "hybrid_fusion_method": "lambdamart",
                    "hybrid_top_k": 8,
                    "simple_query_top_k": 3,
                    "rerank_top_k": 8,
                }
            )
        self.settings = resolved_settings.normalized()
        self.profile = _normalize_profile_name(profile)
        self.chunks = [dict(chunk) for chunk in chunks]
        self.aliases = [dict(alias) for alias in aliases]
        self.query_expansions = _normalize_query_expansions(query_expansions)
        self.embeddings = _normalize_matrix(matrix) if matrix.size else matrix
        self.embed_query_fn = embed_query_fn
        self.embed_queries_fn = embed_queries_fn
        self.embedding_model = embedding_model
        self.sources = [dict(source) for source in (sources or [])]
        self.metadata = dict(metadata or {})
        self.base_source_ids = {
            str(chunk.get("source_id") or chunk.get("source") or "")
            for chunk in self.chunks
            if str(chunk.get("source_id") or chunk.get("source") or "")
        }
        self.bm25 = Bm25Index(
            self.chunks,
            k1=self.settings.hybrid_bm25_k1,
            b=self.settings.hybrid_bm25_b,
        )
        self._chunk_token_sets = [
            set(counts.keys()) for counts in self.bm25.term_frequencies
        ]
        self._index_lock = threading.RLock()
        self._query_embedding_cache: Dict[str, np.ndarray] = {}
        self._query_embedding_cache_limit = 256
        self._cross_encoder: Optional[CrossEncoderReranker] = None

    @classmethod
    def from_profile(
        cls,
        profile: str = DEFAULT_PROFILE,
        project_root: Path = PROJECT_ROOT,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        settings: Optional[RagConfig] = None,
    ) -> "HybridRetriever":
        clean_profile = _normalize_profile_name(profile)
        profile_config = load_profile_retrieval_config(clean_profile, project_root=project_root)
        data_path = profile_config.get("data_path")
        data = {}
        if data_path:
            data = json.loads(Path(data_path).read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError(f"Retrieval data must contain a JSON object: {data_path}")
        chunks = data.get("chunks") or []
        aliases = data.get("aliases") or []
        embeddings: Sequence[Sequence[float]] = []
        if chunks:
            index_dir = Path(profile_config["index_dir"])
            embedding_path = index_dir / "web_embeddings.npy"
            embedding_meta_path = index_dir / "web_embeddings.meta.json"
            fingerprint = _chunks_fingerprint(chunks)
            embeddings = _load_valid_embedding_cache(
                embedding_path=embedding_path,
                meta_path=embedding_meta_path,
                model_name=embedding_model,
                chunk_count=len(chunks),
                fingerprint=fingerprint,
            )
            if embeddings is None:
                embeddings = embed_chunks(chunks, model_name=embedding_model)
                save_embedding_matrix(embeddings, embedding_path)
                embedding_meta_path.parent.mkdir(parents=True, exist_ok=True)
                embedding_meta_path.write_text(
                    json.dumps(
                        {
                            "model": embedding_model,
                            "chunk_count": len(chunks),
                            "fingerprint": fingerprint,
                            "dimensions": len(embeddings[0]),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

        retriever = cls(
            chunks=chunks,
            aliases=aliases,
            embeddings=embeddings,
            embed_query_fn=lambda text: embed_query(text, model_name=embedding_model),
            embed_queries_fn=lambda texts: embed_texts(texts, model_name=embedding_model),
            embedding_model=embedding_model,
            sources=data.get("sources") or [],
            profile=clean_profile,
            query_expansions=profile_config.get("query_expansions") or [],
            metadata={
                **dict(data.get("meta") or {}),
                "profile": clean_profile,
                "label": profile_config["label"],
                "sample_queries": profile_config["sample_queries"],
                "graph_relation_count": len((data.get("graph") or {}).get("relations") or []),
            },
            settings=settings,
        )
        for document in DocumentStore(embedding_model=embedding_model).load_indexed_documents():
            try:
                retriever.add_document(
                    source=document.metadata,
                    chunks=document.chunks,
                    embeddings=document.embeddings,
                )
            except ValueError:
                continue
        return retriever

    def retrieve(
        self,
        question: str,
        retrieval_query: str = "",
        query_variants: Optional[Sequence[str]] = None,
        evidence_query: str = "",
        source_ids: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
        candidate_k: Optional[int] = None,
        retrieval_scope: Optional[RetrievalScope] = None,
    ) -> dict:
        with self._index_lock:
            return self._retrieve_locked(
                question=question,
                retrieval_query=retrieval_query,
                query_variants=query_variants,
                evidence_query=evidence_query,
                source_ids=source_ids,
                top_k=top_k,
                candidate_k=candidate_k,
                retrieval_scope=retrieval_scope,
            )

    def _retrieve_locked(
        self,
        question: str,
        retrieval_query: str = "",
        query_variants: Optional[Sequence[str]] = None,
        evidence_query: str = "",
        source_ids: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
        candidate_k: Optional[int] = None,
        retrieval_scope: Optional[RetrievalScope] = None,
    ) -> dict:
        started_at = perf_counter()
        complexity_decision = classify_query_complexity(
            question,
            query_variants=query_variants or (),
            threshold=self.settings.query_complexity_threshold,
        )
        top_k = max(
            1,
            min(
                int(top_k or self.settings.hybrid_top_k),
                self.settings.hybrid_max_top_k,
            ),
        )
        candidate_k = max(
            top_k,
            min(
                int(candidate_k or self.settings.hybrid_candidate_k),
                self.settings.hybrid_max_candidate_k,
            ),
        )
        allowed_indices = self._allowed_indices(source_ids, retrieval_scope)
        profile_expansions_enabled = self._profile_expansions_enabled(source_ids)
        planned_queries = list(
            dict.fromkeys(
                str(query).strip()
                for query in [retrieval_query, *(query_variants or ()), question]
                if str(query).strip()
            )
        )
        if not self.settings.multi_query_enabled:
            planned_queries = planned_queries[:1]
        else:
            planned_queries = planned_queries[: self.settings.multi_query_max_variants]

        combined_queries = []
        added_terms = []
        matched_aliases = []
        for query in planned_queries:
            if profile_expansions_enabled:
                expanded_query, query_added_terms, query_aliases = expand_retrieval_query(
                    question,
                    query,
                    self.aliases,
                    query_expansions=self.query_expansions,
                )
                added_terms.extend(query_added_terms)
                matched_aliases.extend(query_aliases)
            else:
                expanded_query = query
            if expanded_query and expanded_query not in combined_queries:
                combined_queries.append(expanded_query)
        if not combined_queries:
            combined_queries = [str(question or retrieval_query).strip()]
        added_terms = list(dict.fromkeys(added_terms))
        matched_aliases = list(dict.fromkeys(matched_aliases))

        bm25_started = perf_counter()
        bm25_result_sets = [
            self.bm25.search(
                query,
                top_k=candidate_k,
                allowed_indices=allowed_indices,
            )
            for query in combined_queries
        ]
        bm25_ms = _elapsed_ms(bm25_started)

        embedding_started = perf_counter()
        dense_result_sets = self._embedding_search_many(
            question=question,
            retrieval_queries=combined_queries,
            top_k=candidate_k,
            allowed_indices=allowed_indices,
        )
        embedding_ms = _elapsed_ms(embedding_started)

        fusion_started = perf_counter()
        candidates = reciprocal_rank_fusion_many(
            bm25_result_sets=bm25_result_sets,
            embedding_result_sets=dense_result_sets,
            rrf_k=self.settings.hybrid_rrf_k,
            bm25_weight=self.settings.keyword_weight,
            embedding_weight=self.settings.embedding_weight,
        )
        candidates = candidates[: min(100, max(top_k, candidate_k))]
        if self.settings.hybrid_fusion_method == "rrf":
            for candidate in candidates:
                candidate["rrf_score"] = float(candidate.get("fusion_score", 0.0))
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

        answerability_query = str(evidence_query or "").strip() or " ".join(combined_queries)
        rerank_applied = bool(
            not self.settings.complexity_routing_enabled
            or complexity_decision.is_complex
        )
        if rerank_applied:
            rerank_started = perf_counter()
            rerank_pool_k = min(
                len(candidates),
                max(100, top_k, self.settings.rerank_top_k),
            )
            rerank_pool = candidates[:rerank_pool_k]
            if self.settings.hybrid_fusion_method == "rrf":
                if self._cross_encoder is None:
                    self._cross_encoder = CrossEncoderReranker()
                documents = [
                    str(self.chunks[int(candidate["index"])].get("content") or "")
                    for candidate in rerank_pool
                ]
                scores = self._cross_encoder.score(answerability_query, documents)
                if len(scores) != len(rerank_pool):
                    raise RuntimeError("Cross-Encoder returned a different number of scores")
                reranked = [
                    {**candidate, "rerank_score": float(score), "rerank_applied": True}
                    for candidate, score in zip(rerank_pool, scores)
                ]
                reranked.sort(
                    key=lambda item: (
                        float(item.get("rerank_score", 0.0)),
                        float(item.get("rrf_score", item.get("fusion_score", 0.0))),
                    ),
                    reverse=True,
                )
            else:
                reranked = rerank_candidates(
                    candidates=rerank_pool,
                    chunks=self.chunks,
                    query=answerability_query,
                    top_k=rerank_pool_k,
                    settings=self.settings,
                    chunk_token_sets=self._chunk_token_sets,
                )
            rerank_ms = _elapsed_ms(rerank_started)
        else:
            # A simple factual query does not need an expensive second-stage
            # reranker. RRF already provides a stable rank signal, so keep
            # only the configured direct top-N.
            reranked = []
            for candidate in candidates[: min(top_k, self.settings.simple_query_top_k)]:
                direct = dict(candidate)
                direct["rerank_score"] = float(
                    direct.get(
                        "lambda_mart_fusion_score"
                        if self.settings.hybrid_fusion_method != "rrf"
                        else "rrf_score",
                        direct.get("fusion_score", 0.0),
                    )
                )
                direct["rerank_applied"] = False
                reranked.append(direct)
            rerank_ms = 0.0

        selected_candidates = reranked[:top_k]
        parent_by_id = {
            str(chunk.get("id")): chunk
            for chunk in self.chunks
            if str(chunk.get("chunk_level") or "").lower() == "parent"
        }
        evidence_candidates = [
            {**self.chunks[int(candidate["index"])], **candidate}
            for candidate in selected_candidates
        ]
        evidence_chunks = expand_child_contexts(
            evidence_candidates,
            parent_by_id=parent_by_id,
            limit=top_k,
        )
        candidate_by_child_id = {
            str(candidate.get("id") or self.chunks[int(candidate["index"])].get("id") or ""): candidate
            for candidate in selected_candidates
        }
        contexts = []
        for rank, chunk in enumerate(evidence_chunks, start=1):
            candidate = candidate_by_child_id.get(str(chunk.get("retrieval_child_id") or ""), {})
            contexts.append(
                {
                    "id": str(chunk.get("id") or candidate["index"]),
                    "childChunkId": str(candidate.get("id") or self.chunks[int(candidate["index"])].get("id") or ""),
                    "parentChunkId": str(chunk.get("parent_chunk_id") or chunk.get("id") or ""),
                    "rank": rank,
                    "title": str(chunk.get("title") or chunk.get("id") or "Untitled"),
                    "source": str(chunk.get("source_id") or chunk.get("source") or ""),
                    "page": str(chunk.get("page") or ""),
                    "content": str(chunk.get("content") or ""),
                    "branch": (
                        "RRF fusion (direct top-k)"
                        if not rerank_applied
                        else "RRF fusion + cross-encoder reranker"
                    ),
                    "score": round(float(candidate.get("rerank_score", candidate.get("fusion_score", 0.0))), 6),
                    "bm25Score": round(float(candidate.get("bm25_score", 0.0)), 6),
                    "embeddingScore": round(float(candidate.get("embedding_score", 0.0)), 6),
                    "fusionScore": round(
                        float(
                            candidate.get(
                                "lambda_mart_fusion_score"
                                if self.settings.hybrid_fusion_method != "rrf"
                                else "rrf_score",
                                candidate.get("fusion_score", 0.0),
                            )
                        ),
                        6,
                    ),
                    "rrfScore": round(float(candidate.get("rrf_score", candidate.get("fusion_score", 0.0))), 6),
                    "mappedBm25Score": round(
                        float(candidate.get("mapped_bm25_score", 0.0)),
                        6,
                    ),
                    "mappedEmbeddingScore": round(
                        float(candidate.get("mapped_embedding_score", 0.0)),
                        6,
                    ),
                    "lambdaMARTScore": round(
                        float(candidate.get("lambda_mart_fusion_score", 0.0)),
                        6,
                    ),
                    "rerankScore": round(float(candidate.get("rerank_score", candidate.get("fusion_score", 0.0))), 6),
                    "matchedTerms": list(candidate.get("matched_terms") or []),
                    "documentVersionId": str(chunk.get("document_version_id") or ""),
                    "indexVersionId": str(chunk.get("index_version_id") or ""),
                }
            )

        evidence_evaluation = evaluate_retrieval_evidence(
            contexts,
            settings=self.settings,
            question=answerability_query,
        )

        return {
            "variant": "bm25_embedding_rerank",
            "query": question,
            "retrievalQuery": combined_queries[0],
            "retrievalQueries": combined_queries,
            "evidenceQuery": answerability_query,
            "translation": {
                "detectedLanguage": "zh-TW" if _contains_cjk(question) else "en",
                "addedTerms": added_terms,
            },
            "comparisonGraph": None,
            "contexts": contexts,
            "pipeline": [
                {
                    "name": "Query Planning",
                    "detail": (
                        f"Run {len(combined_queries)} complementary retrieval queries; "
                        f"complexity gate={complexity_decision.label}."
                    ),
                },
                {
                    "name": "BM25",
                    "detail": f"Lexical candidate generation across {len(bm25_result_sets)} queries.",
                },
                {
                    "name": "Embedding",
                    "detail": f"{self.embedding_model}, cosine candidates across {len(dense_result_sets)} queries.",
                },
                {
                    "name": "RRF Merge",
                    "detail": (
                        f"Weighted RRF fused {len(candidates)} unique retrieval units with k={self.settings.hybrid_rrf_k} "
                        f"(BM25 {self.settings.keyword_weight:.2f}, Embedding {self.settings.embedding_weight:.2f}); "
                        f"fusion method={self.settings.hybrid_fusion_method}."
                    ),
                },
                {
                    "name": "Hybrid Relevance Reranker",
                    "detail": (
                        f"{'Complex query: rerank candidates' if rerank_applied else 'Simple query: skip reranker'}; "
                        f"keep top {len(contexts)} (simple cap {self.settings.simple_query_top_k})."
                    ),
                },
                {
                    "name": "Evidence Relevance Gate",
                    "detail": evidence_evaluation["reason"],
                },
            ],
            "timings": {
                "bm25Ms": bm25_ms,
                "embeddingMs": embedding_ms,
                "fusionMs": fusion_ms,
                "rerankMs": rerank_ms,
                "totalMs": _elapsed_ms(started_at),
            },
            "diagnostics": {
                "matchedEntities": [],
                "hubWarning": False,
                "denseMatchedAliases": matched_aliases,
                "embeddingModel": self.embedding_model,
                "reranker": "cross-encoder" if rerank_applied else "skipped-simple-query",
                "rerankApplied": rerank_applied,
                "queryComplexity": complexity_decision.as_dict(),
                "fusionMethod": RRF_FUSION_METHOD if self.settings.hybrid_fusion_method == "rrf" else LAMBDA_MART_FUSION_METHOD,
                "fusionDimension": "reciprocal-rank" if self.settings.hybrid_fusion_method == "rrf" else "[0, 1]",
                "fusionKeywordWeight": self.settings.keyword_weight,
                "fusionEmbeddingWeight": self.settings.embedding_weight,
                "simpleQueryTopK": self.settings.simple_query_top_k,
                "rerankTopK": self.settings.rerank_top_k,
                "candidateCount": len(candidates),
                "retrievalChunkLevel": "child" if any(str(item.get("chunk_level") or "").lower() == "child" for item in self.chunks) else "single",
                "evidenceChunkLevel": "parent" if any(str(item.get("chunk_level") or "").lower() == "child" for item in self.chunks) else "same",
                "queryCount": len(combined_queries),
                "selectedSourceCount": len(set(source_ids or [])),
                "profile": self.profile,
                "profileExpansionsApplied": profile_expansions_enabled,
                "tenantScopeApplied": retrieval_scope is not None,
            },
            "confidence": evidence_evaluation["confidence"],
            "evidenceEvaluation": evidence_evaluation,
        }

    def add_document(
        self,
        source: dict,
        chunks: Sequence[dict],
        embeddings: Sequence[Sequence[float]],
    ) -> bool:
        source_id = str(source.get("source_id") or "").strip()
        if not source_id:
            raise ValueError("Uploaded source requires a source_id.")
        clean_chunks = [dict(chunk) for chunk in chunks]
        if not clean_chunks:
            raise ValueError("Uploaded source requires at least one chunk.")
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(clean_chunks):
            raise ValueError("Uploaded chunk and embedding counts must match.")

        with self._index_lock:
            existing_source_ids = {
                str(chunk.get("source_id") or chunk.get("source") or "")
                for chunk in self.chunks
            }
            if source_id in existing_source_ids:
                return False
            if self.embeddings.size and (
                self.embeddings.ndim != 2 or matrix.shape[1] != self.embeddings.shape[1]
            ):
                raise ValueError("Uploaded embeddings use a different dimension.")

            for chunk in clean_chunks:
                chunk["source_id"] = source_id
                chunk["source"] = source_id
                for scope_key in (
                    "tenant_id",
                    "knowledge_base_id",
                    "document_version_id",
                    "index_version_id",
                ):
                    if source.get(scope_key):
                        chunk[scope_key] = str(source[scope_key])
            self.chunks.extend(clean_chunks)
            normalized_matrix = _normalize_matrix(matrix)
            self.embeddings = (
                np.vstack((self.embeddings, normalized_matrix))
                if self.embeddings.size
                else normalized_matrix
            )
            self.bm25 = Bm25Index(
                self.chunks,
                k1=self.settings.hybrid_bm25_k1,
                b=self.settings.hybrid_bm25_b,
            )
            self._chunk_token_sets = [
                set(counts.keys()) for counts in self.bm25.term_frequencies
            ]
            if not any(item.get("source_id") == source_id for item in self.sources):
                self.sources.append(dict(source))
            return True

    def list_sources(self) -> List[dict]:
        with self._index_lock:
            return [dict(source) for source in self.sources]

    def describe(self) -> dict:
        with self._index_lock:
            chunk_counts = Counter(
                str(chunk.get("source_id") or chunk.get("source") or "")
                for chunk in self.chunks
                if str(chunk.get("source_id") or chunk.get("source") or "")
            )
            sources = []
            known_source_ids = set()
            for source in self.sources:
                item = dict(source)
                source_id = str(item.get("source_id") or "").strip()
                if not source_id:
                    continue
                known_source_ids.add(source_id)
                item["chunk_count"] = int(chunk_counts.get(source_id, item.get("chunk_count", 0)))
                item.setdefault("selected_by_default", source_id in self.base_source_ids)
                item.setdefault("status", "ready")
                sources.append(item)
            for source_id, chunk_count in chunk_counts.items():
                if source_id not in known_source_ids:
                    sources.append(
                        {
                            "source_id": source_id,
                            "name": source_id,
                            "source_type": "document",
                            "chunk_count": int(chunk_count),
                            "selected_by_default": source_id in self.base_source_ids,
                            "status": "ready",
                        }
                    )

            metadata = dict(self.metadata)
            metadata.update(
                {
                    "profile": self.profile,
                    "chunk_count": len(self.chunks),
                    "alias_count": len(self.aliases),
                    "source_count": len(sources),
                }
            )
            return {"meta": metadata, "sources": sources}

    def _allowed_indices(
        self,
        source_ids: Optional[Sequence[str]],
        retrieval_scope: Optional[RetrievalScope] = None,
    ) -> List[int]:
        allowed_sources = None
        if source_ids is not None:
            allowed_sources = {
                str(source_id).strip()
                for source_id in source_ids
                if str(source_id).strip()
            }
            if not allowed_sources:
                return []
        return [
            index
            for index, chunk in enumerate(self.chunks)
            if (allowed_sources is None or (
                str(chunk.get("source_id") or chunk.get("source") or "")
                in allowed_sources
            ))
            and (retrieval_scope is None or retrieval_scope.matches(chunk))
        ]

    def _profile_expansions_enabled(self, source_ids: Optional[Sequence[str]]) -> bool:
        if not self.query_expansions and not self.aliases:
            return False
        if source_ids is None:
            return bool(self.base_source_ids)
        selected_sources = {
            str(source_id).strip()
            for source_id in source_ids
            if str(source_id).strip()
        }
        if not selected_sources:
            return False
        return bool(selected_sources & self.base_source_ids)

    def _embedding_search(
        self,
        question: str,
        retrieval_query: str,
        top_k: int,
        allowed_indices: Sequence[int],
        query_vector: Optional[Sequence[float]] = None,
    ) -> List[dict]:
        if not allowed_indices:
            return []
        dense_query = str(retrieval_query or question).strip()
        if query_vector is None:
            query_vector = self._query_vector(dense_query)
        else:
            query_vector = _normalize_vector(np.asarray(query_vector, dtype=np.float32))
        index_array = np.asarray(allowed_indices, dtype=np.int64)
        scores = np.einsum(
            "ij,j->i",
            self.embeddings[index_array],
            query_vector,
            dtype=np.float32,
        )
        result_count = min(top_k, len(index_array))
        if result_count <= 0:
            return []
        local_order = np.argpartition(-scores, result_count - 1)[:result_count]
        local_order = local_order[np.argsort(-scores[local_order])]
        return [
            {
                "index": int(index_array[local_index]),
                "score": float(scores[local_index]),
            }
            for local_index in local_order
        ]

    def _embedding_search_many(
        self,
        question: str,
        retrieval_queries: Sequence[str],
        top_k: int,
        allowed_indices: Sequence[int],
    ) -> List[List[dict]]:
        if not allowed_indices:
            return [[] for _ in retrieval_queries]

        queries = [
            str(query or question).strip()
            for query in retrieval_queries
        ]
        if self.embed_queries_fn is None:
            return [
                self._embedding_search(
                    question=question,
                    retrieval_query=query,
                    top_k=top_k,
                    allowed_indices=allowed_indices,
                )
                for query in queries
            ]

        missing_queries = [
            query for query in queries if query not in self._query_embedding_cache
        ]
        if missing_queries:
            vectors = self.embed_queries_fn(missing_queries)
            if len(vectors) != len(missing_queries):
                raise ValueError("Batch embedding count does not match query count.")
            for query, vector in zip(missing_queries, vectors):
                self._remember_query_vector(query, vector)

        return [
            self._embedding_search(
                question=question,
                retrieval_query=query,
                top_k=top_k,
                allowed_indices=allowed_indices,
                query_vector=self._query_embedding_cache[query],
            )
            for query in queries
        ]

    def _query_vector(self, query: str) -> np.ndarray:
        cached = self._query_embedding_cache.get(query)
        if cached is not None:
            return cached
        vector = self._remember_query_vector(query, self.embed_query_fn(query))
        return vector

    def _remember_query_vector(
        self,
        query: str,
        vector: Sequence[float],
    ) -> np.ndarray:
        normalized = _normalize_vector(np.asarray(vector, dtype=np.float32))
        if self._query_embedding_cache_limit > 0:
            if len(self._query_embedding_cache) >= self._query_embedding_cache_limit:
                self._query_embedding_cache.pop(next(iter(self._query_embedding_cache)))
            self._query_embedding_cache[query] = normalized
        return normalized


def get_hybrid_retriever(profile: str = None) -> HybridRetriever:
    clean_profile = _normalize_profile_name(
        profile or os.getenv("RAG_PROFILE") or DEFAULT_PROFILE
    )
    retriever = _RETRIEVERS.get(clean_profile)
    if retriever is None:
        with _RETRIEVER_LOCK:
            retriever = _RETRIEVERS.get(clean_profile)
            if retriever is None:
                retriever = HybridRetriever.from_profile(clean_profile)
                _RETRIEVERS[clean_profile] = retriever
    return retriever


def register_uploaded_document(
    source: dict,
    chunks: Sequence[dict],
    embeddings: Sequence[Sequence[float]],
) -> None:
    with _RETRIEVER_LOCK:
        retrievers = list(_RETRIEVERS.values())
    for retriever in retrievers:
        retriever.add_document(source=source, chunks=chunks, embeddings=embeddings)


def tokenize_bm25(text: str) -> List[str]:
    clean_text = str(text or "").lower()
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", clean_text)
        if token and token not in _STOP_WORDS
    ]
    for sequence in re.findall(r"[\u4e00-\u9fff]+", clean_text):
        if len(sequence) == 1:
            tokens.append(sequence)
            continue
        for size in (2, 3):
            tokens.extend(
                sequence[index : index + size]
                for index in range(max(0, len(sequence) - size + 1))
            )
    return tokens


def expand_retrieval_query(
    question: str,
    retrieval_query: str,
    aliases: Sequence[dict],
    query_expansions: Optional[Sequence[dict]] = None,
):
    original_parts = [str(question or "").strip(), str(retrieval_query or "").strip()]
    base_query = " ".join(dict.fromkeys(part for part in original_parts if part))
    concept_terms = _concept_expansion_terms(question, query_expansions or [])
    lower_query = base_query.lower()
    query_tokens = set(tokenize_bm25(base_query))
    additions = list(concept_terms)
    matched_aliases = []

    for record in aliases:
        primary_terms = [record.get("canonical"), *(record.get("aliases") or [])]
        support_terms = [*(record.get("related_terms") or []), *(record.get("triggers") or [])]
        primary_terms = [str(term).strip() for term in primary_terms if str(term or "").strip()]
        support_terms = [str(term).strip() for term in support_terms if str(term or "").strip()]

        matched = any(_query_contains_term(lower_query, query_tokens, term) for term in primary_terms)
        if not matched:
            matched = any(len(term) > 2 and term.lower() in lower_query for term in support_terms)
        if not matched:
            continue

        canonical = str(record.get("canonical") or "").strip()
        if canonical:
            matched_aliases.append(canonical)
        additions.extend(primary_terms)
        additions.extend(support_terms)

    unique_additions = list(dict.fromkeys(term for term in additions if term))
    combined_query = " ".join([base_query, *unique_additions]).strip()
    return combined_query, unique_additions, list(dict.fromkeys(matched_aliases))


def _normalize_fusion_weights(bm25_weight: float, embedding_weight: float) -> tuple[float, float]:
    bm25 = max(0.0, float(bm25_weight))
    embedding = max(0.0, float(embedding_weight))
    total = bm25 + embedding
    if total <= 0.0:
        return 0.5, 0.5
    return bm25 / total, embedding / total


def reciprocal_rank_fusion(
    bm25_results: Sequence[dict],
    embedding_results: Sequence[dict],
    rrf_k: Optional[int] = None,
    bm25_weight: Optional[float] = None,
    embedding_weight: Optional[float] = None,
) -> List[dict]:
    return reciprocal_rank_fusion_many(
        bm25_result_sets=[bm25_results],
        embedding_result_sets=[embedding_results],
        rrf_k=rrf_k,
        bm25_weight=bm25_weight,
        embedding_weight=embedding_weight,
    )


def reciprocal_rank_fusion_many(
    bm25_result_sets: Sequence[Sequence[dict]],
    embedding_result_sets: Sequence[Sequence[dict]],
    rrf_k: Optional[int] = None,
    bm25_weight: Optional[float] = None,
    embedding_weight: Optional[float] = None,
) -> List[dict]:
    """Fuse sparse and dense rankings from complementary query variants."""

    settings = RagConfig.from_env()
    rrf_k = max(1, int(rrf_k or settings.hybrid_rrf_k))
    bm25_weight, embedding_weight = _normalize_fusion_weights(
        settings.keyword_weight if bm25_weight is None else bm25_weight,
        settings.embedding_weight if embedding_weight is None else embedding_weight,
    )
    candidates: Dict[int, dict] = {}
    result_sets = [
        ("bm25", results) for results in bm25_result_sets
    ] + [
        ("embedding", results) for results in embedding_result_sets
    ]
    for branch, results in result_sets:
        for rank, result in enumerate(results, start=1):
            index = int(result["index"])
            candidate = candidates.setdefault(
                index,
                {
                    "index": index,
                    "fusion_score": 0.0,
                    "bm25_score": 0.0,
                    "embedding_score": 0.0,
                    "matched_terms": [],
                },
            )
            branch_weight = bm25_weight if branch == "bm25" else embedding_weight
            candidate["fusion_score"] += branch_weight / (rrf_k + rank)
            if branch == "bm25":
                candidate["bm25_score"] = max(
                    candidate["bm25_score"],
                    float(result.get("score", 0.0)),
                )
                candidate["matched_terms"] = list(
                    dict.fromkeys(
                        [
                            *candidate["matched_terms"],
                            *(result.get("matched_terms") or []),
                        ]
                    )
                )
            else:
                candidate["embedding_score"] = max(
                    candidate["embedding_score"],
                    float(result.get("score", 0.0)),
                )

    for candidate in candidates.values():
        candidate["rrf_score"] = float(candidate.get("fusion_score", 0.0))
    return sorted(candidates.values(), key=lambda item: item["fusion_score"], reverse=True)


def rerank_candidates(
    candidates: Sequence[dict],
    chunks: Sequence[dict],
    query: str,
    top_k: int,
    settings: Optional[RagConfig] = None,
    chunk_token_sets: Optional[Sequence[Set[str]]] = None,
) -> List[dict]:
    if not candidates:
        return []
    maximum_bm25 = max((float(item.get("bm25_score", 0.0)) for item in candidates), default=1.0)
    embedding_values = [float(item.get("embedding_score", 0.0)) for item in candidates]
    minimum_embedding = min(embedding_values, default=0.0)
    maximum_embedding = max(embedding_values, default=1.0)
    embedding_span = max(maximum_embedding - minimum_embedding, 1e-9)
    maximum_fusion = max(
        (
            float(
                item.get(
                    "lambda_mart_fusion_score",
                    item.get("fusion_score", 0.0),
                )
            )
            for item in candidates
        ),
        default=1.0,
    )
    query_terms = list(dict.fromkeys(tokenize_bm25(query)))

    settings = (settings or RagConfig.from_env()).normalized()
    reranked = []
    for candidate in candidates:
        chunk_index = int(candidate["index"])
        if chunk_token_sets is not None and chunk_index < len(chunk_token_sets):
            text_tokens = chunk_token_sets[chunk_index]
        else:
            text_tokens = set(tokenize_bm25(_chunk_text(chunks[chunk_index])))
        covered_terms = [term for term in query_terms if term in text_tokens]
        coverage = len(covered_terms) / max(1, len(query_terms))
        normalized_bm25 = (
            float(candidate.get("bm25_score", 0.0)) / maximum_bm25 if maximum_bm25 else 0.0
        )
        normalized_embedding = (
            float(candidate.get("embedding_score", 0.0)) - minimum_embedding
        ) / embedding_span
        normalized_fusion = (
            float(
                candidate.get(
                    "lambda_mart_fusion_score",
                    candidate.get("fusion_score", 0.0),
                )
            )
            / maximum_fusion
            if maximum_fusion
            else 0.0
        )
        exact_phrase_bonus = _exact_phrase_bonus(query, _chunk_text(chunks[int(candidate["index"])]))
        rerank_score = (
            settings.rerank_fusion_weight * normalized_fusion
            + settings.rerank_bm25_weight * normalized_bm25
            + settings.rerank_embedding_weight * normalized_embedding
            + settings.rerank_coverage_weight * coverage
            + settings.rerank_phrase_weight * exact_phrase_bonus
        )
        reranked.append(
            {
                **candidate,
                "matched_terms": list(dict.fromkeys([*candidate.get("matched_terms", []), *covered_terms]))[:20],
                "rerank_score": rerank_score,
            }
        )

    return sorted(reranked, key=lambda item: item["rerank_score"], reverse=True)[:top_k]


def estimate_confidence(contexts: Sequence[dict], question: str = "") -> str:
    return evaluate_retrieval_evidence(contexts, question=question)["confidence"]


def evaluate_retrieval_evidence(
    contexts: Sequence[dict],
    settings: Optional[RagConfig] = None,
    question: str = "",
) -> dict:
    """Grade retrieved evidence using raw lexical and semantic signals.

    Rerank scores are normalized within each candidate set, so they cannot by
    themselves show whether the best candidate is actually relevant. This gate
    intentionally uses the pre-normalization BM25 and embedding signals.
    """
    settings = (settings or RagConfig.from_env()).normalized()
    if not contexts:
        return {
            "status": "irrelevant",
            "sufficient": False,
            "confidence": "low",
            "reason": "檢索器沒有返回可用片段。",
            "signals": {"bm25": 0.0, "embedding": 0.0, "matchedTerms": 0},
        }

    top = contexts[0]
    raw_signal_keys = {"bm25Score", "embeddingScore", "matchedTerms"}
    if not any(key in top for key in raw_signal_keys):
        return {
            "status": "unknown",
            "sufficient": True,
            "confidence": "medium",
            "reason": "檢索來源未提供原始相關性分數，保留既有相容模式。",
            "signals": {"bm25": 0.0, "embedding": 0.0, "matchedTerms": 0},
        }

    bm25_score = _safe_float(top.get("bm25Score"))
    embedding_score = _safe_float(top.get("embeddingScore"))
    matched_terms = [str(term).strip().lower() for term in (top.get("matchedTerms") or [])]
    generic_terms = {
        "資料", "資訊", "文件", "內容", "說明", "時間", "今天", "目前", "這份",
        "問題", "什麼", "怎麼", "如何", "是否", "有沒有", "沒有", "可以", "使用", "請問",
    }
    meaningful_terms = [
        term for term in matched_terms
        if len(term) > 1 and term not in generic_terms
    ]
    matched_term_count = len(matched_terms)
    meaningful_term_count = len(meaningful_terms)
    requires_answer_value = _requires_answer_value(question)
    relaxed_mode = settings.evidence_gate_mode == "relaxed"
    answer_bearing_evidence = (
        _has_answer_bearing_evidence(contexts, question)
        if requires_answer_value
        else None
    )
    relaxed_relevance = (
        _has_relaxed_relevance(contexts)
        if relaxed_mode
        else False
    )
    lexical_match = (
        bm25_score >= settings.hybrid_min_bm25_score and meaningful_term_count > 0
    )
    strong_semantic_match = embedding_score >= settings.hybrid_high_embedding_score
    semantic_match = embedding_score >= settings.hybrid_min_embedding_score

    if requires_answer_value and not answer_bearing_evidence and not relaxed_relevance:
        status = "ambiguous"
        sufficient = False
        confidence = "low"
        reason = (
            "片段雖可能與主題相關，但沒有同時包含問題主體與所需的數值／期間證據，"
            "不足以回答精確門檻問題。"
        )
    elif relaxed_mode and relaxed_relevance:
        status = "relevant"
        sufficient = True
        confidence = "medium"
        reason = (
            "放寬 Gate：前八個候選片段至少一個具有足夠的原始詞彙／語意相關性；"
            "回答仍必須受檢索證據限制。"
        )
    elif strong_semantic_match or (lexical_match and (semantic_match or meaningful_term_count >= 2)):
        status = "relevant"
        sufficient = True
        confidence = "high"
        reason = "檢索片段具有明確的詞彙或語意相關性，可進入 grounded generation。"
    elif lexical_match:
        status = "ambiguous"
        sufficient = True
        confidence = "medium"
        reason = "檢索片段具有部分相關性；回答時必須受檢索證據限制。"
    elif semantic_match:
        status = "ambiguous"
        sufficient = False
        confidence = "low"
        reason = "片段只有弱語意相似度，缺少具辨識力的詞彙命中，不足以可靠引用。"
    else:
        status = "irrelevant"
        sufficient = False
        confidence = "low"
        reason = "最高排名片段缺乏足夠的詞彙與語意相關性，不應交給模型引用。"

    return {
        "status": status,
        "sufficient": sufficient,
        "confidence": confidence,
        "reason": reason,
        "signals": {
            "bm25": round(bm25_score, 6),
            "embedding": round(embedding_score, 6),
            "matchedTerms": matched_term_count,
            "meaningfulMatchedTerms": meaningful_term_count,
            "requiresAnswerValue": requires_answer_value,
            "answerBearingEvidence": answer_bearing_evidence,
            "gateMode": settings.evidence_gate_mode,
            "relaxedRelevance": relaxed_relevance,
        },
    }


def _requires_answer_value(question: str) -> bool:
    return bool(
        re.search(
            r"(?:多少|幾(?:天|日|年|月|小時|分鐘|公里|公尺|人|項|次)|"
            r"天數|上限|下限|期限|預告期|費率|比例|距離|時長|金額|數量|公式)",
            str(question or ""),
            flags=re.IGNORECASE,
        )
    )


def _has_answer_bearing_evidence(contexts: Sequence[dict], question: str) -> bool:
    anchors = [
        term
        for term in extract_focus_terms(question)
        if 2 <= len(term) <= 16 and term not in _ANSWER_INTENT_TERMS
    ]
    if not anchors:
        anchors = [
            token for token in tokenize_bm25(question)
            if len(token) >= 2 and token not in _ANSWER_INTENT_TERMS
        ]
    value_pattern = _answer_value_pattern(question)
    for context in contexts[:8]:
        text = re.sub(r"\s+", " ", str(context.get("content") or ""))
        for anchor in anchors:
            start = text.lower().find(anchor.lower())
            while start >= 0:
                window = text[max(0, start - 100) : start + len(anchor) + 140]
                if re.search(value_pattern, window, flags=re.IGNORECASE):
                    return True
                start = text.lower().find(anchor.lower(), start + 1)
    return False


def _has_relaxed_relevance(contexts: Sequence[dict]) -> bool:
    """Return whether any retained context has independent raw relevance.

    The original gate only evaluates rank 1 and, for value questions, also
    requires an answer value to occur in a narrow anchor window.  That is too
    brittle for grouped rules, comparisons, and evidence whose useful clause
    is rank 2/3.  Relaxed mode still rejects generic-only matches by requiring
    both a meaningful matched term and a moderate lexical or embedding signal.
    """
    for context in contexts[:8]:
        bm25 = _safe_float(context.get("bm25Score"))
        embedding = _safe_float(context.get("embeddingScore"))
        matched_terms = [str(term).strip().lower() for term in (context.get("matchedTerms") or [])]
        meaningful_terms = [
            term for term in matched_terms
            if len(term) > 1 and term not in {
                "資料", "資訊", "文件", "內容", "說明", "時間", "今天", "目前", "這份",
                "問題", "什麼", "怎麼", "如何", "是否", "有沒有", "沒有", "可以", "使用", "請問",
            }
        ]
        if len(meaningful_terms) >= 2 and (bm25 >= 0.4 or embedding >= 0.40):
            return True
    return False


def _answer_value_pattern(question: str) -> str:
    text = str(question or "")
    if re.search(r"(?:天數|幾天|幾日|預告期|日數)", text):
        return r"[零〇一二兩三四五六七八九十百千\d,.]+\s*(?:日|天)"
    if re.search(r"(?:公里|公尺|距離)", text):
        return r"[零〇一二兩三四五六七八九十百千萬億\d,.]+\s*(?:公里|公尺|km|m)\b"
    if re.search(r"(?:幾年|幾月|期限|時長)", text):
        return r"[零〇一二兩三四五六七八九十百千\d,.]+\s*(?:年|月|日|天|小時|分鐘)"
    if re.search(r"(?:金額|費率|比例)", text):
        return r"(?:新臺幣|台幣|NT\$|\$)?\s*[零〇一二兩三四五六七八九十百千萬億\d,.]+\s*(?:元|%|％|成|倍)?"
    return r"[零〇一二兩三四五六七八九十百千萬億\d,.]+"


_ANSWER_INTENT_TERMS = {
    "規定", "相關", "相關規定", "天數", "天數上限", "上限", "下限",
    "期限", "多少", "幾天", "幾日", "數量", "金額", "比例", "距離",
    "內容", "條件", "方式", "方法", "程序", "流程", "要求", "標準",
}


def _load_valid_embedding_cache(
    embedding_path: Path,
    meta_path: Path,
    model_name: str,
    chunk_count: int,
    fingerprint: str,
):
    if not embedding_path.exists() or not meta_path.exists():
        return None
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("model") != model_name:
            return None
        if int(metadata.get("chunk_count", -1)) != chunk_count:
            return None
        if metadata.get("fingerprint") != fingerprint:
            return None
        embeddings = load_embedding_matrix(embedding_path)
        if len(embeddings) != chunk_count:
            return None
        return embeddings
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _chunks_fingerprint(chunks: Sequence[dict]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(str(chunk.get("id") or "").encode("utf-8"))
        digest.update(str(chunk.get("content") or "").encode("utf-8"))
    return digest.hexdigest()


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def _chunk_text(chunk: dict) -> str:
    return "\n".join(
        str(value or "")
        for value in (
            chunk.get("parent_title"),
            chunk.get("title"),
            chunk.get("source_id") or chunk.get("source"),
            chunk.get("page"),
            chunk.get("content"),
        )
    )


def _query_contains_term(lower_query: str, query_tokens: Set[str], term: str) -> bool:
    lower_term = term.lower()
    if lower_term in lower_query:
        return True
    term_tokens = tokenize_bm25(lower_term)
    return bool(term_tokens) and all(token in query_tokens for token in term_tokens)


def _concept_expansion_terms(
    question: str,
    query_expansions: Sequence[dict],
) -> List[str]:
    lower_question = str(question or "").lower()
    terms = []
    for record in query_expansions:
        markers = record.get("markers") or []
        expansions = record.get("terms") or []
        if any(marker.lower() in lower_question for marker in markers):
            terms.extend(expansions)
    return list(dict.fromkeys(terms))


def _exact_phrase_bonus(query: str, text: str) -> float:
    lower_text = str(text or "").lower()
    phrases = re.findall(r"[a-z0-9]+(?:\s+[a-z0-9]+){1,5}", str(query or "").lower())
    return 1.0 if any(phrase in lower_text for phrase in phrases) else 0.0


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(text or "")))


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000.0, 3)


def _safe_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def load_profile_retrieval_config(
    profile: str,
    project_root: Path = PROJECT_ROOT,
) -> dict:
    clean_profile = _normalize_profile_name(profile)
    profile_root = Path(project_root) / "profiles" / clean_profile
    config_path = profile_root / PROFILE_RETRIEVAL_CONFIG
    payload = {}
    if config_path.exists():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Profile retrieval config must contain a JSON object: {config_path}")

    data_path = _resolve_optional_profile_path(profile_root, payload.get("data_path"))
    if data_path is not None and not data_path.is_file():
        raise FileNotFoundError(f"Profile retrieval data does not exist: {data_path}")

    return {
        "profile": clean_profile,
        "profile_root": profile_root,
        "index_dir": profile_root / "index",
        "data_path": data_path,
        "label": str(payload.get("label") or _default_profile_label(clean_profile)).strip(),
        "sample_queries": _normalize_string_list(payload.get("sample_queries"), limit=12),
        "query_expansions": _normalize_query_expansions(payload.get("query_expansions")),
    }


def list_profile_retrieval_configs(
    project_root: Path = PROJECT_ROOT,
    default_profile: str = DEFAULT_PROFILE,
) -> List[dict]:
    profiles_root = Path(project_root) / "profiles"
    profile_names = []
    if profiles_root.is_dir():
        profile_names.extend(
            child.name
            for child in profiles_root.iterdir()
            if child.is_dir() and (child / PROFILE_RETRIEVAL_CONFIG).is_file()
        )

    clean_default = _normalize_profile_name(default_profile)
    ordered_names = [clean_default, *sorted(set(profile_names) - {clean_default})]
    return [
        load_profile_retrieval_config(profile_name, project_root=project_root)
        for profile_name in ordered_names
    ]


def _normalize_profile_name(profile: str) -> str:
    clean_profile = str(profile or DEFAULT_PROFILE).strip()
    if not _PROFILE_NAME_PATTERN.fullmatch(clean_profile):
        raise ValueError("Profile names may contain only letters, numbers, underscores, and hyphens.")
    return clean_profile


def _resolve_optional_profile_path(profile_root: Path, value: object) -> Optional[Path]:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)
    return (profile_root / path).resolve(strict=False)


def _default_profile_label(profile: str) -> str:
    return str(profile or DEFAULT_PROFILE).replace("_", " ").replace("-", " ").title()


def _normalize_string_list(value: object, limit: int) -> List[str]:
    if not isinstance(value, list):
        return []
    return list(
        dict.fromkeys(
            str(item).strip()
            for item in value
            if isinstance(item, str) and str(item).strip()
        )
    )[:limit]


def _normalize_query_expansions(records: Optional[Sequence[dict]]) -> List[dict]:
    normalized = []
    for record in records or []:
        if not isinstance(record, dict):
            continue
        markers = [
            str(marker).strip()
            for marker in (record.get("markers") or [])
            if str(marker).strip()
        ]
        terms = [
            str(term).strip()
            for term in (record.get("terms") or [])
            if str(term).strip()
        ]
        if markers and terms:
            normalized.append({"markers": markers, "terms": terms})
    return normalized
