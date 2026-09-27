import unittest

from rag_demo.config import RagConfig
from rag_demo.production.retriever import ProductionHybridRetriever
from rag_demo.production.vector_repository import DenseSearchHit
from rag_demo.rag_pipeline import normalize_contexts
from rag_demo.retrieval_scope import RetrievalScope


class FakeRuntime:
    def __init__(self):
        self.rerank_calls = 0

    def embed_query(self, query):
        return [0.2, 0.8]

    def sparse_query(self, query):
        return {"indices": [1, 2], "values": [0.4, 0.6]}

    def rerank(self, query, documents):
        self.rerank_calls += 1
        return [0.1 if "weak" in document else 0.9 for document in documents]


class FakeVectorRepository:
    def __init__(self):
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(("dense", kwargs))
        return [
            DenseSearchHit("point-a", 0.82, _payload("chunk-a", "weak evidence")),
            DenseSearchHit("point-b", 0.71, _payload("chunk-b", "strong CSM evidence")),
        ]

    def search_sparse(self, **kwargs):
        self.calls.append(("sparse", kwargs))
        return [
            DenseSearchHit("point-b", 4.2, _payload("chunk-b", "strong CSM evidence")),
            DenseSearchHit("point-a", 2.1, _payload("chunk-a", "weak evidence")),
        ]


def _payload(chunk_id, content):
    return {
        "chunk_id": chunk_id,
        "chunk_record_id": f"record-{chunk_id}",
        "document_version_id": "version-1",
        "index_version_id": "index-1",
        "source_id": "source-1",
        "title": "Policy",
        "page": "7",
        "content": content,
    }


class ProductionHybridRetrieverTests(unittest.TestCase):
    def test_planned_variants_and_evidence_query_are_retrieved_with_same_scope(self):
        class QueryRuntime(FakeRuntime):
            def __init__(self):
                super().__init__()
                self.queries = []

            def embed_query(self, query):
                self.queries.append(query)
                return query

            def sparse_query(self, query):
                return query

        class QueryRepository(FakeVectorRepository):
            def search(self, **kwargs):
                self.calls.append(("dense", kwargs))
                query = kwargs["query_vector"]
                chunk_id = "focus" if query == "focused evidence" else "common"
                return [DenseSearchHit(chunk_id, 0.8, _payload(chunk_id, query))]

            def search_sparse(self, **kwargs):
                self.calls.append(("sparse", kwargs))
                query = kwargs["query_sparse_vector"]
                chunk_id = "focus" if query == "focused evidence" else "common"
                return [DenseSearchHit(chunk_id, 4.0, _payload(chunk_id, query))]

        vectors = QueryRepository()
        runtime = QueryRuntime()
        retriever = ProductionHybridRetriever(
            vectors, runtime, settings=RagConfig(multi_query_max_variants=4)
        )
        scope = RetrievalScope("tenant-a", "kb-a", "index-1")

        result = retriever.retrieve(
            question="main question",
            retrieval_query="primary",
            query_variants=("primary", "alternate"),
            evidence_query="focused evidence",
            source_ids=["source-1"],
            top_k=2,
            candidate_k=8,
            retrieval_scope=scope,
        )

        self.assertEqual(
            result["retrievalQueries"],
            ["main question primary", "primary", "alternate", "focused evidence"],
        )
        self.assertEqual(runtime.queries, result["retrievalQueries"])
        self.assertEqual(result["diagnostics"]["perQueryCandidateK"], 2)
        self.assertEqual(len(vectors.calls), 8)
        self.assertTrue(all(kwargs["scope"] is scope for _, kwargs in vectors.calls))
        self.assertTrue(all(kwargs["source_ids"] == ["source-1"] for _, kwargs in vectors.calls))
        self.assertTrue(all(kwargs["top_k"] == 2 for _, kwargs in vectors.calls))
        self.assertIn("focus", [context["id"] for context in result["contexts"]])

    def test_multi_query_can_be_disabled(self):
        vectors = FakeVectorRepository()
        runtime = FakeRuntime()
        retriever = ProductionHybridRetriever(
            vectors, runtime, settings=RagConfig(multi_query_enabled=False)
        )
        result = retriever.retrieve(
            question="main question",
            retrieval_query="primary",
            query_variants=("alternate",),
            evidence_query="focused evidence",
            retrieval_scope=RetrievalScope("tenant-a", "kb-a", "index-1"),
        )

        self.assertEqual(result["retrievalQueries"], ["main question primary"])
        self.assertEqual(len(vectors.calls), 2)

    def test_hybrid_branches_share_scope_and_cross_encoder_sets_final_order(self):
        vectors = FakeVectorRepository()
        runtime = FakeRuntime()
        retriever = ProductionHybridRetriever(vectors, runtime)
        scope = RetrievalScope("tenant-a", "kb-a", "index-1")

        result = retriever.retrieve(
            question="CSM 是什麼？",
            retrieval_query="contractual service margin",
            source_ids=["source-1"],
            top_k=2,
            candidate_k=8,
            retrieval_scope=scope,
        )

        self.assertEqual([call[0] for call in vectors.calls], ["dense", "sparse"])
        for _, kwargs in vectors.calls:
            self.assertIs(kwargs["scope"], scope)
            self.assertEqual(kwargs["source_ids"], ["source-1"])
        self.assertEqual(result["contexts"][0]["id"], "chunk-b")
        self.assertEqual(result["contexts"][0]["bm25Score"], 4.2)
        self.assertEqual(result["contexts"][0]["embeddingScore"], 0.71)
        self.assertEqual(result["contexts"][0]["documentVersionId"], "version-1")
        self.assertTrue(result["diagnostics"]["rerankApplied"])
        self.assertEqual(runtime.rerank_calls, 1)

    def test_simple_production_query_skips_cross_encoder_and_returns_top_three(self):
        class ManyHitsRepository(FakeVectorRepository):
            def search(self, **kwargs):
                return [
                    DenseSearchHit(f"point-{index}", 0.90 - index * 0.05, _payload(f"chunk-{index}", "apple evidence"))
                    for index in range(5)
                ]

            def search_sparse(self, **kwargs):
                return [
                    DenseSearchHit(f"point-{index}", 5.0 - index, _payload(f"chunk-{index}", "apple evidence"))
                    for index in range(5)
                ]

        vectors = ManyHitsRepository()
        runtime = FakeRuntime()
        retriever = ProductionHybridRetriever(vectors, runtime)
        result = retriever.retrieve(
            question="apple",
            top_k=8,
            candidate_k=8,
            retrieval_scope=RetrievalScope("tenant-a", "kb-a", "index-1"),
        )

        self.assertEqual(len(result["contexts"]), 3)
        self.assertFalse(result["diagnostics"]["rerankApplied"])
        self.assertEqual(runtime.rerank_calls, 0)

    def test_missing_active_index_fails_closed_without_vector_query(self):
        vectors = FakeVectorRepository()
        retriever = ProductionHybridRetriever(vectors, FakeRuntime())
        result = retriever.retrieve(
            question="Find evidence",
            retrieval_scope=RetrievalScope("tenant-a", "kb-a"),
        )
        self.assertEqual(result["contexts"], [])
        self.assertEqual(vectors.calls, [])

    def test_pipeline_normalization_preserves_versioned_citation_keys(self):
        normalized = normalize_contexts([{
            "id": "chunk-1",
            "content": "evidence",
            "documentVersionId": "version-3",
            "chunkRecordId": "record-4",
            "indexVersionId": "index-5",
        }])
        self.assertEqual(normalized[0]["documentVersionId"], "version-3")
        self.assertEqual(normalized[0]["chunkRecordId"], "record-4")
        self.assertEqual(normalized[0]["indexVersionId"], "index-5")


if __name__ == "__main__":
    unittest.main()
