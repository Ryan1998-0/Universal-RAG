import unittest

from rag_demo.config import RagConfig
from rag_demo.hybrid_retrieval import HybridRetriever
from rag_demo.parent_child import build_parent_child_index, expand_child_contexts


class ParentChildChunkTests(unittest.TestCase):
    def test_children_fit_inside_parent_and_keep_parent_evidence(self):
        text = "".join(f"第{i}項規定員工權益與申請程序。" for i in range(180))
        index = build_parent_child_index(
            [{"title": "Article 1", "page": "1", "content": text}],
            source_id="source-1",
            filename="policy.txt",
            source_type="text",
            extraction_method="text",
            parent_size_tokens=64,
            child_size_tokens=16,
        )

        self.assertGreater(len(index.parents), 1)
        self.assertGreater(len(index.children), len(index.parents))
        parent_ids = {parent["id"] for parent in index.parents}
        for child in index.children:
            self.assertEqual(child["chunk_level"], "child")
            self.assertIn(child["parent_id"], parent_ids)
            self.assertLessEqual(child["token_count"], 16)
            self.assertIn(child["parent_content"], {parent["content"] for parent in index.parents})

    def test_expansion_deduplicates_children_to_parent_contexts(self):
        index = build_parent_child_index(
            [{"title": "Article 1", "content": "甲。乙。丙。"}],
            source_id="source-1",
            filename="policy.txt",
            source_type="text",
            extraction_method="text",
            parent_size_tokens=32,
            child_size_tokens=4,
        )
        selected = [dict(index.children[0], rrf_score=0.9)] * 2
        expanded = expand_child_contexts(selected, parent_by_id=index.parent_by_id, limit=5)

        self.assertEqual(len(expanded), 1)
        self.assertEqual(expanded[0]["chunk_level"], "parent")
        self.assertEqual(expanded[0]["content"], index.parents[0]["content"])

    def test_hybrid_retrieval_returns_parent_content_for_rrf_child_hits(self):
        index = build_parent_child_index(
            [{"title": "Article 1", "content": "蘋果規定需要保存完整證據。"}],
            source_id="source-1",
            filename="policy.txt",
            source_type="text",
            extraction_method="text",
            parent_size_tokens=64,
            child_size_tokens=8,
        )
        chunks = index.children
        retriever = HybridRetriever(
            chunks=chunks,
            aliases=[],
            embeddings=[[1.0, 0.0] for _ in chunks],
            embed_query_fn=lambda _: [1.0, 0.0],
            settings=RagConfig(
                hybrid_fusion_method="rrf",
                complexity_routing_enabled=True,
                simple_query_top_k=5,
            ),
        )

        result = retriever.retrieve("蘋果", top_k=5, candidate_k=100)

        self.assertEqual(result["diagnostics"]["fusionMethod"], "rrf_v1")
        self.assertEqual(result["diagnostics"]["retrievalChunkLevel"], "child")
        self.assertEqual(result["diagnostics"]["evidenceChunkLevel"], "parent")
        self.assertTrue(result["contexts"])
        self.assertEqual(result["contexts"][0]["content"], index.parents[0]["content"])
        self.assertEqual(result["contexts"][0]["parentChunkId"], index.parents[0]["id"])


if __name__ == "__main__":
    unittest.main()
