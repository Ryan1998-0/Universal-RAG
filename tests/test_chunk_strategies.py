import unittest

from rag_demo.chunk_strategies import (
    CHUNK_STRATEGY_DYNAMIC,
    CHUNK_STRATEGY_HARD,
    estimate_token_count,
    split_sentence_units,
    split_text,
)
from rag_demo.config import RagConfig


class ChunkStrategyTests(unittest.TestCase):
    def test_hard_strategy_uses_fixed_character_windows_without_overlap(self):
        chunks = list(split_text("abcdefghij", 4, 4, strategy=CHUNK_STRATEGY_HARD))
        self.assertEqual(chunks, ["abcd", "efgh", "ij"])

    def test_dynamic_strategy_keeps_sentence_boundaries_and_carries_overlap(self):
        text = "甲" * 29 + "。" + "乙" * 29 + "。" + "丙" * 29 + "。"
        sentences = split_sentence_units(text)
        self.assertEqual(len(sentences), 3)
        chunks = list(
            split_text(
                text,
                chunk_size=65,
                chunk_stride=65,
                strategy=CHUNK_STRATEGY_DYNAMIC,
                overlap_tokens=31,
            )
        )
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0].endswith("。"))
        self.assertTrue(chunks[1].endswith("。"))
        self.assertIn("乙", chunks[0])
        self.assertIn("乙", chunks[1])
        self.assertLessEqual(estimate_token_count(chunks[1]), 65)

    def test_config_exposes_chunking_ab_settings(self):
        settings = RagConfig.from_env(
            {
                "RAG_CHUNK_STRATEGY": "hard",
                "RAG_CHUNK_SIZE": "600",
                "RAG_CHUNK_STRIDE": "600",
                "RAG_CHUNK_OVERLAP_TOKENS": "200",
                "RAG_QUERY_REWRITE_MIN_SIMILARITY": "0.60",
                "RAG_KEYWORD_WEIGHT": "0.50",
                "RAG_EMBEDDING_WEIGHT": "0.50",
                "RAG_RERANK_TOP_K": "8",
            }
        )
        self.assertEqual(settings.chunk_strategy, "hard")
        self.assertEqual(settings.chunk_size, 600)
        self.assertEqual(settings.chunk_stride, 600)
        self.assertEqual(settings.chunk_overlap_tokens, 200)
        self.assertEqual(settings.query_rewrite_min_similarity, 0.60)
        self.assertEqual(settings.keyword_weight, 0.50)
        self.assertEqual(settings.embedding_weight, 0.50)
        self.assertEqual(settings.rerank_top_k, 8)


if __name__ == "__main__":
    unittest.main()
