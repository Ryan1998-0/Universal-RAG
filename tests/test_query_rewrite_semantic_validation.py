import unittest
from unittest.mock import patch

from rag_demo.query_rewriter import (
    decide_and_rewrite_query_for_retrieval,
    validate_rewrite_semantics,
)


class QueryRewriteSemanticValidationTests(unittest.TestCase):
    def test_discards_rewrite_below_cosine_threshold(self):
        vectors = {
            "原始問題": [1.0, 0.0],
            "語意相近改寫": [0.9, 0.1],
            "完全不同內容": [0.0, 1.0],
        }

        def fake_embed(texts, **_kwargs):
            return [vectors[text] for text in texts]

        result = validate_rewrite_semantics(
            "原始問題",
            ["語意相近改寫", "完全不同內容"],
            min_similarity=0.60,
            embedding_fn=fake_embed,
        )

        self.assertEqual(result["status"], "filtered")
        self.assertEqual(result["accepted"], [True, False])
        self.assertGreater(result["similarities"][0], 0.60)
        self.assertLess(result["similarities"][1], 0.60)

    def test_filtered_primary_rewrite_falls_back_to_original_question(self):
        output = "\n".join(
            [
                "語意理解：需要外部法規證據。",
                "是否需要檢索：是",
                "判斷理由：需要法規來源。",
                "向量檢索用查詢：合法 量子物理 完全不同的主題",
            ]
        )

        def fake_embed(texts, **_kwargs):
            return [[1.0, 0.0] if text == "公司這樣做是否合法？" else [0.0, 1.0] for text in texts]

        with patch("rag_demo.query_rewriter.ask_model", return_value=output), patch(
            "rag_demo.query_rewriter.embed_texts", side_effect=fake_embed
        ):
            decision = decide_and_rewrite_query_for_retrieval(
                "公司這樣做是否合法？",
                model="unused",
            )

        self.assertEqual(decision.retrieval_query, "公司這樣做是否合法？")
        self.assertFalse(decision.semantic_accepted)
        self.assertEqual(decision.semantic_validation_status, "filtered")

    def test_disabled_validation_keeps_all_rewrites(self):
        result = validate_rewrite_semantics(
            "原始問題",
            ["完全不同內容"],
            enabled=False,
        )
        self.assertEqual(result["status"], "disabled")
        self.assertEqual(result["accepted"], [True])


if __name__ == "__main__":
    unittest.main()
