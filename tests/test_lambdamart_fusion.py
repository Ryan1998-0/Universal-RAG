import unittest

from rag_demo.lambdamart_fusion import (
    LAMBDA_MART_FUSION_METHOD,
    LambdaMARTScoreMapper,
    fuse_candidates_with_lambdamart,
)


class LambdaMARTFusionTests(unittest.TestCase):
    def test_sparse_and_dense_scores_are_mapped_before_weighted_fusion(self):
        candidates = [
            {"index": 0, "bm25_score": 100.0, "embedding_score": 0.20, "fusion_score": 0.1},
            {"index": 1, "bm25_score": 50.0, "embedding_score": 0.80, "fusion_score": 0.2},
            {"index": 2, "bm25_score": 0.0, "embedding_score": 0.50, "fusion_score": 0.3},
        ]

        fused = fuse_candidates_with_lambdamart(
            candidates,
            keyword_weight=0.3,
            embedding_weight=0.7,
        )

        self.assertEqual(LAMBDA_MART_FUSION_METHOD, "lambdamart_score_mapping_v1")
        self.assertTrue(all(0.0 <= item["mapped_bm25_score"] <= 1.0 for item in fused))
        self.assertTrue(all(0.0 <= item["mapped_embedding_score"] <= 1.0 for item in fused))
        for item in fused:
            expected = 0.3 * item["mapped_bm25_score"] + 0.7 * item["mapped_embedding_score"]
            self.assertAlmostEqual(item["lambda_mart_fusion_score"], expected, places=7)
        # A raw BM25 value is never added to a raw cosine value.
        self.assertNotEqual(fused[0]["lambda_mart_fusion_score"], 100.0 + 0.20)

    def test_zero_branch_misses_stay_at_the_floor(self):
        mapped = LambdaMARTScoreMapper().map_scores([0.0, 2.0, 4.0], branch="bm25")

        self.assertEqual(mapped[0], 0.0)
        self.assertEqual(mapped[-1], 1.0)
        self.assertGreater(mapped[1], 0.0)


if __name__ == "__main__":
    unittest.main()
