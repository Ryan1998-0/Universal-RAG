import unittest

from rag_demo.query_complexity import classify_query_complexity


class QueryComplexityTests(unittest.TestCase):
    def test_short_single_fact_question_is_simple(self):
        decision = classify_query_complexity("蘋果是什麼？")

        self.assertEqual(decision.label, "simple")
        self.assertFalse(decision.is_complex)

    def test_multi_condition_comparison_is_complex(self):
        decision = classify_query_complexity("比較 A 與 B 的差異，並說明各自限制？")

        self.assertEqual(decision.label, "complex")
        self.assertTrue(decision.is_complex)
        self.assertIn("包含多條件或並列語句", decision.reasons)

    def test_acronym_definition_uses_precise_path(self):
        decision = classify_query_complexity("CSM 是什麼？")

        self.assertEqual(decision.label, "complex")
        self.assertIn("專有縮寫定義需要精確比對", decision.reasons)


if __name__ == "__main__":
    unittest.main()
