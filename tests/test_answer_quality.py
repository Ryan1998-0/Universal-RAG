import unittest

from rag_demo.answer_quality import evaluate_answer_quality


class AnswerQualityTests(unittest.TestCase):
    def test_supported_answer_passes_evidence_and_has_no_hallucination(self):
        result = evaluate_answer_quality(
            question="期限？",
            answer="申請期限為三日。來源：[1]",
            contexts=[{"rank": 1, "title": "規則", "content": "申請期限為三日。"}],
            expected_facts=["三日"],
        )

        evidence = result["answer_in_retrieved_chunks"]
        hallucination = result["hallucination"]
        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["score"], 1.0)
        self.assertFalse(hallucination["detected"])
        self.assertEqual(result["expected_fact_coverage"]["answer_score"], 1.0)

    def test_unsupported_claim_and_number_are_hallucination(self):
        result = evaluate_answer_quality(
            question="期限？",
            answer="申請期限為五日。來源：[1]",
            contexts=[{"rank": 1, "title": "規則", "content": "申請期限為三日。"}],
        )

        self.assertFalse(result["answer_in_retrieved_chunks"]["passed"])
        self.assertTrue(result["hallucination"]["detected"])
        self.assertIn("五日", result["hallucination"]["unsupported_numbers"])

    def test_refusal_is_not_marked_as_hallucination(self):
        result = evaluate_answer_quality(
            question="未知問題？",
            answer="根據目前檢索資料無法確認。",
            contexts=[{"rank": 1, "content": "沒有相關內容。"}],
        )

        self.assertFalse(result["answer_in_retrieved_chunks"]["passed"])
        self.assertTrue(result["hallucination"]["hallucination_free"])

    def test_cited_comparative_inference_is_supported_by_expected_facts(self):
        result = evaluate_answer_quality(
            question="兩版本相同嗎？",
            answer="期限為三日。兩者相同。來源：[1]",
            contexts=[{"rank": 1, "title": "規則", "content": "兩版本期限均為三日。"}],
            expected_facts=["三日"],
        )

        self.assertTrue(result["answer_in_retrieved_chunks"]["passed"])
        self.assertFalse(result["hallucination"]["detected"])


if __name__ == "__main__":
    unittest.main()
