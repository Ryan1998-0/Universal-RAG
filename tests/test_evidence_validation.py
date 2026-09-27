import unittest

from rag_demo.evidence_validation import validate_answer_evidence


class EvidenceValidationTests(unittest.TestCase):
    def setUp(self):
        self.contexts = [
            {"rank": 1, "content": "答案內容。第一項有規定。"},
            {"rank": 2, "content": "第二項也有規定。"},
        ]

    def test_accepts_valid_citation(self):
        result = validate_answer_evidence("答案內容。來源：[1]", self.contexts)
        self.assertTrue(result["sufficient"])
        self.assertEqual(result["valid_citations"], [1])

    def test_rejects_unknown_citation(self):
        result = validate_answer_evidence("答案內容。來源：[9]", self.contexts)
        self.assertFalse(result["sufficient"])
        self.assertEqual(result["invalid_citations"], [9])

    def test_accepts_explicit_refusal(self):
        result = validate_answer_evidence(
            "根據目前檢索資料無法確認。",
            self.contexts,
        )
        self.assertEqual(result["status"], "refused")
        self.assertTrue(result["sufficient"])

    def test_accepts_refusal_with_missing_evidence_reason(self):
        answer = "根據目前檢索資料無法確認。缺少的是全勤獎金扣發方式。\n\n來源：[1] [2]"
        result = validate_answer_evidence(answer, self.contexts)
        self.assertEqual(result["status"], "refused")
        self.assertTrue(result["sufficient"])

    def test_rejects_refusal_mixed_with_factual_claim(self):
        answer = "資料不足。不過所有員工都可領一百萬元。來源：[1]"
        result = validate_answer_evidence(answer, self.contexts)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["sufficient"])

    def test_rejects_multiple_claims_with_only_footer_citation(self):
        answer = "第一項有規定。第二項也有規定。來源：[1]"
        result = validate_answer_evidence(answer, self.contexts)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["uncited_claims"], ["第一項有規定。", "第二項也有規定。"])

    def test_accepts_each_sentence_cited_after_punctuation(self):
        answer = "第一項有規定。[1] 第二項也有規定。[2] 來源：[1], [2]"
        result = validate_answer_evidence(answer, self.contexts)
        self.assertTrue(result["sufficient"])
        self.assertEqual(result["uncited_claims"], [])

    def test_rejects_source_footer_without_answer(self):
        result = validate_answer_evidence("來源：[1]", self.contexts)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["sufficient"])

    def test_rejects_unrelated_claim_despite_valid_citation(self):
        result = validate_answer_evidence("員工可以申請補助。[1]", self.contexts)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["unsupported_claims"], ["員工可以申請補助。"])

    def test_rejects_number_not_in_cited_evidence(self):
        contexts = [{"rank": 1, "content": "所有員工都可領一百元。"}]
        result = validate_answer_evidence("所有員工都可領一百萬元。來源：[1]", contexts)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["unsupported_claims"], ["所有員工都可領一百萬元。"])

    def test_matches_chinese_and_arabic_currency_amounts(self):
        contexts = [{"rank": 1, "content": "所有員工都可領一百萬元。"}]
        result = validate_answer_evidence("所有員工都可領100萬元。來源：[1]", contexts)
        self.assertTrue(result["sufficient"])


if __name__ == "__main__":
    unittest.main()
