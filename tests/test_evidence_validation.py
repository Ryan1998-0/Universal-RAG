import unittest

from rag_demo.evidence_validation import validate_answer_evidence


class EvidenceValidationTests(unittest.TestCase):
    def setUp(self):
        self.contexts = [
            {"rank": 1, "content": "文件內容"},
            {"rank": 2, "content": "另一段文件內容"},
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


if __name__ == "__main__":
    unittest.main()
