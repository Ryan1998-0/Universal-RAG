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


if __name__ == "__main__":
    unittest.main()
