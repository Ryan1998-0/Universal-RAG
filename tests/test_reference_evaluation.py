import json
import unittest

from rag_demo.reference_evaluation import (
    QUALITY_RUBRIC,
    _friendly_claude_error,
    evaluate_qwen_with_claude,
    normalize_dimension_scores,
)


class ClaudeReferenceEvaluationTests(unittest.TestCase):
    def test_same_final_prompt_is_sent_to_reference_model_and_scores_are_capped(self):
        calls = []

        def runner(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return {"text": "Claude 參考答案。來源：[1]", "duration_ms": 12}
            judgment = {
                "dimensions": [
                    {"id": item["id"], "score": item["max_score"], "reason": "通過"}
                    for item in QUALITY_RUBRIC
                ],
                "contradicts_evidence": True,
                "critical_unsupported_claim": False,
                "failed_required_abstention": False,
                "summary": "候選答案有一項重大矛盾。",
                "strengths": ["引用格式正確"],
                "improvements": ["修正矛盾步驟"],
            }
            return {"text": json.dumps(judgment, ensure_ascii=False), "duration_ms": 8}

        result = evaluate_qwen_with_claude(
            question="怎麼操作？",
            final_prompt="EXACT FINAL PROMPT",
            system_prompt="EXACT SYSTEM PROMPT",
            qwen_answer="Qwen 回答。來源：[1]",
            contexts=[{"rank": 1, "title": "手冊", "content": "正確步驟"}],
            runner=runner,
        )

        self.assertEqual(calls[0]["prompt"], "EXACT FINAL PROMPT")
        self.assertEqual(calls[0]["system"], "EXACT SYSTEM PROMPT")
        self.assertEqual(result["reference"]["score"], 100)
        self.assertEqual(result["candidate"]["raw_score"], 100)
        self.assertEqual(result["candidate"]["score"], 49)
        self.assertTrue(result["score_cap"]["applied"])

    def test_dimension_scores_are_clamped_and_missing_dimensions_score_zero(self):
        dimensions = normalize_dimension_scores([
            {"id": "grounded_correctness", "score": 99, "reason": "too high"},
            {"id": "language_terminology", "score": -2, "reason": "too low"},
        ])

        self.assertEqual(dimensions[0]["score"], 35)
        self.assertEqual(dimensions[-1]["score"], 0)
        self.assertEqual(sum(item["score"] for item in dimensions), 35)

    def test_expired_login_error_is_sanitized(self):
        error = _friendly_claude_error(
            json.dumps({
                "api_error_status": 401,
                "result": "OAuth access token has expired.",
                "session_id": "must-not-leak",
            }),
            "",
        )

        self.assertIn("登入已過期", error)
        self.assertNotIn("session_id", error)


if __name__ == "__main__":
    unittest.main()
