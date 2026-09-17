import unittest
from unittest.mock import patch

from rag_demo.web_app import (
    _answer_from_browser_contexts,
    _normalize_retrieval_decision,
    _rag_conversation_context,
    _runtime_config_payload,
)


class AdaptiveWebAppTest(unittest.TestCase):
    def test_normalizes_no_retrieval_decision(self):
        decision = _normalize_retrieval_decision({
            "needs_retrieval": False,
            "reason": "一般問題",
            "retrieval_query": "你好",
        })

        self.assertEqual(decision, {
            "needs_retrieval": False,
            "reason": "一般問題",
            "retrieval_query": "你好",
        })

    def test_rejects_non_object_decision(self):
        self.assertIsNone(_normalize_retrieval_decision("false"))

    def test_rag_context_excludes_old_assistant_answers(self):
        context = _rag_conversation_context(
            [
                {"role": "user", "content": "今天幾號？"},
                {"role": "assistant", "content": "資料不足，無法得知今天幾號。"},
                {"role": "user", "content": "CSM 是什麼？"},
            ]
        )

        self.assertIn("今天幾號？", context)
        self.assertIn("CSM 是什麼？", context)
        self.assertNotIn("資料不足", context)

    def test_runtime_config_exposes_data_driven_profiles_and_model(self):
        runtime = _runtime_config_payload()

        profile_ids = {profile["id"] for profile in runtime["profiles"]}
        self.assertIn("labor_standards_act", profile_ids)
        self.assertEqual(runtime["models"][0]["id"], runtime["default_model"])
        self.assertGreater(runtime["retrieval"]["top_k"], 0)

    def test_threshold_answer_bypasses_model_and_uses_matching_range(self):
        contexts = [
            {
                "rank": 1,
                "title": "資遣預告",
                "page": "章節 4",
                "content": (
                    "工作一年以上、三年未滿：二十日前預告。\n"
                    "工作三年以上：三十日前預告。"
                ),
            }
        ]
        with patch("rag_demo.web_app.ask_model") as ask_model:
            answer = _answer_from_browser_contexts(
                question="那剛好滿三年呢？",
                contexts=contexts,
                model="unused",
                history=[
                    {"role": "user", "content": "工作滿兩年被資遣，要提前多久通知？"},
                    {"role": "assistant", "content": "錯誤的舊答案：二十日。"},
                ],
            )

        ask_model.assert_not_called()
        self.assertIn("三十日前預告", answer)
        self.assertNotIn("二十日前預告", answer)


if __name__ == "__main__":
    unittest.main()
