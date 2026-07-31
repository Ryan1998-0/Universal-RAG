import unittest
from datetime import datetime, timezone

from rag_demo.general_answer import (
    answer_current_datetime_question,
    build_qwen_direct_system_prompt,
    build_qwen_rag_system_prompt,
    is_current_datetime_question,
)


class QwenRagSystemPromptTests(unittest.TestCase):
    def test_includes_current_taipei_date_and_time(self):
        prompt = build_qwen_rag_system_prompt(
            datetime(2026, 7, 20, 6, 30, 45, tzinfo=timezone.utc)
        )

        self.assertIn("目前系統日期：2026-07-20", prompt)
        self.assertIn("目前星期：星期一", prompt)
        self.assertIn("目前系統時間：14:30:45", prompt)
        self.assertIn("目前時區：Asia/Taipei", prompt)
        self.assertIn("不可自行推算或依模型記憶猜測", prompt)

    def test_includes_long_term_memories_with_evidence_boundary(self):
        prompt = build_qwen_rag_system_prompt(
            datetime(2026, 7, 20, 6, 30, 45, tzinfo=timezone.utc),
            memories=["使用者偏好繁體中文", "目前專案使用 Qwen 2.5 7B"],
        )

        self.assertIn("### 使用者長期記憶", prompt)
        self.assertIn("- 使用者偏好繁體中文", prompt)
        self.assertIn("不可取代專業文件證據", prompt)

    def test_answers_follow_up_weekday_from_taipei_system_clock(self):
        answer = answer_current_datetime_question(
            "那麼今天是星期幾？",
            datetime(2026, 7, 23, 2, 9, 38, tzinfo=timezone.utc),
        )

        self.assertEqual(answer, "今天是 2026 年 7 月 23 日，星期四。")

    def test_answers_combined_date_and_weekday_question(self):
        answer = answer_current_datetime_question(
            "今天是幾月幾日、星期幾？",
            datetime(2026, 7, 23, 2, 9, 38, tzinfo=timezone.utc),
        )

        self.assertEqual(answer, "今天是 2026 年 7 月 23 日，星期四。")

    def test_answers_short_follow_up_prefix(self):
        answer = answer_current_datetime_question(
            "那今天星期幾？",
            datetime(2026, 7, 23, 2, 9, 38, tzinfo=timezone.utc),
        )

        self.assertEqual(answer, "今天是 2026 年 7 月 23 日，星期四。")

    def test_answers_current_time_from_taipei_system_clock(self):
        answer = answer_current_datetime_question(
            "請問現在幾點？",
            datetime(2026, 7, 23, 2, 9, 38, tzinfo=timezone.utc),
        )

        self.assertEqual(answer, "現在是 2026 年 7 月 23 日（星期四）10:09，時區 Asia/Taipei。")

    def test_timezone_is_runtime_configurable(self):
        answer = answer_current_datetime_question(
            "現在幾點？",
            datetime(2026, 7, 23, 2, 9, 38, tzinfo=timezone.utc),
            timezone_name="UTC",
        )

        self.assertEqual(answer, "現在是 2026 年 7 月 23 日（星期四）02:09，時區 UTC。")

    def test_does_not_capture_domain_questions_that_mention_today(self):
        self.assertFalse(is_current_datetime_question("IFRS 17 今天有哪些新規定？"))
        self.assertIsNone(answer_current_datetime_question("IFRS 17 今天有哪些新規定？"))

    def test_direct_system_prompt_allows_basic_questions_without_document_evidence(self):
        prompt = build_qwen_direct_system_prompt(
            datetime(2026, 7, 23, 2, 9, 38, tzinfo=timezone.utc)
        )

        self.assertIn("基本問題", prompt)
        self.assertIn("算術", prompt)
        self.assertNotIn("請忠實根據提供的檢索資料回答", prompt)
        self.assertIn("目前星期：星期四", prompt)


if __name__ == "__main__":
    unittest.main()
