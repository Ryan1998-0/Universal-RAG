import unittest

from rag_demo.threshold_evidence import (
    render_threshold_answer,
    resolve_threshold_evidence,
)


class ThresholdEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.contexts = [
            {
                "rank": 1,
                "title": "資遣預告與謀職假",
                "page": "章節 4",
                "content": (
                    "繼續工作三個月以上、一年未滿：十日前預告。\n"
                    "繼續工作一年以上、三年未滿：二十日前預告。\n"
                    "繼續工作三年以上：三十日前預告。剛好滿三年屬於三年以上。"
                ),
            },
            {
                "rank": 2,
                "title": "特別休假日數",
                "page": "章節 9",
                "content": (
                    "二年以上、三年未滿：十日。\n"
                    "三年以上、五年未滿：每年十四日。"
                ),
            },
        ]

    def test_resolves_two_year_notice_period(self):
        evidence = resolve_threshold_evidence(
            "工作滿兩年被資遣，要提前多久通知？",
            self.contexts,
        )

        self.assertIsNotNone(evidence)
        self.assertEqual(evidence.result_text, "二十日前預告")
        self.assertEqual(evidence.rank, 1)

    def test_followup_uses_previous_question_only_to_resolve_intent(self):
        evidence = resolve_threshold_evidence(
            "那剛好滿三年呢？",
            self.contexts,
            previous_user_question="工作滿兩年被資遣，要提前多久通知？",
        )

        self.assertIsNotNone(evidence)
        self.assertEqual(evidence.result_text, "三十日前預告")
        self.assertIn("三年以上", evidence.evidence_line)

    def test_prefers_matching_notice_table_over_other_duration_table(self):
        evidence = resolve_threshold_evidence(
            "工作滿兩年被資遣，要提前多久通知？",
            list(reversed(self.contexts)),
        )

        self.assertIsNotNone(evidence)
        self.assertEqual(evidence.title, "資遣預告與謀職假")

    def test_does_not_activate_for_formula_question(self):
        evidence = resolve_threshold_evidence(
            "依新制，工作三年被資遣時怎麼計算？",
            self.contexts,
        )

        self.assertIsNone(evidence)

    def test_full_formula_question_does_not_borrow_previous_threshold_intent(self):
        evidence = resolve_threshold_evidence(
            "適用勞退新制，月平均工資50,000元且年資剛好3年，資遣費是多少？請列出公式。",
            self.contexts,
            previous_user_question="工作滿三年時，要提前幾天通知資遣？",
        )

        self.assertIsNone(evidence)

    def test_renders_direct_grounded_answer(self):
        evidence = resolve_threshold_evidence(
            "工作滿兩年被資遣，要提前多久通知？",
            self.contexts,
        )

        answer = render_threshold_answer(evidence)

        self.assertIn("二十日前預告", answer)
        self.assertIn("來源：[1]", answer)


if __name__ == "__main__":
    unittest.main()
