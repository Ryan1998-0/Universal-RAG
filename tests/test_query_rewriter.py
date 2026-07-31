import unittest
from unittest.mock import patch

from rag_demo.query_rewriter import (
    QUERY_REWRITER_SYSTEM_PROMPT,
    build_rewrite_prompt,
    decide_and_rewrite_query_for_retrieval,
    extract_decision_reason,
    extract_needs_retrieval,
    extract_retrieval_query,
    rewrite_query_for_retrieval,
)


class QueryRewriterDecisionTest(unittest.TestCase):
    def test_router_has_no_corpus_inventory_before_retrieval(self):
        prompt = build_rewrite_prompt(
            "這個規範要求什麼？",
            conversation_context="使用者：請查看剛才提到的規範。",
        )

        self.assertNotIn("候選章節", prompt)
        self.assertIn("不知道可檢索資料有哪些文件", prompt)
        self.assertIn("不知道知識庫有哪些文件", QUERY_REWRITER_SYSTEM_PROMPT)
        self.assertIn("不可把查詢壓縮", QUERY_REWRITER_SYSTEM_PROMPT)
        self.assertIn("延長工作時間", prompt)
        self.assertIn("不良改寫", prompt)

    def test_extracts_retrieval_decision_fields(self):
        output = "\n".join(
            [
                "語意理解：使用者想查 IFRS 17 定義。",
                "是否需要檢索：是",
                "判斷理由：問題涉及準則定義，需要 evidence。",
                "向量檢索用查詢：IFRS 17 定義 insurance contracts",
            ]
        )

        self.assertTrue(extract_needs_retrieval(output))
        self.assertEqual(extract_decision_reason(output), "問題涉及準則定義，需要 evidence。")
        self.assertEqual(extract_retrieval_query(output), "IFRS 17 定義 insurance contracts")

    def test_extracts_no_retrieval_decision(self):
        output = "\n".join(
            [
                "語意理解：使用者只是打招呼。",
                "是否需要檢索：否，不需要",
                "判斷理由：打招呼不需要查詢 knowledge base。",
                "向量檢索用查詢：你好",
            ]
        )

        self.assertFalse(extract_needs_retrieval(output))

    def test_obvious_greeting_skips_model_and_retrieval(self):
        decision = decide_and_rewrite_query_for_retrieval("你好", model="unused")

        self.assertFalse(decision.needs_retrieval)
        self.assertEqual(decision.retrieval_query, "你好")
        self.assertIn("不需要", decision.reason)

    def test_long_greeting_is_not_partially_stripped_into_model_call(self):
        decision = decide_and_rewrite_query_for_retrieval("謝謝你", model="unused")

        self.assertFalse(decision.needs_retrieval)
        self.assertEqual(decision.retrieval_query, "謝謝你")

    def test_current_weekday_skips_retrieval_even_with_ifrs_history(self):
        with patch("rag_demo.query_rewriter.ask_model") as ask_model:
            decision = decide_and_rewrite_query_for_retrieval(
                "那麼今天是星期幾？",
                model="unused",
                conversation_context="使用者：新法規在會計上有什麼需求\n助理：IFRS 17 需要新的會計框架。",
            )

        ask_model.assert_not_called()
        self.assertFalse(decision.needs_retrieval)
        self.assertEqual(decision.retrieval_query, "那麼今天是星期幾？")
        self.assertIn("系統時鐘", decision.reason)

    def test_combined_current_date_and_weekday_skips_router_model(self):
        with patch("rag_demo.query_rewriter.ask_model") as ask_model:
            decision = decide_and_rewrite_query_for_retrieval(
                "今天是幾月幾日、星期幾？",
                model="unused",
            )

        ask_model.assert_not_called()
        self.assertFalse(decision.needs_retrieval)

    def test_ambiguous_legality_question_uses_router_rewrite(self):
        model_output = "\n".join(
            [
                "語意理解：使用者想確認延長工時是否合法。",
                "是否需要檢索：是",
                "判斷理由：需要外部法規證據。",
                "向量檢索用查詢：延長工時 加班 每日每週上限 合法性",
            ]
        )
        with patch("rag_demo.query_rewriter.ask_model", return_value=model_output) as ask_model:
            decision = decide_and_rewrite_query_for_retrieval(
                "公司這樣做是否合法？",
                model="unused",
            )

        ask_model.assert_called_once()
        self.assertTrue(decision.needs_retrieval)
        self.assertIn("延長工時", decision.retrieval_query)

    def test_ambiguous_document_question_uses_router_rewrite(self):
        model_output = "\n".join(
            [
                "語意理解：使用者想確認文件是否記載一般正常工時。",
                "是否需要檢索：是",
                "判斷理由：需要查詢上傳文件。",
                "向量檢索用查詢：一般正常工時 每日 每週 上限",
            ]
        )
        with patch("rag_demo.query_rewriter.ask_model", return_value=model_output) as ask_model:
            decision = decide_and_rewrite_query_for_retrieval(
                "這份資料有沒有提到一般上班時間？",
                model="unused",
            )

        ask_model.assert_called_once()
        self.assertTrue(decision.needs_retrieval)
        self.assertIn("正常工時", decision.retrieval_query)

    def test_duration_boundary_followup_retrieves_without_router_model(self):
        with patch("rag_demo.query_rewriter.ask_model") as ask_model:
            decision = decide_and_rewrite_query_for_retrieval(
                "那剛好滿三年呢？",
                model="unused",
            )

        ask_model.assert_not_called()
        self.assertTrue(decision.needs_retrieval)

    def test_simple_arithmetic_skips_retrieval_without_calling_router_model(self):
        with patch("rag_demo.query_rewriter.ask_model") as ask_model:
            decision = decide_and_rewrite_query_for_retrieval(
                "2 + 2 是多少？",
                model="unused",
                conversation_context="使用者：請解釋 IFRS 17。",
            )

        ask_model.assert_not_called()
        self.assertFalse(decision.needs_retrieval)
        self.assertIn("基本算術", decision.reason)

    def test_explicit_law_question_retrieves_without_calling_router_model(self):
        with patch("rag_demo.query_rewriter.ask_model") as ask_model:
            decision = decide_and_rewrite_query_for_retrieval(
                "依勞動基準法，工作三年以上的終止契約預告期是幾天？",
                model="unused",
            )

        ask_model.assert_not_called()
        self.assertTrue(decision.needs_retrieval)
        self.assertIn("法規", decision.reason)

    def test_precise_domain_value_retrieves_without_calling_router_model(self):
        with patch("rag_demo.query_rewriter.ask_model") as ask_model:
            decision = decide_and_rewrite_query_for_retrieval(
                "毅力號從地球到火星的巡航約花多少天，距離約多少公里？",
                model="unused",
            )

        ask_model.assert_not_called()
        self.assertTrue(decision.needs_retrieval)
        self.assertIn("精確", decision.reason)

    def test_acronym_definition_overrides_a_direct_router_decision(self):
        model_output = "\n".join(
            [
                "語意理解：使用者詢問縮寫定義。",
                "是否需要檢索：否",
                "判斷理由：模型誤認為一般常識。",
                "向量檢索用查詢：CSM contractual service margin 定義",
            ]
        )
        with patch("rag_demo.query_rewriter.ask_model", return_value=model_output) as ask_model:
            decision = decide_and_rewrite_query_for_retrieval(
                "CSM 的定義是什麼？",
                model="unused",
            )

        ask_model.assert_called_once()
        self.assertTrue(decision.needs_retrieval)
        self.assertIn("縮寫", decision.reason)
        self.assertIn("contractual service margin", decision.retrieval_query)

    def test_two_letter_general_term_can_still_use_router_model(self):
        model_output = "\n".join(
            [
                "語意理解：使用者詢問一般 AI 概念。",
                "是否需要檢索：否",
                "判斷理由：可直接回答一般概念。",
                "向量檢索用查詢：AI 定義",
            ]
        )
        with patch("rag_demo.query_rewriter.ask_model", return_value=model_output) as ask_model:
            decision = decide_and_rewrite_query_for_retrieval("AI 是什麼？", model="unused")

        ask_model.assert_called_once()
        self.assertFalse(decision.needs_retrieval)

    def test_short_stable_general_fact_can_still_skip_retrieval(self):
        model_output = "\n".join(
            [
                "語意理解：使用者詢問穩定的基礎地理常識。",
                "是否需要檢索：否",
                "判斷理由：這是穩定的一般常識。",
                "向量檢索用查詢：法國首都",
            ]
        )
        with patch("rag_demo.query_rewriter.ask_model", return_value=model_output) as ask_model:
            decision = decide_and_rewrite_query_for_retrieval("法國的首都是哪裡？", model="unused")

        ask_model.assert_called_once()
        self.assertFalse(decision.needs_retrieval)

    def test_text_translation_stays_direct_even_when_text_contains_numbers(self):
        with patch("rag_demo.query_rewriter.ask_model") as ask_model:
            decision = decide_and_rewrite_query_for_retrieval(
                "請翻譯：Perseverance traveled for 203 days.",
                model="unused",
            )

        ask_model.assert_not_called()
        self.assertFalse(decision.needs_retrieval)

    def test_legacy_rewrite_function_still_returns_string(self):
        rewritten = rewrite_query_for_retrieval("你好", model="unused")

        self.assertEqual(rewritten, "你好")

    def test_greeting_prefix_is_removed_before_model_decision(self):
        model_output = "\n".join(
            [
                "語意理解：使用者想查 CSM 的定義。",
                "是否需要檢索：是",
                "判斷理由：問題涉及 IFRS 17 術語，需要 evidence。",
                "向量檢索用查詢：IFRS17 CSM 定義",
            ]
        )
        with patch("rag_demo.query_rewriter.ask_model", return_value=model_output) as ask_model:
            decision = decide_and_rewrite_query_for_retrieval("你好，請問 IFRS17 的 CSM 是什麼？", model="unused")

        prompt = ask_model.call_args.args[0]
        self.assertIn("使用者問題：\nIFRS17 的 CSM 是什麼？", prompt)
        self.assertNotIn("使用者問題：\n你好，請問", prompt)
        self.assertTrue(decision.needs_retrieval)

    def test_named_domain_does_not_trigger_a_hidden_corpus_guard(self):
        model_output = "\n".join(
            [
                "語意理解：使用者想用口語描述 IFRS 17 的目的。",
                "是否需要檢索：否",
                "判斷理由：這只是一般解釋。",
                "向量檢索用查詢：IFRS 17 objective",
            ]
        )
        with patch("rag_demo.query_rewriter.ask_model", return_value=model_output):
            decision = decide_and_rewrite_query_for_retrieval(
                "When explaining the purpose of IFRS 17, how would you describe the standard's objective?",
                model="unused",
            )

        self.assertFalse(decision.needs_retrieval)
        self.assertEqual(decision.reason, "這只是一般解釋。")

    def test_query_rewrite_keeps_question_derived_synonyms(self):
        model_output = "\n".join(
            [
                "語意理解：使用者需要精確的術語定義。",
                "是否需要檢索：是",
                "判斷理由：需要外部證據。",
                "向量檢索用查詢：CSM contractual service margin 定義",
            ]
        )
        with patch("rag_demo.query_rewriter.ask_model", return_value=model_output):
            decision = decide_and_rewrite_query_for_retrieval("CSM 定義是什麼？", model="unused")

        self.assertIn("contractual service margin", decision.retrieval_query)


if __name__ == "__main__":
    unittest.main()
