import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rag_demo.query_rewriter import QueryRewriteDecision


class SelfRagReflectionParsingTest(unittest.TestCase):
    def test_parse_passage_reflections_preserves_labels_and_reasons(self):
        from rag_demo.self_rag_reflection import parse_passage_reflections

        output = "\n".join(
            [
                "來源 1：[Relevant][Supported] 理由：直接說明 CSM 的定義。",
                "來源 2：[Irrelevant][Unsupported] 理由：只提到過渡規定。",
            ]
        )

        reflections = parse_passage_reflections(output, chunk_count=3)

        self.assertEqual(len(reflections), 3)
        self.assertEqual(reflections[0].source_index, 1)
        self.assertEqual(reflections[0].relevance_label, "[Relevant]")
        self.assertEqual(reflections[0].support_label, "[Supported]")
        self.assertIn("CSM", reflections[0].reason)
        self.assertEqual(reflections[1].relevance_label, "[Irrelevant]")
        self.assertEqual(reflections[2].relevance_label, "[Irrelevant]")
        self.assertIn("未被", reflections[2].reason)

    def test_parse_answer_support_critique_extracts_retry_decision(self):
        from rag_demo.self_rag_reflection import parse_answer_support_critique

        output = "\n".join(
            [
                "[Supported]：否",
                "[Utility]：2/5",
                "需要重新檢索：是",
                "重新檢索查詢：IFRS17 CSM insurance contract service margin measurement",
                "判斷理由：答案提到 CSM，但來源沒有足夠定義。",
            ]
        )

        critique = parse_answer_support_critique(output)

        self.assertFalse(critique.is_supported)
        self.assertEqual(critique.support_label, "[Unsupported]")
        self.assertEqual(critique.utility_score, 2)
        self.assertEqual(critique.utility_label, "[Utility:2/5]")
        self.assertTrue(critique.retry_needed)
        self.assertIn("CSM", critique.retry_query)
        self.assertIn("來源沒有足夠定義", critique.reason)

    def test_parse_utility_critique_extracts_score(self):
        from rag_demo.self_rag_reflection import parse_utility_critique

        output = "\n".join(
            [
                "[Utility]：4/5",
                "需要重新檢索：否",
                "判斷理由：回答清楚且有根據來源。",
            ]
        )

        critique = parse_utility_critique(output)

        self.assertEqual(critique.score, 4)
        self.assertEqual(critique.utility_label, "[Utility:4/5]")
        self.assertFalse(critique.retry_needed)


class SelfRagControllerTest(unittest.TestCase):
    def test_self_rag_controller_skips_retrieval_for_no_retrieval_decision(self):
        from rag_demo.query import _format_self_rag_no_retrieval_output

        output = _format_self_rag_no_retrieval_output(
            question="你好",
            rewrite_decision=QueryRewriteDecision(
                needs_retrieval=False,
                reason="打招呼不需要查詢 knowledge base。",
                retrieval_query="你好",
            ),
            answer="你好，我可以協助回答資料庫相關問題。",
            timing={"query_rewrite": 0.01, "general_fallback": 0.02, "total": 0.03},
        )

        self.assertIn("[Retrieve]=false", output)
        self.assertIn("Skip Retrieval", output)
        self.assertIn("Engineering Self-RAG Controller", output)

    def test_self_rag_controller_retries_when_answer_is_unsupported(self):
        from rag_demo.query import answer_question_self_rag
        from rag_demo.self_rag_reflection import AnswerSupportCritique, PassageReflection, UtilityCritique

        chunks = [
            {
                "id": "c1",
                "source": "ifrs17.md",
                "parent_title": "IFRS 17",
                "title": "CSM",
                "content": "CSM 是 contractual service margin。",
                "score": 1.0,
            }
        ]
        fake_kb = SimpleNamespace(raw_dir=Path("/tmp/unused/raw"), index_dir=Path("/tmp/unused/index"))

        with patch("rag_demo.query.active_knowledge_base", return_value=fake_kb), patch(
            "rag_demo.query.load_knowledge_base_chunks", return_value=chunks
        ), patch("rag_demo.query.decide_and_rewrite_query_for_retrieval") as decide, patch(
            "rag_demo.query._rrf_parent_context_results", side_effect=[chunks, chunks]
        ), patch("rag_demo.query.critique_passages") as critique_passages, patch(
            "rag_demo.query.answer_with_qa_agent", side_effect=["資料不足。", "CSM 是 contractual service margin。"]
        ), patch("rag_demo.query.critique_answer_support") as critique_answer, patch(
            "rag_demo.query.score_answer_utility"
        ) as score_utility:
            decide.return_value = QueryRewriteDecision(
                needs_retrieval=True,
                reason="IFRS 17 問題需要 evidence。",
                retrieval_query="IFRS17 CSM",
            )
            critique_passages.return_value = [
                PassageReflection(1, "[Relevant]", "[Supported]", "來源直接提到 CSM。")
            ]
            critique_answer.side_effect = [
                AnswerSupportCritique(
                    support_label="[Unsupported]",
                    is_supported=False,
                    retry_needed=True,
                    retry_query="IFRS17 CSM contractual service margin",
                    reason="第一版答案資料不足。",
                ),
                AnswerSupportCritique(
                    support_label="[Supported]",
                    is_supported=True,
                    retry_needed=False,
                    retry_query="",
                    reason="答案被來源支撐。",
                ),
            ]
            score_utility.return_value = UtilityCritique(
                score=4,
                utility_label="[Utility:4/5]",
                retry_needed=False,
                retry_query="",
                reason="回答清楚。",
            )

            output = answer_question_self_rag("IFRS17 的 CSM 是什麼？", model="unused", top_k=1, max_attempts=2)

        self.assertIn("[Retrieve]=true", output)
        self.assertIn("Attempt 1", output)
        self.assertIn("Attempt 2", output)
        self.assertIn("[Unsupported]", output)
        self.assertIn("[Supported]", output)
        self.assertIn("answer_critic_attempt_1", output)

    def test_self_rag_controller_uses_combined_critic_instead_of_utility_agent(self):
        from rag_demo.query import answer_question_self_rag
        from rag_demo.self_rag_reflection import AnswerSupportCritique, PassageReflection

        chunks = [
            {
                "id": "c1",
                "source": "ifrs17.md",
                "parent_title": "IFRS 17",
                "title": "Objective",
                "content": "IFRS 17 establishes principles for recognition, measurement, presentation and disclosure.",
                "score": 1.0,
            }
        ]
        fake_kb = SimpleNamespace(raw_dir=Path("/tmp/unused/raw"), index_dir=Path("/tmp/unused/index"))

        with patch("rag_demo.query.active_knowledge_base", return_value=fake_kb), patch(
            "rag_demo.query.load_knowledge_base_chunks", return_value=chunks
        ), patch("rag_demo.query.decide_and_rewrite_query_for_retrieval") as decide, patch(
            "rag_demo.query._rrf_parent_context_results", return_value=chunks
        ), patch("rag_demo.query.critique_passages") as critique_passages, patch(
            "rag_demo.query.answer_with_qa_agent",
            return_value="IFRS 17 establishes principles for recognition, measurement, presentation and disclosure.",
        ), patch("rag_demo.query.critique_answer_support") as critique_answer, patch(
            "rag_demo.query.score_answer_utility", side_effect=AssertionError("separate Utility Critic should not run")
        ) as score_utility:
            decide.return_value = QueryRewriteDecision(
                needs_retrieval=True,
                reason="IFRS 17 問題需要 evidence。",
                retrieval_query="IFRS17 objective",
            )
            critique_passages.return_value = [
                PassageReflection(1, "[Relevant]", "[Supported]", "來源直接支撐答案。")
            ]
            critique_answer.return_value = AnswerSupportCritique(
                support_label="[Supported]",
                is_supported=True,
                retry_needed=False,
                retry_query="",
                reason="答案被來源支撐。",
                utility_score=4,
                utility_label="[Utility:4/5]",
            )

            output = answer_question_self_rag("What is the objective of IFRS 17?", model="unused", top_k=1)

        score_utility.assert_not_called()
        self.assertIn("[Utility:4/5]", output)
        self.assertNotIn("utility_scoring_attempt_1", output)

    def test_answer_support_is_downgraded_without_supporting_passages(self):
        from rag_demo.query import _self_rag_enforce_passage_support
        from rag_demo.self_rag_reflection import AnswerSupportCritique

        critique = AnswerSupportCritique(
            support_label="[Supported]",
            is_supported=True,
            retry_needed=False,
            retry_query="",
            reason="模型誤判為有支撐。",
        )

        enforced = _self_rag_enforce_passage_support(
            question="IFRS17 的 CSM 是什麼？",
            answer_critique=critique,
            passage_reflections=[],
        )

        self.assertEqual(enforced.support_label, "[Unsupported]")
        self.assertFalse(enforced.is_supported)
        self.assertTrue(enforced.retry_needed)
        self.assertIn("沒有 supporting passage", enforced.reason)


if __name__ == "__main__":
    unittest.main()
