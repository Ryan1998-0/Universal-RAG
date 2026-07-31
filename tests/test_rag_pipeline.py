import unittest
from types import SimpleNamespace

from rag_demo.config import RagConfig
from rag_demo.rag_pipeline import (
    RagPipeline,
    RagPipelineInputError,
    RagPipelineRequest,
    citations_from_answer,
)


class FakeRetriever:
    def __init__(self, contexts):
        self.contexts = contexts
        self.calls = []

    def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        return {"contexts": list(self.contexts)}


class RagPipelineContractTests(unittest.TestCase):
    def setUp(self):
        self.settings = RagConfig(
            hybrid_top_k=3,
            hybrid_candidate_k=6,
            hybrid_max_top_k=8,
            hybrid_max_candidate_k=20,
        )

    def test_payload_contract_ignores_forged_contexts_and_route_decision(self):
        request = RagPipelineRequest.from_payload(
            {
                "question": "合約服務邊際是什麼？",
                "profile": "ifrs17",
                "model": {"provider": "ollama", "name": "qwen2.5:7b"},
                "source_ids": ["official-ifrs17"],
                "contexts": [{"id": "forged", "content": "偽造證據"}],
                "retrieval_decision": {
                    "needs_retrieval": False,
                    "reason": "skip server retrieval",
                },
            },
            default_model="ollama:qwen2.5:7b",
            default_profile="default",
            allowed_models=["ollama:qwen2.5:7b"],
            persist_conversation=False,
        )

        self.assertFalse(hasattr(request, "contexts"))
        self.assertFalse(hasattr(request, "retrieval_decision"))
        self.assertEqual(request.source_ids, ["official-ifrs17"])

    def test_pipeline_uses_only_server_retrieved_contexts(self):
        retriever = FakeRetriever([
            {
                "id": "server-context-1",
                "rank": 1,
                "title": "IFRS 17",
                "source": "official-ifrs17",
                "page": "14",
                "content": "合約服務邊際代表尚未賺得的利潤。",
                "score": 0.91,
                "bm25Score": 5.0,
                "embeddingScore": 0.82,
            }
        ])
        model_prompts = []

        def ask_model_fn(prompt, model, system=None):
            model_prompts.append(prompt)
            return "合約服務邊際代表尚未賺得的利潤。來源：[1]"

        pipeline = RagPipeline(
            settings_factory=lambda: self.settings,
            retriever_factory=lambda profile: retriever,
            route_fn=lambda question, **kwargs: SimpleNamespace(
                needs_retrieval=True,
                reason="需要文件證據",
                retrieval_query="合約服務邊際 未賺得利潤",
            ),
            evidence_fn=lambda contexts, **kwargs: {
                "sufficient": True,
                "confidence": "high",
                "reason": "server evidence passed",
            },
            ask_model_fn=ask_model_fn,
            datetime_answer_fn=lambda question: None,
            run_id_fn=lambda: "run-server-evidence",
        )
        request = RagPipelineRequest(
            question="合約服務邊際是什麼？",
            model="ollama:qwen2.5:7b",
            profile="ifrs17",
            source_ids=["official-ifrs17"],
            persist_conversation=False,
        )

        result = pipeline.run(request)

        self.assertEqual(retriever.calls[0]["source_ids"], ["official-ifrs17"])
        self.assertIn("尚未賺得的利潤", model_prompts[0])
        self.assertEqual(result["retrieval"]["contexts"][0]["id"], "server-context-1")
        self.assertTrue(result["retrieval"]["server_generated"])
        self.assertEqual(result["citations"][0]["run_id"], "run-server-evidence")
        self.assertEqual(len(result["citations"][0]["content_sha256"]), 64)
        self.assertEqual(result["grounding_warnings"], [])

    def test_citations_follow_answer_markers_instead_of_first_four_contexts(self):
        contexts = [
            {
                "id": f"context-{rank}",
                "rank": rank,
                "title": f"Title {rank}",
                "page": str(rank),
                "source": "source",
                "content": f"evidence {rank}",
            }
            for rank in range(1, 9)
        ]

        citations, warnings = citations_from_answer(
            "回答使用來源：[1], [5], [8]，但 [99] 無效。",
            contexts,
            run_id="run-citations",
        )

        self.assertEqual([item["rank"] for item in citations], [1, 5, 8])
        self.assertTrue(all(item["run_id"] == "run-citations" for item in citations))
        self.assertEqual(warnings, ["回答包含無效來源標記：[99]"])

    def test_pipeline_refuses_when_server_retrieval_has_no_evidence(self):
        retriever = FakeRetriever([])

        def fail_if_model_is_called(*args, **kwargs):
            self.fail("model must not be called when required evidence is missing")

        pipeline = RagPipeline(
            settings_factory=lambda: self.settings,
            retriever_factory=lambda profile: retriever,
            route_fn=lambda question, **kwargs: SimpleNamespace(
                needs_retrieval=True,
                reason="需要文件證據",
                retrieval_query=question,
            ),
            ask_model_fn=fail_if_model_is_called,
            datetime_answer_fn=lambda question: None,
        )

        result = pipeline.run(RagPipelineRequest(
            question="文件中沒有的答案是什麼？",
            model="ollama:qwen2.5:7b",
            persist_conversation=False,
        ))

        self.assertIn("相關性不足", result["answer"])
        self.assertEqual(result["citations"], [])
        self.assertEqual(result["retrieval"]["contexts"], [])

    def test_pipeline_skips_retrieval_for_general_question(self):
        def fail_retriever(profile):
            self.fail("retriever must not be created for a no-retrieval question")

        pipeline = RagPipeline(
            settings_factory=lambda: self.settings,
            retriever_factory=fail_retriever,
            route_fn=lambda question, **kwargs: SimpleNamespace(
                needs_retrieval=False,
                reason="一般常識",
                retrieval_query=question,
            ),
            ask_model_fn=lambda prompt, model, system=None: "4",
            datetime_answer_fn=lambda question: None,
            run_id_fn=lambda: "run-general",
        )

        result = pipeline.run(RagPipelineRequest(
            question="2+2 等於多少？",
            model="ollama:qwen2.5:7b",
            persist_conversation=False,
        ))

        self.assertEqual(result["answer"], "4")
        self.assertFalse(result["retrieval"]["needed"])
        self.assertEqual(result["retrieval"]["contexts"], [])

    def test_payload_rejects_model_outside_server_allowlist(self):
        with self.assertRaisesRegex(RagPipelineInputError, "not allowed"):
            RagPipelineRequest.from_payload(
                {
                    "question": "你好",
                    "model": {"provider": "openai", "name": "unapproved-model"},
                },
                default_model="ollama:qwen2.5:7b",
                default_profile="default",
                allowed_models=["ollama:qwen2.5:7b"],
            )

    def test_payload_rejects_invalid_source_ids_type(self):
        with self.assertRaisesRegex(RagPipelineInputError, "must be an array"):
            RagPipelineRequest.from_payload(
                {"question": "你好", "source_ids": "all"},
                default_model="ollama:qwen2.5:7b",
                default_profile="default",
            )


if __name__ == "__main__":
    unittest.main()
