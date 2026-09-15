import unittest
from types import SimpleNamespace

from rag_demo.config import RagConfig
from rag_demo.rag_pipeline import RagPipeline, RagPipelineRequest


class _Retriever:
    def retrieve(self, **kwargs):
        return {
            "contexts": [{
                "id": "chunk-1",
                "rank": 1,
                "title": "規則",
                "source": "source-1",
                "page": "1",
                "content": "申請期限為三日。",
            }],
            "timings": {
                "bm25Ms": 1.0,
                "embeddingMs": 2.0,
                "fusionMs": 0.5,
                "rerankMs": 0.0,
                "totalMs": 4.0,
            },
        }


class PipelineTimingContractTests(unittest.TestCase):
    def test_pipeline_returns_correlated_stage_timings(self):
        pipeline = RagPipeline(
            settings_factory=lambda: RagConfig(
                hybrid_top_k=3,
                hybrid_candidate_k=6,
                hybrid_max_top_k=8,
                hybrid_max_candidate_k=20,
            ),
            retriever_factory=lambda _profile: _Retriever(),
            route_fn=lambda question, **kwargs: SimpleNamespace(
                needs_retrieval=True,
                reason="需要文件證據",
                retrieval_query=question,
                query_variants=(question,),
            ),
            evidence_fn=lambda contexts, **kwargs: {
                "sufficient": True,
                "confidence": "high",
                "reason": "evidence passed",
            },
            ask_model_fn=lambda prompt, model, system=None: "申請期限為三日。來源：[1]",
            datetime_answer_fn=lambda question: None,
            run_id_fn=lambda: "run-timing",
        )
        result = pipeline.run(RagPipelineRequest(
            question="申請期限？",
            model="ollama:qwen2.5:7b",
            persist_conversation=False,
            request_id="req-timing",
        ))

        self.assertEqual(result["timing_trace"]["runId"], "run-timing")
        self.assertEqual(result["timing_trace"]["requestId"], "req-timing")
        names = {stage["name"] for stage in result["timing_trace"]["stages"]}
        self.assertTrue({
            "query.route",
            "retrieval.bm25",
            "retrieval.embedding",
            "retrieval",
            "evidence.gate",
            "model.generation",
            "generation",
            "total",
        }.issubset(names))
        self.assertEqual(result["retrieval"]["timings"]["bm25Ms"], 1.0)


if __name__ == "__main__":
    unittest.main()
