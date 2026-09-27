import json
import unittest
from pathlib import Path

from scripts.run_prompt_injection_eval import evaluate
from scripts.run_prompt_injection_answer_probe import _classify, load_manifest, run_probe


CORPUS = Path(__file__).resolve().parents[1] / "evals/prompt_injection/corpus.json"
ANSWER_TEMPLATE = CORPUS.parent / "answer-manifest.template.json"


class PromptInjectionEvalTests(unittest.TestCase):
    def test_frozen_corpus_reports_all_attack_and_benign_controls(self):
        report = evaluate(json.loads(CORPUS.read_text(encoding="utf-8")))
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["summary"]["true_positive"], 11)
        self.assertEqual(report["summary"]["true_negative"], 3)
        self.assertEqual(report["summary"]["false_negative"], 0)

    def test_answer_probe_requires_configured_sources_and_rejects_followed_attack(self):
        corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
        with self.assertRaisesRegex(ValueError, "configure the staging"):
            load_manifest(ANSWER_TEMPLATE, corpus)

        attack = next(case for case in corpus["cases"] if case["expected_quarantine"])
        source_id = "source-1"
        self.assertEqual(
            _classify(attack, source_id, 200, {"answer": attack["forbidden_output"]}),
            "attack_followed",
        )
        self.assertEqual(
            _classify(attack, source_id, 200, {
                "answer": attack["expected_fact"] + "。[1]",
                "retrieval": {"needed": True},
                "citations": [{"source": source_id}],
                "evidence_validation": {"status": "passed", "sufficient": True},
            }),
            "safe_answer",
        )

    def test_answer_probe_counts_safe_refusals_and_checks_active_index(self):
        corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
        attacks = [case for case in corpus["cases"] if case["expected_quarantine"]]
        manifest = {
            "knowledge_base_id": "kb-staging",
            "active_index_id": "index-current",
            "source_ids": {case["id"]: f"source-{index}" for index, case in enumerate(attacks)},
        }

        class FakeClient:
            def __init__(self, index_id):
                self.index_id = index_id
                self.answer_calls = 0

            def request(self, path, *, method="GET", payload=None):
                if path == "/health/ready":
                    return 200, {"status": "ready"}
                if path == "/v1/knowledge-bases":
                    return 200, {"items": [{"id": "kb-staging", "active_index_version_id": self.index_id}]}
                if path == "/v1/runtime":
                    return 200, {"default_model": "ollama:test"}
                if path == "/v1/models":
                    return 200, {"components": {"embedding": {"model": "embed"}, "reranker": {"model": "rerank"}}}
                self.answer_calls += 1
                return 200, {"answer": "根據目前檢索資料無法確認。", "citations": []}

        client = FakeClient("index-current")
        report = run_probe(manifest, corpus, client)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["summary"]["safe_refusal"], len(attacks))
        self.assertEqual(client.answer_calls, len(attacks))

        stale = FakeClient("index-old")
        self.assertEqual(run_probe(manifest, corpus, stale)["status"], "failed")
        self.assertEqual(stale.answer_calls, 0)


if __name__ == "__main__":
    unittest.main()
