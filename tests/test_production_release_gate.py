import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_production_release_gate import _check_answer, _corpus_provenance, load_manifest, run_gate


TEMPLATE = Path(__file__).resolve().parents[1] / "evals/production_release_gate/manifest.template.json"


def configured_manifest():
    text = TEMPLATE.read_text(encoding="utf-8").replace("SET_STAGING_KNOWLEDGE_BASE_ID", "staging-kb")
    text = text.replace("SET_OTHER_TENANT_KNOWLEDGE_BASE_ID", "other-tenant-kb")
    text = text.replace("SET_POLICY_2024_SOURCE_ID", "source-2024")
    text = text.replace("SET_POLICY_2026_SOURCE_ID", "source-2026")
    text = text.replace("SET_WORKFLOW_REQUEST_SOURCE_ID", "source-workflow-request")
    text = text.replace("SET_WORKFLOW_APPROVAL_SOURCE_ID", "source-workflow-approval")
    text = text.replace("SET_OCR_NOTICE_SOURCE_ID", "source-ocr")
    text = text.replace("SET_IMAGE_SHA256_DIGEST", "sha256:deployment")
    text = text.replace("SET_GENERATION_MODEL_REVISION_OR_DIGEST", "sha256:generation")
    text = text.replace("SET_EMBEDDING_MODEL_REVISION", "embedding-revision")
    text = text.replace("SET_RERANKER_MODEL_REVISION", "reranker-revision")
    text = text.replace("SET_ACTIVE_INDEX_VERSION_ID", "index-1")
    text = text.replace("SET_PARSER_VERSION", "canonical-v1")
    text = text.replace("SET_CHUNK_SCHEMA_VERSION", "parent-child-v1")
    text = text.replace("SET_STAGING_HARDWARE_PROFILE", "local-test")
    return json.loads(text)


class FakeClient:
    def __init__(self, manifest, unauthorized_status=404, active_index="index-1"):
        self.manifest = manifest
        self.unauthorized_status = unauthorized_status
        self.active_index = active_index

    def request(self, path, *, method="GET", payload=None):
        if path == "/health/ready":
            return 200, {"status": "ready"}
        if path == "/v1/runtime":
            return 200, {"default_model": "ollama:qwen2.5:7b", "allowed_models": ["ollama:qwen2.5:7b"]}
        if path == "/v1/models":
            return 200, {"components": {
                "embedding": {"model": "example/embed"},
                "sparse_embedding": {"model": "Qdrant/bm25"},
                "reranker": {"model": "example/reranker"},
            }}
        if path == "/v1/knowledge-bases":
            return 200, {"items": [{"id": "staging-kb", "active_index_version_id": self.active_index}]}
        if payload["knowledge_base_id"] == "other-tenant-kb":
            return self.unauthorized_status, {}
        case = next(item for item in self.manifest["cases"] if item["question"] == payload["question"])
        expected = case["expect"]
        if expected["status"] == "refused":
            return 200, {
                "run_id": case["id"],
                "answer": "根據目前檢索資料無法確認。",
                "retrieval": {"needed": True},
                "citations": [],
            }
        sources = expected["allowed_citation_source_ids"]
        citations = [
            {"rank": index, "source": source}
            for index, source in enumerate(sources[:expected["min_citations"]], start=1)
        ]
        answer = "、".join(group[0] for group in expected["required_fact_groups"])
        return 200, {
            "run_id": case["id"],
            "answer": answer,
            "retrieval": {"needed": True},
            "citations": citations,
            "evidence_validation": {
                "status": "passed", "sufficient": True,
                "valid_citations": [citation["rank"] for citation in citations],
                "uncited_claims": [], "unsupported_claims": [],
            },
        }


class ProductionReleaseGateTests(unittest.TestCase):
    def test_template_requires_real_staging_ids(self):
        with self.assertRaisesRegex(ValueError, "real staging knowledge_base_id"):
            load_manifest(TEMPLATE)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(configured_manifest(), ensure_ascii=False), encoding="utf-8")
            self.assertEqual(len(load_manifest(path)["cases"]), 7)

            invalid = configured_manifest()
            invalid["cases"][0]["expect"]["required_fact_groups"] = [[]]
            path.write_text(json.dumps(invalid, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid required_fact_groups"):
                load_manifest(path)

    def test_seeded_cases_and_cross_tenant_denial_are_required(self):
        manifest = configured_manifest()
        report = run_gate(manifest, FakeClient(manifest))
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["summary"]["passed_cases"], 7)
        self.assertEqual(report["summary"]["latency"]["sample_count"], 7)
        self.assertEqual(len(_corpus_provenance(manifest)["fixture_sources"]), 5)

        unsafe = run_gate(manifest, FakeClient(manifest, unauthorized_status=200))
        self.assertEqual(unsafe["status"], "failed")
        self.assertFalse(unsafe["unauthorized_checks"][0]["passed"])

        stale_index = run_gate(manifest, FakeClient(manifest, active_index="index-old"))
        self.assertEqual(stale_index["status"], "failed")
        self.assertEqual(stale_index["summary"]["case_count"], 0)

    def test_valid_http_response_with_unsupported_claim_fails(self):
        case = configured_manifest()["cases"][0]
        response = {
            "run_id": "run-1", "answer": "申請期限為三日。[1]",
            "retrieval": {"needed": True},
            "citations": [{"rank": 1, "source": "source-2024"}],
            "evidence_validation": {
                "status": "failed", "sufficient": False,
                "valid_citations": [1],
                "unsupported_claims": ["申請期限為三日。"],
            },
        }
        self.assertIn("evidence validation did not pass", _check_answer(case, 200, response))

    def test_unready_service_fails_without_sending_answer_requests(self):
        class UnreadyClient:
            def request(self, path, *, method="GET", payload=None):
                if path != "/health/ready":
                    raise AssertionError("answer request should not run")
                return 503, {"status": "not_ready"}

        report = run_gate(configured_manifest(), UnreadyClient())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["summary"]["case_count"], 0)


if __name__ == "__main__":
    unittest.main()
