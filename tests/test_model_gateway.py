import os
import unittest

from rag_demo.model_gateway import (
    ModelGateway,
    resolve_model_for_node,
    validate_model_spec,
)


class ModelGatewayTests(unittest.TestCase):
    def test_node_environment_override_is_replaceable(self):
        previous = os.environ.get("RAG_MODEL_QUERY_REWRITE")
        os.environ["RAG_MODEL_QUERY_REWRITE"] = "ollama:qwen2.5:14b"
        try:
            self.assertEqual(
                resolve_model_for_node(
                    "query_rewrite",
                    requested_model="ollama:qwen2.5:7b",
                ),
                "ollama:qwen2.5:14b",
            )
        finally:
            if previous is None:
                os.environ.pop("RAG_MODEL_QUERY_REWRITE", None)
            else:
                os.environ["RAG_MODEL_QUERY_REWRITE"] = previous

    def test_runtime_override_and_invocation(self):
        calls = []

        def fake_model(prompt, model, system=None):
            calls.append((prompt, model, system))
            return "ok"

        gateway = ModelGateway(
            default_model="ollama:qwen2.5:7b",
            allowed_models=["ollama:qwen2.5:7b", "ollama:qwen2.5:14b"],
            ask_model_fn=fake_model,
        )
        binding = gateway.set_override("generation", "ollama:qwen2.5:14b")
        self.assertEqual(binding.model, "ollama:qwen2.5:14b")
        self.assertEqual(
            gateway.invoke("hello", node="generation", system="system"),
            "ok",
        )
        self.assertEqual(calls[0][1], "ollama:qwen2.5:14b")
        self.assertEqual(calls[0][2], "system")
        gateway.clear_override("generation")
        self.assertEqual(gateway.binding("generation").source, "default")
        self.assertIn("query_rewrite", {item["node"] for item in gateway.describe()["nodes"]})

    def test_model_validation_rejects_models_outside_allowlist(self):
        result = validate_model_spec(
            "ollama:qwen2.5:7b",
            ["ollama:qwen2.5:7b"],
        )
        self.assertTrue(result["allowed"])
        rejected = validate_model_spec(
            "openai:gpt-5.5",
            ["ollama:qwen2.5:7b"],
        )
        self.assertFalse(rejected["allowed"])

    def test_vision_without_configuration_is_empty(self):
        previous = os.environ.pop("RAG_VLM_MODEL", None)
        try:
            self.assertEqual(resolve_model_for_node("vision"), "")
        finally:
            if previous is not None:
                os.environ["RAG_VLM_MODEL"] = previous


if __name__ == "__main__":
    unittest.main()
