import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rag_demo.model_providers import ask_model, model_request_timeout
from rag_demo.ollama_client import (
    ask_ollama,
    ask_ollama_vision,
    build_ollama_payload,
    build_ollama_vision_payload,
    ollama_base_url,
)


class OllamaClientTest(unittest.TestCase):
    def test_uses_deterministic_generation_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            payload = build_ollama_payload("test")

        self.assertEqual(payload["options"]["temperature"], 0.0)
        self.assertEqual(payload["options"]["seed"], 42)

    def test_generation_defaults_remain_configurable(self):
        with patch.dict(
            os.environ,
            {"RAG_OLLAMA_TEMPERATURE": "0.2", "RAG_OLLAMA_SEED": "7"},
            clear=True,
        ):
            payload = build_ollama_payload("test")

        self.assertEqual(payload["options"]["temperature"], 0.2)
        self.assertEqual(payload["options"]["seed"], 7)

    def test_inference_url_is_configurable_and_validated(self):
        with patch.dict(os.environ, {"RAG_OLLAMA_URL": "https://inference.example.com/"}):
            self.assertEqual(ollama_base_url(), "https://inference.example.com")
        with patch.dict(os.environ, {"RAG_OLLAMA_URL": "file:///tmp/socket"}):
            with self.assertRaises(RuntimeError):
                ollama_base_url()

    def test_ollama_socket_timeout_is_explicit(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"response":"ok"}'
        with patch("urllib.request.urlopen", return_value=response) as urlopen:
            self.assertEqual(ask_ollama("test", timeout_seconds=7), "ok")
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 7.0)

    def test_vision_payload_embeds_image_and_uses_vision_limits(self):
        with patch.dict(
            os.environ,
            {
                "RAG_OLLAMA_VISION_NUM_CTX": "4096",
                "RAG_OLLAMA_VISION_NUM_PREDICT": "333",
                "RAG_OLLAMA_VISION_KEEP_ALIVE": "45m",
            },
            clear=True,
        ):
            payload = build_ollama_vision_payload("read this", b"image", "vision-model")

        self.assertEqual(payload["model"], "vision-model")
        self.assertEqual(payload["images"], ["aW1hZ2U="])
        self.assertEqual(payload["keep_alive"], "45m")
        self.assertEqual(payload["options"]["num_ctx"], 4096)
        self.assertEqual(payload["options"]["num_predict"], 333)

    def test_vision_request_uses_explicit_timeout(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"response":"screen details"}'
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "screen.png"
            image_path.write_bytes(b"image")
            with patch("urllib.request.urlopen", return_value=response) as urlopen:
                answer = ask_ollama_vision(
                    image_path,
                    prompt="describe",
                    model="vision-model",
                    timeout_seconds=17,
                )

        self.assertEqual(answer, "screen details")
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 17.0)

    def test_model_timeout_context_reaches_provider(self):
        with patch("rag_demo.model_providers.ask_ollama", return_value="ok") as provider:
            with model_request_timeout(9):
                self.assertEqual(ask_model("test"), "ok")
        self.assertEqual(provider.call_args.kwargs["timeout_seconds"], 9.0)


if __name__ == "__main__":
    unittest.main()
