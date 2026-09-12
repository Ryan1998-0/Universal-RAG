import os
import unittest
from unittest.mock import patch

from rag_demo.langchain_integration import (
    LangChainUnavailableError,
    _message_content,
    langchain_backend_available,
)
from rag_demo.model_providers import ask_model


class LangChainIntegrationTests(unittest.TestCase):
    def test_backend_is_optional_in_current_environment(self):
        # The test environment does not need LangChain to run the existing
        # direct API path.
        self.assertIsInstance(langchain_backend_available(), bool)

    def test_message_content_handles_text_blocks(self):
        self.assertEqual(
            _message_content({"content": [{"text": "第一段"}, {"text": "第二段"}]}),
            "第一段\n第二段",
        )

    def test_explicit_langchain_backend_reports_missing_optional_dependency(self):
        with patch.dict(os.environ, {"RAG_MODEL_BACKEND": "langchain"}, clear=False):
            with patch(
                "rag_demo.langchain_integration.ask_langchain",
                side_effect=LangChainUnavailableError("missing"),
            ):
                with self.assertRaises(LangChainUnavailableError):
                    ask_model("test")

    def test_invalid_backend_is_rejected(self):
        with patch.dict(os.environ, {"RAG_MODEL_BACKEND": "invalid"}, clear=False):
            with self.assertRaises(ValueError):
                ask_model("test")


if __name__ == "__main__":
    unittest.main()
