from unittest.mock import Mock

import httpx
import pytest

from rag_demo.production.inference_probe import OllamaInferenceProbe


def test_inference_probe_checks_models_without_following_redirects(monkeypatch):
    get = Mock(return_value=httpx.Response(200, json={"models": [{"name": "qwen2.5:7b"}]}, request=httpx.Request("GET", "http://ollama/api/tags")))
    monkeypatch.setattr(httpx, "get", get)
    OllamaInferenceProbe("http://ollama", ["ollama:qwen2.5:7b"]).ping()
    assert get.call_args.kwargs["follow_redirects"] is False
    assert get.call_args.kwargs["timeout"] == 5
    with pytest.raises(RuntimeError, match="unavailable"):
        OllamaInferenceProbe("http://ollama", ["ollama:missing"]).ping()


def test_inference_probe_rejects_redirect_and_non_http_endpoint(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(302, headers={"Location": "ftp://untrusted.test"}, request=httpx.Request("GET", "http://ollama/api/tags")))
    with pytest.raises(httpx.HTTPStatusError):
        OllamaInferenceProbe("http://ollama", ["ollama:qwen2.5:7b"]).ping()
    with pytest.raises(ValueError):
        OllamaInferenceProbe("file:///etc", [])
