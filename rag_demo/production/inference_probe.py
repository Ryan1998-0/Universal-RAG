from urllib.parse import urlsplit

import httpx


class OllamaInferenceProbe:
    def __init__(self, base_url: str, model_names):
        self.base_url = str(base_url or "").rstrip("/")
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("inference endpoint must be an HTTP(S) URL")
        self.model_names = {
            str(model).partition(":")[2]
            for model in model_names
            if str(model).startswith("ollama:")
        }

    def ping(self) -> None:
        response = httpx.get(
            f"{self.base_url}/api/tags",
            headers={"Accept": "application/json"},
            timeout=5,
            follow_redirects=False,
        )
        response.raise_for_status()
        payload = response.json()
        available = {
            str(item.get("name") or item.get("model") or "")
            for item in payload.get("models", [])
            if isinstance(item, dict)
        }
        missing = sorted(
            model
            for model in self.model_names
            if model not in available and not any(name.startswith(f"{model}:") for name in available)
        )
        if missing:
            raise RuntimeError("configured inference model is unavailable")
