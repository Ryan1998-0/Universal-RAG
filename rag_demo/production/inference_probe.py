import json
import urllib.request


class OllamaInferenceProbe:
    def __init__(self, base_url: str, model_names):
        self.base_url = str(base_url or "").rstrip("/")
        self.model_names = {
            str(model).partition(":")[2]
            for model in model_names
            if str(model).startswith("ollama:")
        }

    def ping(self) -> None:
        request = urllib.request.Request(
            f"{self.base_url}/api/tags",
            headers={"Accept": "application/json"},
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
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
