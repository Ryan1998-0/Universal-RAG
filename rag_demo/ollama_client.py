import json
import os
from typing import Dict, Optional
import urllib.request
from urllib.parse import urlparse


def build_ollama_payload(
    prompt: str,
    model: str = "qwen2.5:7b",
    system: Optional[str] = None,
) -> Dict[str, object]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": _float_env("RAG_OLLAMA_TEMPERATURE", 0.0),
            "seed": _int_env("RAG_OLLAMA_SEED", 42),
            "num_ctx": 8192,
            "num_predict": _int_env("RAG_OLLAMA_NUM_PREDICT", 768),
        },
    }
    if system:
        payload["system"] = system
    return payload


def ask_ollama(
    prompt: str,
    model: str = "qwen2.5:7b",
    system: Optional[str] = None,
    timeout_seconds: float = 120.0,
) -> str:
    payload = build_ollama_payload(prompt=prompt, model=model, system=system)
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{ollama_base_url()}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=max(1.0, float(timeout_seconds))) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["response"].strip()


def ollama_base_url() -> str:
    base_url = os.getenv("RAG_OLLAMA_URL", "http://127.0.0.1:11434").strip().rstrip("/")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError("RAG_OLLAMA_URL must be an absolute HTTP(S) URL")
    return base_url


def _int_env(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


def _float_env(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default
