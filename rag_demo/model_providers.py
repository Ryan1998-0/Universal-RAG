import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Dict, Optional
import urllib.request

from rag_demo.ollama_client import ask_ollama


_MODEL_REQUEST_TIMEOUT_SECONDS: ContextVar[float] = ContextVar(
    "model_request_timeout_seconds",
    default=120.0,
)


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str


def parse_model_spec(model_spec: str) -> ModelSpec:
    if not model_spec:
        return ModelSpec(provider="ollama", model="qwen2.5:7b")

    provider, separator, model = model_spec.partition(":")
    if separator and provider in {"ollama", "openai", "anthropic"}:
        return ModelSpec(provider=provider, model=model)

    return ModelSpec(provider="ollama", model=model_spec)


def ask_model(
    prompt: str,
    model: str = "ollama:qwen2.5:7b",
    system: Optional[str] = None,
) -> str:
    spec = parse_model_spec(model)
    timeout_seconds = _MODEL_REQUEST_TIMEOUT_SECONDS.get()
    if spec.provider == "ollama":
        return ask_ollama(
            prompt=prompt,
            model=spec.model,
            system=system,
            timeout_seconds=timeout_seconds,
        )
    if spec.provider == "openai":
        return ask_openai(
            prompt=prompt,
            model=spec.model,
            system=system,
            timeout_seconds=timeout_seconds,
        )
    if spec.provider == "anthropic":
        return ask_anthropic(
            prompt=prompt,
            model=spec.model,
            system=system,
            timeout_seconds=timeout_seconds,
        )
    raise ValueError(f"Unsupported model provider: {spec.provider}")


def build_openai_payload(
    prompt: str,
    model: str,
    system: Optional[str] = None,
) -> Dict[str, object]:
    input_messages = []
    if system:
        input_messages.append({"role": "system", "content": system})
    input_messages.append({"role": "user", "content": prompt})
    return {
        "model": model,
        "input": input_messages,
    }


def ask_openai(
    prompt: str,
    model: str,
    system: Optional[str] = None,
    timeout_seconds: float = 120.0,
) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for openai:* models.")

    request = _json_request(
        "https://api.openai.com/v1/responses",
        build_openai_payload(prompt=prompt, model=model, system=system),
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=max(1.0, float(timeout_seconds))) as response:
        body = json.loads(response.read().decode("utf-8"))
    return _extract_openai_text(body)


def build_anthropic_payload(
    prompt: str,
    model: str,
    system: Optional[str] = None,
) -> Dict[str, object]:
    payload = {
        "model": model,
        "max_tokens": 2048,
        "temperature": 0.1,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        payload["system"] = system
    return payload


def ask_anthropic(
    prompt: str,
    model: str,
    system: Optional[str] = None,
    timeout_seconds: float = 120.0,
) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for anthropic:* models.")

    request = _json_request(
        "https://api.anthropic.com/v1/messages",
        build_anthropic_payload(prompt=prompt, model=model, system=system),
        {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=max(1.0, float(timeout_seconds))) as response:
        body = json.loads(response.read().decode("utf-8"))
    return _extract_anthropic_text(body)


def _json_request(url: str, payload: Dict[str, object], headers: Dict[str, str]):
    return urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )


def _extract_openai_text(body: Dict[str, object]) -> str:
    output_text = body.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    parts = []
    for item in body.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                parts.append(content["text"])
    text = "\n".join(parts).strip()
    if text:
        return text
    raise RuntimeError("OpenAI response did not contain text output.")


def _extract_anthropic_text(body: Dict[str, object]) -> str:
    parts = []
    for content in body.get("content", []):
        if isinstance(content, dict) and content.get("type") == "text":
            parts.append(str(content.get("text", "")))
    text = "\n".join(part for part in parts if part).strip()
    if text:
        return text
    raise RuntimeError("Anthropic response did not contain text output.")


@contextmanager
def model_request_timeout(timeout_seconds: float):
    token = _MODEL_REQUEST_TIMEOUT_SECONDS.set(max(1.0, float(timeout_seconds)))
    try:
        yield
    finally:
        _MODEL_REQUEST_TIMEOUT_SECONDS.reset(token)
