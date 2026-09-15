"""Replaceable model-node interface used by the RAG pipeline and API.

Every place that may call a language or vision model is represented by a named
node.  The default implementation still delegates to ``model_providers`` so
existing Ollama/OpenAI/Anthropic/Codex behavior is preserved.  Applications can
inject another callable or use the API/environment bindings later without
rewriting pipeline nodes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable, Optional

from rag_demo.model_providers import ModelSpec, ask_model, parse_model_spec
from rag_demo.observability import TimingTrace


MODEL_NODE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "generation": {
        "label": "回答生成",
        "env": "RAG_MODEL_GENERATION",
        "fallback": "RAG_MODEL",
        "kind": "language_model",
    },
    "query_rewrite": {
        "label": "問題改寫與路由",
        "env": "RAG_MODEL_QUERY_REWRITE",
        "fallback": "RAG_MODEL",
        "kind": "language_model",
    },
    "keyword_extraction": {
        "label": "關鍵字抽取",
        "env": "RAG_MODEL_KEYWORD_EXTRACTION",
        "fallback": "RAG_MODEL",
        "kind": "language_model",
    },
    "question_extraction": {
        "label": "問題欄位抽取",
        "env": "RAG_MODEL_QUESTION_EXTRACTION",
        "fallback": "RAG_MODEL",
        "kind": "language_model",
    },
    "evidence_extraction": {
        "label": "證據抽取",
        "env": "RAG_MODEL_EVIDENCE_EXTRACTION",
        "fallback": "RAG_MODEL",
        "kind": "language_model",
    },
    "context_summary": {
        "label": "Context 摘要",
        "env": "RAG_MODEL_CONTEXT_SUMMARY",
        "fallback": "RAG_MODEL",
        "kind": "language_model",
    },
    "evaluation": {
        "label": "回答評估",
        "env": "RAG_MODEL_EVALUATION",
        "fallback": "RAG_MODEL",
        "kind": "language_model",
    },
    "vision": {
        "label": "圖片理解",
        "env": "RAG_VLM_MODEL",
        "fallback": "",
        "kind": "vision_model",
    },
}

SUPPORTED_MODEL_PROVIDERS = frozenset({"ollama", "openai", "anthropic", "codex"})


@dataclass(frozen=True)
class ModelBinding:
    node: str
    model: str
    source: str
    env_var: str
    kind: str
    replaceable: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "model": self.model,
            "source": self.source,
            "env_var": self.env_var,
            "kind": self.kind,
            "replaceable": self.replaceable,
        }


def normalize_model_node(node: str) -> str:
    clean = str(node or "").strip().lower().replace("-", "_")
    if clean not in MODEL_NODE_DEFINITIONS:
        raise ValueError(
            "unsupported model node; choose one of: "
            + ", ".join(sorted(MODEL_NODE_DEFINITIONS))
        )
    return clean


def resolve_model_for_node(
    node: str = "generation",
    *,
    requested_model: Optional[str] = None,
    default_model: Optional[str] = None,
    prefer_requested: bool = False,
) -> str:
    """Resolve a model for a node when no process-local gateway is injected.

    This function is intentionally small so standalone pipeline nodes can use
    the same environment contract as the API without depending on FastAPI.
    """

    clean_node = normalize_model_node(node)
    definition = MODEL_NODE_DEFINITIONS[clean_node]
    requested = str(requested_model or "").strip()
    if requested and prefer_requested:
        return requested
    env_model = str(os.getenv(definition["env"], "")).strip()
    if env_model:
        return env_model
    if requested:
        return requested
    fallback_name = definition.get("fallback")
    if fallback_name:
        fallback = str(os.getenv(fallback_name, "")).strip()
        if fallback:
            return fallback
    if clean_node == "vision":
        return ""
    return str(default_model or os.getenv("RAG_MODEL", "ollama:qwen2.5:7b")).strip()


def validate_model_spec(model: str, allowed_models: Optional[list[str]] = None) -> dict[str, Any]:
    clean = str(model or "").strip()
    if not clean:
        raise ValueError("model is required")
    spec: ModelSpec = parse_model_spec(clean)
    if spec.provider not in SUPPORTED_MODEL_PROVIDERS:
        raise ValueError(f"unsupported model provider: {spec.provider}")
    normalized = f"{spec.provider}:{spec.model}"
    allowed = list(allowed_models or [])
    allowed_result = not allowed or normalized in allowed or clean in allowed
    return {
        "requested": clean,
        "normalized": normalized,
        "provider": spec.provider,
        "name": spec.model,
        "allowed": allowed_result,
    }


class ModelGateway:
    """Thread-safe model binding registry with an injectable invocation hook."""

    def __init__(
        self,
        *,
        default_model: str = "ollama:qwen2.5:7b",
        allowed_models: Optional[list[str]] = None,
        ask_model_fn: Callable[..., str] = ask_model,
    ):
        self.default_model = str(default_model or "ollama:qwen2.5:7b").strip()
        self.allowed_models = list(dict.fromkeys(allowed_models or [self.default_model]))
        if self.default_model not in self.allowed_models:
            self.allowed_models.insert(0, self.default_model)
        self._ask_model_fn = ask_model_fn
        self._overrides: dict[str, str] = {}
        self._lock = RLock()

    def resolve(self, node: str = "generation", requested_model: Optional[str] = None) -> str:
        clean_node = normalize_model_node(node)
        definition = MODEL_NODE_DEFINITIONS[clean_node]
        with self._lock:
            override = self._overrides.get(clean_node, "").strip()
        if override:
            return override
        env_model = str(os.getenv(definition["env"], "")).strip()
        if env_model:
            return env_model
        requested = str(requested_model or "").strip()
        if requested:
            return requested
        fallback_name = definition.get("fallback")
        if fallback_name:
            fallback = str(os.getenv(fallback_name, "")).strip()
            if fallback:
                return fallback
        if clean_node == "vision":
            # Vision is optional.  Returning the text-generation default here
            # would silently route image work to the wrong adapter.
            return ""
        return self.default_model

    def binding(self, node: str = "generation", requested_model: Optional[str] = None) -> ModelBinding:
        clean_node = normalize_model_node(node)
        definition = MODEL_NODE_DEFINITIONS[clean_node]
        with self._lock:
            has_override = clean_node in self._overrides
        env_model = str(os.getenv(definition["env"], "")).strip()
        requested = str(requested_model or "").strip()
        if has_override:
            source = "api_override"
        elif env_model:
            source = "environment"
        elif requested:
            source = "request"
        else:
            source = "default"
        return ModelBinding(
            node=clean_node,
            model=self.resolve(clean_node, requested_model=requested_model),
            source=source,
            env_var=str(definition["env"]),
            kind=str(definition["kind"]),
        )

    def set_override(self, node: str, model: str) -> ModelBinding:
        clean_node = normalize_model_node(node)
        validation = validate_model_spec(model, self.allowed_models)
        if not validation["allowed"]:
            raise ValueError("model is not in the server allowlist")
        with self._lock:
            self._overrides[clean_node] = validation["normalized"]
        return self.binding(clean_node)

    def clear_override(self, node: str) -> ModelBinding:
        clean_node = normalize_model_node(node)
        with self._lock:
            self._overrides.pop(clean_node, None)
        return self.binding(clean_node)

    def describe(self) -> dict[str, Any]:
        return {
            "default_model": self.default_model,
            "allowed_models": list(self.allowed_models),
            "nodes": [
                self.binding(node).as_dict()
                for node in sorted(MODEL_NODE_DEFINITIONS)
            ],
            "replacement": {
                "per_request_field": "model",
                "environment_override": True,
                "runtime_override_scope": "process",
                "runtime_override_persistent": False,
                "restart_for_environment_change": True,
            },
        }

    def invoke(
        self,
        prompt: str,
        *,
        node: str = "generation",
        model: Optional[str] = None,
        system: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        trace: Optional[TimingTrace] = None,
    ) -> str:
        """Invoke a model through one replaceable, timed node interface."""

        clean_node = normalize_model_node(node)
        resolved_model = self.resolve(clean_node, requested_model=model)
        timer_trace = trace or TimingTrace(component="model_gateway")
        metadata = {"node": clean_node, "model": resolved_model}
        with timer_trace.stage("model." + clean_node, **metadata):
            kwargs: dict[str, Any] = {"model": resolved_model, "system": system}
            if timeout_seconds is not None:
                # Existing providers read timeout from their context manager;
                # callers can still wrap this invocation with that context.
                kwargs["timeout_seconds"] = timeout_seconds
            try:
                return self._ask_model_fn(prompt, **kwargs)
            except TypeError as exc:
                # Test doubles and legacy adapters commonly accept only the
                # prompt/model/system trio.  Keep that compatibility explicit.
                if "timeout_seconds" not in kwargs or "timeout_seconds" not in str(exc):
                    raise
                kwargs.pop("timeout_seconds", None)
                return self._ask_model_fn(prompt, **kwargs)
