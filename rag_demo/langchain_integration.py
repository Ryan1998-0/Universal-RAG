"""Optional LangChain integrations for the Universal-RAG pipeline.

The project keeps retrieval, ACL and evidence policies in first-party modules
because those rules are domain-specific. This module adapts the model layer to
LangChain's standard chat model interface without making LangChain a hard
runtime dependency for the existing local demo.
"""

from __future__ import annotations

import os
from typing import Any, Optional


class LangChainUnavailableError(RuntimeError):
    """Raised when the optional LangChain backend is not installed."""


def _temperature() -> float:
    try:
        return float(os.getenv("RAG_OLLAMA_TEMPERATURE", "0"))
    except (TypeError, ValueError):
        return 0.0


def _integer_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def create_chat_model(model_spec: str, timeout_seconds: float = 120.0) -> Any:
    """Create a LangChain chat model for an existing ``provider:model`` spec.

    Provider packages are imported lazily so the legacy backend and all tests
    continue to work when the optional extra is not installed.
    """

    from rag_demo.model_providers import parse_model_spec

    spec = parse_model_spec(model_spec)
    temperature = _temperature()
    if spec.provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise LangChainUnavailableError(
                "LangChain Ollama integration is not installed. "
                "Install the [langchain] extra."
            ) from exc

        from rag_demo.ollama_client import ollama_base_url

        # ``client_kwargs`` is supported by the Ollama integration and keeps
        # the request timeout explicit. Older versions may not accept it, so
        # retry construction without it for compatibility.
        common = {
            "model": spec.model,
            "base_url": ollama_base_url(),
            "temperature": temperature,
            "num_ctx": _integer_env("RAG_OLLAMA_NUM_CTX", 8192),
            "num_predict": _integer_env("RAG_OLLAMA_NUM_PREDICT", 768),
        }
        try:
            return ChatOllama(
                **common,
                client_kwargs={"timeout": max(1.0, float(timeout_seconds))},
            )
        except TypeError:
            return ChatOllama(**common)

    if spec.provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise LangChainUnavailableError(
                "LangChain OpenAI integration is not installed. "
                "Install the [langchain] extra."
            ) from exc
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for openai:* models.")
        return ChatOpenAI(
            model=spec.model,
            temperature=temperature,
            timeout=max(1.0, float(timeout_seconds)),
        )

    if spec.provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise LangChainUnavailableError(
                "LangChain Anthropic integration is not installed. "
                "Install the [langchain] extra."
            ) from exc
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is required for anthropic:* models.")
        return ChatAnthropic(
            model=spec.model,
            max_tokens=2048,
            temperature=0.1,
            timeout=max(1.0, float(timeout_seconds)),
        )

    raise ValueError(f"Unsupported model provider: {spec.provider}")


def _message_content(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        return _message_content(content.get("content", ""))
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return str(content or "").strip()


def ask_langchain(
    prompt: str,
    model: str = "ollama:qwen2.5:7b",
    system: Optional[str] = None,
    timeout_seconds: float = 120.0,
) -> str:
    """Invoke a LangChain chat model using the project's existing prompt API."""

    try:
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError as exc:
        raise LangChainUnavailableError(
            "LangChain core is not installed. Install the [langchain] extra."
        ) from exc

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system or "You are a helpful assistant."),
            ("human", "{user_prompt}"),
        ]
    )
    prompt_value = prompt_template.invoke({"user_prompt": prompt})
    response = create_chat_model(model, timeout_seconds=timeout_seconds).invoke(
        prompt_value.to_messages()
    )
    text = _message_content(response)
    if not text:
        raise RuntimeError("LangChain model did not return text output.")
    return text


def build_langchain_retriever(
    profile: str,
    *,
    retriever_factory: Any,
    source_ids: Optional[list[str]] = None,
    top_k: int = 8,
    candidate_k: int = 24,
    retrieval_scope: Any = None,
) -> Any:
    """Adapt the project's hybrid retriever to LangChain's Retriever API.

    The adapter keeps the richer multi-query call available on the first-party
    retriever. ``invoke`` is useful for simple semantic search and LangChain
    tooling, while the production pipeline can pass its full retrieval plan.
    """

    try:
        from langchain_core.documents import Document
        from langchain_core.retrievers import BaseRetriever
    except ImportError as exc:
        raise LangChainUnavailableError(
            "LangChain core is not installed. Install the [langchain] extra."
        ) from exc

    class HybridRetriever(BaseRetriever):
        profile: str
        top_k: int = 8
        candidate_k: int = 24
        source_ids: tuple[str, ...] = ()
        retrieval_scope: Any = None

        def _get_relevant_documents(self, query: str, *, run_manager: Any = None):
            del run_manager
            result = retriever_factory(self.profile).retrieve(
                question=query,
                retrieval_query=query,
                query_variants=(query,),
                evidence_query=query,
                source_ids=list(self.source_ids) or None,
                top_k=self.top_k,
                candidate_k=self.candidate_k,
                retrieval_scope=self.retrieval_scope,
            )
            documents = []
            raw_contexts = result.get("contexts", []) if isinstance(result, dict) else []
            for context in raw_contexts:
                if not isinstance(context, dict):
                    continue
                content = str(context.get("content") or "").strip()
                if not content:
                    continue
                metadata = {
                    key: context[key]
                    for key in (
                        "id", "rank", "title", "source", "page",
                        "score", "bm25Score", "embeddingScore", "rerankScore",
                    )
                    if key in context
                }
                documents.append(Document(page_content=content, metadata=metadata))
            return documents

    return HybridRetriever(
        profile=profile,
        top_k=max(1, int(top_k)),
        candidate_k=max(1, int(candidate_k)),
        source_ids=tuple(str(item) for item in (source_ids or [])),
        retrieval_scope=retrieval_scope,
    )


def langchain_backend_available() -> bool:
    """Return whether the core LangChain package can be imported."""

    try:
        import langchain_core  # noqa: F401
    except ImportError:
        return False
    return True
