from __future__ import annotations

import json
import time

from rag_demo.production.config import get_production_settings
from rag_demo.production.embedding_runtime import FastEmbedRuntime
from rag_demo.production.file_security import ClamAvScanner
from rag_demo.production.inference_probe import OllamaInferenceProbe
from rag_demo.production.object_storage import S3ObjectStorage
from rag_demo.production.vector_repository import QdrantChunkRepository


def _retry(name: str, operation, attempts: int = 60, delay_seconds: float = 2.0):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(delay_seconds)
    raise RuntimeError(f"{name} bootstrap failed") from last_error


def bootstrap() -> dict:
    settings = get_production_settings()
    storage = S3ObjectStorage.from_settings(settings)
    vectors = QdrantChunkRepository.from_settings(settings)
    scanner = ClamAvScanner(settings.clamav_host or "", settings.clamav_port)
    inference = OllamaInferenceProbe(
        settings.inference_base_url,
        settings.allowed_models,
    )

    _retry("object storage", storage.ensure_bucket)
    _retry(
        "Qdrant",
        lambda: vectors.ensure_collection(settings.embedding_dimensions),
    )
    _retry("ClamAV", scanner.ping)
    _retry("inference", inference.ping)

    runtime = FastEmbedRuntime.from_settings(settings)
    observed_dimensions = runtime.embedding_dimensions()
    if observed_dimensions != settings.embedding_dimensions:
        raise RuntimeError(
            "configured embedding dimensions do not match the embedding model"
        )
    runtime.sparse_documents(["production model readiness check"])
    runtime.rerank(
        "production readiness",
        ["production model readiness check"],
    )
    close = getattr(vectors.client, "close", None)
    if close is not None:
        close()
    return {
        "status": "ready",
        "bucket": settings.s3_bucket,
        "collection": settings.qdrant_collection,
        "embedding_dimensions": observed_dimensions,
        "models_prefetched": True,
    }


def main() -> None:
    print(json.dumps(bootstrap(), ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
