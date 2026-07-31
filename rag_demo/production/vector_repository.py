from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence
from uuid import NAMESPACE_URL, uuid5

from rag_demo.production.index_manifest import index_entries_sha256
from rag_demo.retrieval_scope import RetrievalScope


class VectorRepositoryError(RuntimeError):
    pass


@dataclass(frozen=True)
class DenseSearchHit:
    point_id: str
    score: float
    payload: dict

    def as_chunk(self) -> dict:
        result = dict(self.payload)
        result["qdrant_point_id"] = self.point_id
        result["embedding_score"] = self.score
        result["score"] = self.score
        result["retrieval_method"] = "qdrant_dense"
        return result


class QdrantChunkRepository:
    """Qdrant adapter that refuses unscoped reads, writes, and deletes."""

    def __init__(self, client, collection_name: str):
        self.client = client
        self.collection_name = str(collection_name or "").strip()
        if not self.collection_name:
            raise ValueError("collection_name is required")

    @classmethod
    def from_settings(cls, settings) -> "QdrantChunkRepository":
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise VectorRepositoryError("qdrant-client is required") from exc
        return cls(
            client=QdrantClient(
                url=settings.qdrant_url,
                api_key=(
                    settings.qdrant_api_key.get_secret_value()
                    if settings.qdrant_api_key
                    else None
                ),
                timeout=10,
            ),
            collection_name=settings.qdrant_collection,
        )

    def ping(self) -> None:
        self.client.get_collection(collection_name=self.collection_name)

    def ensure_collection(self, vector_size: int) -> None:
        if vector_size <= 0:
            raise ValueError("vector_size must be positive")
        models = _models()
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=int(vector_size),
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )
        for field_name in (
            "tenant_id",
            "knowledge_base_id",
            "index_version_id",
            "document_version_id",
            "source_id",
        ):
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )

    def upsert_chunks(
        self,
        *,
        scope: RetrievalScope,
        chunks: Sequence[dict],
        vectors: Sequence[Sequence[float]],
        sparse_vectors: Optional[Sequence[object]] = None,
    ) -> list[str]:
        _require_versioned_scope(scope)
        if len(chunks) != len(vectors):
            raise ValueError("chunk and vector counts must match")
        if sparse_vectors is not None and len(chunks) != len(sparse_vectors):
            raise ValueError("chunk and sparse vector counts must match")
        prepared = []
        point_ids = []
        expected_dimensions = None
        for ordinal, (chunk, vector) in enumerate(zip(chunks, vectors)):
            clean_vector = [float(value) for value in vector]
            if not clean_vector:
                raise ValueError("vectors must not be empty")
            if expected_dimensions is None:
                expected_dimensions = len(clean_vector)
            elif len(clean_vector) != expected_dimensions:
                raise ValueError("all vectors must have the same dimensions")

            payload = _chunk_payload(scope, chunk, ordinal)
            point_id = _point_id(scope, payload["chunk_id"])
            sparse_vector = (
                _as_sparse_vector(sparse_vectors[ordinal])
                if sparse_vectors is not None
                else None
            )
            prepared.append((point_id, clean_vector, sparse_vector, payload))
            point_ids.append(point_id)

        models = _models()
        points = []
        for point_id, clean_vector, sparse_vector, payload in prepared:
            point_vectors = {"dense": clean_vector}
            if sparse_vector is not None:
                point_vectors["bm25"] = sparse_vector
            points.append(models.PointStruct(
                id=point_id,
                vector=point_vectors,
                payload=payload,
            ))
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
        return point_ids

    def search(
        self,
        *,
        scope: RetrievalScope,
        query_vector: Sequence[float],
        top_k: int,
        source_ids: Optional[Sequence[str]] = None,
    ) -> list[DenseSearchHit]:
        _require_versioned_scope(scope)
        clean_vector = [float(value) for value in query_vector]
        if not clean_vector:
            raise ValueError("query_vector must not be empty")
        limit = max(1, min(int(top_k), 200))
        if source_ids is not None and not list(source_ids):
            return []
        query_filter = _scope_filter(scope, source_ids=source_ids)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=clean_vector,
            using="dense",
            query_filter=query_filter,
            with_payload=True,
            limit=limit,
        )
        return [
            DenseSearchHit(
                point_id=str(point.id),
                score=float(point.score),
                payload=dict(point.payload or {}),
            )
            for point in response.points
        ]

    def search_sparse(
        self,
        *,
        scope: RetrievalScope,
        query_sparse_vector: object,
        top_k: int,
        source_ids: Optional[Sequence[str]] = None,
    ) -> list[DenseSearchHit]:
        _require_versioned_scope(scope)
        if source_ids is not None and not list(source_ids):
            return []
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=_as_sparse_vector(query_sparse_vector),
            using="bm25",
            query_filter=_scope_filter(scope, source_ids=source_ids),
            with_payload=True,
            limit=max(1, min(int(top_k), 200)),
        )
        return [
            DenseSearchHit(
                point_id=str(point.id),
                score=float(point.score),
                payload=dict(point.payload or {}),
            )
            for point in response.points
        ]

    def delete_index(self, scope: RetrievalScope) -> None:
        _require_versioned_scope(scope)
        models = _models()
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=_scope_filter(scope)),
            wait=True,
        )

    def count_index(self, scope: RetrievalScope) -> int:
        _require_versioned_scope(scope)
        result = self.client.count(
            collection_name=self.collection_name,
            count_filter=_scope_filter(scope),
            exact=True,
        )
        return int(result.count)

    def index_entries_sha256(self, scope: RetrievalScope) -> str:
        _require_versioned_scope(scope)
        entries = []
        offset = None
        payload_fields = [
            "chunk_id",
            "chunk_record_id",
            "document_id",
            "document_version_id",
            "content",
            "content_sha256",
        ]
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=_scope_filter(scope),
                limit=256,
                offset=offset,
                with_payload=payload_fields,
                with_vectors=False,
            )
            for point in points:
                payload = dict(point.payload or {})
                entries.append({
                    "qdrant_point_id": str(point.id),
                    "chunk_id": payload.get("chunk_id"),
                    "chunk_record_id": payload.get("chunk_record_id"),
                    "document_id": payload.get("document_id"),
                    "document_version_id": payload.get("document_version_id"),
                    "content": payload.get("content"),
                    "content_sha256": payload.get("content_sha256"),
                })
            if offset is None:
                break
        return index_entries_sha256(entries)

    def delete_document_version(
        self,
        *,
        scope: RetrievalScope,
        document_version_id: str,
    ) -> None:
        _require_versioned_scope(scope)
        clean_version = str(document_version_id or "").strip()
        if not clean_version:
            raise ValueError("document_version_id is required")
        models = _models()
        query_filter = _scope_filter(scope)
        query_filter.must.append(models.FieldCondition(
            key="document_version_id",
            match=models.MatchValue(value=clean_version),
        ))
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=query_filter),
            wait=True,
        )


def _scope_filter(
    scope: RetrievalScope,
    source_ids: Optional[Sequence[str]] = None,
):
    models = _models()
    must = [
        models.FieldCondition(
            key="tenant_id",
            match=models.MatchValue(value=scope.tenant_id),
        ),
        models.FieldCondition(
            key="knowledge_base_id",
            match=models.MatchValue(value=scope.knowledge_base_id),
        ),
        models.FieldCondition(
            key="index_version_id",
            match=models.MatchValue(value=scope.index_version_id),
        ),
    ]
    if source_ids is not None:
        clean_sources = list(dict.fromkeys(
            str(source_id).strip()
            for source_id in source_ids
            if str(source_id).strip()
        ))
        must.append(models.FieldCondition(
            key="source_id",
            match=models.MatchAny(any=clean_sources),
        ))
    return models.Filter(must=must)


def _chunk_payload(scope: RetrievalScope, chunk: dict, ordinal: int) -> dict:
    for field_name, expected in (
        ("tenant_id", scope.tenant_id),
        ("knowledge_base_id", scope.knowledge_base_id),
        ("index_version_id", scope.index_version_id),
    ):
        supplied = str(chunk.get(field_name) or "").strip()
        if supplied and supplied != expected:
            raise ValueError(f"chunk {field_name} is outside the retrieval scope")

    chunk_id = str(chunk.get("id") or chunk.get("chunk_id") or "").strip()
    document_id = str(chunk.get("document_id") or "").strip()
    document_version_id = str(chunk.get("document_version_id") or "").strip()
    source_id = str(chunk.get("source_id") or chunk.get("source") or "").strip()
    if not all((chunk_id, document_id, document_version_id, source_id)):
        raise ValueError(
            "chunks require id, document_id, document_version_id, and source_id"
        )
    content = str(chunk.get("content") or "")
    digest = str(chunk.get("content_sha256") or "").strip().lower()
    computed_digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if digest and digest != computed_digest:
        raise ValueError("chunk content_sha256 does not match content")
    return {
        "tenant_id": scope.tenant_id,
        "knowledge_base_id": scope.knowledge_base_id,
        "index_version_id": scope.index_version_id,
        "document_id": document_id,
        "document_version_id": document_version_id,
        "source_id": source_id,
        "chunk_id": chunk_id,
        "chunk_record_id": str(chunk.get("chunk_record_id") or ""),
        "ordinal": int(chunk.get("ordinal", chunk.get("chunk_index", ordinal))),
        "page": str(chunk.get("page") or chunk.get("page_start") or ""),
        "title": str(chunk.get("title") or ""),
        "parent_title": str(chunk.get("parent_title") or ""),
        "content": content,
        "content_sha256": computed_digest,
    }


def _point_id(scope: RetrievalScope, chunk_id: str) -> str:
    stable_name = "/".join((
        scope.tenant_id,
        scope.knowledge_base_id,
        scope.index_version_id,
        chunk_id,
    ))
    return str(uuid5(NAMESPACE_URL, stable_name))


def _as_sparse_vector(value: object):
    models = _models()
    if isinstance(value, models.SparseVector):
        return value
    indices = getattr(value, "indices", None)
    values = getattr(value, "values", None)
    if isinstance(value, dict):
        indices = value.get("indices")
        values = value.get("values")
    clean_indices = [int(item) for item in (indices or [])]
    clean_values = [float(item) for item in (values or [])]
    if not clean_indices or len(clean_indices) != len(clean_values):
        raise ValueError("sparse vectors require equal non-empty indices and values")
    return models.SparseVector(indices=clean_indices, values=clean_values)


def _require_versioned_scope(scope: RetrievalScope) -> None:
    if not isinstance(scope, RetrievalScope) or not scope.index_version_id:
        raise ValueError("a tenant, knowledge base, and index version scope is required")


def _models():
    try:
        from qdrant_client import models
    except ImportError as exc:
        raise VectorRepositoryError("qdrant-client is required") from exc
    return models
