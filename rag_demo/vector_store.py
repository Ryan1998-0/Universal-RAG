from pathlib import Path
from typing import Callable, List, Optional, Sequence, Set

from rag_demo.chunking import Chunk
from rag_demo.embeddings import embed_query, load_embedding_matrix
from rag_demo.retrieval_scope import RetrievalScope


DEFAULT_QDRANT_COLLECTION = "rag_chunks"
DEFAULT_QDRANT_DIR = "qdrant"


class QdrantVectorStore:
    def __init__(
        self,
        client,
        collection_name: str,
        chunks_by_point_id,
        embed_query_fn: Callable[[str], Sequence[float]] = embed_query,
    ):
        self.client = client
        self.collection_name = collection_name
        self.chunks_by_point_id = chunks_by_point_id
        self.embed_query_fn = embed_query_fn

    @classmethod
    def in_memory(
        cls,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        collection_name: str = DEFAULT_QDRANT_COLLECTION,
        embed_query_fn: Callable[[str], Sequence[float]] = embed_query,
    ) -> "QdrantVectorStore":
        client, models = _qdrant_modules()
        qdrant = client(":memory:")
        _recreate_collection(qdrant, models, collection_name, embeddings)
        _upsert_chunks(qdrant, models, collection_name, chunks, embeddings)
        return cls(
            qdrant,
            collection_name,
            {index: chunk for index, chunk in enumerate(chunks)},
            embed_query_fn=embed_query_fn,
        )

    @classmethod
    def local(
        cls,
        index_dir: Path,
        chunks: Sequence[Chunk],
        embeddings: Optional[Sequence[Sequence[float]]] = None,
        collection_name: str = DEFAULT_QDRANT_COLLECTION,
        rebuild: bool = False,
        embed_query_fn: Callable[[str], Sequence[float]] = embed_query,
    ) -> "QdrantVectorStore":
        client, models = _qdrant_modules()
        qdrant_path = Path(index_dir) / DEFAULT_QDRANT_DIR
        qdrant_path.mkdir(parents=True, exist_ok=True)
        qdrant = client(path=str(qdrant_path))

        if rebuild or not _collection_exists(qdrant, collection_name):
            if embeddings is None:
                embedding_path = Path(index_dir) / "embeddings.npy"
                if not embedding_path.exists():
                    raise RuntimeError("embeddings.npy is required to build the Qdrant vector store.")
                embeddings = load_embedding_matrix(embedding_path)
            _recreate_collection(qdrant, models, collection_name, embeddings)
            _upsert_chunks(qdrant, models, collection_name, chunks, embeddings)

        return cls(
            qdrant,
            collection_name,
            {index: chunk for index, chunk in enumerate(chunks)},
            embed_query_fn=embed_query_fn,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        allowed_chunk_ids: Optional[Set[str]] = None,
        retrieval_scope: Optional[RetrievalScope] = None,
    ) -> List[Chunk]:
        _, models = _qdrant_modules()
        if allowed_chunk_ids is not None and not allowed_chunk_ids:
            return []
        must_conditions = []
        if retrieval_scope is not None:
            must_conditions.extend([
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(value=retrieval_scope.tenant_id),
                ),
                models.FieldCondition(
                    key="knowledge_base_id",
                    match=models.MatchValue(value=retrieval_scope.knowledge_base_id),
                ),
            ])
            if retrieval_scope.index_version_id:
                must_conditions.append(models.FieldCondition(
                    key="index_version_id",
                    match=models.MatchValue(value=retrieval_scope.index_version_id),
                ))
        if allowed_chunk_ids:
            must_conditions.append(models.FieldCondition(
                key="chunk_id",
                match=models.MatchAny(any=sorted(allowed_chunk_ids)),
            ))
        query_filter = (
            models.Filter(must=must_conditions)
            if must_conditions
            else None
        )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=[float(value) for value in self.embed_query_fn(query)],
            query_filter=query_filter,
            limit=top_k,
        )
        results = []
        for point in response.points:
            chunk = dict(self.chunks_by_point_id[int(point.id)])
            chunk["embedding_score"] = float(point.score)
            chunk["score"] = float(point.score)
            chunk["retrieval_method"] = "qdrant_dense"
            results.append(chunk)
        return results


def load_or_build_qdrant_vector_store(
    index_dir: Path,
    chunks: Sequence[Chunk],
    embeddings: Optional[Sequence[Sequence[float]]] = None,
    collection_name: str = DEFAULT_QDRANT_COLLECTION,
    rebuild: bool = False,
) -> Optional[QdrantVectorStore]:
    if embeddings is None and not (Path(index_dir) / "embeddings.npy").exists():
        return None
    return QdrantVectorStore.local(
        index_dir=Path(index_dir),
        chunks=chunks,
        embeddings=embeddings,
        collection_name=collection_name,
        rebuild=rebuild,
    )


def build_qdrant_vector_store(
    index_dir: Path,
    chunks: Sequence[Chunk],
    embeddings: Sequence[Sequence[float]],
    collection_name: str = DEFAULT_QDRANT_COLLECTION,
) -> QdrantVectorStore:
    return QdrantVectorStore.local(
        index_dir=Path(index_dir),
        chunks=chunks,
        embeddings=embeddings,
        collection_name=collection_name,
        rebuild=True,
    )


def _qdrant_modules():
    try:
        from qdrant_client import QdrantClient, models
    except ImportError as exc:
        raise RuntimeError(
            "qdrant-client is required for vector DB retrieval. "
            "Install it with: python3 -m pip install qdrant-client"
        ) from exc
    return QdrantClient, models


def _recreate_collection(client, models, collection_name: str, embeddings: Sequence[Sequence[float]]) -> None:
    vector_size = len(embeddings[0])
    if _collection_exists(client, collection_name):
        client.delete_collection(collection_name=collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )


def _upsert_chunks(client, models, collection_name: str, chunks, embeddings) -> None:
    points = [
        models.PointStruct(
            id=index,
            vector=[float(value) for value in embedding],
            payload={
                "chunk_id": chunk["id"],
                "source": chunk.get("source", ""),
                "title": chunk.get("title", ""),
                "parent_title": chunk.get("parent_title", ""),
                "chunk_index": int(chunk.get("chunk_index", index)),
                "tenant_id": chunk.get("tenant_id", ""),
                "knowledge_base_id": chunk.get("knowledge_base_id", ""),
                "document_version_id": chunk.get("document_version_id", ""),
                "index_version_id": chunk.get("index_version_id", ""),
            },
        )
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
    client.upsert(collection_name=collection_name, points=points)


def _collection_exists(client, collection_name: str) -> bool:
    try:
        return bool(client.collection_exists(collection_name))
    except AttributeError:
        try:
            client.get_collection(collection_name=collection_name)
            return True
        except Exception:
            return False
