import io
import hashlib
import unittest
from types import SimpleNamespace

from rag_demo.production.object_storage import (
    ObjectStorageError,
    S3ObjectStorage,
    document_object_key,
)
from rag_demo.production.vector_repository import QdrantChunkRepository, VectorRepositoryError
from rag_demo.retrieval_scope import RetrievalScope


class FakeS3Client:
    def __init__(self):
        self.calls = []
        self.objects = {}

    def head_bucket(self, **kwargs):
        self.calls.append(("head_bucket", kwargs))

    def put_object(self, **kwargs):
        payload = kwargs["Body"].read()
        self.calls.append(("put_object", {**kwargs, "Body": payload}))
        self.objects[kwargs["Key"]] = payload

    def get_object(self, **kwargs):
        payload = self.objects[kwargs["Key"]]
        return {"ContentLength": len(payload), "Body": io.BytesIO(payload)}

    def generate_presigned_url(self, operation, Params, ExpiresIn):
        self.calls.append(("presign", {"operation": operation, "params": Params, "ttl": ExpiresIn}))
        return f"https://signed.invalid/{Params['Key']}"

    def delete_object(self, **kwargs):
        self.calls.append(("delete", kwargs))


class FakeQdrantClient:
    def __init__(self):
        self.upserts = []
        self.queries = []
        self.deletes = []
        self.scrolls = []

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    def query_points(self, **kwargs):
        self.queries.append(kwargs)
        return SimpleNamespace(points=[
            SimpleNamespace(
                id="point-1",
                score=0.83,
                payload={"chunk_id": "chunk-1", "content": "evidence"},
            )
        ])

    def delete(self, **kwargs):
        self.deletes.append(kwargs)

    def get_collection(self, **kwargs):
        return {"status": "green"}

    def scroll(self, **kwargs):
        self.scrolls.append(kwargs)
        content = "evidence"
        return ([SimpleNamespace(
            id="point-1",
            payload={
                "chunk_id": "chunk-1",
                "chunk_record_id": "record-1",
                "document_id": "document-1",
                "document_version_id": "version-1",
                "content": content,
                "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
            },
        )], None)


def filter_values(query_filter):
    result = {}
    for condition in query_filter.must:
        match = condition.match
        result[condition.key] = getattr(match, "value", None)
        if result[condition.key] is None:
            result[condition.key] = list(getattr(match, "any", []) or [])
    return result


class ProductionObjectStorageTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeS3Client()
        self.storage = S3ObjectStorage(self.client, "rag-documents")

    def test_document_keys_are_tenant_scoped_and_ignore_user_path(self):
        key = document_object_key(
            tenant_id="tenant-a",
            knowledge_base_id="kb-1",
            document_id="doc-1",
            version_id="version-1",
            filename="../../payroll.pdf",
        )
        self.assertEqual(
            key,
            "tenants/tenant-a/knowledge-bases/kb-1/documents/doc-1/versions/version-1/original.pdf",
        )
        self.assertNotIn("..", key)

    def test_put_checks_sha_and_private_key(self):
        stored = self.storage.put_document_version(
            tenant_id="tenant-a",
            knowledge_base_id="kb-1",
            document_id="doc-1",
            version_id="version-1",
            filename="policy.pdf",
            content_type="application/pdf",
            body=b"safe-payload",
        )
        self.assertEqual(stored.size_bytes, 12)
        self.assertEqual(self.client.objects[stored.key], b"safe-payload")
        with self.assertRaises(ObjectStorageError):
            self.storage.put_document_version(
                tenant_id="tenant-a",
                knowledge_base_id="kb-1",
                document_id="doc-1",
                version_id="version-2",
                filename="policy.pdf",
                content_type="application/pdf",
                body=b"changed",
                expected_sha256="0" * 64,
            )

    def test_download_limit_and_signed_url_ttl_are_enforced(self):
        key = document_object_key(
            tenant_id="tenant-a",
            knowledge_base_id="kb-1",
            document_id="doc-1",
            version_id="version-1",
            filename="policy.pdf",
        )
        self.client.objects[key] = b"12345"
        self.assertEqual(self.storage.download_bytes(key, max_bytes=5), b"12345")
        with self.assertRaises(ObjectStorageError):
            self.storage.download_bytes(key, max_bytes=4)
        self.storage.presign_download(key, expires_seconds=9999)
        self.assertEqual(self.client.calls[-1][1]["ttl"], 900)


class ProductionVectorRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeQdrantClient()
        self.repository = QdrantChunkRepository(self.client, "rag_chunks")
        self.scope = RetrievalScope(
            tenant_id="tenant-a",
            knowledge_base_id="kb-1",
            index_version_id="index-7",
        )

    def test_search_always_filters_tenant_kb_index_and_sources(self):
        hits = self.repository.search(
            scope=self.scope,
            query_vector=[0.2, 0.8],
            top_k=8,
            source_ids=["doc-a", "doc-b"],
        )
        values = filter_values(self.client.queries[0]["query_filter"])
        self.assertEqual(values["tenant_id"], "tenant-a")
        self.assertEqual(values["knowledge_base_id"], "kb-1")
        self.assertEqual(values["index_version_id"], "index-7")
        self.assertEqual(values["source_id"], ["doc-a", "doc-b"])
        self.assertEqual(hits[0].as_chunk()["embedding_score"], 0.83)

    def test_existing_collection_rejects_wrong_embedding_dimensions(self):
        class ExistingCollectionClient(FakeQdrantClient):
            def __init__(self, dense_size):
                super().__init__()
                self.dense_size = dense_size

            def collection_exists(self, name):
                return True

            def get_collection(self, **kwargs):
                vectors = {"dense": SimpleNamespace(size=self.dense_size)}
                return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=vectors)))

            def create_payload_index(self, **kwargs):
                return None

        incompatible = QdrantChunkRepository(ExistingCollectionClient(768), "rag_chunks")
        with self.assertRaisesRegex(VectorRepositoryError, "incompatible dense vector size"):
            incompatible.ensure_collection(384)

        compatible = QdrantChunkRepository(ExistingCollectionClient(384), "rag_chunks")
        compatible.ensure_collection(384)

    def test_empty_source_selection_returns_no_data_without_query(self):
        self.assertEqual(
            self.repository.search(
                scope=self.scope,
                query_vector=[1.0, 0.0],
                top_k=3,
                source_ids=[],
            ),
            [],
        )
        self.assertEqual(self.client.queries, [])

    def test_upsert_rejects_cross_tenant_payload_and_requires_version(self):
        chunk = {
            "id": "chunk-1",
            "document_id": "document-1",
            "document_version_id": "version-1",
            "source_id": "source-1",
            "tenant_id": "tenant-b",
            "content": "evidence",
        }
        with self.assertRaises(ValueError):
            self.repository.upsert_chunks(scope=self.scope, chunks=[chunk], vectors=[[1.0, 0.0]])
        with self.assertRaises(ValueError):
            self.repository.search(
                scope=RetrievalScope("tenant-a", "kb-1"),
                query_vector=[1.0, 0.0],
                top_k=3,
            )

    def test_delete_index_uses_the_same_mandatory_scope(self):
        self.repository.delete_index(self.scope)
        values = filter_values(self.client.deletes[0]["points_selector"].filter)
        self.assertEqual(
            values,
            {
                "tenant_id": "tenant-a",
                "knowledge_base_id": "kb-1",
                "index_version_id": "index-7",
            },
        )

    def test_index_fingerprint_scrolls_only_the_versioned_scope(self):
        digest = self.repository.index_entries_sha256(self.scope)
        self.assertEqual(len(digest), 64)
        call = self.client.scrolls[0]
        self.assertFalse(call["with_vectors"])
        values = filter_values(call["scroll_filter"])
        self.assertEqual(
            values,
            {
                "tenant_id": "tenant-a",
                "knowledge_base_id": "kb-1",
                "index_version_id": "index-7",
            },
        )


if __name__ == "__main__":
    unittest.main()
