from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Optional


_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SAFE_EXTENSIONS = {
    ".bmp",
    ".docx",
    ".heic",
    ".jpeg",
    ".jpg",
    ".json",
    ".markdown",
    ".md",
    ".pdf",
    ".png",
    ".tif",
    ".tiff",
    ".txt",
    ".webp",
}


class ObjectStorageError(RuntimeError):
    pass


@dataclass(frozen=True)
class StoredObject:
    key: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ObjectInfo:
    key: str
    size_bytes: int
    content_type: str
    sha256: str


class S3ObjectStorage:
    """Private S3-compatible storage with tenant-scoped object keys."""

    def __init__(self, client, bucket: str, presign_client=None):
        self.client = client
        self.presign_client = presign_client or client
        self.bucket = _required_component(bucket, "bucket", max_length=255)

    @classmethod
    def from_settings(cls, settings) -> "S3ObjectStorage":
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise ObjectStorageError("boto3 is required for object storage") from exc

        credentials = {
            "aws_access_key_id": settings.s3_access_key_id,
            "aws_secret_access_key": (
                settings.s3_secret_access_key.get_secret_value()
                if settings.s3_secret_access_key
                else None
            ),
            "region_name": settings.s3_region,
            "config": Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
                retries={"max_attempts": 4, "mode": "standard"},
                connect_timeout=5,
                read_timeout=30,
            ),
        }
        client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            **credentials,
        )
        public_endpoint = str(settings.s3_public_endpoint_url or "").strip()
        presign_client = (
            boto3.client("s3", endpoint_url=public_endpoint, **credentials)
            if public_endpoint
            else client
        )
        return cls(
            client=client,
            presign_client=presign_client,
            bucket=settings.s3_bucket,
        )

    def ping(self) -> None:
        self.client.head_bucket(Bucket=self.bucket)

    def ensure_bucket(self) -> None:
        try:
            self.ping()
        except Exception:
            self.client.create_bucket(Bucket=self.bucket)
            self.ping()

    def presign_upload(
        self,
        *,
        key: str,
        content_type: str,
        expected_sha256: str,
        expires_seconds: int = 900,
    ) -> dict:
        safe_key = _required_object_key(key)
        digest = str(expected_sha256 or "").strip().lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
        ttl = max(60, min(int(expires_seconds), 3600))
        required_headers = {
            "Content-Type": str(content_type or "application/octet-stream"),
            "x-amz-meta-sha256": digest,
        }
        url = self.presign_client.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": self.bucket,
                "Key": safe_key,
                "ContentType": required_headers["Content-Type"],
                "Metadata": {"sha256": digest},
            },
            ExpiresIn=ttl,
        )
        return {"url": str(url), "headers": required_headers, "expires_in": ttl}

    def put_reserved_upload(
        self,
        *,
        key: str,
        body: BinaryIO,
        content_type: str,
        size_bytes: int,
        sha256: str,
    ) -> ObjectInfo:
        safe_key = _required_object_key(key)
        clean_size = int(size_bytes)
        clean_digest = str(sha256 or "").strip().lower()
        if clean_size <= 0:
            raise ValueError("size_bytes must be positive")
        if len(clean_digest) != 64 or any(
            character not in "0123456789abcdef" for character in clean_digest
        ):
            raise ValueError("sha256 must be a lowercase SHA-256 digest")
        if hasattr(body, "seek"):
            body.seek(0)
        clean_content_type = str(content_type or "application/octet-stream")
        self.client.put_object(
            Bucket=self.bucket,
            Key=safe_key,
            Body=body,
            ContentLength=clean_size,
            ContentType=clean_content_type,
            Metadata={"sha256": clean_digest},
        )
        return ObjectInfo(
            key=safe_key,
            size_bytes=clean_size,
            content_type=clean_content_type,
            sha256=clean_digest,
        )

    def head(self, key: str) -> ObjectInfo:
        safe_key = _required_object_key(key)
        response = self.client.head_object(Bucket=self.bucket, Key=safe_key)
        metadata = dict(response.get("Metadata") or {})
        return ObjectInfo(
            key=safe_key,
            size_bytes=int(response.get("ContentLength") or 0),
            content_type=str(response.get("ContentType") or "application/octet-stream"),
            sha256=str(metadata.get("sha256") or "").lower(),
        )

    def put_document_version(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        document_id: str,
        version_id: str,
        filename: str,
        content_type: str,
        body: bytes | BinaryIO,
        expected_sha256: str = "",
    ) -> StoredObject:
        key = document_object_key(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
            version_id=version_id,
            filename=filename,
        )
        payload = body.read() if hasattr(body, "read") else bytes(body)
        digest = hashlib.sha256(payload).hexdigest()
        if expected_sha256 and digest != str(expected_sha256).lower():
            raise ObjectStorageError("object checksum does not match the expected SHA-256")
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=BytesIO(payload),
            ContentLength=len(payload),
            ContentType=str(content_type or "application/octet-stream"),
            Metadata={"sha256": digest},
        )
        return StoredObject(key=key, size_bytes=len(payload), sha256=digest)

    def put_extracted_artifact(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        document_id: str,
        version_id: str,
        body: bytes,
        artifact_name: str = "canonical.json",
        content_type: str = "application/json",
    ) -> StoredObject:
        key = extracted_object_key(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
            version_id=version_id,
            artifact_name=artifact_name,
        )
        payload = bytes(body)
        digest = hashlib.sha256(payload).hexdigest()
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=BytesIO(payload),
            ContentLength=len(payload),
            ContentType=content_type,
            Metadata={"sha256": digest},
        )
        return StoredObject(key=key, size_bytes=len(payload), sha256=digest)

    def put_index_manifest(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        index_version_id: str,
        body: bytes,
    ) -> StoredObject:
        key = index_manifest_object_key(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            index_version_id=index_version_id,
        )
        payload = bytes(body)
        digest = hashlib.sha256(payload).hexdigest()
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=BytesIO(payload),
            ContentLength=len(payload),
            ContentType="application/json",
            Metadata={"sha256": digest, "immutable": "true"},
        )
        return StoredObject(key=key, size_bytes=len(payload), sha256=digest)

    def download_bytes(self, key: str, max_bytes: int) -> bytes:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        safe_key = _required_object_key(key)
        response = self.client.get_object(Bucket=self.bucket, Key=safe_key)
        declared_size = int(response.get("ContentLength") or 0)
        if declared_size > max_bytes:
            raise ObjectStorageError("stored object exceeds the permitted size")
        payload = response["Body"].read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise ObjectStorageError("stored object exceeds the permitted size")
        return payload

    def presign_download(self, key: str, expires_seconds: int = 300) -> str:
        safe_key = _required_object_key(key)
        ttl = max(30, min(int(expires_seconds), 900))
        return str(self.presign_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": safe_key},
            ExpiresIn=ttl,
        ))

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=_required_object_key(key))


def document_object_key(
    *,
    tenant_id: str,
    knowledge_base_id: str,
    document_id: str,
    version_id: str,
    filename: str,
) -> str:
    suffix = Path(str(filename or "")).suffix.lower()
    if suffix not in _SAFE_EXTENSIONS:
        suffix = ".bin"
    return _version_prefix(
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        version_id=version_id,
    ) + f"/original{suffix}"


def upload_object_key(
    *,
    tenant_id: str,
    knowledge_base_id: str,
    upload_session_id: str,
    filename: str,
) -> str:
    suffix = Path(str(filename or "")).suffix.lower()
    if suffix not in _SAFE_EXTENSIONS:
        suffix = ".bin"
    return (
        f"tenants/{_required_component(tenant_id, 'tenant_id')}"
        f"/knowledge-bases/{_required_component(knowledge_base_id, 'knowledge_base_id')}"
        f"/uploads/{_required_component(upload_session_id, 'upload_session_id')}"
        f"/original{suffix}"
    )


def extracted_object_key(
    *,
    tenant_id: str,
    knowledge_base_id: str,
    document_id: str,
    version_id: str,
    artifact_name: str,
) -> str:
    safe_name = _required_component(artifact_name, "artifact_name", max_length=160)
    return _version_prefix(
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        version_id=version_id,
    ) + f"/extracted/{safe_name}"


def index_manifest_object_key(
    *,
    tenant_id: str,
    knowledge_base_id: str,
    index_version_id: str,
) -> str:
    return (
        f"tenants/{_required_component(tenant_id, 'tenant_id')}"
        f"/knowledge-bases/{_required_component(knowledge_base_id, 'knowledge_base_id')}"
        f"/indexes/{_required_component(index_version_id, 'index_version_id')}"
        "/manifest.json"
    )


def _version_prefix(
    *,
    tenant_id: str,
    knowledge_base_id: str,
    document_id: str,
    version_id: str,
) -> str:
    values = {
        "tenant_id": tenant_id,
        "knowledge_base_id": knowledge_base_id,
        "document_id": document_id,
        "version_id": version_id,
    }
    clean = {name: _required_component(value, name) for name, value in values.items()}
    return (
        f"tenants/{clean['tenant_id']}/knowledge-bases/{clean['knowledge_base_id']}"
        f"/documents/{clean['document_id']}/versions/{clean['version_id']}"
    )


def _required_component(value: str, name: str, max_length: int = 128) -> str:
    clean = str(value or "").strip()
    if len(clean) > max_length or not _ID_PATTERN.fullmatch(clean):
        raise ValueError(f"{name} contains invalid characters")
    return clean


def _required_object_key(value: str) -> str:
    clean = str(value or "").strip()
    if (
        not clean.startswith("tenants/")
        or clean.startswith("/")
        or "//" in clean
        or any(part in {"", ".", ".."} for part in clean.split("/"))
    ):
        raise ValueError("object key is invalid")
    return clean
