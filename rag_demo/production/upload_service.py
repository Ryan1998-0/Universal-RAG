from __future__ import annotations

import re
from datetime import timedelta
from pathlib import Path
from typing import BinaryIO, Optional

from rag_demo.document_pipeline import SUPPORTED_FORMATS
from rag_demo.production.database import new_id, utc_now
from rag_demo.production.object_storage import upload_object_key


_IDEMPOTENCY_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{8,128}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class UploadValidationError(ValueError):
    pass


class UploadService:
    def __init__(self, repository, object_storage, settings, dispatcher=None):
        self.repository = repository
        self.object_storage = object_storage
        self.settings = settings
        self.dispatcher = dispatcher

    def reserve(
        self,
        *,
        principal,
        authorized,
        idempotency_key: str,
        filename: str,
        content_type: str,
        size_bytes: int,
        sha256: str,
        folder_id: Optional[str] = None,
        target_document_id: Optional[str] = None,
    ) -> dict:
        clean_idempotency = str(idempotency_key or "").strip()
        if not _IDEMPOTENCY_PATTERN.fullmatch(clean_idempotency):
            raise UploadValidationError(
                "Idempotency-Key must contain 8 to 128 safe characters"
            )
        clean_filename = Path(str(filename or "").strip()).name
        extension = Path(clean_filename).suffix.lower()
        if not clean_filename or extension not in SUPPORTED_FORMATS:
            raise UploadValidationError("file format is not supported")
        clean_content_type = str(content_type or "").strip().lower()
        expected_mime = SUPPORTED_FORMATS[extension]["mime_type"]
        if clean_content_type and clean_content_type != expected_mime:
            raise UploadValidationError("declared MIME type does not match the extension")
        clean_content_type = expected_mime
        clean_sha256 = str(sha256 or "").strip().lower()
        if not _SHA256_PATTERN.fullmatch(clean_sha256):
            raise UploadValidationError("sha256 must contain exactly 64 lowercase hex characters")
        clean_size = int(size_bytes)
        if clean_size <= 0 or clean_size > self.settings.max_upload_bytes:
            raise UploadValidationError("file size is outside the configured upload limit")

        upload_id = new_id()
        object_key = upload_object_key(
            tenant_id=principal.tenant_id,
            knowledge_base_id=authorized.id,
            upload_session_id=upload_id,
            filename=clean_filename,
        )
        reservation = self.repository.reserve_upload(
            principal=principal,
            authorized=authorized,
            upload_id=upload_id,
            idempotency_key=clean_idempotency,
            filename=clean_filename,
            declared_mime_type=clean_content_type,
            expected_size_bytes=clean_size,
            expected_sha256=clean_sha256,
            object_key=object_key,
            expires_at=utc_now() + timedelta(
                seconds=self.settings.upload_session_ttl_seconds
            ),
            folder_id=folder_id,
            target_document_id=target_document_id,
        )
        if self.settings.upload_mode == "presigned":
            upload_target = {
                "mode": "presigned",
                **self.object_storage.presign_upload(
                    key=reservation.object_key,
                    content_type=reservation.declared_mime_type,
                    expected_sha256=reservation.expected_sha256,
                    expires_seconds=self.settings.upload_session_ttl_seconds,
                ),
            }
        else:
            upload_target = {
                "mode": "proxy",
                "url": f"/v1/uploads/{reservation.id}/content",
                "method": "PUT",
                "headers": {"Content-Type": reservation.declared_mime_type},
                "expires_in": self.settings.upload_session_ttl_seconds,
            }
        return {
            "upload_id": reservation.id,
            "state": reservation.state,
            "upload": upload_target,
            "expires_at": reservation.expires_at.isoformat(),
        }

    def get_reservation(self, *, principal, upload_id: str):
        return self.repository.get_upload_reservation(
            principal=principal,
            upload_id=upload_id,
        )

    def store_proxy_content(
        self,
        *,
        principal,
        reservation,
        body: BinaryIO,
        size_bytes: int,
        sha256: str,
        content_type: str,
    ) -> dict:
        clean_type = str(content_type or "").split(";", 1)[0].strip().lower()
        if (
            int(size_bytes) != int(reservation.expected_size_bytes)
            or str(sha256 or "").lower() != reservation.expected_sha256
            or clean_type != reservation.declared_mime_type.lower()
        ):
            raise UploadValidationError(
                "uploaded content does not match the reserved size, hash, or MIME type"
            )
        stored = self.object_storage.put_reserved_upload(
            key=reservation.object_key,
            body=body,
            content_type=clean_type,
            size_bytes=size_bytes,
            sha256=sha256,
        )
        self.repository.mark_upload_stored(
            principal=principal,
            upload_id=reservation.id,
            observed_size_bytes=stored.size_bytes,
            observed_sha256=stored.sha256,
            observed_mime_type=stored.content_type,
        )
        return {
            "upload_id": reservation.id,
            "state": "uploaded",
            "size_bytes": stored.size_bytes,
            "sha256": stored.sha256,
        }

    def complete(self, *, principal, upload_id: str) -> dict:
        object_key = self.repository.get_upload_object_key(
            principal=principal,
            upload_id=upload_id,
        )
        object_info = self.object_storage.head(object_key)
        completed = self.repository.complete_upload(
            principal=principal,
            upload_id=upload_id,
            observed_size_bytes=object_info.size_bytes,
            observed_sha256=object_info.sha256,
            observed_mime_type=object_info.content_type,
            parser_version=self.settings.parser_version,
            chunk_schema_version=self.settings.chunk_schema_version,
            embedding_model=self.settings.embedding_model,
        )
        dispatch_state = "already_queued" if completed.duplicate else "pending"
        if self.dispatcher is not None and not completed.duplicate:
            try:
                task_id = self.dispatcher.dispatch_ingestion(completed.ingestion_job_id)
                self.repository.record_ingestion_dispatch(
                    job_id=completed.ingestion_job_id,
                    task_id=task_id,
                )
                dispatch_state = "queued"
            except Exception:
                # The durable queued row remains available for a scheduler sweep.
                dispatch_state = "pending"
        return {
            "upload_id": completed.upload_id,
            "document_id": completed.document_id,
            "document_version_id": completed.document_version_id,
            "ingestion_job_id": completed.ingestion_job_id,
            "duplicate": completed.duplicate,
            "dispatch_state": dispatch_state,
        }
