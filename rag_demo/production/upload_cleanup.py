from __future__ import annotations


class UploadCleanupService:
    def __init__(self, repository, object_storage):
        self.repository = repository
        self.object_storage = object_storage

    def sweep(self, *, limit: int = 100, grace_seconds: int = 300) -> dict:
        candidates = self.repository.expired_upload_cleanup_candidates(
            limit=limit,
            grace_seconds=grace_seconds,
        )
        cleaned = []
        failed = []
        for candidate in candidates:
            try:
                self.object_storage.delete(candidate["object_key"])
                self.repository.mark_upload_cleanup_complete(
                    upload_id=candidate["upload_id"],
                    object_key=candidate["object_key"],
                )
                cleaned.append(candidate["upload_id"])
            except Exception as exc:
                failed.append({
                    "upload_id": candidate["upload_id"],
                    "error_class": exc.__class__.__name__,
                })
        return {"cleaned": cleaned, "failed": failed}
