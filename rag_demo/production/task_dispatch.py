from uuid import uuid4


class CeleryTaskDispatcher:
    def __init__(self, celery_app):
        self.celery_app = celery_app

    @classmethod
    def from_settings(cls, settings) -> "CeleryTaskDispatcher":
        try:
            from celery import Celery
        except ImportError as exc:
            raise RuntimeError("celery is required for background ingestion") from exc
        app = Celery(
            "ifrs17-rag-dispatcher",
            broker=settings.redis_url,
            backend=settings.redis_url,
        )
        return cls(app)

    def dispatch_ingestion(self, job_id: str) -> str:
        task_id = f"ingestion-{job_id}-{uuid4().hex[:12]}"
        self.celery_app.send_task(
            "rag_demo.process_ingestion",
            args=[job_id],
            task_id=task_id,
        )
        return task_id

    def dispatch_cleanup(self, outbox_id: str) -> str:
        task_id = f"cleanup-{outbox_id}-{uuid4().hex[:12]}"
        self.celery_app.send_task(
            "rag_demo.process_deletion_outbox",
            args=[outbox_id],
            task_id=task_id,
        )
        return task_id

    def dispatch_index_build(self, job_id: str) -> str:
        task_id = f"index-build-{job_id}-{uuid4().hex[:12]}"
        self.celery_app.send_task(
            "rag_demo.process_index_build",
            args=[job_id],
            task_id=task_id,
        )
        return task_id
