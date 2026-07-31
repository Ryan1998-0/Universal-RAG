from celery import Celery

from rag_demo.production.config import get_production_settings


def build_celery_app(settings=None) -> Celery:
    resolved = settings or get_production_settings()
    app = Celery(
        "universal-rag-worker",
        broker=resolved.redis_url,
        backend=resolved.redis_url,
        include=["rag_demo.worker.tasks"],
    )
    app.conf.update(
        broker_connection_retry_on_startup=True,
        enable_utc=True,
        timezone="UTC",
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        task_track_started=True,
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        result_expires=3600,
        worker_prefetch_multiplier=1,
        task_soft_time_limit=1100,
        task_time_limit=1200,
        beat_schedule={
            "sweep-durable-ingestion-jobs": {
                "task": "rag_demo.sweep_ingestion_jobs",
                "schedule": 30.0,
            },
            "sweep-durable-deletion-outbox": {
                "task": "rag_demo.sweep_deletion_outbox",
                "schedule": 30.0,
            },
            "sweep-durable-index-build-jobs": {
                "task": "rag_demo.sweep_index_build_jobs",
                "schedule": 30.0,
            },
            "verify-beat-to-worker-path": {
                "task": "rag_demo.record_pipeline_heartbeat",
                "schedule": 15.0,
            },
            "sweep-expired-upload-objects": {
                "task": "rag_demo.sweep_expired_uploads",
                "schedule": 60.0,
            },
        },
    )
    return app


celery_app = build_celery_app()
