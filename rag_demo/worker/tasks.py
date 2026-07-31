from __future__ import annotations

import socket
from dataclasses import dataclass
from functools import lru_cache

from rag_demo.production.config import get_production_settings
from rag_demo.production.background_health import RedisBackgroundHealth
from rag_demo.production.database import create_database_engine, create_session_factory
from rag_demo.production.deletion import DeletionService, SqlAlchemyDeletionRepository
from rag_demo.production.file_security import ClamAvScanner
from rag_demo.production.ingestion import IngestionService, SqlAlchemyIngestionRepository
from rag_demo.production.embedding_runtime import FastEmbedRuntime
from rag_demo.production.indexing import IndexingService, SqlAlchemyIndexingRepository
from rag_demo.production.object_storage import S3ObjectStorage
from rag_demo.production.repository import SqlAlchemyTenantRepository
from rag_demo.production.task_dispatch import CeleryTaskDispatcher
from rag_demo.production.upload_cleanup import UploadCleanupService
from rag_demo.production.vector_repository import QdrantChunkRepository
from rag_demo.worker.celery_app import celery_app


@dataclass(frozen=True)
class WorkerRuntime:
    ingestion_repository: SqlAlchemyIngestionRepository
    tenant_repository: SqlAlchemyTenantRepository
    ingestion_service: IngestionService
    indexing_repository: SqlAlchemyIndexingRepository
    indexing_service: IndexingService
    deletion_repository: SqlAlchemyDeletionRepository
    deletion_service: DeletionService
    dispatcher: CeleryTaskDispatcher
    upload_cleanup_service: UploadCleanupService


@lru_cache(maxsize=1)
def worker_runtime() -> WorkerRuntime:
    settings = get_production_settings()
    engine = create_database_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    storage = S3ObjectStorage.from_settings(settings)
    ingestion_repository = SqlAlchemyIngestionRepository(session_factory)
    tenant_repository = SqlAlchemyTenantRepository(session_factory)
    indexing_repository = SqlAlchemyIndexingRepository(session_factory)
    deletion_repository = SqlAlchemyDeletionRepository(session_factory)
    vector_repository = QdrantChunkRepository.from_settings(settings)
    embedding_runtime = FastEmbedRuntime.from_settings(settings)
    return WorkerRuntime(
        ingestion_repository=ingestion_repository,
        tenant_repository=tenant_repository,
        ingestion_service=IngestionService(
            repository=ingestion_repository,
            object_storage=storage,
            malware_scanner=ClamAvScanner(
                host=settings.clamav_host or "",
                port=settings.clamav_port,
            ),
            max_upload_bytes=settings.max_upload_bytes,
            lease_seconds=settings.ingestion_lease_seconds,
            heartbeat_interval_seconds=(
                settings.ingestion_heartbeat_interval_seconds
            ),
            prompt_injection_policy=settings.prompt_injection_policy,
        ),
        indexing_repository=indexing_repository,
        indexing_service=IndexingService(
            repository=indexing_repository,
            tenant_repository=tenant_repository,
            object_storage=storage,
            vector_repository=vector_repository,
            embedding_runtime=embedding_runtime,
            batch_size=settings.index_batch_size,
            max_canonical_bytes=settings.max_upload_bytes * 4,
        ),
        deletion_repository=deletion_repository,
        deletion_service=DeletionService(
            repository=deletion_repository,
            object_storage=storage,
            vector_repository=vector_repository,
        ),
        dispatcher=CeleryTaskDispatcher(celery_app),
        upload_cleanup_service=UploadCleanupService(
            repository=tenant_repository,
            object_storage=storage,
        ),
    )


@lru_cache(maxsize=1)
def background_health_runtime() -> RedisBackgroundHealth:
    return RedisBackgroundHealth.from_settings(get_production_settings())


@celery_app.task(bind=True, name="rag_demo.record_pipeline_heartbeat")
def record_pipeline_heartbeat(self) -> dict:
    worker_id = f"{socket.gethostname()}:{self.request.id}"
    background_health_runtime().mark_alive(worker_id=worker_id)
    return {"status": "alive"}


@celery_app.task(name="rag_demo.sweep_expired_uploads")
def sweep_expired_uploads(limit: int = 100) -> dict:
    return worker_runtime().upload_cleanup_service.sweep(limit=limit)


@celery_app.task(
    bind=True,
    name="rag_demo.process_ingestion",
    acks_late=True,
    reject_on_worker_lost=True,
)
def process_ingestion(self, job_id: str) -> dict:
    worker_id = f"{socket.gethostname()}:{self.request.id}"
    return worker_runtime().ingestion_service.process(
        job_id=str(job_id),
        worker_id=worker_id,
    )


@celery_app.task(
    bind=True,
    name="rag_demo.process_index_build",
    acks_late=True,
    reject_on_worker_lost=True,
)
def process_index_build(self, job_id: str) -> dict:
    worker_id = f"{socket.gethostname()}:{self.request.id}"
    return worker_runtime().indexing_service.process(
        job_id=str(job_id),
        worker_id=worker_id,
    )


@celery_app.task(
    bind=True,
    name="rag_demo.process_deletion_outbox",
    acks_late=True,
    reject_on_worker_lost=True,
)
def process_deletion_outbox(self, outbox_id: str) -> dict:
    worker_id = f"{socket.gethostname()}:{self.request.id}"
    return worker_runtime().deletion_service.process(
        outbox_id=str(outbox_id),
        worker_id=worker_id,
    )


@celery_app.task(name="rag_demo.sweep_ingestion_jobs")
def sweep_ingestion_jobs(limit: int = 100) -> dict:
    runtime = worker_runtime()
    dispatched = []
    failed = []
    for job_id in runtime.ingestion_repository.dispatchable_job_ids(limit=limit):
        try:
            task_id = runtime.dispatcher.dispatch_ingestion(job_id)
            runtime.tenant_repository.record_ingestion_dispatch(
                job_id=job_id,
                task_id=task_id,
            )
            dispatched.append(job_id)
        except Exception as exc:
            failed.append({"job_id": job_id, "error_class": exc.__class__.__name__})
    return {"dispatched": dispatched, "failed": failed}


@celery_app.task(name="rag_demo.sweep_index_build_jobs")
def sweep_index_build_jobs(limit: int = 50) -> dict:
    runtime = worker_runtime()
    dispatched = []
    failed = []
    for job_id in runtime.indexing_repository.dispatchable_job_ids(limit=limit):
        try:
            task_id = runtime.dispatcher.dispatch_index_build(job_id)
            runtime.indexing_repository.record_dispatch(
                job_id=job_id,
                task_id=task_id,
            )
            dispatched.append(job_id)
        except Exception as exc:
            failed.append({"job_id": job_id, "error_class": exc.__class__.__name__})
    return {"dispatched": dispatched, "failed": failed}


@celery_app.task(name="rag_demo.sweep_deletion_outbox")
def sweep_deletion_outbox(limit: int = 100) -> dict:
    runtime = worker_runtime()
    dispatched = []
    failed = []
    for outbox_id in runtime.deletion_repository.dispatchable_ids(limit=limit):
        try:
            task_id = runtime.dispatcher.dispatch_cleanup(outbox_id)
            runtime.deletion_repository.record_dispatch(
                outbox_id=outbox_id,
                task_id=task_id,
            )
            dispatched.append(outbox_id)
        except Exception as exc:
            failed.append({"outbox_id": outbox_id, "error_class": exc.__class__.__name__})
    return {"dispatched": dispatched, "failed": failed}
