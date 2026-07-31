from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

from sqlalchemy import and_, or_, select

from rag_demo.document_pipeline import (
    DocumentPipelineError,
    build_document_chunks,
    extract_document,
)
from rag_demo.production.database import (
    DocumentRecord,
    DocumentVersionRecord,
    IngestionJobRecord,
    KnowledgeBaseRecord,
    utc_now,
)
from rag_demo.production.file_security import (
    FileSecurityError,
    inspect_file,
    scan_prompt_injection,
)
from rag_demo.production.lease_heartbeat import LeaseHeartbeat


@dataclass(frozen=True)
class IngestionWorkItem:
    job_id: str
    tenant_id: str
    knowledge_base_id: str
    document_id: str
    document_version_id: str
    source_id: str
    filename: str
    object_key: str
    sha256: str
    mime_type: str
    attempt: int
    max_attempts: int


class SqlAlchemyIngestionRepository:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def claim(
        self,
        *,
        job_id: str,
        worker_id: str,
        lease_seconds: int = 600,
    ) -> Optional[IngestionWorkItem]:
        now = utc_now()
        with self.session_factory.begin() as session:
            row = session.execute(
                select(
                    IngestionJobRecord,
                    DocumentVersionRecord,
                    DocumentRecord,
                    KnowledgeBaseRecord,
                )
                .join(
                    DocumentVersionRecord,
                    DocumentVersionRecord.id == IngestionJobRecord.document_version_id,
                )
                .join(DocumentRecord, DocumentRecord.id == DocumentVersionRecord.document_id)
                .join(
                    KnowledgeBaseRecord,
                    KnowledgeBaseRecord.id == DocumentRecord.knowledge_base_id,
                )
                .where(IngestionJobRecord.id == job_id)
                .with_for_update()
            ).one_or_none()
            if row is None:
                return None
            job, version, document, knowledge_base = row
            if not (
                job.tenant_id
                == version.tenant_id
                == document.tenant_id
                == knowledge_base.tenant_id
            ):
                raise RuntimeError("ingestion job scope is inconsistent")
            if job.status in {"succeeded", "dead", "cancelled"}:
                return None
            if (
                job.status == "running"
                and job.lease_expires_at is not None
                and _as_utc(job.lease_expires_at) > now
                and job.lease_owner != worker_id
            ):
                return None
            if (
                job.status == "retry_wait"
                and job.next_attempt_at is not None
                and _as_utc(job.next_attempt_at) > now
            ):
                return None

            job.status = "running"
            job.stage = "scanning"
            job.attempt += 1
            job.lease_owner = worker_id
            job.lease_expires_at = now + timedelta(seconds=max(60, int(lease_seconds)))
            job.heartbeat_at = now
            job.started_at = job.started_at or now
            job.error_code = ""
            job.error_detail = ""
            job.error_class = ""
            version.status = "scanning"
            return IngestionWorkItem(
                job_id=job.id,
                tenant_id=job.tenant_id,
                knowledge_base_id=document.knowledge_base_id,
                document_id=document.id,
                document_version_id=version.id,
                source_id=document.source_id,
                filename=document.name,
                object_key=version.object_key,
                sha256=version.sha256,
                mime_type=version.mime_type,
                attempt=job.attempt,
                max_attempts=job.max_attempts,
            )

    def dispatchable_job_ids(
        self,
        *,
        limit: int = 100,
        redispatch_after_seconds: int = 90,
    ) -> list[str]:
        now = utc_now()
        stale_dispatch = now - timedelta(
            seconds=max(30, int(redispatch_after_seconds))
        )
        with self.session_factory() as session:
            return list(session.scalars(
                select(IngestionJobRecord.id)
                .where(or_(
                    and_(
                        IngestionJobRecord.status == "queued",
                        or_(
                            IngestionJobRecord.last_dispatched_at.is_(None),
                            IngestionJobRecord.last_dispatched_at <= stale_dispatch,
                        ),
                    ),
                    and_(
                        IngestionJobRecord.status == "retry_wait",
                        or_(
                            IngestionJobRecord.next_attempt_at.is_(None),
                            IngestionJobRecord.next_attempt_at <= now,
                        ),
                    ),
                    and_(
                        IngestionJobRecord.status == "running",
                        IngestionJobRecord.lease_expires_at.is_not(None),
                        IngestionJobRecord.lease_expires_at <= now,
                    ),
                ))
                .order_by(IngestionJobRecord.created_at, IngestionJobRecord.id)
                .limit(max(1, min(int(limit), 1000)))
            ))

    def heartbeat(
        self,
        *,
        job_id: str,
        worker_id: str,
        stage: str,
        lease_seconds: int = 600,
    ) -> None:
        now = utc_now()
        with self.session_factory.begin() as session:
            job = session.scalar(
                select(IngestionJobRecord)
                .where(
                    IngestionJobRecord.id == job_id,
                    IngestionJobRecord.status == "running",
                    IngestionJobRecord.lease_owner == worker_id,
                )
                .with_for_update()
            )
            if job is None:
                raise RuntimeError("ingestion lease was lost")
            version = session.get(DocumentVersionRecord, job.document_version_id)
            job.stage = str(stage)[:40]
            job.heartbeat_at = now
            job.lease_expires_at = now + timedelta(
                seconds=max(60, int(lease_seconds))
            )
            if version is not None:
                version.status = str(stage)[:30]

    def complete(
        self,
        *,
        job_id: str,
        worker_id: str,
        extracted_object_key: str,
        page_count: int,
        metadata: dict,
    ) -> None:
        now = utc_now()
        with self.session_factory.begin() as session:
            job = session.scalar(
                select(IngestionJobRecord)
                .where(
                    IngestionJobRecord.id == job_id,
                    IngestionJobRecord.status == "running",
                    IngestionJobRecord.lease_owner == worker_id,
                )
                .with_for_update()
            )
            if job is None:
                raise RuntimeError("ingestion lease was lost")
            version = session.get(DocumentVersionRecord, job.document_version_id)
            document = session.get(DocumentRecord, version.document_id) if version else None
            if version is None or document is None:
                raise RuntimeError("ingestion document disappeared")
            version.extracted_object_key = extracted_object_key
            version.page_count = int(page_count)
            version.metadata_json = dict(metadata)
            version.status = "parsed"
            document.status = "parsed"
            job.status = "succeeded"
            job.stage = "ready_for_index"
            job.completed_at = now
            job.heartbeat_at = now
            job.lease_owner = ""
            job.lease_expires_at = None
            job.next_attempt_at = None

    def fail(
        self,
        *,
        job_id: str,
        worker_id: str,
        error_code: str,
        error_class: str,
        error_detail: str,
        permanent: bool,
    ) -> str:
        now = utc_now()
        with self.session_factory.begin() as session:
            job = session.scalar(
                select(IngestionJobRecord)
                .where(
                    IngestionJobRecord.id == job_id,
                    IngestionJobRecord.lease_owner == worker_id,
                )
                .with_for_update()
            )
            if job is None:
                return "lease_lost"
            terminal = permanent or job.attempt >= job.max_attempts
            job.status = "dead" if terminal else "retry_wait"
            job.stage = "failed"
            job.error_code = str(error_code)[:80]
            job.error_class = str(error_class)[:120]
            job.error_detail = _safe_error_detail(error_detail)
            job.lease_owner = ""
            job.lease_expires_at = None
            job.heartbeat_at = now
            if terminal:
                job.completed_at = now
                job.next_attempt_at = None
                version = session.get(DocumentVersionRecord, job.document_version_id)
                if version is not None:
                    version.status = "failed"
                    document = session.get(DocumentRecord, version.document_id)
                    if document is not None and document.current_version_id is None:
                        document.status = "failed"
            else:
                delay_seconds = min(3600, 30 * (2 ** max(0, job.attempt - 1)))
                job.next_attempt_at = now + timedelta(seconds=delay_seconds)
            return job.status

    def quarantine(
        self,
        *,
        job_id: str,
        worker_id: str,
        findings: list[str],
    ) -> str:
        now = utc_now()
        safe_findings = sorted({str(item)[:80] for item in findings if str(item)})
        with self.session_factory.begin() as session:
            job = session.scalar(
                select(IngestionJobRecord)
                .where(
                    IngestionJobRecord.id == job_id,
                    IngestionJobRecord.lease_owner == worker_id,
                )
                .with_for_update()
            )
            if job is None:
                return "lease_lost"
            job.status = "dead"
            job.stage = "quarantined"
            job.error_code = "PROMPT_INJECTION_DETECTED"
            job.error_class = "PromptInjectionDetectedError"
            job.error_detail = "Detected document instruction patterns: " + ", ".join(
                safe_findings
            )
            job.lease_owner = ""
            job.lease_expires_at = None
            job.heartbeat_at = now
            job.completed_at = now
            job.next_attempt_at = None
            version = session.get(DocumentVersionRecord, job.document_version_id)
            if version is not None:
                version.status = "quarantined"
                version.metadata_json = {
                    **dict(version.metadata_json or {}),
                    "security": {
                        "quarantined": True,
                        "prompt_injection_findings": safe_findings,
                    },
                }
                document = session.get(DocumentRecord, version.document_id)
                if document is not None and document.current_version_id is None:
                    document.status = "quarantined"
            return "quarantined"


class IngestionService:
    def __init__(
        self,
        repository,
        object_storage,
        malware_scanner,
        max_upload_bytes: int,
        lease_seconds: int = 600,
        heartbeat_interval_seconds: float = 60,
        prompt_injection_policy: str = "quarantine",
    ):
        self.repository = repository
        self.object_storage = object_storage
        self.malware_scanner = malware_scanner
        self.max_upload_bytes = int(max_upload_bytes)
        self.lease_seconds = max(60, int(lease_seconds))
        self.heartbeat_interval_seconds = max(
            0.01,
            float(heartbeat_interval_seconds),
        )
        policy = str(prompt_injection_policy or "").strip().lower()
        if policy not in {"quarantine", "flag"}:
            raise ValueError("prompt_injection_policy must be quarantine or flag")
        self.prompt_injection_policy = policy

    def process(self, *, job_id: str, worker_id: str) -> dict:
        work = self.repository.claim(
            job_id=job_id,
            worker_id=worker_id,
            lease_seconds=self.lease_seconds,
        )
        if work is None:
            return {"job_id": job_id, "status": "not_claimed"}
        heartbeat = LeaseHeartbeat(
            lambda stage: self.repository.heartbeat(
                job_id=work.job_id,
                worker_id=worker_id,
                stage=stage,
                lease_seconds=self.lease_seconds,
            ),
            initial_stage="scanning",
            interval_seconds=self.heartbeat_interval_seconds,
        )
        heartbeat.start()
        try:
            payload = self.object_storage.download_bytes(
                work.object_key,
                max_bytes=self.max_upload_bytes,
            )
            if hashlib.sha256(payload).hexdigest() != work.sha256:
                raise FileSecurityError("uploaded object SHA-256 does not match")
            self.malware_scanner.scan(payload)
            inspection = inspect_file(
                filename=work.filename,
                payload=payload,
                declared_mime_type=work.mime_type,
            )

            heartbeat.set_stage("parsing")
            self.repository.heartbeat(
                job_id=work.job_id,
                worker_id=worker_id,
                stage="parsing",
                lease_seconds=self.lease_seconds,
            )
            extraction = extract_document(
                filename=work.filename,
                payload=payload,
                extension=inspection.extension,
            )
            heartbeat.raise_if_failed()
            heartbeat.set_stage("chunking")
            self.repository.heartbeat(
                job_id=work.job_id,
                worker_id=worker_id,
                stage="chunking",
                lease_seconds=self.lease_seconds,
            )
            chunks = build_document_chunks(
                extraction.units,
                source_id=work.source_id,
                filename=work.filename,
                source_type=extraction.source_type,
                extraction_method=extraction.method,
            )
            if not chunks:
                raise DocumentPipelineError("no indexable text was extracted")
            extracted_text = "\n".join(
                str(unit.get("content") or "") for unit in extraction.units
            )
            injection_findings = sorted(set(
                inspection.prompt_injection_findings
            ).union(scan_prompt_injection(extracted_text)))
            if injection_findings and self.prompt_injection_policy == "quarantine":
                raise PromptInjectionDetectedError(injection_findings)
            canonical = {
                "schema_version": "canonical-document-v1",
                "tenant_id": work.tenant_id,
                "knowledge_base_id": work.knowledge_base_id,
                "document_id": work.document_id,
                "document_version_id": work.document_version_id,
                "source_id": work.source_id,
                "filename": work.filename,
                "mime_type": inspection.detected_mime_type,
                "extraction": {
                    "method": extraction.method,
                    "source_type": extraction.source_type,
                    "ocr_used": extraction.ocr_used,
                    "warnings": list(extraction.warnings),
                },
                "prompt_injection_findings": injection_findings,
                "units": extraction.units,
                "chunks": chunks,
            }
            artifact = self.object_storage.put_extracted_artifact(
                tenant_id=work.tenant_id,
                knowledge_base_id=work.knowledge_base_id,
                document_id=work.document_id,
                version_id=work.document_version_id,
                body=json.dumps(canonical, ensure_ascii=False).encode("utf-8"),
            )
            heartbeat.stop()
            page_count = len({
                str(unit.get("page"))
                for unit in extraction.units
                if str(unit.get("page") or "").strip()
            })
            self.repository.complete(
                job_id=work.job_id,
                worker_id=worker_id,
                extracted_object_key=artifact.key,
                page_count=page_count,
                metadata={
                    "chunk_count": len(chunks),
                    "source_type": extraction.source_type,
                    "extraction_method": extraction.method,
                    "ocr_used": extraction.ocr_used,
                    "prompt_injection_findings": injection_findings,
                    "canonical_sha256": artifact.sha256,
                },
            )
            return {
                "job_id": work.job_id,
                "status": "succeeded",
                "chunk_count": len(chunks),
            }
        except PromptInjectionDetectedError as exc:
            heartbeat.cancel()
            status = self.repository.quarantine(
                job_id=work.job_id,
                worker_id=worker_id,
                findings=exc.findings,
            )
            return {"job_id": work.job_id, "status": status}
        except (FileSecurityError, DocumentPipelineError) as exc:
            heartbeat.cancel()
            status = self.repository.fail(
                job_id=work.job_id,
                worker_id=worker_id,
                error_code="DOCUMENT_REJECTED",
                error_class=exc.__class__.__name__,
                error_detail=str(exc),
                permanent=True,
            )
            return {"job_id": work.job_id, "status": status}
        except Exception as exc:
            heartbeat.cancel()
            status = self.repository.fail(
                job_id=work.job_id,
                worker_id=worker_id,
                error_code="INGESTION_DEPENDENCY_FAILED",
                error_class=exc.__class__.__name__,
                error_detail=str(exc),
                permanent=False,
            )
            return {"job_id": work.job_id, "status": status}
        finally:
            heartbeat.cancel()


def _safe_error_detail(value: str) -> str:
    clean = " ".join(str(value or "").split())
    clean = clean.replace("/Users/", "/redacted/").replace("/app/", "/redacted/")
    return clean[:1000]


class PromptInjectionDetectedError(FileSecurityError):
    def __init__(self, findings: list[str]):
        self.findings = list(findings)
        super().__init__("document contains instruction-like prompt injection patterns")


def _as_utc(value):
    if value.tzinfo is None:
        return value.replace(tzinfo=utc_now().tzinfo)
    return value
