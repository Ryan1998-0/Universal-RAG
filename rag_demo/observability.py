"""Small, dependency-free observability helpers for the RAG pipeline.

The application already returns timing information to API callers.  This
module makes that information consistent across API requests, pipeline nodes,
workers and model adapters, while writing safe JSON Lines logs for diagnosis.
It intentionally never logs prompts, document content, answers, credentials or
tokens.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any, Mapping, Optional


LOGGER = logging.getLogger("rag_demo")
_CONFIG_LOCK = RLock()
_FILE_HANDLER_MARKER = "rag_demo_json_file_handler"
_SENSITIVE_KEY = re.compile(
    r"(?:authorization|cookie|password|secret|token|api[_-]?key|credential|prompt|content|answer|document_text|message|exception|question(?:$|[_-])|query(?:$|[_-])|text(?:$|[_-]))",
    re.IGNORECASE,
)


def _safe_value(key: str, value: Any) -> Any:
    """Return a log-safe representation without leaking user or document data."""

    if _SENSITIVE_KEY.search(str(key)):
        if value is None:
            return None
        try:
            length = len(value)
        except TypeError:
            length = len(str(value))
        return {"redacted": True, "length": length}
    if isinstance(value, Mapping):
        return {str(k): _safe_value(str(k), v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_value(key, item) for item in value[:100]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)[:500]


class JsonLogFormatter(logging.Formatter):
    """Format both ordinary log messages and structured event JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any]
        message = record.getMessage()
        try:
            parsed = json.loads(message)
            payload = dict(parsed) if isinstance(parsed, dict) else {"message": message}
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = {"message": message}
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        payload.setdefault("level", record.levelname)
        payload.setdefault("logger", record.name)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(_safe_value("event", payload), ensure_ascii=False, sort_keys=True)


def configure_logging(
    *,
    log_dir: Optional[str | Path] = None,
    level: Optional[str] = None,
    filename: Optional[str] = None,
    max_bytes: Optional[int] = None,
    backup_count: Optional[int] = None,
) -> Path:
    """Configure one rotating JSONL file handler and a stderr handler.

    Calling this repeatedly is safe.  Environment variables are intentionally
    used as the primary interface so the API, worker and local process can use
    the same settings without importing the production settings model.
    """

    resolved_dir = Path(
        log_dir
        or os.getenv("RAG_LOG_DIR", "logs")
    ).expanduser()
    if not resolved_dir.is_absolute():
        resolved_dir = Path.cwd() / resolved_dir
    resolved_dir.mkdir(parents=True, exist_ok=True)
    resolved_filename = str(filename or os.getenv("RAG_LOG_FILE", "rag.log")).strip()
    if not resolved_filename or Path(resolved_filename).name != resolved_filename:
        resolved_filename = "rag.log"
    log_path = resolved_dir / resolved_filename
    numeric_level = str(level or os.getenv("RAG_LOG_LEVEL", "INFO")).upper()
    log_level = getattr(logging, numeric_level, logging.INFO)
    rotation_size = max(
        1024,
        int(max_bytes or os.getenv("RAG_LOG_MAX_BYTES", str(20 * 1024 * 1024))),
    )
    rotations = max(
        1,
        int(backup_count or os.getenv("RAG_LOG_BACKUP_COUNT", "5")),
    )

    with _CONFIG_LOCK:
        LOGGER.setLevel(log_level)
        LOGGER.propagate = False
        existing_file = next(
            (
                handler
                for handler in LOGGER.handlers
                if getattr(handler, _FILE_HANDLER_MARKER, False)
            ),
            None,
        )
        if existing_file is None or Path(getattr(existing_file, "baseFilename", "")) != log_path:
            if existing_file is not None:
                LOGGER.removeHandler(existing_file)
                existing_file.close()
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=rotation_size,
                backupCount=rotations,
                encoding="utf-8",
            )
            setattr(file_handler, _FILE_HANDLER_MARKER, True)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(JsonLogFormatter())
            LOGGER.addHandler(file_handler)
        else:
            existing_file.setLevel(log_level)

        if not any(getattr(handler, "_rag_demo_console_handler", False) for handler in LOGGER.handlers):
            console_handler = logging.StreamHandler(sys.stderr)
            setattr(console_handler, "_rag_demo_console_handler", True)
            console_handler.setLevel(log_level)
            console_handler.setFormatter(JsonLogFormatter())
            LOGGER.addHandler(console_handler)
    return log_path


def log_event(event: str, **fields: Any) -> None:
    """Write a sanitized structured event to the configured RAG logger."""

    LOGGER.info(
        json.dumps(
            {
                "event": str(event),
                **{str(key): _safe_value(str(key), value) for key, value in fields.items()},
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


@dataclass
class TimingTrace:
    """Thread-safe collection of node timings for one operation."""

    run_id: str = ""
    request_id: str = ""
    component: str = "rag"
    stages: list[dict[str, Any]] = field(default_factory=list)
    _lock: RLock = field(default_factory=RLock, repr=False)

    def stage(self, name: str, **metadata: Any) -> "StageTimer":
        return StageTimer(self, name, metadata)

    def record(
        self,
        name: str,
        duration_ms: float,
        *,
        status: str = "completed",
        **metadata: Any,
    ) -> dict[str, Any]:
        entry = {
            "name": str(name),
            "durationMs": round(max(0.0, float(duration_ms)), 3),
            "status": str(status),
            **{str(key): value for key, value in metadata.items() if value is not None},
        }
        with self._lock:
            self.stages.append(entry)
        log_event(
            "pipeline.stage." + ("failed" if status == "failed" else "completed"),
            run_id=self.run_id,
            request_id=self.request_id,
            component=self.component,
            stage=entry["name"],
            duration_ms=entry["durationMs"],
            status=entry["status"],
            **{key: value for key, value in metadata.items() if value is not None},
        )
        return entry

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            stages = [dict(stage) for stage in self.stages]
        return {
            "runId": self.run_id,
            "requestId": self.request_id,
            "component": self.component,
            "stages": stages,
            "stageCount": len(stages),
        }


class StageTimer:
    def __init__(self, trace: TimingTrace, name: str, metadata: Optional[Mapping[str, Any]] = None):
        self.trace = trace
        self.name = str(name)
        self.metadata = dict(metadata or {})
        self.started_at = 0.0
        self.duration_ms = 0.0
        self.entry: Optional[dict[str, Any]] = None

    def __enter__(self) -> "StageTimer":
        self.started_at = perf_counter()
        return self

    def __exit__(self, exc_type, _exc_value, _traceback) -> bool:
        self.duration_ms = (perf_counter() - self.started_at) * 1000.0
        self.entry = self.trace.record(
            self.name,
            self.duration_ms,
            status="failed" if exc_type else "completed",
            error_class=exc_type.__name__ if exc_type else None,
            **self.metadata,
        )
        return False


def elapsed_ms(started_at: float) -> float:
    return round(max(0.0, (perf_counter() - started_at) * 1000.0), 3)
