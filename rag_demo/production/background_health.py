from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone


class RedisBackgroundHealth:
    def __init__(self, redis_client, *, key: str, ttl_seconds: int):
        self.redis_client = redis_client
        self.key = str(key)
        self.ttl_seconds = max(30, int(ttl_seconds))
        self.started_at = datetime.now(timezone.utc)

    @classmethod
    def from_settings(cls, settings):
        try:
            from redis import Redis
        except ImportError as exc:
            raise RuntimeError("redis is required for background health checks") from exc
        return cls(
            Redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
                health_check_interval=30,
            ),
            key=f"rag:health:{settings.environment}:pipeline",
            ttl_seconds=settings.background_heartbeat_ttl_seconds,
        )

    def mark_alive(self, *, worker_id: str = "") -> None:
        payload = json.dumps({
            "worker_id": str(worker_id)[:160],
            "observed_at": datetime.now(timezone.utc).isoformat(),
        })
        self.redis_client.set(self.key, payload, ex=self.ttl_seconds)

    def ping(self) -> None:
        payload = self.redis_client.get(self.key)
        if not payload:
            raise RuntimeError("Celery beat-to-worker heartbeat is stale")
        try:
            observed = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Celery heartbeat payload is invalid") from exc
        try:
            observed_at = datetime.fromisoformat(str(observed.get("observed_at") or ""))
        except ValueError as exc:
            raise RuntimeError("Celery heartbeat timestamp is invalid") from exc
        if observed_at.tzinfo is None:
            raise RuntimeError("Celery heartbeat timestamp has no timezone")
        now = datetime.now(timezone.utc)
        observed_at = observed_at.astimezone(timezone.utc)
        if (
            observed_at <= self.started_at
            or observed_at > now + timedelta(seconds=5)
            or now - observed_at >= timedelta(seconds=self.ttl_seconds)
        ):
            raise RuntimeError("Celery beat-to-worker heartbeat is stale")
