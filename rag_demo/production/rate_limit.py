import hashlib
import threading
import time
from dataclasses import dataclass
from typing import Protocol

from rag_demo.production.auth import Principal
from rag_demo.production.config import ProductionSettings


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    remaining: int
    retry_after_seconds: int


class RateLimiter(Protocol):
    def allow(self, principal: Principal, action: str) -> RateLimitDecision:
        ...


class InMemoryRateLimiter:
    def __init__(self, limit: int, window_seconds: int = 60, clock=time.monotonic):
        self.limit = max(1, int(limit))
        self.window_seconds = max(1, int(window_seconds))
        self.clock = clock
        self._entries = {}
        self._lock = threading.Lock()

    def allow(self, principal: Principal, action: str) -> RateLimitDecision:
        key = _safe_key(principal, action)
        now = float(self.clock())
        with self._lock:
            count, expires_at = self._entries.get(
                key,
                (0, now + self.window_seconds),
            )
            if now >= expires_at:
                count = 0
                expires_at = now + self.window_seconds
            count += 1
            self._entries[key] = (count, expires_at)
            retry_after = max(1, int(expires_at - now + 0.999))
            return RateLimitDecision(
                allowed=count <= self.limit,
                remaining=max(0, self.limit - count),
                retry_after_seconds=retry_after,
            )


class RedisRateLimiter:
    _SCRIPT = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then
  redis.call('EXPIRE', KEYS[1], ARGV[2])
end
local ttl = redis.call('TTL', KEYS[1])
return {count, ttl}
"""

    def __init__(self, redis_client, limit: int, window_seconds: int = 60):
        self.redis_client = redis_client
        self.limit = max(1, int(limit))
        self.window_seconds = max(1, int(window_seconds))

    @classmethod
    def from_url(cls, url: str, limit: int, window_seconds: int = 60):
        try:
            from redis import Redis
        except ImportError as exc:
            raise RuntimeError("redis is required for production rate limiting") from exc
        return cls(
            Redis.from_url(
                url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
                health_check_interval=30,
            ),
            limit=limit,
            window_seconds=window_seconds,
        )

    def allow(self, principal: Principal, action: str) -> RateLimitDecision:
        raw_count, raw_ttl = self.redis_client.eval(
            self._SCRIPT,
            1,
            f"rag:rate:{_safe_key(principal, action)}",
            self.limit,
            self.window_seconds,
        )
        count = int(raw_count)
        ttl = max(1, int(raw_ttl))
        return RateLimitDecision(
            allowed=count <= self.limit,
            remaining=max(0, self.limit - count),
            retry_after_seconds=ttl,
        )

    def ping(self) -> None:
        self.redis_client.ping()


def build_rate_limiter(settings: ProductionSettings) -> RateLimiter:
    if settings.redis_url:
        return RedisRateLimiter.from_url(
            settings.redis_url,
            limit=settings.ask_rate_limit_per_minute,
        )
    return InMemoryRateLimiter(limit=settings.ask_rate_limit_per_minute)


def _safe_key(principal: Principal, action: str) -> str:
    identity = "\0".join((principal.tenant_id, principal.subject, str(action)))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()
