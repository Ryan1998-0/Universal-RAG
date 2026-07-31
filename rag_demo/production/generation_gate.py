from __future__ import annotations

import asyncio
from typing import Protocol
from uuid import uuid4


class GenerationCapacityTimeout(TimeoutError):
    pass


class GenerationGate(Protocol):
    async def acquire(self, timeout_seconds: float) -> str:
        ...

    async def release(self, token: str) -> None:
        ...


class InMemoryGenerationGate:
    def __init__(self, limit: int):
        self._semaphore = asyncio.Semaphore(max(1, int(limit)))

    async def acquire(self, timeout_seconds: float) -> str:
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=max(0.001, float(timeout_seconds)),
            )
        except asyncio.TimeoutError as exc:
            raise GenerationCapacityTimeout("generation capacity is full") from exc
        return uuid4().hex

    async def release(self, token: str) -> None:
        del token
        self._semaphore.release()


class RedisGenerationGate:
    _ACQUIRE_SCRIPT = """
local now_parts = redis.call('TIME')
local now_ms = (tonumber(now_parts[1]) * 1000) + math.floor(tonumber(now_parts[2]) / 1000)
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now_ms)
local count = redis.call('ZCARD', KEYS[1])
if count < tonumber(ARGV[1]) then
  local expires_at = now_ms + tonumber(ARGV[2])
  redis.call('ZADD', KEYS[1], expires_at, ARGV[3])
  redis.call('PEXPIRE', KEYS[1], tonumber(ARGV[2]) + 1000)
  return {1, count + 1, 0}
end
local earliest = redis.call('ZRANGE', KEYS[1], 0, 0, 'WITHSCORES')
local retry_ms = tonumber(ARGV[2])
if earliest[2] then
  retry_ms = math.max(1, tonumber(earliest[2]) - now_ms)
end
return {0, count, retry_ms}
"""
    _RELEASE_SCRIPT = """
local removed = redis.call('ZREM', KEYS[1], ARGV[1])
if redis.call('ZCARD', KEYS[1]) == 0 then
  redis.call('DEL', KEYS[1])
end
return removed
"""

    def __init__(
        self,
        redis_client,
        *,
        limit: int,
        lease_seconds: float,
        key: str = "rag:generation:global",
        poll_interval_seconds: float = 0.1,
    ):
        self.redis_client = redis_client
        self.limit = max(1, int(limit))
        self.lease_milliseconds = max(1000, int(float(lease_seconds) * 1000))
        self.key = str(key)
        self.poll_interval_seconds = max(0.01, float(poll_interval_seconds))

    @classmethod
    def from_url(cls, url: str, *, limit: int, lease_seconds: float):
        try:
            from redis import Redis
        except ImportError as exc:
            raise RuntimeError("redis is required for global generation limits") from exc
        return cls(
            Redis.from_url(
                url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
                health_check_interval=30,
            ),
            limit=limit,
            lease_seconds=lease_seconds,
        )

    async def acquire(self, timeout_seconds: float) -> str:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.001, float(timeout_seconds))
        token = uuid4().hex
        while True:
            acquired, _count, retry_ms = await asyncio.to_thread(
                self.redis_client.eval,
                self._ACQUIRE_SCRIPT,
                1,
                self.key,
                self.limit,
                self.lease_milliseconds,
                token,
            )
            if int(acquired) == 1:
                return token
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise GenerationCapacityTimeout("generation capacity is full")
            retry_seconds = max(0.001, int(retry_ms) / 1000)
            await asyncio.sleep(
                min(self.poll_interval_seconds, retry_seconds, remaining)
            )

    async def release(self, token: str) -> None:
        await asyncio.to_thread(
            self.redis_client.eval,
            self._RELEASE_SCRIPT,
            1,
            self.key,
            str(token),
        )

    def ping(self) -> None:
        self.redis_client.ping()


def build_generation_gate(settings) -> GenerationGate:
    if settings.redis_url:
        return RedisGenerationGate.from_url(
            settings.redis_url,
            limit=settings.max_concurrent_generations,
            lease_seconds=settings.request_timeout_seconds + 30,
        )
    return InMemoryGenerationGate(settings.max_concurrent_generations)
