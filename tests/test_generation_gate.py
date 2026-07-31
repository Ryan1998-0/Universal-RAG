import asyncio
import time
import unittest

from rag_demo.production.generation_gate import (
    GenerationCapacityTimeout,
    RedisGenerationGate,
)


class FakeRedis:
    def __init__(self):
        self.leases = {}

    def eval(self, script, key_count, key, *args):
        self._remove_expired()
        if "ZREMRANGEBYSCORE" in script:
            limit, lease_milliseconds, token = args
            if len(self.leases) >= int(limit):
                retry_ms = max(
                    1,
                    int((min(self.leases.values()) - time.monotonic()) * 1000),
                )
                return [0, len(self.leases), retry_ms]
            self.leases[str(token)] = (
                time.monotonic() + int(lease_milliseconds) / 1000
            )
            return [1, len(self.leases), 0]
        token = str(args[0])
        removed = int(self.leases.pop(token, None) is not None)
        return removed

    def ping(self):
        return True

    def _remove_expired(self):
        now = time.monotonic()
        self.leases = {
            token: expires_at
            for token, expires_at in self.leases.items()
            if expires_at > now
        }


class RedisGenerationGateTests(unittest.TestCase):
    def test_limit_is_shared_across_gate_instances(self):
        async def scenario():
            redis = FakeRedis()
            first_gate = RedisGenerationGate(
                redis,
                limit=1,
                lease_seconds=5,
                poll_interval_seconds=0.01,
            )
            second_gate = RedisGenerationGate(
                redis,
                limit=1,
                lease_seconds=5,
                poll_interval_seconds=0.01,
            )
            first_token = await first_gate.acquire(0.1)
            with self.assertRaises(GenerationCapacityTimeout):
                await second_gate.acquire(0.03)
            await first_gate.release(first_token)
            second_token = await second_gate.acquire(0.1)
            await second_gate.release(second_token)
            self.assertEqual(redis.leases, {})

        asyncio.run(scenario())

    def test_abandoned_lease_expires(self):
        async def scenario():
            redis = FakeRedis()
            first_gate = RedisGenerationGate(
                redis,
                limit=1,
                lease_seconds=0.02,
                poll_interval_seconds=0.01,
            )
            second_gate = RedisGenerationGate(
                redis,
                limit=1,
                lease_seconds=0.02,
                poll_interval_seconds=0.01,
            )
            await first_gate.acquire(0.1)
            second_token = await second_gate.acquire(1.2)
            await second_gate.release(second_token)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
