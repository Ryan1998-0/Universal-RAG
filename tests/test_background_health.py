import json
import unittest

from rag_demo.production.background_health import RedisBackgroundHealth


class FakeRedis:
    def __init__(self):
        self.values = {}

    def set(self, key, value, ex):
        self.values[key] = (value, ex)

    def get(self, key):
        value = self.values.get(key)
        return value[0] if value else None


class BackgroundHealthTests(unittest.TestCase):
    def test_beat_to_worker_heartbeat_is_required(self):
        redis = FakeRedis()
        probe = RedisBackgroundHealth(
            redis,
            key="rag:health:test:pipeline",
            ttl_seconds=90,
        )
        with self.assertRaises(RuntimeError):
            probe.ping()

        probe.mark_alive(worker_id="worker-a")
        probe.ping()
        payload, ttl = redis.values["rag:health:test:pipeline"]
        self.assertEqual(ttl, 90)
        self.assertEqual(json.loads(payload)["worker_id"], "worker-a")

    def test_invalid_heartbeat_payload_fails_closed(self):
        redis = FakeRedis()
        redis.values["rag:health:test:pipeline"] = ("not-json", 90)
        probe = RedisBackgroundHealth(
            redis,
            key="rag:health:test:pipeline",
            ttl_seconds=90,
        )
        with self.assertRaises(RuntimeError):
            probe.ping()


if __name__ == "__main__":
    unittest.main()
