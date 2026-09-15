import json
import logging
import tempfile
import unittest

from rag_demo.observability import TimingTrace, configure_logging, log_event


class ObservabilityTests(unittest.TestCase):
    def test_json_log_is_written_and_sensitive_values_are_redacted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = configure_logging(log_dir=directory, filename="test-rag.log")
            trace = TimingTrace(run_id="run-1", request_id="req-1", component="test")
            trace.record("retrieval.bm25", 4.25, candidates=5)
            log_event("test.payload", prompt="do not write this", content="private text")
            logger = logging.getLogger("rag_demo")
            for handler in logger.handlers:
                handler.flush()

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertGreaterEqual(len(lines), 2)
            payload = [json.loads(line) for line in lines]
            self.assertTrue(any(item.get("stage") == "retrieval.bm25" for item in payload))
            raw = path.read_text(encoding="utf-8")
            self.assertNotIn("do not write this", raw)
            self.assertNotIn("private text", raw)
            self.assertIn('"redacted": true', raw)
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
                handler.close()

    def test_stage_context_records_duration_and_failure(self):
        trace = TimingTrace(component="test")
        with trace.stage("ok"):
            pass
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with trace.stage("failed"):
                raise RuntimeError("boom")
        stages = trace.as_dict()["stages"]
        self.assertEqual([stage["name"] for stage in stages], ["ok", "failed"])
        self.assertEqual(stages[0]["status"], "completed")
        self.assertEqual(stages[1]["status"], "failed")
        self.assertGreaterEqual(stages[0]["durationMs"], 0)


if __name__ == "__main__":
    unittest.main()
