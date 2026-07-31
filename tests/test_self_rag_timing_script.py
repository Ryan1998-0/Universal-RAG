import unittest


class SelfRagTimingScriptTest(unittest.TestCase):
    def test_extract_timing_nodes_from_rag_output(self):
        from evals.self_rag_timing.measure_self_rag_overhead import extract_timing_nodes

        output = "\n".join(
            [
                "節點耗時：",
                "- query_rewrite: 0.50s",
                "- retrieval_attempt_1: 0.12s",
                "- answer_critic_attempt_1: 1.75s",
                "- total: 3.10s",
            ]
        )

        timings = extract_timing_nodes(output)

        self.assertEqual(timings["query_rewrite"], 0.5)
        self.assertEqual(timings["retrieval_attempt_1"], 0.12)
        self.assertEqual(timings["answer_critic_attempt_1"], 1.75)
        self.assertEqual(timings["total"], 3.1)

    def test_summarize_pair_reports_overhead(self):
        from evals.self_rag_timing.measure_self_rag_overhead import summarize_pair

        summary = summarize_pair(
            question="IFRS17 的 CSM 是什麼？",
            baseline={"ok": True, "timings": {"total": 2.0}},
            self_rag={"ok": True, "timings": {"total": 5.0}},
        )

        self.assertEqual(summary["baseline_total_seconds"], 2.0)
        self.assertEqual(summary["self_rag_total_seconds"], 5.0)
        self.assertEqual(summary["overhead_seconds"], 3.0)
        self.assertEqual(summary["overhead_multiplier"], 2.5)


if __name__ == "__main__":
    unittest.main()
