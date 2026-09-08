import json
import tempfile
import unittest
from pathlib import Path

from rag_demo.hybrid_retrieval import (
    Bm25Index,
    HybridRetriever,
    evaluate_retrieval_evidence,
    expand_retrieval_query,
    list_profile_retrieval_configs,
    load_profile_retrieval_config,
    reciprocal_rank_fusion,
)
from rag_demo.config import RagConfig
from rag_demo.retrieval_scope import RetrievalScope


class HybridRetrievalTest(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {
                "id": "apple-1",
                "source_id": "fruit-guide",
                "title": "Apple guide",
                "page": "page 1",
                "content": "An apple is a crisp red fruit used in pies.",
            },
            {
                "id": "banana-1",
                "source_id": "fruit-guide",
                "title": "Banana guide",
                "page": "page 2",
                "content": "A banana is a soft yellow fruit rich in potassium.",
            },
            {
                "id": "contract-1",
                "source_id": "accounting-guide",
                "title": "Contract accounting",
                "page": "page 3",
                "content": "Contract liabilities are recognized under the accounting standard.",
            },
        ]
        self.retriever = HybridRetriever(
            chunks=self.chunks,
            aliases=[],
            embeddings=[
                [1.0, 0.0],
                [0.0, 1.0],
                [0.7, 0.7],
            ],
            embed_query_fn=lambda _: [1.0, 0.0],
            embedding_model="fake-multilingual-embedding",
        )

    def test_bm25_ranks_matching_document_first(self):
        results = Bm25Index(self.chunks).search("apple fruit", top_k=3)

        self.assertEqual(results[0]["index"], 0)
        self.assertIn("apple", results[0]["matched_terms"])

    def test_hybrid_retrieval_fuses_embedding_and_bm25_then_reranks(self):
        result = self.retriever.retrieve("What is an apple?", top_k=2, candidate_k=3)

        self.assertEqual(result["variant"], "bm25_embedding_rerank")
        self.assertEqual(result["contexts"][0]["id"], "apple-1")
        self.assertGreater(result["contexts"][0]["bm25Score"], 0)
        self.assertGreater(result["contexts"][0]["embeddingScore"], 0)
        self.assertGreater(result["contexts"][0]["rerankScore"], 0)
        self.assertEqual(
            [step["name"] for step in result["pipeline"]],
            [
                "Query Planning",
                "BM25",
                "Embedding",
                "RRF Merge",
                "Hybrid Relevance Reranker",
                "Evidence Relevance Gate",
            ],
        )
        self.assertTrue(result["evidenceEvaluation"]["sufficient"])

    def test_multi_query_runs_each_variant_and_exposes_diagnostics(self):
        seen_queries = []
        retriever = HybridRetriever(
            chunks=self.chunks,
            aliases=[],
            embeddings=[[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]],
            embed_query_fn=lambda query: seen_queries.append(query) or [1.0, 0.0],
            embedding_model="fake",
        )

        result = retriever.retrieve(
            "What fruit is red?",
            retrieval_query="red fruit",
            query_variants=["red fruit", "apple crisp pies", "fruit color"],
            evidence_query="red fruit apple",
            top_k=2,
            candidate_k=3,
        )

        self.assertEqual(result["retrievalQueries"], ["red fruit", "apple crisp pies", "fruit color", "What fruit is red?"])
        self.assertEqual(seen_queries, result["retrievalQueries"])
        self.assertEqual(result["diagnostics"]["queryCount"], 4)

    def test_evidence_gate_rejects_normalized_top_result_without_raw_relevance(self):
        evaluation = evaluate_retrieval_evidence(
            [
                {
                    "rerankScore": 0.56,
                    "bm25Score": 0.0,
                    "embeddingScore": 0.13,
                    "matchedTerms": [],
                }
            ]
        )

        self.assertEqual(evaluation["status"], "irrelevant")
        self.assertFalse(evaluation["sufficient"])
        self.assertEqual(evaluation["confidence"], "low")

    def test_evidence_gate_accepts_cross_language_semantic_match(self):
        evaluation = evaluate_retrieval_evidence(
            [
                {
                    "rerankScore": 0.56,
                    "bm25Score": 0.0,
                    "embeddingScore": 0.53,
                    "matchedTerms": [],
                }
            ]
        )

        self.assertEqual(evaluation["status"], "relevant")
        self.assertTrue(evaluation["sufficient"])
        self.assertEqual(evaluation["confidence"], "high")

    def test_evidence_gate_rejects_generic_lexical_matches(self):
        evaluation = evaluate_retrieval_evidence(
            [
                {
                    "bm25Score": 12.7,
                    "embeddingScore": 0.408,
                    "matchedTerms": ["資料", "時間"],
                }
            ],
            question="台北捷運末班車時間是幾點？",
        )

        self.assertEqual(evaluation["status"], "ambiguous")
        self.assertFalse(evaluation["sufficient"])
        self.assertEqual(evaluation["signals"]["meaningfulMatchedTerms"], 0)

    def test_document_scoped_question_still_requires_real_relevance(self):
        evaluation = evaluate_retrieval_evidence(
            [
                {
                    "bm25Score": 0.0,
                    "embeddingScore": 0.356,
                    "matchedTerms": [],
                }
            ],
            question="這份資料有沒有提到一般上班時間？",
        )

        self.assertEqual(evaluation["status"], "ambiguous")
        self.assertFalse(evaluation["sufficient"])
        self.assertEqual(evaluation["confidence"], "low")

    def test_evidence_gate_accepts_meaningful_lexical_match(self):
        evaluation = evaluate_retrieval_evidence(
            [
                {
                    "bm25Score": 9.0,
                    "embeddingScore": 0.12,
                    "matchedTerms": ["資遣費"],
                }
            ],
            question="資遣費怎麼計算？",
        )

        self.assertTrue(evaluation["sufficient"])
        self.assertEqual(evaluation["confidence"], "medium")

    def test_precision_question_requires_subject_and_value_in_same_evidence_window(self):
        evaluation = evaluate_retrieval_evidence(
            [
                {
                    "content": "工作規則應記載考勤、請假、獎懲及升遷。",
                    "bm25Score": 8.0,
                    "embeddingScore": 0.72,
                    "matchedTerms": ["規定", "事假"],
                },
                {
                    "content": "勞工有正當事由得請假；事假以外期間的工資標準另定。",
                    "bm25Score": 4.0,
                    "embeddingScore": 0.66,
                    "matchedTerms": ["事假"],
                },
            ],
            question="事假天數上限是多少？",
        )

        self.assertFalse(evaluation["sufficient"])
        self.assertFalse(evaluation["signals"]["answerBearingEvidence"])

    def test_precision_question_accepts_answer_bearing_subject_value_window(self):
        evaluation = evaluate_retrieval_evidence(
            [
                {
                    "content": "勞工因事故必須親自處理者，得請事假，一年內合計不得超過十四日。",
                    "bm25Score": 8.0,
                    "embeddingScore": 0.72,
                    "matchedTerms": ["事假", "上限"],
                }
            ],
            question="事假天數上限是多少？",
        )

        self.assertTrue(evaluation["sufficient"])
        self.assertTrue(evaluation["signals"]["answerBearingEvidence"])

    def test_relaxed_gate_accepts_relevant_lower_rank_context(self):
        evaluation = evaluate_retrieval_evidence(
            [
                {
                    "content": "文件標題與來源資訊。",
                    "bm25Score": 0.39,
                    "embeddingScore": 0.21,
                    "matchedTerms": ["文件", "來源"],
                },
                {
                    "content": "勞工依法令規定應給予公假者，工資照給，其假期視實際需要定之。",
                    "bm25Score": 0.44,
                    "embeddingScore": 0.41,
                    "matchedTerms": ["公假", "假期", "實際需要"],
                },
            ],
            settings=RagConfig(evidence_gate_mode="relaxed"),
            question="公假的假期依什麼決定？",
        )

        self.assertTrue(evaluation["sufficient"])
        self.assertEqual(evaluation["signals"]["gateMode"], "relaxed")
        self.assertTrue(evaluation["signals"]["relaxedRelevance"])

    def test_relaxed_gate_still_rejects_generic_only_match(self):
        evaluation = evaluate_retrieval_evidence(
            [
                {
                    "content": "一般資料與時間說明。",
                    "bm25Score": 12.7,
                    "embeddingScore": 0.408,
                    "matchedTerms": ["資料", "時間"],
                }
            ],
            settings=RagConfig(evidence_gate_mode="relaxed"),
            question="台北捷運末班車時間是幾點？",
        )

        self.assertFalse(evaluation["sufficient"])
        self.assertFalse(evaluation["signals"]["relaxedRelevance"])

    def test_source_filter_is_applied_to_both_retrieval_branches(self):
        result = self.retriever.retrieve(
            "apple",
            source_ids=["accounting-guide"],
            top_k=2,
            candidate_k=3,
        )

        self.assertEqual([context["source"] for context in result["contexts"]], ["accounting-guide"])

    def test_explicit_empty_source_selection_retrieves_nothing(self):
        result = self.retriever.retrieve(
            "apple",
            source_ids=[],
            top_k=2,
            candidate_k=3,
        )

        self.assertEqual(result["contexts"], [])
        self.assertEqual(result["diagnostics"]["selectedSourceCount"], 0)
        self.assertFalse(result["diagnostics"]["profileExpansionsApplied"])

    def test_tenant_scope_blocks_same_source_id_from_other_tenant(self):
        chunks = [
            {
                "id": "tenant-a-chunk",
                "source_id": "shared-source",
                "tenant_id": "tenant-a",
                "knowledge_base_id": "kb-a",
                "index_version_id": "index-a",
                "title": "Tenant A",
                "content": "Shared keyword tenant A evidence.",
            },
            {
                "id": "tenant-b-chunk",
                "source_id": "shared-source",
                "tenant_id": "tenant-b",
                "knowledge_base_id": "kb-b",
                "index_version_id": "index-b",
                "title": "Tenant B",
                "content": "Shared keyword tenant B secret.",
            },
        ]
        retriever = HybridRetriever(
            chunks=chunks,
            aliases=[],
            embeddings=[[1.0, 0.0], [1.0, 0.0]],
            embed_query_fn=lambda _: [1.0, 0.0],
            embedding_model="fake",
        )

        result = retriever.retrieve(
            "shared keyword",
            source_ids=["shared-source"],
            retrieval_scope=RetrievalScope(
                tenant_id="tenant-a",
                knowledge_base_id="kb-a",
                index_version_id="index-a",
            ),
            top_k=2,
            candidate_k=2,
        )

        self.assertEqual([item["id"] for item in result["contexts"]], ["tenant-a-chunk"])
        self.assertTrue(result["diagnostics"]["tenantScopeApplied"])

    def test_rrf_keeps_candidates_from_both_branches(self):
        fused = reciprocal_rank_fusion(
            bm25_results=[{"index": 0, "score": 3.0, "matched_terms": ["apple"]}],
            embedding_results=[{"index": 2, "score": 0.9}],
        )

        self.assertEqual({candidate["index"] for candidate in fused}, {0, 2})

    def test_query_expansion_is_supplied_by_profile_data(self):
        query, added_terms, _ = expand_retrieval_query(
            "蘋果可以保存多久？",
            "",
            [],
            query_expansions=[
                {
                    "markers": ["保存"],
                    "terms": ["storage life", "refrigeration"],
                }
            ],
        )

        self.assertIn("storage life", query)
        self.assertIn("refrigeration", added_terms)

    def test_query_expansion_has_no_domain_terms_without_profile_data(self):
        query, added_terms, _ = expand_retrieval_query("制度的適用範圍是什麼？", "", [])

        self.assertEqual(query, "制度的適用範圍是什麼？")
        self.assertEqual(added_terms, [])

    def test_profile_retrieval_rules_are_loaded_from_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_root = root / "profiles" / "legal_docs"
            profile_root.mkdir(parents=True)
            data_path = root / "corpus.json"
            data_path.write_text('{"chunks": [], "aliases": [], "sources": []}', encoding="utf-8")
            (profile_root / "retrieval.json").write_text(
                json.dumps(
                    {
                        "data_path": "../../corpus.json",
                        "label": "Legal Documents",
                        "sample_queries": ["How is severance calculated?"],
                        "query_expansions": [
                            {"markers": ["資遣"], "terms": ["severance pay"]}
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            config = load_profile_retrieval_config("legal_docs", project_root=root)

        self.assertEqual(config["data_path"], data_path.resolve())
        self.assertEqual(config["label"], "Legal Documents")
        self.assertEqual(config["sample_queries"], ["How is severance calculated?"])
        self.assertEqual(
            config["query_expansions"],
            [{"markers": ["資遣"], "terms": ["severance pay"]}],
        )

    def test_profile_discovery_is_driven_by_retrieval_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_root = root / "profiles" / "employee_handbook"
            profile_root.mkdir(parents=True)
            (profile_root / "retrieval.json").write_text(
                json.dumps({"label": "Employee Handbook"}),
                encoding="utf-8",
            )

            profiles = list_profile_retrieval_configs(
                project_root=root,
                default_profile="employee_handbook",
            )

        self.assertEqual([profile["profile"] for profile in profiles], ["employee_handbook"])
        self.assertEqual(profiles[0]["label"], "Employee Handbook")

    def test_uploaded_document_can_be_added_and_filtered_without_restart(self):
        source = {
            "source_id": "upload-0123456789abcdefabcd",
            "name": "Ryan project.docx",
            "source_type": "word",
        }
        added = self.retriever.add_document(
            source=source,
            chunks=[
                {
                    "id": "upload-0123456789abcdefabcd::0",
                    "source_id": source["source_id"],
                    "title": "Project secret",
                    "content": "The unique project code is AURORA-731.",
                }
            ],
            embeddings=[[1.0, 0.0]],
        )
        self.retriever.aliases = [
            {
                "canonical": "risk adjustment for non-financial risk",
                "aliases": ["RA"],
            }
        ]

        result = self.retriever.retrieve(
            "AURORA-731",
            source_ids=[source["source_id"]],
            top_k=2,
            candidate_k=3,
        )

        self.assertTrue(added)
        self.assertEqual(result["contexts"][0]["source"], source["source_id"])
        self.assertIn("AURORA-731", result["contexts"][0]["content"])
        self.assertNotIn("risk adjustment", result["retrievalQuery"])
        self.assertFalse(result["diagnostics"]["profileExpansionsApplied"])
        self.assertIn(source, self.retriever.list_sources())
        self.assertFalse(
            self.retriever.add_document(
                source=source,
                chunks=[{"id": "duplicate", "content": "duplicate"}],
                embeddings=[[1.0, 0.0]],
            )
        )


if __name__ == "__main__":
    unittest.main()
