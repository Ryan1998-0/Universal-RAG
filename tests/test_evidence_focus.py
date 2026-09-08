from rag_demo.config import RagConfig
from rag_demo.evidence_focus import focus_retrieved_evidence


def test_evidence_focus_prefers_answer_bearing_sentence_with_low_dense_weight():
    settings = RagConfig(
        evidence_focus_enabled=True,
        evidence_focus_top_k=2,
        evidence_focus_max_chars=480,
        evidence_focus_keyword_weight=0.8,
        evidence_focus_embedding_weight=0.2,
    ).normalized()
    contexts = [
        {
            "id": "chunk-1",
            "rank": 1,
            "title": "規則",
            "source": "source-1",
            "page": "全文",
            "content": "婚假八日，工資照給。\n勞工因有事故必須親自處理，得請事假，一年內合計不得超過十四日。事假期間不給工資。\n其他背景說明。",
        }
    ]

    focused, trace = focus_retrieved_evidence(
        "事假一年內最多幾日？是否給薪？",
        contexts,
        settings=settings,
        embed_query_fn=lambda _: [1.0, 0.0],
        embed_texts_fn=lambda texts: [[0.0, 1.0] for _ in texts],
    )

    assert focused
    assert "十四日" in focused[0]["content"]
    assert "不給工資" in focused[0]["content"]
    assert trace["candidateCount"] == 6
    assert trace["keywordWeight"] == 0.8
    assert trace["embeddingWeight"] == 0.2


def test_evidence_focus_preserves_source_and_original_rank():
    settings = RagConfig(evidence_focus_top_k=1).normalized()
    contexts = [
        {
            "id": "chunk-9",
            "rank": 2,
            "title": "勞工請假規則",
            "source": "official",
            "page": "全文",
            "content": "第 7 條\n事假期間不給工資。",
        }
    ]

    focused, _ = focus_retrieved_evidence(
        "事假是否給薪？",
        contexts,
        settings=settings,
        embed_query_fn=lambda _: [1.0],
        embed_texts_fn=lambda texts: [[1.0] for _ in texts],
    )

    assert focused[0]["source"] == "official"
    assert focused[0]["originalRank"] == 2
    assert focused[0]["focus"] is True
