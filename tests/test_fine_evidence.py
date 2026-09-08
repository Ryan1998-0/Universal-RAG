from rag_demo.config import RagConfig
from rag_demo.fine_evidence import _fine_units, retrieve_fine_evidence


def test_fine_evidence_reembeds_small_units_and_merges_same_article():
    settings = RagConfig(
        fine_evidence_enabled=True,
        fine_evidence_chunk_chars=240,
        fine_evidence_top_k=4,
        fine_evidence_max_groups=2,
        fine_evidence_min_embedding_score=0.45,
    ).normalized()
    contexts = [
        {
            "id": "parent-1",
            "rank": 1,
            "title": "勞工請假規則",
            "source": "official",
            "page": "全文",
            "content": (
                "第 7 條 勞工因有事故必須親自處理，得請事假，一年內合計不得超過十四日。"
                "事假期間不給工資。"
                "第 8 條 勞工依法令規定應給予公假者，工資照給，其假期視實際需要定之。"
            ),
        }
    ]

    focused, trace = retrieve_fine_evidence(
        "事假一年內最多幾日？事假期間是否給薪？",
        contexts,
        settings=settings,
        embed_query_fn=lambda _: [1.0, 0.0],
        embed_texts_fn=lambda texts: [
            [1.0, 0.0] if "事假" in text and "十四日" in text else [0.3, 0.1]
            for text in texts
        ],
    )

    assert focused
    assert "十四日" in focused[0]["content"]
    assert "不給工資" in focused[0]["content"]
    assert focused[0]["fineGrouped"] is True
    assert trace["mode"] == "fine-embedding-group"
    assert trace["candidateCount"] >= 3


def test_fine_evidence_rejects_semantic_only_distractor_after_best_hit():
    settings = RagConfig(
        fine_evidence_enabled=True,
        fine_evidence_top_k=3,
        fine_evidence_max_groups=2,
        fine_evidence_min_embedding_score=0.70,
    ).normalized()
    contexts = [
        {
            "id": "parent-1",
            "rank": 1,
            "title": "規則",
            "source": "official",
            "content": "第 9-1 條 一年內普通傷病假未超過十日者，雇主不得為不利處分。",
        },
        {
            "id": "parent-2",
            "rank": 2,
            "title": "其他假別",
            "source": "official",
            "content": "婚假八日，工資照給。喪假六日，工資照給。",
        },
    ]

    focused, _ = retrieve_fine_evidence(
        "普通傷病假未超過幾日不得不利處分？",
        contexts,
        settings=settings,
        embed_query_fn=lambda _: [1.0, 0.0],
        embed_texts_fn=lambda texts: [
            [0.8, 0.0] if "普通傷病假" in text else [0.69, 0.724]
            for text in texts
        ],
    )

    assert focused[0]["source"] == "official"
    assert "十日" in focused[0]["content"]
    assert not any("婚假" in item["content"] for item in focused)


def test_fine_evidence_fraction_sizes_second_pass_from_each_parent_chunk():
    content = "第 404 條 " + ("普通傷病假規定與數值證據。" * 28)
    parent_chars = len(content)
    units = _fine_units(
        [{"id": "parent-1", "source": "official", "content": content}],
        max_chars=240,
        overlap_chars=32,
        chunk_fraction=1 / 3,
    )

    assert units
    assert max(len(unit["content"]) for unit in units) <= round(parent_chars / 3)
