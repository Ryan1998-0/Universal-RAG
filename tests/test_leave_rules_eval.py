import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_closed_book_oracle_eval.py"
SPEC = importlib.util.spec_from_file_location("leave_rules_eval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_refusal_cannot_pass_by_repeating_expected_words():
    item = {
        "required": [["工資照給"], ["實際需要"]],
        "forbidden": [],
    }
    answer = "根據目前檢索資料無法確認。條文寫著工資照給，假期視實際需要定之。"

    result = MODULE.score_answer(answer, item)

    assert result["score"] == 0
    assert "根據目前檢索資料無法確認" in result["forbidden_hits"]


def test_grounded_answer_passes_all_required_fact_groups():
    item = {
        "required": [["十四日", "14日"], ["不給工資", "無薪"]],
        "forbidden": ["三十日"],
    }

    result = MODULE.score_answer("事假一年最多十四日，期間不給工資。", item)

    assert result["score"] == 100
    assert result["forbidden_hits"] == []


def test_question_relevant_grounded_score_does_not_require_oracle_extras():
    item = {
        "id": "corrupt-middle-bereavement",
        "oracle_context": "祖父母、子女喪亡者，給予喪假六十六日，工資照給。",
        "required": [["六十六日"], ["工資照給"]],
        "forbidden": [],
    }

    result = MODULE.score_grounded_answer(
        "六十六日。",
        item,
        evidence_text="祖父母、子女喪亡者，給予喪假六十六日，工資照給。",
    )

    assert result["score"] == 100
    assert result["metric"] == "question-relevant-grounded-facts"


def test_question_relevant_grounded_score_rejects_unsupported_number():
    item = {
        "id": "corrupt-inpatient-sick-limit",
        "oracle_context": "住院者，二十年內合計不得超過十年。",
        "required": [["二十年"], ["十年"]],
        "forbidden": [],
    }

    result = MODULE.score_grounded_answer(
        "二十年內合計不得超過二十年。",
        item,
        evidence_text="住院者，二十年內合計不得超過十年。",
    )

    assert result["score"] == 0
    assert any("未在證據出現的數值" in hit for hit in result["forbidden_hits"])


def test_question_relevant_grounded_score_accepts_semantic_punctuation_variant():
    item = {
        "id": "corrupt-family-care-exception",
        "oracle_context": "除本法或其他法律另有規定者外，得依前項規定請事假。",
        "required": [["本法或其他法律另有規定"]],
        "forbidden": [],
    }

    result = MODULE.score_grounded_answer(
        "除《本法》或其他法律另有規定外，得依前項規定辦理。",
        item,
        evidence_text="除本法或其他法律另有規定者外，得依前項規定請事假。",
    )

    assert result["score"] == 100
