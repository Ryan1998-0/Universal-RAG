import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class RetrievalPlan:
    primary_query: str
    query_variants: Tuple[str, ...]
    evidence_query: str
    intent_labels: Tuple[str, ...]


def build_retrieval_plan(
    question: str,
    primary_query: str = "",
    rewritten_query: str = "",
    max_variants: int = 10,
) -> RetrievalPlan:
    """Build an agent-style retrieval plan without adding another model call."""

    focus_terms = extract_focus_terms(question)
    intent = infer_intent_terms(question)
    intent_terms = intent["terms"]
    subject_terms = focus_terms[:5]

    variants = _unique_nonempty(
        [
            primary_query,
            question,
            rewritten_query,
            " ".join([*focus_terms, *intent_terms]),
            " ".join([*subject_terms, *intent_terms]),
            *_pairwise_focus_queries(focus_terms, intent_terms),
        ]
    )[:max_variants]

    if not variants:
        variants = (_clean_spaces(question),)

    evidence_query = " ".join(
        _unique_terms(
            [
                question,
                primary_query,
                rewritten_query,
                *focus_terms,
                *intent_terms,
            ]
        )
    )

    return RetrievalPlan(
        primary_query=variants[0],
        query_variants=tuple(variants),
        evidence_query=evidence_query,
        intent_labels=tuple(intent["labels"]),
    )


def extract_focus_terms(question: str) -> List[str]:
    text = _clean_spaces(question)
    terms = []
    terms.extend(_quoted_terms(text))
    terms.extend(_split_chinese_focus_terms(text))
    terms.extend(_compound_subterms(terms))
    return _unique_terms(term for term in terms if _is_useful_term(term))


def infer_intent_terms(question: str):
    compact = re.sub(r"\s+", "", str(question))
    labels = []
    terms = []

    if any(marker in compact for marker in ("為什麼", "為何", "原因", "怎麼會", "如何影響")):
        labels.append("cause")
        terms.extend(["原因", "背景", "導致", "因為", "所以", "影響", "結果"])

    if any(marker in compact for marker in ("如何", "怎麼", "透過", "方式", "方法")):
        labels.append("method")
        terms.extend(["方式", "方法", "過程", "透過", "經由", "使用"])

    if any(marker in compact for marker in ("失敗", "錯", "錯誤", "證明", "被證明", "不成立", "模型")):
        labels.append("failure_reason")
        terms.extend(["失敗", "錯誤", "不成立", "被證明", "模型", "假設", "缺陷", "預測"])

    if any(marker in compact for marker in ("說明", "表明", "代表", "意味", "實驗", "比喻")):
        labels.append("explanation")
        terms.extend(["說明", "表明", "代表", "意味", "實驗", "比喻", "問題"])

    if any(marker in compact for marker in ("不同", "差異", "相比", "比較")):
        labels.append("comparison")
        terms.extend(["不同", "差異", "判斷", "觀點", "看法", "認為"])

    if any(marker in compact for marker in ("為什麼要", "目的", "為了", "保住", "保存", "保留", "避免")):
        labels.append("purpose")
        terms.extend(["目的", "為了", "需要", "避免", "保存", "保留", "資料", "資訊"])

    if not labels:
        labels.append("fact")

    return {"labels": _unique_terms(labels), "terms": _unique_terms(terms)}


def _quoted_terms(text: str) -> List[str]:
    return re.findall(r"[「『《]([^」』》]{1,30})[」』》]", text)


def _split_chinese_focus_terms(text: str) -> List[str]:
    normalized = text
    for phrase in _STOP_PHRASES:
        normalized = normalized.replace(phrase, " ")
    normalized = re.sub(r"[，。！？!?、：:；;（）()\[\]【】\n\r\t]+", " ", normalized)
    return [term.strip() for term in normalized.split() if term.strip()]


def _compound_subterms(terms: Iterable[str]) -> List[str]:
    subterms = []
    for term in terms:
        compact = re.sub(r"\s+", "", str(term))
        if len(compact) < 4:
            continue
        if compact.endswith("實驗"):
            subterms.extend([compact[:-2], "實驗"])
        if compact.endswith("模型"):
            subterms.extend([compact[:-2], "模型"])
        if compact.endswith("行動"):
            subterms.extend([compact[:-2], "行動"])
        if compact.endswith("材料"):
            subterms.extend([compact[:-2], "材料"])
        if compact.endswith("文明"):
            subterms.extend([compact[:-2], "文明"])
    return subterms


def _pairwise_focus_queries(focus_terms: Sequence[str], intent_terms: Sequence[str]) -> List[str]:
    if not focus_terms:
        return []

    queries = []
    head = list(focus_terms[:4])
    tail_intent = list(intent_terms[:5])
    if head:
        queries.append(" ".join([*head, *tail_intent]))

    for index, term in enumerate(focus_terms[:5]):
        neighbors = focus_terms[max(0, index - 1) : index + 2]
        queries.append(" ".join([*neighbors, *tail_intent[:3]]))
    return queries


def _unique_nonempty(values: Iterable[str]) -> List[str]:
    return [value for value in _unique_terms(_clean_spaces(value) for value in values) if value]


def _unique_terms(values: Iterable[str]) -> List[str]:
    seen = set()
    unique = []
    for value in values:
        cleaned = _clean_spaces(value)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        unique.append(cleaned)
    return unique


def _clean_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _is_useful_term(term: str) -> bool:
    compact = re.sub(r"\s+", "", str(term))
    if len(compact) < 2 or len(compact) > 24:
        return False
    return compact not in _STOP_TERMS


_STOP_PHRASES = (
    "請問",
    "小說中",
    "小說裡",
    "為什麼要",
    "為什麼會",
    "為什麼",
    "為何",
    "如何",
    "怎麼",
    "什麼樣的",
    "什麼",
    "哪些",
    "哪一項",
    "第一次",
    "最初",
    "一開始",
    "最後",
    "當前",
    "目前",
    "所處",
    "面臨",
    "使用",
    "透過",
    "通過",
    "接觸到",
    "接觸",
    "介紹",
    "研究",
    "展現",
    "特殊能力",
    "思考方式",
    "能力",
    "想向",
    "想要",
    "說明",
    "代表",
    "被證明是",
    "被證明",
    "的是",
    "是",
    "的",
    "了",
    "與",
    "和",
    "及",
    "或",
    "在",
    "時",
    "中",
    "裡",
    "內",
    "對",
    "向",
    "用",
    "有",
    "要",
    "會",
    "很",
    "能",
    "不",
    "嗎",
    "？",
    "?",
)


_STOP_TERMS = {
    "問題",
    "原因",
    "背景",
    "主要",
    "小說",
    "遊戲",
    "內容",
    "資料",
    "資訊",
    "方式",
    "方法",
    "不同",
    "差異",
    "影響",
    "結果",
}
