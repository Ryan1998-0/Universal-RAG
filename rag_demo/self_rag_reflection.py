import re
from dataclasses import dataclass
from typing import Callable, List

from rag_demo.chunking import Chunk
from rag_demo.model_providers import ask_model
from rag_demo.prompting import render_retrieved_context


PASSAGE_CRITIC_SYSTEM_PROMPT = """你是工程版 Self-RAG 的 Passage Critic。
你的任務不是回答問題，而是逐一判斷檢索片段是否和問題相關，以及是否能支撐答案。
請使用繁體中文，並只根據提供的檢索片段判斷。
"""

ANSWER_SUPPORT_CRITIC_SYSTEM_PROMPT = """你是工程版 Self-RAG 的 Answer Support Critic。
你的任務是判斷回答是否被檢索片段支撐，並在不足時提出下一次檢索查詢。
請使用繁體中文，嚴格區分來源支撐和模型推測。
"""

UTILITY_CRITIC_SYSTEM_PROMPT = """你是工程版 Self-RAG 的 Utility Critic。
你的任務是評估回答對使用者是否有用、清楚、可追溯來源。
請使用 1 到 5 分評分，5 分代表最有用。
"""


@dataclass(frozen=True)
class PassageReflection:
    source_index: int
    relevance_label: str
    support_label: str
    reason: str
    raw_output: str = ""

    @property
    def is_relevant(self) -> bool:
        return self.relevance_label in {"[Relevant]", "[Partially Relevant]"}

    @property
    def is_supported(self) -> bool:
        return self.support_label in {"[Supported]", "[Partially Supported]"}


@dataclass(frozen=True)
class AnswerSupportCritique:
    support_label: str
    is_supported: bool
    retry_needed: bool
    retry_query: str
    reason: str
    utility_score: int = 0
    utility_label: str = ""
    raw_output: str = ""


@dataclass(frozen=True)
class UtilityCritique:
    score: int
    utility_label: str
    retry_needed: bool
    retry_query: str
    reason: str
    raw_output: str = ""


def build_passage_critique_prompt(question: str, chunks: List[Chunk]) -> str:
    return f"""請逐一評估檢索片段是否能支撐回答。

輸出格式請固定如下，每個來源一行：
來源 1：[Relevant 或 Irrelevant][Supported 或 Unsupported] 理由：...
來源 2：[Relevant 或 Irrelevant][Supported 或 Unsupported] 理由：...

判斷規則：
- [Relevant]：來源和問題主題直接相關。
- [Irrelevant]：來源和問題主題無關或只共享泛用詞。
- [Supported]：來源足以支撐回答問題的關鍵事實。
- [Unsupported]：來源不足以支撐關鍵事實。

使用者問題：
{question}

檢索片段：
{render_retrieved_context(chunks)}
"""


def build_answer_support_critique_prompt(
    question: str,
    answer: str,
    chunks: List[Chunk],
    passage_reflections: List[PassageReflection],
) -> str:
    passage_text = "\n".join(
        f"來源 {item.source_index}：{item.relevance_label}{item.support_label} 理由：{item.reason}"
        for item in passage_reflections
    )
    return f"""請判斷回答是否被檢索來源支撐。

輸出格式：
[Supported]：是 / 否 / 部分
[Utility]：1/5 到 5/5
需要重新檢索：是 / 否
重新檢索查詢：...
判斷理由：...

使用者問題：
{question}

回答：
{answer}

Passage Critic 結果：
{passage_text or '無'}

檢索片段：
{render_retrieved_context(chunks)}
"""


def build_utility_critique_prompt(
    question: str,
    answer: str,
    answer_critique: AnswerSupportCritique,
) -> str:
    return f"""請評估回答對使用者是否有用。

輸出格式：
[Utility]：1/5 到 5/5
需要重新檢索：是 / 否
重新檢索查詢：...
判斷理由：...

評分標準：
- 5：回答清楚、直接、被來源支撐。
- 3：回答部分有用，但不夠完整或可追溯性不足。
- 1：回答無法回答問題、沒有來源支撐，或可能誤導使用者。

使用者問題：
{question}

回答：
{answer}

Answer Support Critic：
{answer_critique.support_label}, retry_needed={str(answer_critique.retry_needed).lower()}, reason={answer_critique.reason}
"""


def critique_passages(
    question: str,
    chunks: List[Chunk],
    model: str = "qwen2.5:7b",
    ask_model_fn: Callable[..., str] = ask_model,
) -> List[PassageReflection]:
    if not chunks:
        return []
    try:
        output = ask_model_fn(
            build_passage_critique_prompt(question, chunks),
            model=model,
            system=PASSAGE_CRITIC_SYSTEM_PROMPT,
        )
        return parse_passage_reflections(output, chunk_count=len(chunks))
    except Exception as exc:
        return _fallback_passage_reflections(question, chunks, reason=f"Passage Critic 呼叫失敗：{exc}")


def critique_answer_support(
    question: str,
    answer: str,
    chunks: List[Chunk],
    passage_reflections: List[PassageReflection],
    model: str = "qwen2.5:7b",
    ask_model_fn: Callable[..., str] = ask_model,
) -> AnswerSupportCritique:
    try:
        output = ask_model_fn(
            build_answer_support_critique_prompt(question, answer, chunks, passage_reflections),
            model=model,
            system=ANSWER_SUPPORT_CRITIC_SYSTEM_PROMPT,
        )
        return parse_answer_support_critique(output)
    except Exception as exc:
        has_supporting_passage = any(item.is_supported for item in passage_reflections)
        return AnswerSupportCritique(
            support_label="[Supported]" if has_supporting_passage else "[Unsupported]",
            is_supported=has_supporting_passage,
            retry_needed=not has_supporting_passage,
            retry_query=question,
            reason=f"Answer Support Critic 呼叫失敗，改用 passage fallback：{exc}",
            utility_score=4 if has_supporting_passage else 2,
            utility_label=f"[Utility:{4 if has_supporting_passage else 2}/5]",
        )


def score_answer_utility(
    question: str,
    answer: str,
    answer_critique: AnswerSupportCritique,
    model: str = "qwen2.5:7b",
    ask_model_fn: Callable[..., str] = ask_model,
) -> UtilityCritique:
    try:
        output = ask_model_fn(
            build_utility_critique_prompt(question, answer, answer_critique),
            model=model,
            system=UTILITY_CRITIC_SYSTEM_PROMPT,
        )
        return parse_utility_critique(output)
    except Exception as exc:
        score = 4 if answer_critique.is_supported else 2
        return UtilityCritique(
            score=score,
            utility_label=f"[Utility:{score}/5]",
            retry_needed=score <= 2,
            retry_query=answer_critique.retry_query or question,
            reason=f"Utility Critic 呼叫失敗，改用 support fallback：{exc}",
        )


def parse_passage_reflections(output: str, chunk_count: int) -> List[PassageReflection]:
    raw_output = str(output or "")
    reflections = {}
    for raw_line in raw_output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.search(r"(?:來源|source)\s*(\d+)", line, flags=re.IGNORECASE)
        if not match:
            continue
        source_index = int(match.group(1))
        if source_index < 1 or source_index > int(chunk_count or 0):
            continue
        reflections[source_index] = PassageReflection(
            source_index=source_index,
            relevance_label=_parse_relevance_label(line),
            support_label=_parse_support_label(line),
            reason=_extract_reason(line) or line,
            raw_output=raw_output,
        )

    parsed = []
    for source_index in range(1, int(chunk_count or 0) + 1):
        parsed.append(
            reflections.get(
                source_index,
                PassageReflection(
                    source_index=source_index,
                    relevance_label="[Irrelevant]",
                    support_label="[Unsupported]",
                    reason="未被 passage critique 輸出涵蓋。",
                    raw_output=raw_output,
                ),
            )
        )
    return parsed


def parse_answer_support_critique(output: str) -> AnswerSupportCritique:
    raw_output = str(output or "")
    support_line = _find_line(raw_output, ("[Supported]", "是否支撐", "支撐"))
    support_label = _parse_answer_support_label(support_line or raw_output)
    is_supported = support_label == "[Supported]"
    utility_score = _extract_utility_score_or_none(raw_output)
    if utility_score is None:
        utility_score = 4 if is_supported else 2
    retry_line = _find_line(raw_output, ("需要重新檢索", "retry"))
    retry_needed = _extract_yes_no(retry_line, default=not is_supported)
    retry_query = _extract_marker_value(raw_output, ("重新檢索查詢", "retry query"))
    reason = _extract_marker_value(raw_output, ("判斷理由", "理由", "reason")) or "未提供判斷理由。"
    return AnswerSupportCritique(
        support_label=support_label,
        is_supported=is_supported,
        retry_needed=retry_needed,
        retry_query=retry_query,
        reason=reason,
        utility_score=utility_score,
        utility_label=f"[Utility:{utility_score}/5]",
        raw_output=raw_output,
    )


def parse_utility_critique(output: str) -> UtilityCritique:
    raw_output = str(output or "")
    score = _extract_utility_score(raw_output)
    retry_line = _find_line(raw_output, ("需要重新檢索", "retry"))
    retry_needed = _extract_yes_no(retry_line, default=score <= 2)
    retry_query = _extract_marker_value(raw_output, ("重新檢索查詢", "retry query"))
    reason = _extract_marker_value(raw_output, ("判斷理由", "理由", "reason")) or "未提供判斷理由。"
    return UtilityCritique(
        score=score,
        utility_label=f"[Utility:{score}/5]",
        retry_needed=retry_needed,
        retry_query=retry_query,
        reason=reason,
        raw_output=raw_output,
    )


def utility_from_answer_support_critique(answer_critique: AnswerSupportCritique) -> UtilityCritique:
    score = int(answer_critique.utility_score or (4 if answer_critique.is_supported else 2))
    score = max(1, min(5, score))
    return UtilityCritique(
        score=score,
        utility_label=answer_critique.utility_label or f"[Utility:{score}/5]",
        retry_needed=bool(answer_critique.retry_needed or score <= 2),
        retry_query=answer_critique.retry_query,
        reason=answer_critique.reason,
        raw_output=answer_critique.raw_output,
    )


def _fallback_passage_reflections(question: str, chunks: List[Chunk], reason: str) -> List[PassageReflection]:
    terms = _question_terms(question)
    reflections = []
    for index, chunk in enumerate(chunks, start=1):
        haystack = _compact_text(
            f"{chunk.get('parent_title', '')} {chunk.get('title', '')} {chunk.get('content', '')}"
        )
        matched = bool(terms and any(term in haystack for term in terms))
        reflections.append(
            PassageReflection(
                source_index=index,
                relevance_label="[Relevant]" if matched else "[Irrelevant]",
                support_label="[Supported]" if matched else "[Unsupported]",
                reason=reason,
            )
        )
    return reflections


def _parse_relevance_label(line: str) -> str:
    compact = _compact_text(line)
    lowered = line.lower()
    if "[irrelevant]" in lowered or "不相關" in compact or "無關" in compact or "irrelevant" in lowered:
        return "[Irrelevant]"
    if "部分相關" in compact or "partially relevant" in lowered:
        return "[Partially Relevant]"
    if "[relevant]" in lowered or "相關" in compact or "relevant" in lowered:
        return "[Relevant]"
    return "[Irrelevant]"


def _parse_support_label(line: str) -> str:
    compact = _compact_text(line)
    lowered = line.lower()
    negative_terms = ("[unsupported]", "unsupported", "不支撐", "不支持", "無法支撐", "沒有支撐")
    if any(term in lowered or term in compact for term in negative_terms):
        return "[Unsupported]"
    if "部分支撐" in compact or "部分支持" in compact or "partially supported" in lowered:
        return "[Partially Supported]"
    if "[supported]" in lowered or "支撐" in compact or "支持" in compact or "supported" in lowered:
        return "[Supported]"
    return "[Unsupported]"


def _parse_answer_support_label(text: str) -> str:
    compact = _compact_text(text)
    lowered = text.lower()
    if "[unsupported]" in lowered or "否" in compact or "不支撐" in compact or "無法支撐" in compact:
        return "[Unsupported]"
    if "部分" in compact or "partial" in lowered:
        return "[Partially Supported]"
    if "[supported]" in lowered or "是" in compact or "支撐" in compact or "supported" in lowered:
        return "[Supported]"
    return "[Unsupported]"


def _extract_utility_score(output: str) -> int:
    extracted = _extract_utility_score_or_none(output)
    return extracted if extracted is not None else 3


def _extract_utility_score_or_none(output: str):
    patterns = (
        r"\[Utility\]\s*[：:]\s*([1-5])\s*/\s*5",
        r"Utility\s*[：:=]\s*([1-5])\s*/\s*5",
        r"utility[_\s-]*score\s*[：:=]\s*([1-5])",
        r"([1-5])\s*/\s*5",
    )
    for pattern in patterns:
        match = re.search(pattern, output, flags=re.IGNORECASE)
        if match:
            return max(1, min(5, int(match.group(1))))
    return None


def _extract_reason(line: str) -> str:
    return _extract_marker_value(line, ("理由", "reason"))


def _find_line(output: str, markers) -> str:
    lowered_markers = tuple(str(marker).lower() for marker in markers)
    for line in str(output or "").splitlines():
        lowered = line.lower()
        if any(marker in lowered for marker in lowered_markers):
            return line.strip()
    return ""


def _extract_marker_value(output: str, markers) -> str:
    marker_patterns = "|".join(re.escape(str(marker)) for marker in markers)
    pattern = rf"(?:{marker_patterns})\s*[：:=]\s*(.+)"
    for line in str(output or "").splitlines():
        match = re.search(pattern, line, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def _extract_yes_no(line: str, default: bool = False) -> bool:
    compact = _compact_text(line)
    lowered = str(line or "").lower()
    if any(term in compact or term in lowered for term in ("否", "不需要", "不用", "false", "no")):
        return False
    if any(term in compact or term in lowered for term in ("是", "需要", "true", "yes")):
        return True
    return default


def _question_terms(question: str):
    text = _compact_text(question)
    stopwords = {"什麼", "什么", "請問", "問題", "怎麼", "如何", "的"}
    terms = set()
    for length in (5, 4, 3, 2):
        for index in range(max(0, len(text) - length + 1)):
            term = text[index : index + length]
            if term and term not in stopwords:
                terms.add(term)
    return terms


def _compact_text(text: str) -> str:
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", "", str(text or "")).lower()
