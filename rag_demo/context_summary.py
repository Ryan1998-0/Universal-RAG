import re
from typing import Callable, List, Optional

from rag_demo.chunking import Chunk
from rag_demo.evidence_policy import build_evidence_policy, evidence_category, sequence_number, should_include_evidence_line
from rag_demo.model_providers import ask_model
from rag_demo.model_gateway import resolve_model_for_node
from rag_demo.prompting import render_retrieved_context
from rag_demo.retrieval_planner import extract_focus_terms
from rag_demo.retrieval_verifier import extract_evidence_candidates


CONTEXT_SUMMARY_SYSTEM_PROMPT = """你是 RAG Context Summary Agent。

你的任務是把使用者問題與 retrieval chunks 彙整成 Answer Agent 更容易使用的資料。
你不負責最終回答。

請只根據來源整理，不要新增來源沒有的資訊。
"""


def build_context_summary_prompt(question: str, chunks: List[Chunk]) -> str:
    deterministic_evidence = extract_deterministic_evidence(chunks, question=question)
    return f"""你是 Context Summary Agent。請先整理 retrieval chunks，再交給 Answer Agent 使用。

使用者提問：
{question}

程式已先抽出的 deterministic evidence：
{deterministic_evidence}

來源資料：
{render_retrieved_context(chunks)}

請輸出一份繁體中文彙整資料，格式如下：

## 直接 evidence
- 列出能直接回答使用者問題的原文片段或結構化資料。
- 若來源是「A / B」或 key-value 格式，請保留原格式，不要改寫成上位分類。
- 只保留能直接回答問題的最小資訊。
- 每個 evidence 最多一行。

## 輔助資訊
- 列出只能輔助判斷、不能單獨回答問題的資訊。
- 若沒有必要的輔助資訊，寫「無」。

## 不足或衝突
- 若來源不足、彼此衝突或只能部分回答，請明確列出。
- 若沒有不足或衝突，寫「無」。

規則：
- 不可新增來源沒有的資訊。
- 不可使用一般常識補答案。
- 必須保留「程式已先抽出的 deterministic evidence」中的所有項目，不可刪除、改寫或合併成更粗略的分類。
- 你可以補充 deterministic evidence 沒有覆蓋、但來源資料明確支持的資訊。
- 不要輸出最終答案，只做資料彙整。
- 不要摘錄思考過程、推理過程或長段落，除非使用者明確詢問原因、思考或推理。
- 若問題詢問名單、身份、欄位值或對照關係，優先輸出簡短對照表。
- 輸出只能包含指定的三個區塊，不要在「不足或衝突」之後追加整理、推測或最終答案。
- 總字數請盡量控制在 300 字以內。
"""


def extract_deterministic_evidence(chunks: List[Chunk], max_items: int = 40, question: str = "") -> str:
    evidence_items = []
    policy = build_evidence_policy(question)

    for index, chunk in enumerate(chunks, start=1):
        source_label = _source_label(index, chunk)
        sequence = sequence_number(f"{chunk.get('parent_title', '')}\n{chunk.get('title', '')}")
        candidates = [
            *extract_evidence_candidates(str(chunk.get("content", "")), max_candidates=max_items),
            *extract_query_evidence_candidates(str(chunk.get("content", "")), question, max_candidates=max_items),
        ]
        for candidate in candidates:
            if not _is_deterministic_evidence_line(candidate) and not _matches_query_evidence_terms(candidate, question):
                continue
            if not should_include_evidence_line(candidate, policy):
                continue
            normalized = " ".join(candidate.split())
            if not normalized:
                continue
            evidence_items.append(
                (
                    _evidence_sort_key(candidate, question, sequence, index, len(evidence_items)),
                    normalized,
                    f"- [{source_label}] {candidate}",
                )
            )

    if not evidence_items:
        return "無"
    evidence_items.sort(key=lambda item: item[0])
    deduped_lines = []
    seen = set()
    for _, normalized, line in evidence_items:
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped_lines.append(line)
        if len(deduped_lines) >= max_items:
            break
    return "\n".join(deduped_lines)


def extract_query_evidence_candidates(content: str, question: str, max_candidates: int = 12) -> List[str]:
    terms = _query_evidence_terms(question)
    if not terms:
        return []

    candidates = []
    for raw_line in content.splitlines():
        for sentence in re.split(r"(?<=[。！？!?])", raw_line):
            sentence = sentence.strip()
            if not sentence:
                continue
            if not _matches_query_evidence_terms(sentence, question, terms=terms):
                continue
            candidates.append(_clip_around_query_terms(sentence, terms))
            if len(candidates) >= max_candidates:
                return candidates
    return candidates


def _matches_query_evidence_terms(line: str, question: str, terms: List[str] = None) -> bool:
    terms = terms or _query_evidence_terms(question)
    if not terms:
        return False
    hits = [term for term in terms if term in line]
    return any(len(term) >= 4 for term in hits) or len(hits) >= 2


def _query_evidence_terms(question: str) -> List[str]:
    terms = list(extract_focus_terms(question))
    raw_terms = re.split(r"[\s,，。！？!?、：:；;（）()「」『』《》/\\]+", str(question))
    stopwords = {
        "什麼",
        "為什麼",
        "如何",
        "哪些",
        "哪一項",
        "主要",
        "小說",
        "遊戲",
        "文明",
        "問題",
        "背景",
        "來源",
        "目前",
        "確認",
    }
    for raw_term in raw_terms:
        for term in re.findall(r"[\u4e00-\u9fffA-Za-z0-9·]{2,}", raw_term):
            if term in stopwords:
                continue
            if len(term) > 18:
                continue
            terms.append(term)
    return list(dict.fromkeys(terms))


def _clip_around_query_terms(sentence: str, terms: List[str], max_chars: int = 220) -> str:
    if len(sentence) <= max_chars:
        return sentence

    anchors = [sentence.find(term) for term in terms if sentence.find(term) >= 0]
    anchor = min(anchors) if anchors else -1
    if anchor < 0:
        return sentence[:max_chars].strip()

    start = max(0, anchor - max_chars // 2)
    end = min(len(sentence), start + max_chars)
    start = max(0, end - max_chars)
    clipped = sentence[start:end].strip()
    if start > 0:
        clipped = f"...{clipped}"
    if end < len(sentence):
        clipped = f"{clipped}..."
    return clipped


def _evidence_sort_key(line: str, question: str, sequence: int, chunk_index: int, item_index: int):
    policy = build_evidence_policy(question)
    if policy.temporal_order:
        return (sequence, _evidence_priority(line, question), chunk_index, item_index)
    return (_evidence_priority(line, question), chunk_index, item_index)


def _evidence_priority(line: str, question: str = "") -> int:
    return evidence_category(line, build_evidence_policy(question))


def _is_deterministic_evidence_line(line: str) -> bool:
    stripped = line.strip()
    lowered = stripped.lower()
    if not stripped:
        return False
    if "api call failed" in lowered or "fallback" in lowered or "operation was aborted" in lowered:
        return False
    if _looks_like_narrative_evidence_line(stripped):
        return True
    if len(stripped) > 100:
        return False
    if re.match(r"^[【\[][^】\]]+[】\]]$", stripped):
        return True
    if stripped.startswith("{") and stripped.endswith("}"):
        return True
    if stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2:
        return True
    if re.match(r"^[\-*]\s+\S+", stripped):
        return True
    key_value_match = re.match(r"^([^：:]{1,20})[：:]\s*(\S.*)$", stripped)
    if key_value_match:
        key = key_value_match.group(1).strip()
        if " " in key:
            return False
        if key.startswith(("我", "你", "他", "她", "它", "我們", "你們", "他們")):
            return False
        return True
    return False


def _looks_like_narrative_evidence_line(line: str) -> bool:
    if len(line) > 240:
        return False
    return any(
        term in line
        for term in (
            "幽靈",
            "楔子",
            "第一次進入",
            "首次進入",
            "太陽運行",
            "太陽",
            "恆星",
            "行星",
            "嚴寒",
            "酷熱",
            "脫水",
            "毀滅",
            "災難",
            "危機",
            "光年",
            "世紀",
            "另一個時代",
            "另一個世界",
        )
    )


def _source_label(index: int, chunk: Chunk) -> str:
    title = str(chunk.get("title", "")).strip()
    if title:
        return f"來源 {index} / {title}"
    return f"來源 {index}"


def summarize_retrieved_context(
    question: str,
    chunks: List[Chunk],
    model: Optional[str] = None,
    ask_model_fn: Callable = ask_model,
) -> str:
    if not chunks:
        return "沒有可彙整的來源。"

    deterministic_evidence = extract_deterministic_evidence(chunks, question=question)
    output = ask_model_fn(
        build_context_summary_prompt(question, chunks),
        model=resolve_model_for_node(
            "context_summary",
            requested_model=model,
            prefer_requested=bool(model),
        ),
        system=CONTEXT_SUMMARY_SYSTEM_PROMPT,
    )
    summary = _demote_summary_headings(_strip_summary_overflow(output.strip()))
    return f"""## 程式抽取 evidence（最高優先）
以下每一行都是 deterministic evidence；若它能回答使用者問題，Answer Agent 必須使用，不可因 Summary Agent 漏列而回答無法確認。
{deterministic_evidence}

## Summary Agent 補充整理（僅輔助，不可覆蓋上方程式 evidence）
{summary}"""


def build_deterministic_context_summary(chunks: List[Chunk], question: str = "") -> str:
    deterministic_evidence = extract_deterministic_evidence(chunks, question=question)
    return f"""## 程式抽取 evidence（最高優先）
以下每一行都是 deterministic evidence；若它能回答使用者問題，Answer Agent 必須使用。
{deterministic_evidence}"""


def _demote_summary_headings(summary: str) -> str:
    replacements = {
        "## 直接 evidence": "### Summary Agent 直接 evidence（僅輔助）",
        "## 輔助資訊": "### Summary Agent 輔助資訊",
        "## 不足或衝突": "### Summary Agent 不足或衝突",
    }
    for old, new in replacements.items():
        summary = summary.replace(old, new)
    return summary


def _strip_summary_overflow(summary: str) -> str:
    kept_lines = []
    for line in summary.splitlines():
        if line.strip() in {"---", "***"}:
            break
        kept_lines.append(line)
    return "\n".join(kept_lines).strip()
