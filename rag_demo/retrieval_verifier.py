import json
import re
from typing import Dict, List, Optional

from rag_demo.chunking import Chunk
from rag_demo.config import RagConfig
from rag_demo.evidence_policy import answerable_event_spans, has_answerable_event_evidence
from rag_demo.model_gateway import resolve_model_for_node
from rag_demo.model_providers import ask_model
from rag_demo.retrieval_planner import extract_focus_terms


VERIFIER_SYSTEM_PROMPT = """你是 RAG retrieval verifier agent。

你的任務是驗證 retrieval chunks 是否包含足夠 evidence 回答使用者問題。
你不負責回答使用者問題。
你必須先找 evidence，再判斷是否可回答。

請只輸出 JSON，不要輸出 Markdown。
"""


def build_verifier_prompt(
    question: str,
    chunks: List[Chunk],
    context_chars: int = None,
    evidence_summary: str = "",
) -> str:
    config = RagConfig.from_env()
    context_chars = context_chars or config.verifier_context_chars
    evidence_section = ""
    if evidence_summary:
        evidence_section = f"""deterministic evidence：
{evidence_summary}

"""
    return f"""請驗證 retrieval chunks 是否包含足夠 evidence 回答使用者問題。

請先找 evidence，再判斷是否可回答。

重要判斷原則：
- 「相關」不等於「足以回答」；只有找到可支持答案的 evidence，is_answerable 才能是 true。
- evidence 可以出現在正文、標題、章節名稱、metadata、表格列、列表、條列項、檔名、欄位名稱、角色標籤、代號標籤或結構化資料中。
- 如果問題問的是名單、身份、分類、欄位、表格值或對照關係，標題、列表、表格列、metadata、角色標籤都可以是有效 evidence。
- 如果 chunks 只是語意相近，但沒有提供能回答問題的 evidence，is_answerable 為 false。
- 如果 is_answerable 為 false，最終系統會回答「無正確來源資料」或進入 fallback。

輸出 JSON 格式：
{{
  "is_answerable": true,
  "confidence": 0.0,
  "evidence_spans": ["直接支持答案的原文片段"],
  "missing_info": ["仍缺少的資訊，若沒有則輸出空陣列"],
  "reason": "簡短原因"
}}

使用者問題：
{question}

{evidence_section}請優先檢查 deterministic evidence。若 deterministic evidence 已直接支持答案，請勿因 retrieval chunks 文字較長而誤判為不可回答。

retrieval chunks：
{render_verifier_context(chunks, context_chars=context_chars)}
"""


def verify_retrieval(
    question: str,
    chunks: List[Chunk],
    model: Optional[str] = None,
    ask_model_fn=ask_model,
    config: RagConfig = None,
    evidence_summary: str = "",
) -> Dict[str, object]:
    config = config or RagConfig.from_env()
    if not chunks:
        return {
            "is_related": False,
            "is_answerable": False,
            "confidence": 0.0,
            "evidence_spans": [],
            "missing_info": ["no retrieved chunks"],
            "reason": "no retrieved chunks",
        }

    deterministic_decision = assess_deterministic_evidence(question, evidence_summary)
    if deterministic_decision is not None:
        return deterministic_decision

    confidence_decision = assess_retrieval_confidence(chunks, config)
    if confidence_decision is not None:
        return confidence_decision

    output = ask_model_fn(
        build_verifier_prompt(
            question,
            chunks,
            context_chars=config.verifier_context_chars,
            evidence_summary=evidence_summary,
        ),
        model=resolve_model_for_node(
            "evidence_extraction",
            requested_model=model,
            prefer_requested=bool(model),
        ),
        system=VERIFIER_SYSTEM_PROMPT,
    )
    return parse_verifier_output(output)


def assess_deterministic_evidence(question: str, evidence_summary: str):
    narrative_decision = assess_deterministic_narrative_evidence(question, evidence_summary)
    if narrative_decision is not None:
        return narrative_decision

    query_evidence_decision = assess_query_evidence_overlap(question, evidence_summary)
    if query_evidence_decision is not None:
        return query_evidence_decision

    if has_answerable_event_evidence(question, evidence_summary):
        return {
            "is_related": True,
            "is_answerable": True,
            "confidence": 0.95,
            "evidence_spans": answerable_event_spans(evidence_summary),
            "missing_info": [],
            "reason": "deterministic_evidence: structured event fields directly support the answer",
        }

    return None


def assess_query_evidence_overlap(question: str, evidence_summary: str):
    lines = [line.strip() for line in str(evidence_summary).splitlines() if line.strip().startswith("- ")]
    if not lines:
        return None

    terms = _query_evidence_terms(question)
    spans = [line for line in lines if _matches_query_evidence_terms(line, terms)]
    if not spans:
        return None

    return {
        "is_related": True,
        "is_answerable": True,
        "confidence": 0.85,
        "evidence_spans": spans[:5],
        "missing_info": [],
        "reason": "deterministic_evidence: query and alias terms overlap retrieved evidence",
    }


def _query_evidence_terms(question: str):
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


def _matches_query_evidence_terms(line: str, terms) -> bool:
    if not terms:
        return False
    hits = [term for term in terms if term in line]
    return any(len(term) >= 4 for term in hits) or len(hits) >= 2


def assess_deterministic_narrative_evidence(question: str, evidence_summary: str):
    compact_question = re.sub(r"\s+", "", str(question))
    hazard_decision = _assess_first_scene_environment_hazard_evidence(compact_question, evidence_summary)
    if hazard_decision is not None:
        return hazard_decision
    return None


def _assess_first_scene_environment_hazard_evidence(compact_question: str, evidence_summary: str):
    if not _is_first_scene_environment_hazard_question(compact_question):
        return None

    lines = [line.strip() for line in str(evidence_summary).splitlines() if line.strip().startswith("- ")]
    spans = [
        line
        for line in lines
        if _contains_environment_hazard_evidence(line)
    ]
    if not spans:
        return None

    has_world_state = any(term in evidence_summary for term in ("世界", "文明", "環境", "紀元", "星系", "行星"))
    has_hazard = any(term in evidence_summary for term in _ENVIRONMENT_HAZARD_TERMS)
    has_astronomical = any(term in evidence_summary for term in _ASTRONOMICAL_HAZARD_TERMS)
    if not (has_world_state and has_hazard and has_astronomical):
        return None

    return {
        "is_related": True,
        "is_answerable": True,
        "confidence": 0.95,
        "evidence_spans": spans[:5],
        "missing_info": [],
        "reason": "deterministic_evidence: first-scene environment hazard evidence directly supports the answer",
    }


def _is_first_scene_environment_hazard_question(compact_question: str) -> bool:
    has_first_entry = any(term in compact_question for term in ("第一次", "首次", "第一回", "初次"))
    has_entry_action = any(term in compact_question for term in ("進入", "登入", "來到", "抵達", "到達"))
    has_scene_or_world = any(term in compact_question for term in ("遊戲", "世界", "場景", "文明", "星球", "星系", "環境"))
    has_hazard_intent = any(term in compact_question for term in ("災難", "危機", "困境", "風險", "面臨", "遭遇", "天文", "生存"))
    return has_first_entry and has_entry_action and has_scene_or_world and has_hazard_intent


def _contains_environment_hazard_evidence(line: str) -> bool:
    return any(term in line for term in _ENVIRONMENT_HAZARD_TERMS) or any(
        term in line for term in _ASTRONOMICAL_HAZARD_TERMS
    )


def assess_retrieval_confidence(chunks: List[Chunk], config: RagConfig = None):
    config = config or RagConfig.from_env()
    if not chunks:
        return {
            "is_related": False,
            "is_answerable": False,
            "confidence": 0.0,
            "evidence_spans": [],
            "missing_info": ["no retrieved chunks"],
            "reason": "no retrieved chunks",
        }

    top = chunks[0]
    score = float(top.get("score", 0.0))
    keyword_score = float(top.get("keyword_score", 0.0))
    embedding_score = float(top.get("embedding_score", 0.0))

    has_strong_signal = (
        keyword_score >= config.verifier_min_keyword_score
        or embedding_score >= config.verifier_min_embedding_score
    )
    if config.verifier_auto_accept_enabled and score >= config.verifier_auto_accept_score and has_strong_signal:
        return {
            "is_related": True,
            "is_answerable": True,
            "confidence": min(1.0, score),
            "evidence_spans": [],
            "missing_info": [],
            "reason": (
                "auto_accept: retrieval confidence gate passed "
                f"(score={score:.4f}, keyword={keyword_score:.1f}, embedding={embedding_score:.4f})"
            ),
        }

    if (
        score <= config.verifier_auto_reject_score
        and keyword_score <= 0
        and embedding_score < config.verifier_min_embedding_score
    ):
        return {
            "is_related": False,
            "is_answerable": False,
            "confidence": max(0.0, score),
            "evidence_spans": [],
            "missing_info": ["retrieval confidence too low"],
            "reason": (
                "auto_reject: retrieval confidence gate failed "
                f"(score={score:.4f}, keyword={keyword_score:.1f}, embedding={embedding_score:.4f})"
            ),
        }

    return None


def render_verifier_context(chunks: List[Chunk], context_chars: int = 600) -> str:
    if not chunks:
        return "沒有找到相關來源。"

    sources = []
    for index, chunk in enumerate(chunks, start=1):
        content = str(chunk.get("content", ""))
        clipped = _clip_for_verifier(content, context_chars)
        candidates = extract_evidence_candidates(content)
        candidate_text = "\n".join(f"- {item}" for item in candidates) if candidates else "無"
        sources.append(
            "\n".join(
                [
                    f"[來源 {index}]",
                    f"文件：{chunk.get('source', '')}",
                    f"父層章節：{chunk.get('parent_title', '')}",
                    f"章節：{chunk.get('title', '')}",
                    f"score：{chunk.get('score', '')}",
                    f"keyword_score：{chunk.get('keyword_score', '')}",
                    f"embedding_score：{chunk.get('embedding_score', '')}",
                    "evidence 候選：",
                    candidate_text,
                    "內容節錄：",
                    clipped,
                ]
            )
        )

    return "\n\n".join(sources)


def extract_evidence_candidates(content: str, max_candidates: int = 12) -> List[str]:
    candidates = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _is_evidence_candidate_line(line):
            candidates.append(line)
        else:
            candidates.extend(_extract_narrative_evidence_sentences(line, max_candidates - len(candidates)))
        if len(candidates) >= max_candidates:
            break
    return candidates


def _is_evidence_candidate_line(line: str) -> bool:
    if len(line) > 120:
        return False
    if re.match(r"^#{1,6}\s+\S+", line):
        return True
    if re.match(r"^[\-*]\s+\S+", line):
        return True
    if line.startswith("|") and line.endswith("|") and line.count("|") >= 2:
        return True
    if re.match(r"^[【\[][^】\]]+[】\]]$", line):
        return True
    if re.match(r"^[^：:]{1,30}[：:]\s*\S+", line):
        return True
    return False


def _extract_narrative_evidence_sentences(line: str, remaining_slots: int) -> List[str]:
    if remaining_slots <= 0:
        return []
    candidates = []
    for sentence in re.split(r"(?<=[。！？；])", line):
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence = _clip_narrative_sentence(sentence)
        if _looks_like_narrative_evidence(sentence):
            candidates.append(sentence)
        if len(candidates) >= remaining_slots:
            break
    return candidates


def _clip_narrative_sentence(sentence: str, max_chars: int = 220) -> str:
    if len(sentence) <= max_chars:
        return sentence

    anchor = _narrative_clip_anchor(sentence)
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


def _narrative_clip_anchor(sentence: str) -> int:
    for term in (
        "另一個時代",
        "另一個世界",
        "第一次進入",
        "首次進入",
        "太陽運行",
        "嚴寒",
        "酷熱",
        "脫水",
        "毀滅",
        "光年",
        "世紀",
    ):
        index = sentence.find(term)
        if index >= 0:
            return index
    return -1


def _looks_like_narrative_evidence(sentence: str) -> bool:
    return any(
        term in sentence
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


_ENVIRONMENT_HAZARD_TERMS = (
    "災難",
    "危機",
    "困境",
    "毀滅",
    "嚴寒",
    "酷熱",
    "高溫",
    "低溫",
    "寒冷",
    "火海",
    "脫水",
    "生存",
)


_ASTRONOMICAL_HAZARD_TERMS = (
    "天文",
    "太陽",
    "恆星",
    "行星",
    "星系",
    "日出",
    "日落",
    "太陽運行",
    "氣候",
    "隕石",
    "撕裂",
    "吞噬",
)


def _clip_for_verifier(content: str, context_chars: int) -> str:
    if len(content) <= context_chars:
        return content

    head_chars = max(1, context_chars // 2)
    tail_chars = max(1, context_chars - head_chars)
    return (
        f"{content[:head_chars]}\n"
        "...（內容已截斷，保留開頭與結尾供 verifier 判斷相關性）\n"
        f"{content[-tail_chars:]}"
    )


def parse_verifier_output(output: str) -> Dict[str, object]:
    text = output.strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        lowered = output.lower()
        related = "true" in lowered and "false" not in lowered
        return {
            "is_related": related,
            "is_answerable": related,
            "confidence": 0.5 if related else 0.0,
            "evidence_spans": [],
            "missing_info": [],
            "reason": output.strip(),
        }

    is_answerable = parsed.get("is_answerable")
    if is_answerable is None:
        is_answerable = parsed.get("is_related")
    is_answerable = bool(is_answerable)
    evidence_spans = parsed.get("evidence_spans", [])
    if not isinstance(evidence_spans, list):
        evidence_spans = [str(evidence_spans)]
    missing_info = parsed.get("missing_info", [])
    if not isinstance(missing_info, list):
        missing_info = [str(missing_info)]

    return {
        "is_related": is_answerable,
        "is_answerable": is_answerable,
        "confidence": float(parsed.get("confidence", 0.0)),
        "evidence_spans": [str(item) for item in evidence_spans],
        "missing_info": [str(item) for item in missing_info],
        "reason": str(parsed.get("reason", "")),
    }
