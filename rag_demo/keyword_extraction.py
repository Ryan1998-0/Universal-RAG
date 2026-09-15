import json
import re
from typing import Callable, List, Optional

from rag_demo.model_gateway import resolve_model_for_node
from rag_demo.model_providers import ask_model


KEYWORD_EXTRACTION_SYSTEM_PROMPT = """你是 Keyword Extraction Agent。

你的任務是從使用者問題中抽取適合 knowledge base retrieval 的關鍵詞。
你不回答問題，只輸出 JSON。
"""


def build_keyword_extraction_prompt(question: str, max_keywords: int = 12) -> str:
    return f"""請從使用者問題抽取 retrieval keywords。

規則：
- 只抽取能幫助檢索 knowledge base 的名詞、人物、組織、地點、事件、概念、時間、動作或關係詞。
- 保留原問題中的專有名詞。
- 可以補少量同義詞，但不要加入與問題無關的背景知識。
- 最多輸出 {max_keywords} 個 keywords。
- 只輸出 JSON，不要輸出 Markdown。

JSON 格式：
{{
  "keywords": ["keyword1", "keyword2"]
}}

使用者問題：
{question}
"""


def extract_keywords(
    question: str,
    model: Optional[str] = None,
    ask_model_fn: Callable[..., str] = ask_model,
    max_keywords: int = 12,
) -> List[str]:
    output = ask_model_fn(
        build_keyword_extraction_prompt(question, max_keywords=max_keywords),
        model=resolve_model_for_node(
            "keyword_extraction",
            requested_model=model,
            prefer_requested=bool(model),
        ),
        system=KEYWORD_EXTRACTION_SYSTEM_PROMPT,
    )
    keywords = parse_keywords_output(output)
    if not keywords:
        keywords = fallback_keywords(question)
    return keywords[:max_keywords]


def parse_keywords_output(output: str) -> List[str]:
    payload = _parse_json_object(output)
    if isinstance(payload, dict):
        raw_keywords = payload.get("keywords", [])
    elif isinstance(payload, list):
        raw_keywords = payload
    else:
        raw_keywords = []

    keywords = []
    for item in raw_keywords:
        cleaned = _clean_keyword(str(item))
        if cleaned:
            keywords.append(cleaned)
    return list(dict.fromkeys(keywords))


def fallback_keywords(question: str) -> List[str]:
    terms = []
    for term in re.findall(r"[\u4e00-\u9fffA-Za-z0-9·:：]{2,}", str(question)):
        cleaned = _clean_keyword(term)
        if cleaned and cleaned not in _STOP_TERMS:
            terms.append(cleaned)
    return list(dict.fromkeys(terms))


def _parse_json_object(output: str):
    text = str(output or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return {}


def _clean_keyword(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" \t\r\n,，。！？!?、;；")


_STOP_TERMS = {
    "什麼",
    "為什麼",
    "如何",
    "哪些",
    "哪個",
    "哪一個",
    "問題",
    "提問",
    "回答",
    "說明",
    "分析",
}
