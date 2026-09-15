import json
import re
from typing import Callable, Optional

from rag_demo.model_gateway import resolve_model_for_node
from rag_demo.model_providers import ask_model


QUESTION_EXTRACTION_SYSTEM_PROMPT = """你是 Question Extraction Agent。

你的任務是理解使用者真正想問的問題，將口語、含糊或多餘背景整理成清楚的核心問題。
你不回答問題，只輸出 JSON。
"""


def build_question_extraction_prompt(question: str) -> str:
    return f"""請分析使用者真正想問的問題，輸出一個清楚、可回答、適合 QA Agent 使用的 refined_question。

規則：
- 保留使用者原本的語意與限制，不要新增問題中沒有的要求。
- 如果使用者一次問多個子問題，請在 refined_question 中保留這些子問題。
- 如果原問題已經清楚，可以原樣整理。
- 只輸出 JSON，不要輸出 Markdown。

JSON 格式：
{{
  "refined_question": "整理後的真正問題",
  "intent": "簡短描述問題意圖"
}}

使用者原始問題：
{question}
"""


def extract_real_question(
    question: str,
    model: Optional[str] = None,
    ask_model_fn: Callable[..., str] = ask_model,
) -> str:
    output = ask_model_fn(
        build_question_extraction_prompt(question),
        model=resolve_model_for_node(
            "question_extraction",
            requested_model=model,
            prefer_requested=bool(model),
        ),
        system=QUESTION_EXTRACTION_SYSTEM_PROMPT,
    )
    refined_question = parse_question_output(output)
    return refined_question or " ".join(str(question).split())


def parse_question_output(output: str) -> str:
    payload = _parse_json_object(output)
    if not isinstance(payload, dict):
        return ""
    refined = str(payload.get("refined_question", "")).strip()
    return re.sub(r"\s+", " ", refined)


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
