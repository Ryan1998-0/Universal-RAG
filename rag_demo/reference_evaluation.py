from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from time import perf_counter
from typing import Callable, Dict, List, Optional, Sequence


REFERENCE_EVALUATION_SCHEMA = "claude-reference-evaluation-v1"
DEFAULT_CLAUDE_MODEL = "sonnet"
DEFAULT_CLAUDE_TIMEOUT_SECONDS = 240.0
QUALITY_RUBRIC = (
    {
        "id": "grounded_correctness",
        "label": "事實正確與證據一致",
        "max_score": 35,
        "description": "每個實質主張都與檢索證據一致，不杜撰步驟、欄位、數值或結果。",
    },
    {
        "id": "completeness",
        "label": "關鍵資訊完整性",
        "max_score": 25,
        "description": "涵蓋回答問題所需的關鍵步驟、條件與例外，不遺漏重要證據。",
    },
    {
        "id": "citation_traceability",
        "label": "引用與可追溯性",
        "max_score": 15,
        "description": "來源編號有效，而且能直接支持相鄰主張。",
    },
    {
        "id": "instruction_compliance",
        "label": "回答規則遵循",
        "max_score": 10,
        "description": "遵守最終 prompt 的限制、格式、證據不足處理與禁止事項。",
    },
    {
        "id": "actionability",
        "label": "清楚與可操作性",
        "max_score": 10,
        "description": "步驟順序清楚、具體，使用者可以據此完成任務。",
    },
    {
        "id": "language_terminology",
        "label": "繁中與術語品質",
        "max_score": 5,
        "description": "使用自然繁體中文，保留正確介面名稱、代碼與專業術語。",
    },
)


def claude_reference_evaluation_enabled() -> bool:
    value = str(os.getenv("RAG_CLAUDE_REFERENCE_EVALUATION", "0")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def evaluate_qwen_with_claude(
    *,
    question: str,
    final_prompt: str,
    system_prompt: str,
    qwen_answer: str,
    contexts: Sequence[dict],
    claude_model: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
    runner: Optional[Callable[..., dict]] = None,
) -> dict:
    model = str(
        claude_model or os.getenv("RAG_CLAUDE_REFERENCE_MODEL", DEFAULT_CLAUDE_MODEL)
    ).strip() or DEFAULT_CLAUDE_MODEL
    timeout = float(
        timeout_seconds
        or os.getenv("RAG_CLAUDE_REFERENCE_TIMEOUT_SECONDS", DEFAULT_CLAUDE_TIMEOUT_SECONDS)
    )
    run_cli = runner or run_claude_cli
    started_at = perf_counter()

    reference_result = run_cli(
        prompt=final_prompt,
        system=system_prompt,
        model=model,
        timeout_seconds=timeout,
    )
    reference_answer = str(reference_result.get("text") or "").strip()
    if not reference_answer:
        raise RuntimeError("Claude reference answer was empty.")

    judge_result = run_cli(
        prompt=build_judge_prompt(
            question=question,
            contexts=contexts,
            reference_answer=reference_answer,
            qwen_answer=qwen_answer,
        ),
        system=build_judge_system_prompt(),
        model=model,
        timeout_seconds=timeout,
        json_schema=judge_json_schema(),
    )
    judgment = parse_judgment(judge_result.get("text"))
    scored_dimensions = normalize_dimension_scores(judgment.get("dimensions"))
    raw_score = sum(item["score"] for item in scored_dimensions)
    applied_cap = 100
    cap_reasons: List[str] = []
    if judgment.get("contradicts_evidence"):
        applied_cap = min(applied_cap, 49)
        cap_reasons.append("存在與檢索證據矛盾的重大主張")
    if judgment.get("critical_unsupported_claim"):
        applied_cap = min(applied_cap, 59)
        cap_reasons.append("存在會影響操作結果的無證據重大主張")
    if judgment.get("failed_required_abstention"):
        applied_cap = min(applied_cap, 40)
        cap_reasons.append("證據不足時未依規則拒答或標示不足")
    final_score = min(raw_score, applied_cap)

    return {
        "schema_version": REFERENCE_EVALUATION_SCHEMA,
        "status": "completed",
        "reference": {
            "provider": "claude-cli",
            "model": model,
            "score": 100,
            "answer": reference_answer,
            "duration_ms": reference_result.get("duration_ms"),
        },
        "candidate": {
            "provider": "ollama",
            "label": "Qwen",
            "score": final_score,
            "raw_score": raw_score,
        },
        "dimensions": scored_dimensions,
        "score_cap": {
            "applied": applied_cap < 100,
            "maximum": applied_cap,
            "reasons": cap_reasons,
        },
        "summary": str(judgment.get("summary") or "").strip(),
        "strengths": _string_list(judgment.get("strengths"), limit=5),
        "improvements": _string_list(judgment.get("improvements"), limit=5),
        "method": {
            "reference_baseline": 100,
            "reference_is_normalized_baseline": True,
            "evidence_has_priority_over_reference_style": True,
            "judge_provider": "claude-cli",
            "judge_model": model,
            "final_prompt_sha256": hashlib.sha256(final_prompt.encode("utf-8")).hexdigest(),
            "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        },
        "timings": {
            "reference_ms": reference_result.get("duration_ms"),
            "judge_ms": judge_result.get("duration_ms"),
            "total_ms": round((perf_counter() - started_at) * 1000.0, 2),
        },
    }


def unavailable_reference_evaluation(reason: str) -> dict:
    return {
        "schema_version": REFERENCE_EVALUATION_SCHEMA,
        "status": "unavailable",
        "reason": str(reason or "Claude reference evaluation is unavailable."),
        "reference": {"provider": "claude-cli", "score": 100},
        "candidate": {"provider": "ollama", "label": "Qwen", "score": None},
        "dimensions": [],
    }


def build_judge_system_prompt() -> str:
    return """你是嚴格的 RAG 回答品質評審。
檢索證據是事實最高準據；Claude 參考答案只作為完整性與表達的 100 分相對基準。
不可因候選答案與參考答案措辭不同而扣分，也不可把參考答案中無證據支持的內容視為正確。
請逐項依評分上限給整數分數，並誠實標記重大錯誤。只輸出符合 JSON Schema 的物件。"""


def build_judge_prompt(
    *,
    question: str,
    contexts: Sequence[dict],
    reference_answer: str,
    qwen_answer: str,
) -> str:
    evidence = "\n\n".join(
        f"[{item.get('rank', index)}] {item.get('title', '')} {item.get('page', '')}\n"
        f"{str(item.get('content') or '')}"
        for index, item in enumerate(contexts, start=1)
    )
    rubric_text = "\n".join(
        f"- {item['id']}（0-{item['max_score']}）：{item['label']}；{item['description']}"
        for item in QUALITY_RUBRIC
    )
    return f"""### 使用者問題
{question}

### 檢索證據
{evidence}

### Claude 參考答案（基準正規化為 100 分）
{reference_answer}

### Qwen 候選答案
{qwen_answer}

### 固定評分標準
{rubric_text}

請評估 Qwen，不要重寫答案。每個 dimensions 項目必須使用上述 id，分數不得超過該項上限。
重大旗標定義：
- contradicts_evidence：候選答案有會改變結論或操作的內容與證據明確矛盾。
- critical_unsupported_claim：候選答案加入會影響操作、數值或結論但沒有證據支持的重大內容。
- failed_required_abstention：檢索證據不足，候選答案卻沒有依最終 prompt 說明無法確認。
"""


def judge_json_schema() -> dict:
    dimension_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "score": {"type": "integer"},
            "reason": {"type": "string"},
        },
        "required": ["id", "score", "reason"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "dimensions": {
                "type": "array",
                "items": dimension_schema,
                "minItems": len(QUALITY_RUBRIC),
                "maxItems": len(QUALITY_RUBRIC),
            },
            "contradicts_evidence": {"type": "boolean"},
            "critical_unsupported_claim": {"type": "boolean"},
            "failed_required_abstention": {"type": "boolean"},
            "summary": {"type": "string"},
            "strengths": {"type": "array", "items": {"type": "string"}},
            "improvements": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "dimensions",
            "contradicts_evidence",
            "critical_unsupported_claim",
            "failed_required_abstention",
            "summary",
            "strengths",
            "improvements",
        ],
        "additionalProperties": False,
    }


def run_claude_cli(
    *,
    prompt: str,
    system: str,
    model: str,
    timeout_seconds: float,
    json_schema: Optional[dict] = None,
) -> dict:
    executable = shutil.which("claude")
    if not executable:
        raise RuntimeError("找不到 Claude Code CLI，無法建立 Claude 參考答案。")
    command = [
        executable,
        "--print",
        "--output-format",
        "json",
        "--model",
        model,
        "--tools",
        "",
        "--no-session-persistence",
        "--permission-mode",
        "dontAsk",
        "--system-prompt",
        system,
    ]
    if json_schema is not None:
        command.extend(["--json-schema", json.dumps(json_schema, ensure_ascii=False)])
    completed = subprocess.run(
        command,
        input=prompt,
        text=True,
        capture_output=True,
        timeout=max(1.0, float(timeout_seconds)),
        check=False,
    )
    if completed.returncode != 0:
        error = _friendly_claude_error(completed.stdout, completed.stderr)
        raise RuntimeError(error)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Claude CLI 未回傳有效 JSON。") from exc
    if payload.get("is_error"):
        raise RuntimeError(f"Claude CLI 回報錯誤：{payload.get('result') or 'unknown error'}")
    return {
        "text": str(payload.get("result") or "").strip(),
        "duration_ms": payload.get("duration_ms"),
        "model_usage": payload.get("modelUsage") or payload.get("usage") or {},
    }


def parse_judgment(raw: object) -> dict:
    if isinstance(raw, dict):
        return raw
    text = str(raw or "").strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError("Claude judge 未回傳有效 JSON 評分。") from exc
    if not isinstance(value, dict):
        raise RuntimeError("Claude judge 評分必須是 JSON 物件。")
    return value


def _friendly_claude_error(stdout: str, stderr: str) -> str:
    for candidate in (stdout, stderr):
        try:
            payload = json.loads(str(candidate or "").strip())
        except json.JSONDecodeError:
            continue
        status = payload.get("api_error_status")
        result = str(payload.get("result") or "")
        if status == 401 or "token has expired" in result.lower():
            return "Claude Code 登入已過期，請執行 claude auth login 後重試。"
        if result:
            return f"Claude Code 無法完成評測：{result[:240]}"
    clean_error = str(stderr or stdout or "").strip()
    if clean_error:
        return f"Claude Code 無法完成評測：{clean_error[:240]}"
    return "Claude Code 無法完成評測，請檢查登入與網路狀態。"


def normalize_dimension_scores(raw_dimensions: object) -> List[Dict[str, object]]:
    by_id = {
        str(item.get("id")): item
        for item in (raw_dimensions or [])
        if isinstance(item, dict)
    }
    normalized = []
    for rubric_item in QUALITY_RUBRIC:
        raw = by_id.get(rubric_item["id"], {})
        try:
            score = int(raw.get("score", 0))
        except (TypeError, ValueError):
            score = 0
        score = max(0, min(score, int(rubric_item["max_score"])))
        normalized.append({
            **rubric_item,
            "score": score,
            "reason": str(raw.get("reason") or "未提供評分理由。").strip(),
        })
    return normalized


def _string_list(value: object, limit: int) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()][:limit]
