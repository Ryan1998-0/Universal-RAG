import re
from dataclasses import dataclass
from typing import Optional, Sequence


_NUMBER_PATTERN = r"(?:\d+(?:\.\d+)?|[零〇一二兩三四五六七八九十百半]+)"
_DURATION_PATTERN = re.compile(
    rf"(?P<number>{_NUMBER_PATTERN})\s*(?P<unit>年|個月|月)"
)
_LOWER_BOUND_PATTERN = re.compile(
    rf"(?P<number>{_NUMBER_PATTERN})\s*(?P<unit>年|個月|月)\s*(?P<operator>以上|起|超過)"
)
_UPPER_BOUND_PATTERN = re.compile(
    rf"(?P<number>{_NUMBER_PATTERN})\s*(?P<unit>年|個月|月)\s*(?P<operator>未滿|以下|以內)"
)
_THRESHOLD_INTENT_TERMS = (
    "多久",
    "幾天",
    "幾日",
    "預告",
    "通知",
    "休假",
    "特休",
    "資格",
    "適用",
    "上限",
    "下限",
)
_GENERIC_NGRAMS = {
    "什麼",
    "怎麼",
    "如何",
    "是否",
    "有沒",
    "沒有",
    "可以",
    "資料",
    "文件",
    "內容",
    "問題",
    "剛好",
    "正好",
    "以上",
    "未滿",
}


@dataclass(frozen=True)
class ThresholdEvidence:
    rank: int
    title: str
    page: str
    evidence_line: str
    result_text: str
    target_months: float
    lower_months: Optional[float]
    upper_months: Optional[float]
    score: float


def resolve_threshold_evidence(
    question: str,
    contexts: Sequence[dict],
    previous_user_question: str = "",
) -> Optional[ThresholdEvidence]:
    """Resolve a duration threshold from retrieved range statements.

    This is deliberately domain-neutral: it understands units and range words,
    then uses query-to-evidence overlap to choose among competing tables.
    """
    if _asks_for_formula_or_amount(question):
        return None

    target_months = _extract_target_months(question)
    intent_text = str(question or "")
    if not _has_threshold_intent(intent_text) and _is_elliptical_followup(question):
        intent_text = " ".join(
            part for part in (previous_user_question, question) if str(part).strip()
        )
    if target_months is None or not _has_threshold_intent(intent_text):
        return None

    query_ngrams = _meaningful_ngrams(intent_text)
    candidates = []
    for context in contexts:
        title = str(context.get("title") or "")
        page = str(context.get("page") or "")
        rank = _safe_rank(context.get("rank"))
        for line in _evidence_lines(context.get("content")):
            parsed = _parse_range_line(line)
            if parsed is None:
                continue
            lower_months, lower_inclusive, upper_months, upper_inclusive, result_text = parsed
            if not _contains_target(
                target_months,
                lower_months,
                lower_inclusive,
                upper_months,
                upper_inclusive,
            ):
                continue

            evidence_ngrams = _meaningful_ngrams(f"{title} {line}")
            overlap = len(query_ngrams & evidence_ngrams)
            if overlap == 0:
                continue
            score = float(overlap) + (1.0 / max(1, rank))
            candidates.append(
                ThresholdEvidence(
                    rank=rank,
                    title=title,
                    page=page,
                    evidence_line=line,
                    result_text=result_text,
                    target_months=target_months,
                    lower_months=lower_months,
                    upper_months=upper_months,
                    score=score,
                )
            )

    if not candidates:
        return None
    return max(candidates, key=lambda item: (item.score, -item.rank))


def render_threshold_answer(evidence: ThresholdEvidence) -> str:
    return f"依檢索資料，{evidence.evidence_line}\n\n來源：[{evidence.rank}]"


def _extract_target_months(text: str) -> Optional[float]:
    matches = list(_DURATION_PATTERN.finditer(str(text or "")))
    if not matches:
        return None

    last = matches[-1]
    value = _parse_number(last.group("number"))
    if value is None:
        return None
    return value * 12.0 if last.group("unit") == "年" else value


def _has_threshold_intent(text: str) -> bool:
    clean_text = str(text or "")
    return any(term in clean_text for term in _THRESHOLD_INTENT_TERMS)


def _is_elliptical_followup(text: str) -> bool:
    clean_text = re.sub(r"[\s？?。.!！,，、]", "", str(text or ""))
    cjk_length = len(re.findall(r"[\u4e00-\u9fff]", clean_text))
    explicit_subject_terms = ("公式", "計算", "工資", "費用", "金額", "資遣費", "加班費")
    return (
        cjk_length <= 10
        and not any(term in clean_text for term in explicit_subject_terms)
        and bool(re.search(r"^(?:那|如果|若|剛好|正好|已經|滿)", clean_text))
    )


def _asks_for_formula_or_amount(text: str) -> bool:
    clean_text = str(text or "")
    return any(
        term in clean_text
        for term in ("公式", "怎麼計算", "如何計算", "金額", "月平均工資", "總額", "費用是多少")
    )


def _evidence_lines(content: object):
    for raw_line in re.split(r"[\r\n]+", str(content or "")):
        line = raw_line.strip()
        if "：" in line or ":" in line:
            yield line


def _parse_range_line(line: str):
    separator = "：" if "：" in line else ":"
    range_text, result_text = line.split(separator, 1)
    lower_match = _LOWER_BOUND_PATTERN.search(range_text)
    upper_match = _UPPER_BOUND_PATTERN.search(range_text)
    if lower_match is None and upper_match is None:
        return None

    lower_months = _duration_match_to_months(lower_match)
    upper_months = _duration_match_to_months(upper_match)
    lower_inclusive = lower_match is not None and lower_match.group("operator") in {"以上", "起"}
    upper_inclusive = upper_match is not None and upper_match.group("operator") in {"以下", "以內"}
    clean_result = result_text.split("。", 1)[0].strip(" ；;")
    if not clean_result:
        return None
    return lower_months, lower_inclusive, upper_months, upper_inclusive, clean_result


def _duration_match_to_months(match) -> Optional[float]:
    if match is None:
        return None
    value = _parse_number(match.group("number"))
    if value is None:
        return None
    return value * 12.0 if match.group("unit") == "年" else value


def _contains_target(
    target: float,
    lower: Optional[float],
    lower_inclusive: bool,
    upper: Optional[float],
    upper_inclusive: bool,
) -> bool:
    if lower is not None:
        if target < lower or (target == lower and not lower_inclusive):
            return False
    if upper is not None:
        if target > upper or (target == upper and not upper_inclusive):
            return False
    return True


def _meaningful_ngrams(text: str) -> set:
    compact = "".join(re.findall(r"[\u4e00-\u9fff]", str(text or "")))
    ngrams = {
        compact[index:index + size]
        for size in (2, 3)
        for index in range(max(0, len(compact) - size + 1))
    }
    return {term for term in ngrams if term not in _GENERIC_NGRAMS}


def _parse_number(value: str) -> Optional[float]:
    text = str(value or "").strip().replace("兩", "二").replace("〇", "零")
    try:
        return float(text)
    except ValueError:
        pass
    if text == "半":
        return 0.5

    digits = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9}
    total = 0
    current = 0
    for character in text:
        if character in digits:
            current = digits[character]
        elif character == "十":
            total += (current or 1) * 10
            current = 0
        elif character == "百":
            total += (current or 1) * 100
            current = 0
        else:
            return None
    return float(total + current)


def _safe_rank(value: object) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1
