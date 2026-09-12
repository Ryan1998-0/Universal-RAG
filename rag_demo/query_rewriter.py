import os
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from rag_demo.config import RagConfig
from rag_demo.embeddings import DEFAULT_EMBEDDING_MODEL, embed_texts
from rag_demo.general_answer import CURRENT_DATETIME_REASON, is_current_datetime_question
from rag_demo.model_providers import ask_model


QUERY_REWRITER_SYSTEM_PROMPT = """你是 RAG 系統的 query-only 問題分析與檢索規劃器。

你只有兩項可用能力：
1. 讓本機模型直接回答。
2. 呼叫一個通用檢索器尋找外部證據。

在檢索發生前，你不知道知識庫有哪些文件、標題、章節或內容，也不可猜測。你的任務不是回答問題，而是只根據「目前問題」判斷是否需要外部證據；需要時，先重述問題、拆成可驗證子問題，再產生多個互補且不依賴語料清單的檢索查詢。

請遵守：
- 使用繁體中文。
- 不要聲稱某份文件或某個章節一定存在。
- 問題需要私有資料、上傳文件、精確來源、時效性資訊、專門事實或模型不應憑記憶確定的資訊時，選擇檢索。
- 模型記憶不算外部證據。涉及法律、政策、標準、特定任務、產品、機構，或要求精確日期、數值、限制、期限、距離時，即使答案看似熟悉，也選擇檢索。
- 寒暄、改寫、翻譯、摘要使用者已完整提供的文字、基本計算、系統日期時間、穩定的一般常識與不需要外部證據的推理，選擇直接回答。
- 「穩定的一般常識」只包含不要求特定來源或精確專業數值的基礎知識；不可把法規要求、任務數據或產品規格歸類為一般常識。
- 不確定能否可靠直接回答時，選擇檢索；檢索後會由另一個節點判斷證據是否足夠。
- 需要檢索時，可以加入由問題本身推導出的同義詞與上位概念，但不可使用未知的文件標題或章節名稱。
- 改寫時必須保留問題中的具體動作、對象與限制，再補上口語動作的中性同義詞；不可把查詢壓縮成只有「合法性、規定、資料、要求」等抽象詞。
- 子問題必須對應回答所需的不同證據欄位，例如對象、條件、數值、期間、例外或程序；簡單問題不可為了湊數而虛構子問題。
- 產生 2 至 5 個互補查詢：至少包含原意完整查詢、精確關鍵詞查詢、答案承載詞查詢；不得只改變語序製造重複查詢。
- 對話背景只可補回目前問題省略的主題，不可沿用先前助理的答案或把答案內容塞進查詢。
- 不要直接回答使用者問題。
"""


@dataclass(frozen=True)
class QueryRewriteDecision:
    needs_retrieval: bool
    reason: str
    retrieval_query: str
    raw_output: str = ""
    sub_questions: Tuple[str, ...] = ()
    query_variants: Tuple[str, ...] = ()
    semantic_similarity: Optional[float] = None
    semantic_accepted: bool = True
    semantic_validation_status: str = "skipped"
    semantic_validation_reason: str = ""
    semantic_validation_threshold: Optional[float] = None


def build_rewrite_prompt(question: str, conversation_context: str = "") -> str:
    context_text = str(conversation_context or "").strip() or "- 無"

    return f"""請只根據目前問題判斷是否需要外部檢索，再在需要時拆解問題並產生適合 embedding / hybrid search 的多個查詢。

要求：
1. 用一句話摘要使用者真正想問的內容，不要展示思考過程。
2. 判斷可靠回答是否需要目前問題之外的外部證據。
3. 需要檢索：私有或上傳資料、精確文件內容與引文、近期或會變動資訊、專門事實、法律與政策要求、特定任務或產品的數據，或明確要求搜尋與來源。模型曾看過或似乎記得，不代表不需要證據。
4. 不需要檢索：寒暄、系統能力、基本算術、日期時間、純文字處理、使用者已提供完成任務所需的全部內容，或不要求精確專業細節的穩定基礎常識。
5. 對話背景只能解析代名詞或省略；不可因舊問題提到某個領域，就把無關的目前問題送去檢索。
6. 你不知道可檢索資料有哪些文件、標題或章節，不可推測或引用語料清單。
7. 需要檢索時，列出 1 至 4 個可獨立驗證的子問題；簡單問題只列 1 個。
8. 需要檢索時，列出 2 至 5 個互補查詢，其中至少有：完整語意、精確詞彙、答案承載詞（例如數值、期間、條件、程序）。
9. 「向量檢索用查詢」填入最完整的主要查詢，並同時列出其餘檢索查詢。
10. 不需要檢索時，不列子問題與額外查詢，向量檢索用查詢保留目前問題。
11. 不要回答問題，只產生決策與檢索文字。

口語改寫示例（只示範語意展開，不代表知識庫一定有相關文件）：
- 「老闆要我晚一點下班，這樣可以嗎？」可改寫為「晚下班 延長工作時間 加班 工時限制 是否允許」。
- 「做了幾年後被公司叫走，可以拿什麼？」可改寫為「工作年資 公司終止工作關係 資遣 可領給付」。
- 不良改寫：「公司要求合法性」。這會遺失原問題的具體動作，禁止使用。

輸出格式：
語意理解：...
是否需要檢索：是 / 否
判斷理由：...
子問題1：...
子問題2：...
向量檢索用查詢：...
檢索查詢1：...
檢索查詢2：...
檢索查詢3：...

對話背景（只用於解析目前問題中的代名詞或省略，不可把舊問題當成目前問題）：
{context_text}

使用者問題：
{question}
"""


def rewrite_query_for_retrieval(
    question: str,
    model: str = "qwen2.5:7b",
) -> str:
    return decide_and_rewrite_query_for_retrieval(
        question,
        model=model,
    ).retrieval_query


def decide_and_rewrite_query_for_retrieval(
    question: str,
    model: str = "qwen2.5:7b",
    conversation_context: str = "",
) -> QueryRewriteDecision:
    decision_question = strip_greeting_prefix(question)
    deterministic_decision = deterministic_retrieval_decision(decision_question, original_question=question)
    if deterministic_decision is not None and not deterministic_decision.needs_retrieval:
        return deterministic_decision

    output = ask_model(
        build_rewrite_prompt(
            decision_question,
            conversation_context=conversation_context,
        ),
        model=model,
        system=QUERY_REWRITER_SYSTEM_PROMPT,
    )
    retrieval_queries = extract_retrieval_queries(output)
    retrieval_query = extract_retrieval_query(output) or (
        retrieval_queries[0] if retrieval_queries else decision_question or question
    )
    sanitized_query = sanitize_retrieval_query(question, retrieval_query)
    needs_retrieval = extract_needs_retrieval(output)
    reason = extract_decision_reason(output)
    if deterministic_decision is not None and deterministic_decision.needs_retrieval:
        needs_retrieval = True
        reason = deterministic_decision.reason
    acronym_reason = ambiguous_acronym_definition_reason(decision_question)
    if acronym_reason:
        needs_retrieval = True
        reason = acronym_reason
    candidate_queries = [
        sanitized_query,
        *(
            sanitize_retrieval_query(question, query)
            for query in retrieval_queries
            if str(query).strip()
        ),
    ]
    semantic_validation = validate_rewrite_semantics(
        original_question=question,
        rewritten_queries=candidate_queries,
    )
    validated_queries = semantic_validation["queries"]
    validated_acceptance = semantic_validation["accepted"]
    primary_similarity = semantic_validation["similarities"][0] if semantic_validation["similarities"] else None
    primary_accepted = bool(validated_acceptance[0]) if validated_acceptance else True
    accepted_queries = [
        query
        for query, accepted in zip(validated_queries[1:], validated_acceptance[1:])
        if accepted
    ]
    primary_query = validated_queries[0] if primary_accepted and validated_queries else question
    if not primary_accepted:
        primary_query = question
    accepted_queries = list(dict.fromkeys([primary_query, *accepted_queries]))
    return QueryRewriteDecision(
        needs_retrieval=needs_retrieval,
        reason=reason,
        retrieval_query=primary_query,
        raw_output=output,
        sub_questions=tuple(extract_sub_questions(output)),
        query_variants=tuple(accepted_queries),
        semantic_similarity=primary_similarity,
        semantic_accepted=primary_accepted,
        semantic_validation_status=semantic_validation["status"],
        semantic_validation_reason=semantic_validation["reason"],
        semantic_validation_threshold=semantic_validation["threshold"],
    )


def validate_rewrite_semantics(
    original_question: str,
    rewritten_queries: Sequence[str],
    min_similarity: Optional[float] = None,
    enabled: Optional[bool] = None,
    embedding_fn: Optional[Callable[..., Sequence[Sequence[float]]]] = None,
    model_name: Optional[str] = None,
) -> dict:
    """Keep only rewrites semantically close to the user's original query.

    The primary rewrite and every auxiliary query are compared in one batched
    embedding call. A candidate below the configured cosine threshold is
    discarded; the original question is retained as the safe retrieval
    fallback. If the embedding service is unavailable, the rewrite is kept and
    the result is marked ``unavailable`` so a temporary model outage does not
    disable retrieval entirely.
    """

    config = RagConfig.from_env().normalized()
    is_enabled = config.query_rewrite_semantic_enabled if enabled is None else bool(enabled)
    threshold = (
        config.query_rewrite_min_similarity
        if min_similarity is None
        else min(1.0, max(-1.0, float(min_similarity)))
    )
    threshold = min(1.0, max(-1.0, float(threshold)))
    original = str(original_question or "").strip()
    candidates = list(dict.fromkeys(str(query or "").strip() for query in rewritten_queries if str(query or "").strip()))
    if not candidates:
        candidates = [original]
    if not is_enabled:
        return {
            "status": "disabled",
            "reason": "semantic query rewrite validation is disabled",
            "threshold": threshold,
            "queries": candidates,
            "similarities": [None for _ in candidates],
            "accepted": [True for _ in candidates],
        }

    similarities: List[Optional[float]] = []
    needs_embedding = []
    for index, candidate in enumerate(candidates):
        if _compact_query(original) == _compact_query(candidate):
            similarities.append(1.0)
        else:
            similarities.append(None)
            needs_embedding.append(index)
    if needs_embedding:
        try:
            embedder = embedding_fn or embed_texts
            texts = [original, *(candidates[index] for index in needs_embedding)]
            try:
                vectors = embedder(
                    texts,
                    model_name=(model_name or os.getenv("RAG_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL),
                )
            except TypeError:
                # Small test doubles and custom adapters may accept only the
                # text sequence; the production embed_texts supports model_name.
                vectors = embedder(texts)
            original_vector = vectors[0]
            for vector_index, candidate_index in enumerate(needs_embedding, start=1):
                similarities[candidate_index] = round(
                    _cosine_similarity(original_vector, vectors[vector_index]),
                    6,
                )
        except Exception as exc:
            return {
                "status": "unavailable",
                "reason": f"embedding validation unavailable; rewrites preserved: {type(exc).__name__}",
                "threshold": threshold,
                "queries": candidates,
                "similarities": similarities,
                "accepted": [True for _ in candidates],
            }

    accepted = [
        similarity is not None and similarity >= threshold
        for similarity in similarities
    ]
    return {
        "status": "passed" if all(accepted) else "filtered",
        "reason": (
            f"cosine similarity threshold {threshold:.2f}; "
            f"kept {sum(accepted)}/{len(accepted)} rewrites"
        ),
        "threshold": threshold,
        "queries": candidates,
        "similarities": similarities,
        "accepted": accepted,
    }


def _compact_query(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) == 0:
        raise ValueError("embedding dimensions differ")
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = sum(float(value) * float(value) for value in left) ** 0.5
    right_norm = sum(float(value) * float(value) for value in right) ** 0.5
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise ValueError("embedding vector is empty")
    return dot / (left_norm * right_norm)


def deterministic_retrieval_decision(question: str, original_question: str = None):
    original_question = str(original_question if original_question is not None else question or "")
    compact = re.sub(r"\s+", "", str(question or "")).strip().lower()
    if not compact:
        return QueryRewriteDecision(False, "空白問題或純寒暄不需要檢索。", original_question)

    if is_current_datetime_question(original_question):
        return QueryRewriteDecision(False, CURRENT_DATETIME_REASON, original_question)

    if is_simple_arithmetic_question(original_question):
        return QueryRewriteDecision(False, "基本算術可由一般問答模型直接回答，不需要查詢 knowledge base。", original_question)

    no_retrieval_exact = {
        "你好",
        "嗨",
        "哈囉",
        "hello",
        "hi",
        "謝謝",
        "謝謝你",
        "thanks",
        "thankyou",
        "測試",
        "test",
    }
    if compact in no_retrieval_exact:
        return QueryRewriteDecision(False, "明確屬於打招呼、致謝或測試，不需要查詢 knowledge base。", original_question)

    no_retrieval_patterns = (
        r"^(你是誰|你可以做什麼|你能做什麼|你會做什麼|你有什麼功能)[？?。!！]*$",
        r"^(請)?(幫我)?(翻譯|改寫|潤稿|整理格式)[：:].+",
    )
    for pattern in no_retrieval_patterns:
        if re.search(pattern, str(question).strip(), flags=re.IGNORECASE):
            return QueryRewriteDecision(False, "問題可直接處理，不依賴 knowledge base evidence。", original_question)

    evidence_reason = verifiable_external_evidence_reason(question)
    if evidence_reason:
        return QueryRewriteDecision(True, evidence_reason, original_question)
    return None


def verifiable_external_evidence_reason(question: str) -> str:
    """Return a conservative retrieval reason for precise, source-backed facts.

    This guard only inspects the question. It has no corpus inventory and exists
    to prevent the router model from treating familiar-looking domain facts as
    stable general knowledge.
    """

    text = re.sub(r"\s+", " ", str(question or "")).strip()
    if not text:
        return ""

    source_backed_patterns = (
        r"(?:依|根據|按照).{0,24}(?:法律|法規|法|條例|準則|規範|政策|文件|報告|研究|論文|官方資料)",
        r"(?:法律|法規|條例|準則|規範|政策|官方資料).{0,24}(?:規定|要求|內容|定義|指出|顯示|期限|上限|下限)",
        r"(?:來源|引文|引用|citation|cite|查證|查詢|搜尋|search|官方來源)",
    )
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in source_backed_patterns):
        return "問題明確依賴可核對的法規、文件或來源，應先檢索外部證據。"

    precision_pattern = (
        r"(?:多少|多久|何時|何年|哪年|哪一天|"
        r"幾(?:天|日|年|月|小時|分鐘|秒|公里|公尺|人|項|次)|"
        r"(?:剛好|正好|已經)?(?:滿|超過|未滿)[一二兩三四五六七八九十百\d.]+(?:年|個月|月)|"
        r"哪(?:一|兩|幾|\d+)項|上限|下限|期限|預告期|費率|比例|距離|時長|"
        r"日期|金額|數量|版本|型號|計算|公式)"
    )
    if not re.search(precision_pattern, text, flags=re.IGNORECASE):
        return ""

    cjk_length = len(re.findall(r"[\u4e00-\u9fff]", text))
    has_named_latin_term = bool(re.search(r"\b[A-Z][A-Za-z0-9-]{1,}\b", text))
    has_domain_subject = bool(
        re.search(
            r"(?:[\u4e00-\u9fff]{2,}(?:號|任務|法案|條例|準則|標準|規範|政策|模型|產品|公司|機構|研究))",
            text,
        )
    )
    has_duration_boundary = bool(
        re.search(
            r"(?:剛好|正好|已經)?(?:滿|超過|未滿)[一二兩三四五六七八九十百\d.]+(?:年|個月|月)",
            text,
        )
    )
    if cjk_length >= 12 or has_named_latin_term or has_domain_subject or has_duration_boundary:
        return "問題要求可核對的精確事實或數值，為避免模型憑記憶誤答，應先檢索外部證據。"
    return ""


def ambiguous_acronym_definition_reason(question: str) -> str:
    text = re.sub(r"\s+", " ", str(question or "")).strip()
    asks_for_definition = bool(
        re.search(
            r"(?:定義|代表什麼|是什麼意思|意思是什麼|全名|縮寫|"
            r"what\s+is|what\s+does\s+.+\s+mean|define|definition|meaning\s+of)",
            text,
            flags=re.IGNORECASE,
        )
    )
    has_ambiguous_acronym = bool(re.search(r"\b[A-Z][A-Z0-9-]{2,12}\b", text))
    if asks_for_definition and has_ambiguous_acronym:
        return "問題要求解釋可能具有多重含義的縮寫，應先檢索證據以確認目前領域語境。"
    return ""


def is_simple_arithmetic_question(question: str) -> bool:
    compact = re.sub(r"\s+", "", str(question or "")).lower()
    compact = re.sub(r"[？?。.!！,，]", "", compact)
    compact = re.sub(r"^(?:請問|請計算|幫我算|計算)", "", compact)
    compact = re.sub(r"(?:等於多少|是多少|答案是什麼|等於|的答案)$", "", compact)
    expression = compact.replace("×", "*").replace("÷", "/").replace("％", "%")
    return bool(re.fullmatch(r"[\d.()+\-*/%^=]+", expression)) and bool(re.search(r"[+\-*/%^=]", expression))


def strip_greeting_prefix(question: str) -> str:
    text = str(question or "").strip()
    if not text:
        return ""

    greeting_pattern = (
        r"^(?:"
        r"thank you|謝謝你|你好|您好|哈囉|哈啰|thanks|hello|hey|謝謝|嗨|hi"
        r")[，,。.!！\s]*"
        r"(?:請問|想問一下|我想問|可以問一下|麻煩你|麻煩|請你|請幫我|幫我)?"
        r"[，,。:：\s]*"
    )
    stripped = re.sub(greeting_pattern, "", text, count=1, flags=re.IGNORECASE).strip()
    return stripped or text


def extract_retrieval_query(output: str) -> str:
    marker = "向量檢索用查詢："
    for line in output.splitlines():
        if marker in line:
            return line.split(marker, 1)[1].strip()
    return output.strip().splitlines()[-1].strip() if output.strip() else ""


def extract_retrieval_queries(output: str):
    queries = []
    for line in str(output or "").splitlines():
        match = re.match(r"\s*(?:檢索查詢|搜尋查詢)\s*\d*\s*[：:]\s*(.+?)\s*$", line)
        if match:
            query = match.group(1).strip(" -•\t")
            if query:
                queries.append(query)
    primary = extract_retrieval_query(output)
    return list(dict.fromkeys([query for query in [primary, *queries] if query]))[:6]


def extract_sub_questions(output: str):
    questions = []
    for line in str(output or "").splitlines():
        match = re.match(r"\s*子問題\s*\d*\s*[：:]\s*(.+?)\s*$", line)
        if match:
            question = match.group(1).strip(" -•\t")
            if question:
                questions.append(question)
    return list(dict.fromkeys(questions))[:4]


def extract_needs_retrieval(output: str) -> bool:
    marker = "是否需要檢索"
    for line in str(output or "").splitlines():
        if marker not in line:
            continue
        value = line.split("：", 1)[-1].strip() if "：" in line else line.split(":", 1)[-1].strip()
        compact = re.sub(r"\s+", "", value).lower()
        if any(term in compact for term in ("否", "不需要", "不用", "false", "no")):
            return False
        if any(term in compact for term in ("是", "需要", "true", "yes")):
            return True
    return True


def extract_decision_reason(output: str) -> str:
    marker = "判斷理由："
    for line in str(output or "").splitlines():
        if marker in line:
            return line.split(marker, 1)[1].strip()
    marker = "判斷理由:"
    for line in str(output or "").splitlines():
        if marker in line:
            return line.split(marker, 1)[1].strip()
    return "未提供判斷理由。"


def sanitize_retrieval_query(
    original_question: str,
    retrieval_query: str,
) -> str:
    sanitized = re.sub(r"\s+", " ", str(retrieval_query or "")).strip()[:500]
    if sanitized and _has_core_overlap(original_question, sanitized):
        return sanitized
    return original_question


def _has_core_overlap(original_question: str, retrieval_query: str) -> bool:
    question_terms = _core_terms(original_question)
    compact_query = re.sub(r"\s+", "", retrieval_query)
    return any(term in compact_query for term in question_terms)


def _core_terms(text: str):
    compact = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", "", text)
    stopwords = {"公司", "規定", "相關", "哪些", "什麼", "怎麼樣", "有沒有"}
    terms = set()
    for length in (4, 3, 2):
        for i in range(max(0, len(compact) - length + 1)):
            term = compact[i : i + length]
            if term not in stopwords:
                terms.add(term)
    return terms
