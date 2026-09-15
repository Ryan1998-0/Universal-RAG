import re
from typing import Callable, List, Optional

from rag_demo.chunking import Chunk
from rag_demo.model_gateway import resolve_model_for_node
from rag_demo.model_providers import ask_model


ANSWER_CUE_TERMS = [
    "原因",
    "目的",
    "結果",
    "後果",
    "處置",
    "處理",
    "做法",
    "工具",
    "材料",
    "對象",
    "時間",
    "地點",
    "數字",
    "名稱",
    "條件",
    "限制",
    "例外",
    "期限",
    "罰則",
    "法條",
    "條文",
]

HIGH_PRIORITY_CUE_TERMS = [
    "明確規定",
    "應",
    "不得",
    "例外",
    "期限",
    "罰鍰",
    "處罰",
    "適用",
]


EVIDENCE_EXTRACTION_SYSTEM_PROMPT = """你是 Evidence Extraction Agent。

你的任務是只根據 retrieved chunks，抽出能直接回答問題的事實證據。
不要回答問題，不要加入來源外資訊，不要做延伸評論。
"""


def build_evidence_extraction_prompt(
    original_question: str,
    refined_question: str,
    keywords: List[str],
    chunks: List[Chunk],
) -> str:
    keyword_text = ", ".join(str(keyword) for keyword in keywords) if keywords else "無"
    evidence_context = render_evidence_context(
        chunks,
        question=f"{original_question}\n{refined_question}",
        keywords=keywords,
    )
    return f"""請從 Retrieved Chunks 中抽取能回答 Question Extraction Agent Output 的 evidence。

## Original Question
{original_question}

## Question Extraction Agent Output
{refined_question}

## Keyword Extraction Agent Output
{keyword_text}

## Retrieved Chunks
{evidence_context}

## Evidence Extraction Rules
- 只抽取 Retrieved Chunks 明確支持的事實。
- 不要回答問題，只列 evidence。
- 不要自由改寫或推論；優先保留 Retrieved Chunks 的原文關鍵詞。
- 每條 evidence 必須保留來源編號，例如：[來源 2]。
- 每條 evidence 必須包含一段 20 到 80 字的「短原文摘錄」。
- 短原文摘錄要能直接支持 Question Extraction Agent Output。
- 如果多個來源有互補資訊，請分別列出，不要合併到看不出來源。
- 如果某段只是在同章節中出現相關名詞、但不能直接回答問題，請不要抽取。
- 如果問題問「如何處置、如何處理、結果、後果、目的、原因、核心做法、作用、目標」，必須抽取對應的處置、處理、結果、後果、目的、原因、做法、作用或目標；不要只抽背景或前因。
- 如果問題問「如何處置/處理」，請優先尋找明確處置、處分、核准、駁回、懲罰、釋放、例外、後續結果等句子；不要只抽態度或背景描述。
- 如果同一段或相鄰句同時出現前提與例外、限制與結果，必須一起抽取；不要只抽前提或預設處置。
- 如果問題問「核心做法」，請優先尋找實際操作方法、工具、材料和作用對象。
- 如果問題問「核心做法」，不要只抽任務目標；必須抽取具體操作，例如工具/材料、設置位置、如何作用於目標、造成什麼結果。
- 如果原文中有數字、專名、術語、法條、期限、條件、例外或關鍵判斷詞，請保留在原文摘錄和支持點中。
- 如果同一來源中同時有背景與答案句，請優先抽答案句。
- 如果沒有足夠 evidence，請只輸出：無法從檢索來源抽取足夠 evidence。
- 請使用繁體中文，不要使用簡體字。
- 請輸出 3 到 8 條 bullet points。
- 每條格式固定為：- [來源 n] 章節：...｜原文摘錄：...｜支持點：...
"""


def extract_evidence(
    original_question: str,
    refined_question: str,
    keywords: List[str],
    chunks: List[Chunk],
    model: Optional[str] = None,
    ask_model_fn: Callable[..., str] = ask_model,
) -> str:
    return ask_model_fn(
        build_evidence_extraction_prompt(
            original_question=original_question,
            refined_question=refined_question,
            keywords=keywords,
            chunks=chunks,
        ),
        model=resolve_model_for_node(
            "evidence_extraction",
            requested_model=model,
            prefer_requested=bool(model),
        ),
        system=EVIDENCE_EXTRACTION_SYSTEM_PROMPT,
    ).strip()


def render_evidence_context(
    chunks: List[Chunk],
    question: str,
    keywords: List[str],
    max_chars_per_chunk: int = 900,
    max_snippets_per_chunk: int = 5,
) -> str:
    if not chunks:
        return "沒有找到相關來源。"

    terms = _focus_terms(" ".join([question, " ".join(str(keyword) for keyword in keywords)]))
    sources = []
    for index, chunk in enumerate(chunks, start=1):
        snippets = _select_focused_snippets(str(chunk["content"]), terms, max_snippets_per_chunk)
        content = "\n".join(f"- {snippet}" for snippet in snippets)
        if len(content) > max_chars_per_chunk:
            content = content[:max_chars_per_chunk].rstrip() + "..."
        sources.append(
            "\n".join(
                [
                    f"[來源 {index}]",
                    f"文件：{chunk['source']}",
                    f"父層章節：{chunk.get('parent_title', '')}",
                    f"章節：{chunk['title']}",
                    "候選摘錄：",
                    content or str(chunk["content"])[:max_chars_per_chunk],
                ]
            )
        )
    return "\n\n".join(sources)


def _select_focused_snippets(content: str, terms: List[str], max_snippets: int) -> List[str]:
    snippets = _split_snippets(content)
    if not snippets:
        return []

    scored = []
    for position, snippet in enumerate(snippets):
        score = _snippet_score(snippet, terms)
        if score > 0:
            scored.append((score, position, snippet))

    if not scored:
        return snippets[:max_snippets]

    selected = sorted(scored, key=lambda item: (-item[0], item[1]))[:max_snippets]
    return [snippet for _, _, snippet in sorted(selected, key=lambda item: item[1])]


def _split_snippets(content: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", str(content)).strip()
    if not normalized:
        return []
    pieces = re.split(r"(?<=[。！？!?；;])\s*", normalized)
    snippets = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if len(piece) <= 220:
            snippets.append(piece)
            continue
        for start in range(0, len(piece), 180):
            segment = piece[start : start + 220].strip()
            if segment:
                snippets.append(segment)
    return snippets


def _snippet_score(snippet: str, terms: List[str]) -> int:
    compact_snippet = _compact(snippet)
    score = 0
    for term in terms:
        compact_term = _compact(term)
        if not compact_term:
            continue
        if compact_term in compact_snippet:
            score += max(1, min(len(compact_term), 8))
    for cue in ANSWER_CUE_TERMS:
        compact_cue = _compact(cue)
        if compact_cue and compact_cue in compact_snippet:
            score += 12
    for cue in HIGH_PRIORITY_CUE_TERMS:
        compact_cue = _compact(cue)
        if compact_cue and compact_cue in compact_snippet:
            score += 30
    return score


def _focus_terms(text: str) -> List[str]:
    raw_terms = re.findall(r"[A-Za-z0-9_.+-]+|[\u4e00-\u9fff]{2,}", str(text))
    stop_terms = {
        "什麼",
        "為什麼",
        "如何",
        "哪裡",
        "哪種",
        "問題",
        "真正",
        "得到",
        "提到",
        "回答",
        "來源",
    }
    terms = []
    for term in raw_terms:
        cleaned = term.strip().lower()
        if len(cleaned) < 2 or cleaned in stop_terms:
            continue
        terms.append(cleaned)
    return list(dict.fromkeys(terms))[:32]


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", str(text)).lower()
