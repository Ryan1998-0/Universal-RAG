from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ArchitectureStep:
    name: str
    purpose: str
    input: str
    output: str


@dataclass(frozen=True)
class RagArchitecture:
    name: str
    steps: Tuple[ArchitectureStep, ...]


V2_RAG_ARCHITECTURE = RagArchitecture(
    name="V2 Retrieval-first RAG",
    steps=(
        ArchitectureStep(
            name="Question",
            purpose="接收使用者原始問題，保留完整語意與原始專名。",
            input="使用者輸入",
            output="original question",
        ),
        ArchitectureStep(
            name="Query Rewrite / Retrieval Decision",
            purpose="先判斷問題是否需要查詢 knowledge base；需要檢索時才產生輔助檢索訊號。",
            input="original question, candidate sections",
            output="needs_retrieval, rewritten query",
        ),
        ArchitectureStep(
            name="Metadata Filter",
            purpose="依章節、文件型態、作者、部門等 metadata 縮小搜尋空間。",
            input="original question, rewritten query, chunks",
            output="filtered chunks",
        ),
        ArchitectureStep(
            name="Entity Extraction",
            purpose="抽取 query 與 chunks 中的人物、組織、地點、事件與概念。",
            input="question, chunks, alias records",
            output="entities",
        ),
        ArchitectureStep(
            name="Query Classifier",
            purpose="判斷問題屬於內容型、關係型或混合型，決定是否啟用 graph branch。",
            input="original question, entities",
            output="content / relation / hybrid",
        ),
        ArchitectureStep(
            name="Graph Retrieval",
            purpose="補強人物關係、組織關係與多跳關係證據。",
            input="question, query type, entities, graph",
            output="graph context chunks",
        ),
        ArchitectureStep(
            name="BM25 + Dense",
            purpose="BM25 與 Dense 都使用原始問題查詢，降低 query rewrite drift。",
            input="original question, filtered chunks, embeddings",
            output="ranked vector candidates",
        ),
        ArchitectureStep(
            name="RRF Merge",
            purpose="合併 BM25 與 Dense ranked lists，提高 recall。",
            input="BM25 candidates, Dense candidates",
            output="merged candidates",
        ),
        ArchitectureStep(
            name="Graph / Vector Merge",
            purpose="把 graph context 與 vector candidates 合併並去重。",
            input="graph context chunks, merged candidates",
            output="retrieval candidates",
        ),
        ArchitectureStep(
            name="Reranker",
            purpose="依問題與候選 chunks 的對應程度重排，提高 precision。",
            input="retrieval candidates, question",
            output="reranked candidates",
        ),
        ArchitectureStep(
            name="Parent Chunk Expansion",
            purpose="補回同父層附近 chunk，避免答案證據被切碎。",
            input="reranked candidates, all chunks",
            output="expanded context",
        ),
        ArchitectureStep(
            name="Top-K Context",
            purpose="控制最終交給 LLM 的 context 數量。",
            input="expanded context",
            output="top context chunks",
        ),
        ArchitectureStep(
            name="LLM / QA Agent",
            purpose="只根據檢索來源產生最終回答。",
            input="question, evidence, top context chunks",
            output="final answer",
        ),
    ),
)


def workflow_text(architecture: RagArchitecture = V2_RAG_ARCHITECTURE) -> str:
    return "\n↓\n".join(step.name for step in architecture.steps)
