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
            name="Child Chunk BM25 + Dense",
            purpose="使用 128 token 子 Chunk 產生精細的字面與語意候選。",
            input="original question, filtered child chunks, embeddings",
            output="ranked child candidates",
        ),
        ArchitectureStep(
            name="RRF Fusion Top-100",
            purpose="以 reciprocal rank fusion 合併 BM25 與 Dense 排名，保留前 100 個子 Chunk 候選。",
            input="BM25 child candidates, Dense child candidates",
            output="RRF ranked child candidates",
        ),
        ArchitectureStep(
            name="Graph / Vector Merge",
            purpose="把 graph context 與 vector candidates 合併並去重。",
            input="graph context chunks, merged candidates",
            output="retrieval candidates",
        ),
        ArchitectureStep(
            name="Query Complexity Gate",
            purpose="簡單問題直接取 Top-5；複雜問題才啟動 Cross-Encoder 重排。",
            input="retrieval candidates, question",
            output="direct top-3 or rerank candidates",
        ),
        ArchitectureStep(
            name="Reranker",
            purpose="對複雜問題的前 100 個子 Chunk 執行 Cross-Encoder，取 Top-5。",
            input="retrieval candidates, question",
            output="reranked candidates",
        ),
        ArchitectureStep(
            name="Parent Chunk Expansion",
            purpose="把命中的子 Chunk 展開成 512 token 父 Chunk，提供完整語境給模型。",
            input="selected child candidates, parent mapping",
            output="parent evidence",
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
