# Adaptive RAG 路由研究與落地方案

更新日期：2026-07-23

## 設計前提

檢索前路由器只知道「有一個通用檢索器可以呼叫」，不知道知識庫有哪些文件、標題、章節或內容。文件清單不得作為是否檢索的判斷依據。

## 研究結論

### Adaptive-RAG

- 論文：[Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity](https://arxiv.org/abs/2403.14403)
- 官方程式：[starsuzi/Adaptive-RAG](https://github.com/starsuzi/Adaptive-RAG)
- 核心做法：先依問題複雜度，在「不檢索、單次檢索、多步檢索」之間路由。
- 本專案借鑑：檢索前先做 query-only 決策。目前先實作「直接回答 / 單次檢索」兩條路徑，未來再加入多步檢索。

### Self-RAG

- 論文：[Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)
- 官方程式：[AkariAsai/self-rag](https://github.com/AkariAsai/self-rag)（約 2.4k stars）
- 核心做法：不是每題固定取回相同數量片段，而是按需檢索，並對證據與回答做反思。
- 本專案借鑑：保留「按需檢索」原則。原論文需要特別訓練的 reflection tokens，不直接套到一般 Qwen 2.5 7B。

### CRAG

- 論文：[Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884)
- 核心做法：檢索後增加輕量 evaluator，先判斷取回文件是否相關，再決定生成、修正檢索或使用其他來源。
- 本專案借鑑：在 BM25 + Embedding + RRF + Rerank 後加入 Evidence Relevance Gate。證據不足時停止引用，不把看似排名第一但實際無關的片段交給 Qwen。

### 高星工程架構

- [LangGraph](https://github.com/langchain-ai/langgraph)（約 37.9k stars）：官方 Agentic RAG 範例採「是否呼叫 retriever tool -> 檢索 -> grade documents -> 生成或改寫問題」的條件圖。本專案採用相同的節點邊界，但維持現有輕量 Python 服務，不額外引入框架。
- [Haystack](https://github.com/deepset-ai/haystack)（約 26k stars）：借鑑元件單一職責與顯式路由的 pipeline 設計。
- [LlamaIndex](https://github.com/run-llama/llama_index)（約 51k stars）：常見 RouterQueryEngine 會根據資料源描述選路。這適合多資料源選擇，但不符合本專案「路由器不可預知語料內容」的前提，因此不採用於檢索前判斷。

GitHub star 數為 2026-07-23 查詢快照，之後會變動。

## 採用架構

```text
目前問題 + 僅供代名詞解析的對話背景
                    |
                    v
          Query-only Retrieval Router
              |                 |
          DIRECT            RETRIEVE
              |                 |
      本機 Qwen 直接回答       通用查詢改寫
                                |
                    BM25 + Embedding + RRF + Rerank
                                |
                    Evidence Relevance Gate
                         |               |
                    sufficient       insufficient
                         |               |
                 Grounded Qwen      明確回報證據不足
```

## 本次實作

1. `/api/route` 不再接收或讀取 `section_titles`。
2. 移除 IFRS / IFRS 17 的硬編碼 domain guard。
3. 路由 prompt 明確宣告不知道語料庫內容，只能根據目前問題判斷是否需要外部證據。
4. 查詢改寫只能使用問題本身與自然同義詞，不得從文件標題借詞。
5. 檢索後以原始 BM25、Embedding 與 matched terms 評估證據；不以候選集合內正規化後的 rerank 分數單獨判斷。
6. Evidence Gate 判定不相關時，後端不呼叫 grounded generation，也不產生引用。

## 評估注意事項

目前 Evidence Gate 是低成本規則式 evaluator，門檻依本機 multilingual MiniLM 的分數校準。正式產品應建立一組 `DIRECT / RETRIEVE` 路由資料與 `relevant / irrelevant` 檢索資料，分別量測：

- 路由 precision / recall
- 檢索證據 precision@k、recall@k
- 最終回答 groundedness 與 citation correctness
- 延遲與額外模型呼叫成本

門檻必須用評估集調整，不能把目前數值視為跨模型通用常數。
