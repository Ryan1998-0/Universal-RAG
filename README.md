# 泛用 RAG 工作台

可在本機執行的多格式 RAG 系統。知識庫與模型皆可替換，不綁定特定領域；IFRS 17 只是內附的其中一組範例資料。

[線上介面展示](https://ryan1998-0.github.io/Universal-RAG/rag-demo/)

> 線上版本用來展示介面。若要使用文件上傳、OCR、本機模型、對話紀錄與持久化索引，請啟動本機服務。

> 想先掌握目前範圍、驗證結果與入口，請看精簡版[專案總覽](docs/project_overview_zh.md)。

## 核心功能

- 自適應路由：簡單問題直接回答，需要文件證據時先分析、拆成子問題與多個互補查詢。
- 父子 Chunk 混合檢索：索引建立 1024 token 父 Chunk 與 256 token 子 Chunk；BM25 與 Embedding 只對子 Chunk 搜尋，使用 RRF 融合排名並保留前 100 個候選，最後把命中的子 Chunk 展開成父 Chunk 作為模型證據。
- 問題複雜度路由：簡單問題直接取 Top 5；複雜問題才對前 100 個子 Chunk 呼叫 Cross-Encoder，重排後取 Top 5，再回傳對應父 Chunk。
- 可選的第二階段證據聚焦：對初次召回的 chunk 切句／雙句視窗，以 lexical 80% + embedding 20% 縮小送給模型的證據範圍。
- 多格式匯入：PDF、圖片、DOCX、TXT、Markdown、JSON。
- 圖片與掃描 PDF OCR、DOCX 內嵌圖片 OCR／本機 VLM、父子分塊、來源引用與證據不足拒答。
- 知識庫、資料夾與文件選取，不同領域可分開管理。

分塊可用 `RAG_CHUNK_STRATEGY` 做 A/B 比較：`hard` 是固定 600 字元、無邊界修正且不重疊；`dynamic` 會偵測句子邊界，以 `RAG_CHUNK_SIZE` 作為近似 token 預算，並以 `RAG_CHUNK_OVERLAP_TOKENS=200` 保留前一段的完整句子。Token 計數器是相容的多語近似值，實際重疊量會受完整句子長度影響。預設範例使用 dynamic；兩種策略都會在重新建立索引時生效。

內建 20 題比較可用 `uv run --python 3.12 --all-extras python scripts/run_chunking_ab_eval.py` 重跑，結果會寫入 [Chunking A/B 報告](evals/chunking_ab/latest-report.md) 與 JSON 原始資料。

Qwen 2.5 7B 一題一請求回答品質測試可用 `python scripts/run_qwen_quality_eval.py` 重跑，結果會寫入 [Qwen 品質報告](evals/qwen_quality/latest-report.md) 與 JSON 原始資料。測試固定 dynamic 分塊與檢索，分別統計答案在檢索切片中的支持率與幻覺檢出率；若要重現舊的 GPT-5.5 結果，可執行 `python scripts/run_gpt55_subagent_quality_eval.py --model codex:gpt-5.5 --output evals/gpt55_subagent_quality/latest-results.json --report evals/gpt55_subagent_quality/latest-report.md`。

Query Rewriter 產生的主要查詢與輔助查詢會先和使用者原始問題做批次 embedding cosine similarity 比對。低於 `RAG_QUERY_REWRITE_MIN_SIMILARITY`（範例為 `0.60`）的改寫會被移除，主要查詢則回退原始問題；可用 `RAG_QUERY_REWRITE_SEMANTIC_ENABLED=0` 暫停這層校驗。

- 文件查詢 ACL：角色對文件來源授權，後端檢索前強制過濾，未授權證據不會交給模型。
- SQLite 對話紀錄與長期記憶。
- 預設使用本機 Ollama `qwen2.5:7b` 生成回答，模型可由 `RAG_MODEL` 與 `RAG_ALLOWED_MODELS` 替換；需要比較大型模型時仍可指定 `codex:gpt-5.5`。

## RAG 實驗超參數

下列參數集中控制本次 RAG 實驗。修改 `.env` 或環境變數後，重新建立索引即可讓新的分塊設定生效；查詢階段的參數會在下一次請求套用。

| 超參數 | 環境變數 | 預設值 | 作用 |
| --- | --- | ---: | --- |
| 父 Chunk token 數量 | `RAG_PARENT_CHUNK_SIZE_TOKENS` | `1024` | 作為模型證據的父 Chunk 近似 token 上限。 |
| 子 Chunk token 數量 | `RAG_CHILD_CHUNK_SIZE_TOKENS` | `256` | BM25 與向量檢索使用的精細子 Chunk 近似 token 上限。 |
| 父子 Chunk 模式 | `RAG_PARENT_CHILD_ENABLED` | `1` | 啟用父子索引與子 Chunk 檢索、父 Chunk 證據展開。 |
| 父子 Chunk 重疊數量 | `RAG_PARENT_CHUNK_OVERLAP_TOKENS` / `RAG_CHILD_CHUNK_OVERLAP_TOKENS` | `0 / 0` | 父、子 Chunk 的句子重疊 token 預算。 |
| 問題重寫語意校驗門檻 | `RAG_QUERY_REWRITE_MIN_SIMILARITY` | `0.60` | 原始問題與改寫問題的 cosine similarity 低於門檻時捨棄改寫。 |
| 關鍵字融合權重 | `RAG_KEYWORD_WEIGHT` | `0.50` | 混合檢索分數映射到共同維度後的 BM25 權重。 |
| 向量融合權重 | `RAG_EMBEDDING_WEIGHT` | `0.50` | 混合檢索分數映射到共同維度後的 Dense 權重。兩者會自動正規化。 |
| 混合檢索候選數量 | `RAG_HYBRID_CANDIDATE_K` | `100` | BM25／向量各自召回並經 RRF 融合後保留的候選上限。 |
| 最終證據 Top-K | `RAG_HYBRID_TOP_K` | `5`（優化設定） | 簡單題直取或複雜題重排後，送入模型的父 Chunk 上限。 |
| 混合檢索融合方式 | `RAG_HYBRID_FUSION_METHOD` | `rrf`（優化設定） | 使用 RRF 融合排名；舊 Profile 可指定 `lambdamart`。 |
| RRF 平滑常數 | `RAG_HYBRID_RRF_K` | `60` | 控制排名倒數分數的平滑程度。 |
| 最終重排 Top-K | `RAG_RERANK_TOP_K` | `5`（優化設定） | 複雜問題由 Cross-Encoder 重排後回傳的子 Chunk 數量，再展開為父 Chunk 證據。 |

相關的分流參數是 `RAG_COMPLEXITY_ROUTING_ENABLED=1`、`RAG_QUERY_COMPLEXITY_THRESHOLD=2.0` 與 `RAG_SIMPLE_QUERY_TOP_K=5`；簡單問題跳過昂貴重排，複雜問題才使用 Cross-Encoder。

## 單步優化消融實驗

`scripts/run_labor_law_ablation.py` 會建立五個可重現版本，除指定步驟外其餘皆固定為無優化基線：

- 無優化基線：hard 600 字元、無問題重寫、BM25 與 Dense 原始分數直接加權相加、基本重排。
- Token 切分優化：只改為 dynamic 600 token、句子邊界與 200 token overlap。
- 問題重寫優化：只開啟問題重寫與語意校驗，其餘維持基線。
- 混合檢索優化：只把兩個分支先映射到共同 `[0, 1]` 維度，再依權重融合。
- 重排優化：只加入問題複雜度分流、簡單問題 Top-5 與 `RAG_RERANK_TOP_K` 候選控制。

```bash
uv run --python 3.12 --all-extras python scripts/fetch_labor_standards_act.py
uv run --python 3.12 --all-extras python scripts/build_labor_law_profile.py
uv run --python 3.12 --all-extras python scripts/run_labor_law_ablation.py
```

測試資料位於 `evals/taiwan_labor_standards_act/`；執行 `build_labor_law_profile.py` 也會建立可在介面選用的 `labor_standards_act` Profile。法規來源是勞動部勞動法令查詢系統的[《勞動基準法》](https://laws.mol.gov.tw/FLAW/PrintFLAWDAT0201.aspx?id=FL014930&ldate=20240731)（民國 113 年 07 月 31 日修正），題庫共 50 題：40 題有明確答案、5 題資訊不足、5 題詢問法規未提供的內容。比較報告會輸出到 `evals/taiwan_labor_standards_act/ablation-report.md`，指標包含證據 fact recall、有答案題全命中率、無答案安全缺漏率、整體判定率與延遲。

這份消融測試先固定在檢索與證據層，不呼叫答案生成模型，避免模型輸出變異掩蓋單一檢索步驟的影響；後續若要比較完整回答品質，可用相同題庫再接上目前預設的 Qwen 或 `codex:gpt-5.5` 子代理與答案／幻覺評分器。

若只需要比較問題與證據檢索，可執行 `python scripts/run_labor_law_retrieval_ab.py`。這會用 50/50 的 BM25／Embedding 權重比較無優化版與全優化版；全優化版會依問題複雜度讓簡單題跳過重排、複雜題才啟動第二階段重排，結果寫入 `evals/taiwan_labor_standards_act/retrieval-ab-report.md`。

若要重現目前的父子 Chunk、RRF 與 Cross-Encoder 評測，可執行 `python scripts/run_labor_law_parent_child_eval.py`。此腳本以 50 題口語化勞基法問題比較無優化版與全優化版，結果寫入 `evals/taiwan_labor_standards_act/parent-child-rerank-report.md` 與 JSON；全優化版只在複雜題啟動 Cross-Encoder，且不呼叫答案生成模型。

若要讓兩個版本都固定取 Top-5，可執行 `python scripts/run_labor_law_retrieval_ab.py --top-k 5 --simple-top-k 5 --output evals/taiwan_labor_standards_act/retrieval-ab-top5-results.json --report evals/taiwan_labor_standards_act/retrieval-ab-top5-report.md`；此測試同樣不呼叫答案生成模型。

若要比較無優化版、全優化版自動複雜度路由與全優化版不重排，可執行 `python scripts/run_labor_law_rerank_ab.py`。此測試使用無優化版 BM25／Embedding 50/50、全優化版 80/20；全優化版會讓簡單問題直接取 Top-5，複雜問題才啟動重排，結果寫入 `evals/taiwan_labor_standards_act/retrieval-rerank-ab-report.md`。

若要直接比較答案生成品質，可執行 `python scripts/run_labor_law_qwen_ab.py`。這會用同一個 `ollama:qwen2.5:7b` 跑無優化版與全優化版各 50 題，並將「答案是否在檢索切片中」與「無幻覺率」分開統計，結果寫入 `evals/taiwan_labor_standards_act/qwen-ab-report.md`。

## 處理流程

```text
使用者問題
  -> 判斷是否需要檢索
     -> 直接回答或日期時間工具
     -> 問題分析與拆解 -> 多個查詢 -> 各自執行 BM25 + Embedding
        -> 子 Chunk BM25 + 向量檢索 -> RRF 融合 -> 前 100 候選
        -> 簡單問題取 Top-5／複雜問題 Cross-Encoder 重排取 Top-5
        -> 子 Chunk 展開為 1024 token 父 Chunk -> 證據檢查
        -> 證據優先排序 -> 僅根據證據生成附來源回答

若初次召回的 chunk 雜訊較多，可設定 `RAG_EVIDENCE_FOCUS_ENABLED=1` 啟用第二階段證據聚焦；
`RAG_EVIDENCE_FOCUS_TOP_K`、`RAG_EVIDENCE_FOCUS_MAX_CHARS`、
`RAG_EVIDENCE_FOCUS_KEYWORD_WEIGHT` 與 `RAG_EVIDENCE_FOCUS_EMBEDDING_WEIGHT` 可調整聚焦範圍與權重。

混合檢索由 `RAG_HYBRID_FUSION_METHOD=rrf` 啟用。BM25 與 Embedding 的結果先依排名套用
reciprocal rank fusion，避免直接相加不同分數尺度；`RAG_HYBRID_RRF_K` 控制 RRF 的平滑常數。
問題複雜度路由由 `RAG_COMPLEXITY_ROUTING_ENABLED`、`RAG_QUERY_COMPLEXITY_THRESHOLD` 與
`RAG_SIMPLE_QUERY_TOP_K` 控制。複雜題的 Cross-Encoder 使用 `RAG_RERANKER_MODEL`，目前預設
為 `BAAI/bge-reranker-base`。LambdaMART 仍可供舊 Profile 以 `RAG_HYBRID_FUSION_METHOD=lambdamart`
相容運行，但目前優化路徑採用 RRF。

若要測試更細的語意證據流程，可改用 `RAG_FINE_EVIDENCE_ENABLED=1`：系統會把初次召回的 parent chunk 再切成相對於各 parent chunk 約 `RAG_FINE_EVIDENCE_CHUNK_FRACTION`（預設 1/3）的細片段（至少 80 字元），重新計算 embedding，以 lexical coverage + 絕對 cosine 門檻篩選，最後只合併同一條文或相鄰的相關片段。`RAG_FINE_EVIDENCE_CHUNK_CHARS` 仍是未指定比例時的固定長度相容設定。這是實驗性流程，預設關閉。

目前預設調校以內建 IFRS 17 100 題檢索題為基準：兩個 query 變體與 32 個候選片段在品質和延遲間取得較佳平衡；第二階段證據聚焦在抽樣測試中增加約 464ms／題且降低命中率，因此維持關閉。若啟用聚焦，系統會在 Evidence Gate 失敗時退回第一階段召回結果。

上傳文件
  -> 格式驗證 -> 文字解析或 OCR -> 標準化
  -> 分塊 -> Embedding -> 持久化索引 -> 加入可選知識庫
```

## 本機啟動

需要 Python 3.12 與 [Ollama](https://ollama.com/)。

```bash
git clone https://github.com/Ryan1998-0/Universal-RAG.git
cd Universal-RAG

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock

ollama pull qwen2.5:7b
ollama pull qwen3-vl:4b-instruct
RAG_VLM_MODEL=qwen3-vl:4b-instruct python -m rag_demo.web_app
```

### LangChain 模型後端（可選）

核心檢索、Evidence Gate 與文件 ACL 維持在專案模組中；Ollama、OpenAI 與 Anthropic
模型呼叫可切換到 LangChain 的標準 Chat Model 介面。安裝額外依賴後，設定
`RAG_MODEL_BACKEND=langchain` 即可使用 `langchain-ollama`、
`langchain-openai` 或 `langchain-anthropic`；未設定時預設使用原本的直接 API。
若設定為 `auto`，則會在套件可用時優先使用 LangChain，否則回退到直接 API。
`codex:gpt-5.5` 會固定走一次性 Codex 子代理 provider，不經過 LangChain。

```bash
pip install -e ".[langchain]"
RAG_MODEL_BACKEND=langchain python -m rag_demo.web_app
```

這種分層讓 LangChain 負責模型與訊息介面，專案仍能控制多查詢混合檢索、
LambdaMART score mapping、Rerank、證據充分性及權限過濾等產品規則。

### GPT-5.5 一題一代理模式（比較用）

設定 `RAG_MODEL=codex:gpt-5.5` 後，每次模型呼叫都以 `codex exec --ephemeral`
建立全新的子代理 session。子代理使用空白暫存工作目錄，停用 shell、瀏覽器、應用程式、
記憶與搜尋功能，只接收目前問題、系統規則與本次檢索切片。即使主程式保存對話供 UI
顯示，也不會把歷史訊息或長期記憶傳給這個 provider；每題回答完成後 session 即結束。

### 證據驗證與對話記憶

回答產生後會執行 deterministic evidence validation，檢查回答引用的 rank
是否存在於本次伺服器檢索結果，並在 API 回傳 `evidence_validation` 狀態、
有效／無效引用與未標記來源的內容。驗證失敗時沿用 fail-closed 規則，不顯示
未通過來源約束的答案。

既有 SQLite `ConversationStore` 仍保存對話與「請記住」的長期記憶；
`rag_demo/langchain_memory.py` 新增 `SqliteChatMessageHistory` 與
`wrap_with_sqlite_message_history`，可直接接到 LangChain 的
`RunnableWithMessageHistory`，讓 Agent 或 Chain 共用相同的對話資料。

若要把同一份最終 RAG prompt 交給 Claude 建立 100 分相對基準，並自動評估 Qwen，請先確定 Claude Code CLI 已登入，再設定 `RAG_CLAUDE_REFERENCE_EVALUATION=1`。固定 rubric、重大錯誤分數上限與資料傳輸邊界見 [Qwen / Claude RAG 回答品質評分](docs/qwen-claude-quality-evaluation.md)。

`RAG_VLM_MODEL` 可省略；省略時 DOCX 內嵌圖片仍會執行 OCR，但不產生畫面語意說明。

### MultiHop-RAG Qwen A/B 評測

五個公開 RAG 測試資料集集中在 `RAG測試題庫/`。MultiHop-RAG 的完整資料位於
`RAG測試題庫/01_MultiHop-RAG/data/`，包含 2,556 題與 609 份文件。執行下列指令會用固定
seed `20260912` 抽出 100 題，讓無優化版與全優化版各自以 Qwen 2.5 7B 完成回答：

```bash
RAG_OLLAMA_NUM_PREDICT=384 RAG_OLLAMA_TEMPERATURE=0 \
  uv run python scripts/run_multihop_qwen_ab.py
```

結果會寫入 [MultiHop-RAG Qwen 報告](evals/multihop_rag_qwen/qwen-ab-report.md)、
[逐題 JSON](evals/multihop_rag_qwen/qwen-ab-results.json) 與 [固定抽樣題目](evals/multihop_rag_qwen/sample-100.json)。
每題不傳入對話歷史或長期記憶；報告分開統計答案正確率、答案是否在檢索切片、證據 fact recall、
安全拒答率、無幻覺率與延遲。

若要隔離檢查回答模型本身，可用 GPT-5.5 搭配題庫標註的正確 evidence prompt 測試 10 題：

```bash
uv run python scripts/run_multihop_gpt55_oracle_eval.py
```

這個測試不執行 BM25、向量檢索或重排；每題直接把 `evidence_list` 注入
`trusted_evidence`，並為每題建立獨立的 `codex:gpt-5.5` ephemeral 子代理。結果會寫入
[GPT-5.5 正確證據報告](evals/multihop_gpt55_oracle/gpt55-report.md)、
[逐題 JSON](evals/multihop_gpt55_oracle/gpt55-results.json) 與
[10 題固定抽樣](evals/multihop_gpt55_oracle/sample-10.json)。

同一組 10 題也已由 GPT-6-Astra 執行，結果見
[GPT-6-Astra 正確證據報告](evals/multihop_gpt6_astra_oracle/gpt6-astra-report.md)、
[逐題 JSON](evals/multihop_gpt6_astra_oracle/gpt6-astra-results.json) 與
[固定題目](evals/multihop_gpt6_astra_oracle/sample-10.json)。
100% 正確答案的四個 gate 與逐題語意核對見
[GPT-6-Astra 100% 正確標準](evals/multihop_gpt6_astra_oracle/accuracy-standard.md)
及其[結構化判定](evals/multihop_gpt6_astra_oracle/accuracy-standard.json)。

若要以相同的 100% 四 gate 標準，比較無優化版與全優化版的實際檢索證據，
可執行：

```bash
uv run python scripts/run_multihop_gpt55_retrieval_standard_eval.py
```

兩個版本會使用同一組 10 題與 GPT-5.5；gold evidence 只在評分階段使用，
不會注入回答 prompt。結果見 [GPT-5.5 檢索 AB 四 gate 報告](evals/multihop_gpt55_retrieval_standard/gpt55-ab-report.md)、
[逐題 JSON](evals/multihop_gpt55_retrieval_standard/gpt55-ab-results.json) 與
[固定題目](evals/multihop_gpt55_retrieval_standard/sample-10.json)。

### MultiHop-RAG 全量檢索評測（2,556 題）

若只要比較證據召回，不需要生成答案，可執行：

```powershell
$env:RAG_CROSS_ENCODER_BATCH_SIZE="100"
uv run --extra production python scripts/run_multihop_retrieval_full_eval.py --cross-encoder-model jinaai/jina-reranker-v1-tiny-en
```

此腳本會把完整 2,556 題依序跑過無優化版與全優化版，流程在證據檢索完成後停止，模型呼叫數固定為 0。無優化版是硬切 600 字元、50/50 BM25／Dense 原始分數相加與基本重排；全優化版是 1024 token parent／256 token child、RRF 前 100 候選、複雜度路由、複雜題 Cross-Encoder Top-5 與 parent evidence 展開。評測中使用英文專用的輕量 Cross-Encoder，方便完成全量測試；產品預設仍可使用 `RAG_RERANKER_MODEL=BAAI/bge-reranker-base`。

結果會寫入 [全量逐題檢索資料](evals/multihop_rag_retrieval_full/retrieval-full-results.json)、[全量彙整報告](evals/multihop_rag_retrieval_full/retrieval-full-report.md) 與 [摘要 JSON](evals/multihop_rag_retrieval_full/retrieval-full-summary.json)。報告會分開列出加權 Gold fact recall、任一／完整證據命中率、題型分組、重排比例與平均／P95 檢索延遲；不把回答正確率或幻覺率混入檢索指標。若執行中斷，可加上 `--resume` 從 checkpoints 繼續。

本次全量實測結果（2026-09-13，2,556 題；可回答 2,255 題、null_query 301 題）如下：

| 版本 | 加權 Gold fact recall | 任一證據命中 | 完整證據命中 | P95 檢索延遲 | 模型呼叫 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 無優化版 | 57.59% | 92.86% | 28.69% | 574.4 ms | 0 |
| 全優化版 | 94.40% | 99.96% | 86.47% | 4,264.0 ms | 0 |

全優化版相較無優化版，Gold fact recall 增加 36.80 個百分點，完整證據命中增加 57.78 個百分點；平均檢索延遲由 481.6 ms 增至 2,969.9 ms。這份結果使用 `jinaai/jina-reranker-v1-tiny-en` 做全量 Cross-Encoder 評測，產品預設的 `BAAI/bge-reranker-base` 不變。

#### 三種證據命中指標如何解讀

這三個指標都是檢索指標，不代表回答模型的正確率或幻覺率。計算時只納入 2,255 題有 Gold evidence 的題目；301 題 `null_query` 沒有標準證據，因此不列入這三個命中率的分母。

| 指標 | 判定方式 | 解讀 |
| --- | --- | --- |
| 任一證據命中率 | 將 Top 5 證據合併後，只要包含至少一個 Gold fact 就算命中 | 系統是否找到任何可用線索 |
| 完整證據命中率 | 將 Top 5 證據合併後，必須包含該題全部 Gold facts 才算命中 | 系統是否找齊回答問題所需的證據 |
| 首個證據命中率 | Top 5 中至少有一個單獨 Chunk 本身包含 Gold fact，就算命中 | 是否存在可以單獨支撐事實的證據 Chunk |

例如一題需要證據 A 與 B：只找到 A 時，任一證據與首個證據會命中，但完整證據不會命中；若 A 的內容被切散在多個 Chunk，合併後可辨識出 A，任一證據可能命中，但沒有單一 Chunk 能獨立命中首個證據。首個證據命中率不是只檢查 Rank 1，而是檢查 Top 5 中是否出現一個可單獨判定的證據；若要衡量第一名是否命中，需另外計算 Top 1 命中率。

本次結果中，無優化版的三項比例為 92.86%、28.69%、79.65%；全優化版為 99.96%、86.47%、99.07%，依序對應任一、完整、首個證據命中率。這表示全優化版不只更容易找到相關線索，也更容易在 Top 5 內找齊多跳問題需要的全部證據。

開啟 [http://127.0.0.1:8765/](http://127.0.0.1:8765/)，上傳文件並勾選本次問答要使用的資料。

### RAG 專案管理器

架構與任務管理入口為 [http://127.0.0.1:8765/architecture.html](http://127.0.0.1:8765/architecture.html)。macOS 可直接雙擊專案根目錄的 `開啟_RAG_專案管理器.command`；它會在需要時啟動本機服務並開啟工作台。詳細說明見 [RAG 專案管理器](RAG_專案管理器.md)。

OCR 品質 Gate、人工校正、版本化金標資料、候選模型評測與 Shadow／灰度發布的受控自我訓練閉環，見 [OCR 受控自我訓練架構](docs/ocr_self_training_architecture.md)。

本機角色與文件查詢 ACL 的設定方式、後端強制點與正式 OIDC/JWT 邊界，見 [文件查詢權限](docs/query_access_control.md)。

## 驗證

```bash
.venv/bin/python -m pytest -q
node --test tests/rag-demo/*.test.mjs tests/frontend/*.test.mjs
```

目前已驗證多格式匯入、資料夾與重啟持久化、混合檢索、對話紀錄、介面操作及容器建置。正式部署前仍需在目標環境完成真實 OIDC、負載、跨租戶安全、備份還原與回滾演練。

- [多格式文件處理流程](docs/multimodal-ingestion-pipeline.md)
- [完整架構流程](docs/hybrid_rag_architecture_flow.md)
- [正式部署目標](docs/production-ready-rag-target.md)
- [硬編碼檢查報告](docs/runtime-hardcoding-audit-2026-07-23.md)

## 範例資料

`profiles/ifrs17` 是可選的領域 Profile，用來展示術語別名與領域查詢擴展。核心 RAG 流程預設使用 `default`，不依賴任何特定資料集。
