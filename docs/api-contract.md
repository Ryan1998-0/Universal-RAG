# RAG 介面與可觀測性契約

這份文件定義目前服務預留的資料輸入、輸出、模型替換與節點計時介面。正式 API 啟用時會依環境設定開關驗證與文件頁面。

## 資料輸入

| 用途 | 方法與路徑 | 說明 |
| --- | --- | --- |
| 問題輸入 | `POST /v1/ask` | 傳入 `question`、`knowledge_base_id`，可選 `source_ids`、`top_k`、`conversation_id` 與請求層級 `model`。伺服器自行建立檢索與證據，不接受外部偽造 contexts。 |
| 建立上傳工作 | `POST /v1/knowledge-bases/{knowledge_base_id}/uploads` | 先預約檔案，傳入檔名、MIME、大小與 SHA-256。 |
| 上傳檔案內容 | `PUT /v1/uploads/{upload_id}/content` | 以 request body 傳入檔案內容；staging 與 production 僅開放 proxy 流程，presigned 直傳尚待完整性與瀏覽器安全驗收。 |
| 完成上傳 | `POST /v1/uploads/{upload_id}/complete` | 驗證物件後建立解析與索引工作。 |

支援 PDF、圖片、DOCX、TXT、Markdown 與 JSON。文件解析、OCR、切分、Embedding 與索引工作會由背景服務處理。

## 資料輸出

`POST /v1/ask` 回傳 `request_id`、`run_id`、`answer`、`citations`、`confidence`、`grounding_warnings`、`evidence_validation`、`model`、`retrieval` 與 `timings`。完整執行紀錄可由 `GET /v1/answer-runs/{run_id}` 讀取。

文件刪除會先使原件下載與歷史引用片段不可用，再由背景工作移除物件與向量。新建 AnswerRun 在 `retrieval_json.citation_lineage_v1` 保存預期的引用 rank、文件版本 ID、Chunk ID、頁碼、驗證旗標與內容 SHA-256（不含全文），讀取時會與目前的 CitationRecord 清單核對；`context_lineage_v1` 另保存最終 `retrieval.contexts` 每筆 Context ID 與文件版本 ID（不含全文）。後者包含經證據聚焦或品質回退後的最終 Context，是可能提供給生成階段的保守來源上界；證據不足時即使清單非空，模型也可能沒有被呼叫。未進入最終清單的 `raw_contexts` 候選不納入來源鏈。來源鏈用 `no_retrieval`、`retrieval_no_context`、`retrieved_contexts` 區分沒有檢索、檢索但沒有 Context，以及有最終 Context 的執行。新建 run 若任一最終 Context 無法唯一對應有效的文件版本，`POST /v1/ask` 會拒絕保存結果並回 `503 RESULT_NOT_PERSISTED`；無檢索且無 Context 的一般回答仍可保存。新 run 的 `retrieval_json` 只保留必要的檢索狀態、Context ID、來源鏈與證據驗證摘要，不保存 `contexts`、`raw_contexts` 或其他可能含原文的檢索診斷欄位。

新建 production AnswerRun 的 `retrieval_json.history_dependency_v1` 保存本輪實際提供給路由階段的近期對話 run ID（最多 10 個、去重，不含訊息文字）；該清單是路由與回答可能使用歷史內容的保守聯集。日期直答與隔離 Codex 子代理未使用歷史，記錄空清單。使用了非空歷史訊息卻無法完整追溯其 run ID、依賴指向不同租戶／使用者／知識庫／對話，或依賴在保存時已撤回時，`POST /v1/ask` 回 `503 RESULT_NOT_PERSISTED`。此內部來源鏈不出現在公開問答回應中。

`canonical-v2` 的新紀錄若缺少 citation、Context 或 history 來源鏈，讀取時會 fail closed；`canonical-v1` 與更舊紀錄保留相容讀取限制，不能宣稱具備這些新增來源證據。

若引用部分遺失或被替換、任一引用或最終 Context 的文件已刪除或遺失、Context 來源鏈不完整、需要檢索卻沒有任何 CitationRecord，或任一歷史依賴 run 已撤回／遺失／形成循環，`GET /v1/answer-runs/{run_id}` 會回傳 `answer_withdrawn: true`、固定撤回文字與 `evidence_validation: null`，也不顯示剩餘引用片段。對話中的助理訊息如缺少有效的 AnswerRun 關聯，或其 AnswerRun 已撤回，也會顯示撤回文字。後續問答的近期語境會排除撤回回答及同輪使用者問題；語境先按上限讀取，再過濾，故實際訊息數可能較少。舊 AnswerRun 沒有 `history_dependency_v1` 時無法證明它是否沿用更早回答；既有引用與 Context 檢查仍適用，但不會向該舊 run 的未知歷史祖先傳播撤回。舊 AnswerRun 沒有 `context_lineage_v1` 時仍沿用原有引用檢查，無法以未引用 Context 的文件刪除事件撤回；更舊的 AnswerRun 若也沒有 `citation_lineage_v1`，則無法偵測部分引用遺失。歷史對話中的使用者訊息仍會顯示。這是讀取時遮罩，不代表已清除資料庫中的回答、對話、Chunk 或舊備份；其保留與清除期限仍須另訂。

歷史引用的 `available` 表示來源文件仍有效；`snippet_available` 表示對應的 ChunkRecord 仍存在，且其內容符合該紀錄的 SHA-256。若 `available: true`、`snippet_available: false`，API 隱藏無法驗證的片段文字，但保留原件連結供使用者查看。`verified` 只表示生成時引用 rank 的檢查結果，不代表目前片段仍可驗證。

`evidence_validation` 會逐句檢查來源 rank，並列出 `uncited_claims` 與 `unsupported_claims`。後者表示主張與所引用片段缺少可檢查的數值或詞彙支持；此確定性檢查無法取代人工或端到端語意評估。

`timings` 保留既有摘要欄位（例如 `routeMs`、`retrieveMs`、`generateMs`、`totalMs`），並新增：

```json
{
  "stages": [
    {"name": "query.route", "durationMs": 12.4, "status": "completed"},
    {"name": "retrieval.bm25", "durationMs": 8.1, "status": "completed"},
    {"name": "retrieval.embedding", "durationMs": 34.7, "status": "completed"},
    {"name": "evidence.gate", "durationMs": 1.2, "status": "completed"},
    {"name": "model.generation", "durationMs": 820.5, "status": "completed"},
    {"name": "total", "durationMs": 890.9, "status": "completed"}
  ]
}
```

每個 stage 都有 `name`、`durationMs` 與 `status`；可選欄位只放節點名稱、候選數量等診斷資訊，不寫入問題、文件內容、prompt、回答或認證資料。

## 模型替換

### 查詢目前綁定

`GET /v1/models` 會列出語言模型節點、允許模型、Embedding／稀疏模型／重排模型，以及對應環境變數。可替換的語言模型節點包括：

`generation`、`query_rewrite`、`keyword_extraction`、`question_extraction`、`evidence_extraction`、`context_summary`、`evaluation`、`vision`。

### 驗證模型

`POST /v1/models/validate`

```json
{"node": "generation", "model": "ollama:qwen2.5:7b"}
```

只驗證 provider、模型格式與伺服器 allowlist，不會呼叫外部模型。

### 執行期替換

Development/Test 中具 `owner` 或 `admin` 角色的使用者可呼叫：

```http
PUT /v1/models/generation
Content-Type: application/json

{"model": "ollama:qwen2.5:14b"}
```

替換只作用於目前 API process，重啟後回到環境設定；`DELETE /v1/models/{node}` 可清除執行期覆寫。由於此覆寫會影響同一 process 的所有租戶，Staging/Production 暫時拒絕這兩個寫入端點，直到具資料庫成員狀態檢查與明確平台管理員權限的持久方案完成。請求層級的 `model` 欄位仍可作為單次呼叫覆寫，但必須通過 allowlist。

可用環境變數：

```text
RAG_MODEL
RAG_ALLOWED_MODELS
RAG_MODEL_GENERATION
RAG_MODEL_QUERY_REWRITE
RAG_MODEL_KEYWORD_EXTRACTION
RAG_MODEL_QUESTION_EXTRACTION
RAG_MODEL_EVIDENCE_EXTRACTION
RAG_MODEL_CONTEXT_SUMMARY
RAG_MODEL_EVALUATION
RAG_VLM_MODEL
RAG_EMBEDDING_MODEL
RAG_SPARSE_EMBEDDING_MODEL
RAG_RERANKER_MODEL
```

## Log 與除錯

服務啟動時會建立旋轉 JSON Lines log：本機執行預設為 `logs/rag.log`；正式容器映像預設為非 root 使用者可寫的 `/var/lib/rag/logs/rag.log`。可用下列環境變數調整：

```text
RAG_LOG_DIR=logs
RAG_LOG_FILE=rag.log
RAG_LOG_LEVEL=INFO
RAG_LOG_MAX_BYTES=20971520
RAG_LOG_BACKUP_COUNT=5
```

每筆 stage log 都帶有 `run_id`、`request_id`、`component`、`stage`、`duration_ms` 與狀態。Log 會自動遮罩 authorization、token、prompt、content、answer 等敏感欄位；`logs/` 與 `*.log` 已加入 Git 忽略規則。
