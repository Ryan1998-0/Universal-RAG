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
