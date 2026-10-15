# 正式 API 串接指南

更新日期：2026-10-15。API 前綴為 `/v1`，正式入口 `https://<APP_DOMAIN>`。此指南配合 [OpenAPI](openapi.json) 與 [維運手冊](production-runbook.md) 使用。

## 身分驗證

伺服器串接請帶 `Authorization: Bearer <OIDC access token>`。Token 需要正確的 Issuer、Audience、有效期限、`sub` 與 `tenant_id`；首次使用前，管理者須用 Provision 指令建立對應使用者、租戶與權限。ID Token 不可當作 API Access Token。

瀏覽器使用 `GET /auth/login` 開始登入；`GET /auth/status` 取得登入狀態；`POST /auth/logout` 登出。不要把 Token 存入 Local Storage。所有使用 Session Cookie 的寫入請求必須來自 `RAG_PUBLIC_BASE_URL` 同一 Origin；會檢查 `Origin`，未提供時只接受 `Sec-Fetch-Site: same-origin`。伺服器以 Bearer Token 串接不需要偽造瀏覽器 Origin。

CORS 預設只允許部署網域，不能用 `*`。Cookie 模式不支援跨站寫入；需要跨來源的應用請由後端以 Bearer Token 串接。

正式環境不允許透過 API 修改全域模型綁定（`409 MODEL_BINDINGS_READ_ONLY`），避免租戶管理者影響其他租戶及多 Worker 設定不一致。請改用環境變數部署；單次問答仍可指定 Allowlist 內的模型。

## 接口目錄

| 用途 | 接口 | 成功狀態 |
|---|---|---|
| 存活／必要依賴健康 | `GET /health/live`、`GET /health/ready` | `200`，依賴未就緒為 `503` |
| 執行期模型、限制及格式 | `GET /v1/runtime` | `200` |
| 知識庫 | `GET /v1/knowledge-bases`、`POST /v1/knowledge-bases` | `200`／`201` |
| 資料夾 | `GET/POST /v1/knowledge-bases/{kb}/folders`、`PUT/DELETE .../folders/{folder}` | 詳見 OpenAPI |
| 文件清單／搬移 | `GET /v1/knowledge-bases/{kb}/documents`、`PUT .../documents/{document}/folder` | `200` |
| 預約上傳 | `POST /v1/knowledge-bases/{kb}/uploads` | `201` |
| 傳送原始檔案 | `PUT /v1/uploads/{upload}/content` | `201` |
| 完成上傳，排入解析 | `POST /v1/uploads/{upload}/complete` | `202` |
| 解析工作狀態 | `GET /v1/ingestion-jobs/{job}` | `200` |
| 建立索引／查詢工作 | `POST /v1/knowledge-bases/{kb}/index-builds`、`GET /v1/index-build-jobs/{job}` | `202`／`200` |
| 問答／執行紀錄 | `POST /v1/ask`、`GET /v1/answer-runs/{run}` | `200` |
| 引用原始檔 | `GET /v1/document-versions/{version}/content` | `200` |
| 對話 | `GET/POST /v1/conversations`、`GET/DELETE /v1/conversations/{conversation}` | 詳見 OpenAPI |
| 刪除文件／工作狀態 | `DELETE /v1/knowledge-bases/{kb}/documents/{document}`、`GET /v1/deletion-jobs/{job}` | `202`／`200` |
| 模型查詢／驗證／覆寫 | `GET /v1/models`、`POST /v1/models/validate`、`PUT/DELETE /v1/models/{node}` | 詳見 OpenAPI |

每次查詢、下載與背景工作都受租戶及知識庫權限限制；知道其他租戶的 ID 不代表可存取。

## 上傳與索引

1. `GET /v1/runtime` 讀取 `max_upload_bytes` 與允許格式。
2. 計算檔案 SHA-256，預約上傳：

```http
POST /v1/knowledge-bases/<knowledge-base-id>/uploads
Authorization: Bearer <access-token>
Content-Type: application/json
Idempotency-Key: upload:unique-operation-id

{"filename":"policy.pdf","content_type":"application/pdf","size_bytes":12345,"sha256":"<64-character-lowercase-hex>"}
```

3. 回應的 `upload_id` 及 `upload.url` 是後續操作依據。Proxy 模式對該 URL 執行 `PUT`，Body 是原始位元組，`Content-Type` 必須與預約一致；不是 Multipart。
4. `POST /v1/uploads/{upload_id}/complete` 回傳 `ingestion_job_id`。輪詢解析工作，直到 `status=succeeded`；`dead` 或 `cancelled` 為失敗。
5. 文件解析完成後，建立索引：

```http
POST /v1/knowledge-bases/<knowledge-base-id>/index-builds
Authorization: Bearer <access-token>
Content-Type: application/json
Idempotency-Key: index:unique-operation-id

{"document_ids":["<document-id>"]}
```

6. 依回應的 `job_id` 查詢 `/v1/index-build-jobs/{job_id}`。只在 `status=succeeded` 後使用新索引；失敗候選不會取代現行索引。

`202` 表示工作已受理，並不代表已完成。輪詢間隔建議 1–3 秒，設定整體 Deadline；失敗時保存 Job ID 與 Request ID。

上傳及建索引的 `Idempotency-Key` 必須保留到該次操作完成；網路逾時重送相同操作時使用同一個 Key。新操作使用新 Key，不能用新 Key 盲目重試所有寫入操作。

## 問答與引用

```http
POST /v1/ask
Authorization: Bearer <access-token>
Content-Type: application/json
X-Request-ID: client-request-001

{"question":"文件規定的申請條件是什麼？","knowledge_base_id":"<knowledge-base-id>","source_ids":[],"top_k":5}
```

回應含 `answer`、`citations`、`run_id`、`request_id`、`confidence`、`grounding_warnings`、`evidence_validation`、`retrieval`、`model` 與 `timings`。使用伺服器回傳的引用 URL 取得已授權的執行紀錄或文件；不要自行拼湊跨租戶來源。證據不足時可能拒答，呼叫端應顯示警示。

請求不接受用戶端傳入 `contexts`、路由結果或 Evidence Gate 決策。可選模型必須在伺服器 Allowlist 內。問答目前為完整 JSON 回應，沒有 SSE 串流；用戶端 Timeout 應大於服務的生成 Deadline 與排隊等待時間。

## 錯誤與重試

API 應用層錯誤統一為：

```json
{"error":{"code":"VALIDATION_ERROR","message":"The request does not match the API schema."},"request_id":"server-request-id"}
```

`X-Request-ID` Header 與 Body 的 `request_id` 相同。合法的用戶端 ID 為 1–64 字元，允許英數、`.`、`_`、`-`；其他值會被伺服器重新產生。驗證錯誤不回傳原始輸入。

| HTTP | 常見錯誤碼 | 呼叫端處理 |
|---|---|---|
| `400` | `INVALID_UPLOAD`、`IDEMPOTENCY_KEY_REQUIRED` | 修正請求 |
| `401` | `AUTH_REQUIRED`、`INVALID_TOKEN` | 重新登入或取得有效 Token |
| `403` | `ACCESS_DENIED`、`ORIGIN_DENIED` | 檢查權限或瀏覽器來源 |
| `404` | `NOT_FOUND`、`RESOURCE_NOT_FOUND` | 檢查資源是否存在及可存取 |
| `409` | `UPLOAD_CONFLICT`、`INDEX_BUILD_CONFLICT` | 重新讀取資源狀態再決定 |
| `413` | `REQUEST_TOO_LARGE`、`UPLOAD_TOO_LARGE` | 減少內容大小 |
| `422` | `VALIDATION_ERROR` | 依 OpenAPI 修正欄位 |
| `429` | `RATE_LIMITED` | 依 `Retry-After` 加上隨機退避 |
| `500` | `INTERNAL_ERROR`、`ANSWER_FAILED` | 保留 Request ID，避免無上限重試 |
| `503` | `GENERATION_CAPACITY_FULL`、`*_UNAVAILABLE` | 等待依賴恢復，有限次退避 |
| `504` | `MODEL_TIMEOUT` | 先確認執行狀態；重送問答可能再次計費 |

Caddy 或上游代理產生的錯誤（例如代理層 `413`）不保證符合 JSON 契約。所有 API 回應預設 `Cache-Control: no-store`；用戶端亦不得快取 Token、問答或文件。

## 契約更新

```bash
python scripts/export_openapi.py
python scripts/export_openapi.py --check
```

匯出在測試設定下離線完成，不連線資料庫、OIDC 或模型。CI 會檢查契約是否與路由同步。OpenAPI 目前完整列出路由、輸入 Schema、Bearer 驗證與共同錯誤格式；部分成功回應仍為動態 JSON，請一併參考本指南及端對端驗收。Production 的 Swagger 與 Schema HTTP 路徑維持關閉。
