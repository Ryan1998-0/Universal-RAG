# 正式部署與維運手冊

更新日期：2026-10-15
適用版本：`0.2.x`
部署模式：單一 Linux VM + Docker Compose + 外部 OIDC + 外部推論服務

## 1. 上線前提

- 一台可執行 Docker Engine 與 Docker Compose v2 的 Linux 主機。
- 一個指向主機的 DNS 名稱，TCP `80/443` 對外開放。
- 一個 OIDC Provider 與已建立的 Web Client。
- 一個可由 Compose 網路連線的 Ollama 或相容推論端點。
- 至少 16 GB RAM；若推論也在同一台主機，容量要另外依模型配置。
- 備份目錄必須位於另一個磁碟、NAS 或加密的遠端儲存，不可只放同一顆系統碟。

第一版是單機部署，不宣稱跨區高可用。目標 RPO 為 24 小時，RTO 為 4 小時。

## 2. OIDC 合約

Provider 必須支援 Authorization Code Flow + PKCE，並設定：

| 項目 | 值 |
|---|---|
| Redirect URI | `https://<APP_DOMAIN>/auth/callback` |
| Web origin | `https://<APP_DOMAIN>` |
| Scope | 至少 `openid`，預設 `openid profile email` |
| Access token | JWT，包含 `iss`、`aud`、`sub`、租戶 Claim 與角色 Claim |
| 租戶 Claim | 預設 `tenant_id`，可用 `RAG_OIDC_TENANT_CLAIM` 修改 |
| 角色 Claim | 預設 `roles`，可用 `RAG_OIDC_ROLES_CLAIM` 修改 |

瀏覽器 Cookie 只保存隨機 Session ID；Access Token 保存在 Redis，Cookie 使用 `HttpOnly`、`Secure`、`SameSite=Lax`。登入 State 為一次性資料，重播會失敗。

`RAG_OIDC_ISSUER` 必須與 Provider 公布及 Token 內的 `iss` **完全相同**，包含結尾 `/`。Issuer、JWKS、Authorization、Token URL 正式環境全部必須使用 HTTPS。`RAG_PUBLIC_BASE_URL` 是純 Origin，不支援子路徑。

登入時會另設短效 HttpOnly Cookie，綁定發起登入的瀏覽器；直接在另一個瀏覽器貼上 Callback 不能建立 Session。同一瀏覽器請完成目前登入後再開始下一次登入。

## 3. 第一次部署

```bash
git clone https://github.com/Ryan1998-0/Universal-RAG.git Universal-RAG
cd Universal-RAG
cp .env.production.example .env.production
chmod 600 .env.production
```

編輯 `.env.production`：

1. 設定正式網域與 ACME Email。
2. 將 PostgreSQL、Redis、Qdrant、Object Storage 密鑰換成獨立的高強度隨機值。
3. 填入 OIDC Issuer、Audience、JWKS、Authorization、Token URL 與 Client。
4. 令 `RAG_CORS_ORIGINS` 與 `https://<APP_DOMAIN>` 完全一致。
5. 填入容器可連線的 `RAG_OLLAMA_URL`。
6. 確認 Embedding 維度與選用模型一致；正式索引建立後不要直接修改維度。

PostgreSQL 與 Redis 密碼會放進連線 URI，請各自使用至少 24 字元的 URL-safe 隨機值，例如每次執行 `openssl rand -hex 32` 產生獨立密碼。不要重複使用同一組密鑰。範例設定中的 `replace-with-*`、`example.com` 都必須換成真實設定。

此 Compose 只公開 Caddy 的 `80/443`，API、資料庫、Redis、Qdrant 與 S3 均不應另設主機 Port。內部網路屬於信任邊界，請勿將不受信任的容器接入。預設為 Proxy Upload；Presigned Upload 尚需外部 S3 入口、CORS 與 Compose 變數配置，預檢會拒絕直接切換。

若主機要執行預檢與 Smoke Script，可建立工具環境：

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --requirement requirements.lock
```

目前鎖定檔包含模型及評估依賴，安裝與映像建置需要充足磁碟空間；Linux PyTorch 依賴也可能包含 CUDA 套件。這不表示推論模型已部署，`RAG_OLLAMA_URL` 仍需另行準備。

檢查設定與建置：

```bash
python -m rag_demo.production.preflight --env-file .env.production
docker compose --env-file .env.production config --quiet
docker compose --env-file .env.production build
docker compose --env-file .env.production up -d
docker compose --env-file .env.production ps
```

`migrate` 會先執行 Alembic，`bootstrap` 會建立 Object Storage Bucket 與 Qdrant Collection；兩者成功後 API、Worker、Beat 與 Caddy 才會啟動。

預檢是唯讀操作，會拒絕示例值、短密碼、重複密鑰及錯誤設定，並透過 Compose 渲染驗證實際傳入 API 的環境。輸出只含欄位名稱與錯誤碼，不列出密鑰。`--skip-docker` 僅驗證本機環境欄位，不驗證 Compose 或 API Settings，也不代表服務已可上線。請勿將 `docker compose config` 的完整輸出貼到公開紀錄，裡面含有密鑰。

容器初次啟動會下載 Embedding／重排模型及 ClamAV 病毒碼，健康檢查可能需數分鐘。若 API 一直不 Ready，先查 Worker／Beat Heartbeat、推論服務與 ClamAV Log。

## 4. 建立第一個租戶與使用者

先從 OIDC Token 確認實際 `sub` 與租戶 Claim，將值填入 `.env.production` 的 `RAG_PROVISION_*`。禁止猜測或使用顯示名稱代替 `sub`。

```bash
docker compose --env-file .env.production \
  --profile admin run --rm provision
```

Provisioning 可重複執行，不會建立重複 Tenant/User/Membership。完成後用瀏覽器登入，確認只能看到該租戶的知識庫。

## 5. 健康檢查與 Smoke Test

```bash
curl --fail --silent https://<APP_DOMAIN>/health/live
curl --fail --silent https://<APP_DOMAIN>/health/ready
```

`/health/ready` 會檢查 PostgreSQL、Redis、Qdrant、Object Storage、推論服務及 Beat-to-Worker Heartbeat。任何必要依賴失敗都應回 `503`。

取得測試帳號的短效 Access Token 後：

```bash
export RAG_SMOKE_BASE_URL=https://<APP_DOMAIN>
export RAG_SMOKE_BEARER_TOKEN='<short-lived-access-token>'
./scripts/smoke_production.py
```

若要連問答一起測：

```bash
export RAG_SMOKE_KNOWLEDGE_BASE_ID='<knowledge-base-id>'
export RAG_SMOKE_QUESTION='今天是星期幾？'
./scripts/smoke_production.py
```

Token 不可放進 Shell History、CI Log 或版本控制；正式自動化應由 Secret Store 注入。

## 6. 正式文件流程

1. 使用者在右側面板選擇資料夾並上傳文件。
2. API 建立 Upload Session，檢查大小、SHA-256、MIME 與完成狀態。
3. Worker 執行 ClamAV、Prompt Injection 掃描、解析/OCR、Canonical JSON 與 Chunking。
4. 文件到達可索引狀態後，使用者勾選要納入知識庫的文件。
5. Index Worker 建立 Dense + Sparse 向量，驗證 PostgreSQL/Qdrant/Manifest 指紋與數量。
6. 驗證成功後以 CAS 原子切換 Active Index；失敗候選不會取代現行索引。

支援：PDF、PNG/JPEG/WebP/TIFF/BMP/HEIC、DOCX、TXT、MD/Markdown、JSON。

## 7. 監控與診斷

外部 Caddy 故意封鎖 `/metrics`。Prometheus 應在私有 Compose 網路抓取 `http://api:8080/metrics`。

重要指標：

- `rag_http_requests_total`
- `rag_http_request_duration_seconds`
- `rag_http_requests_inflight`
- `rag_generation_waiting`
- `rag_generation_active`
- `rag_pipeline_duration_seconds`
- `rag_api_errors_total`

常用診斷：

```bash
docker compose --env-file .env.production ps
docker compose --env-file .env.production logs --since=30m api worker beat
docker compose --env-file .env.production logs --since=30m postgres redis qdrant object-storage clamav
docker compose --env-file .env.production exec redis \
  redis-cli -a "$REDIS_PASSWORD" ping
```

API Log 為 JSON，使用 `request_id` 對應安全錯誤回應與 Audit Event。不得把 Access Token、Cookie、文件全文或 Secret 寫入 Ticket。

初始告警建議：

- Readiness 連續 3 次失敗。
- 5xx 比例 5 分鐘內超過 2%。
- P95 API 延遲超過既定 SLO。
- `rag_generation_waiting` 持續大於 5。
- Ingestion/Index Job 進入 `dead`。
- Beat-to-Worker Heartbeat 過期。
- 磁碟使用率超過 75%，或備份超過 26 小時未成功。

## 8. 受控負載驗證

只在 Staging 執行：

```bash
export RAG_LOAD_CONFIRM_STAGING=yes
export RAG_LOAD_BASE_URL=https://staging.example.com
export RAG_LOAD_BEARER_TOKEN='<short-lived-access-token>'
export RAG_LOAD_KNOWLEDGE_BASE_ID='<knowledge-base-id>'
export RAG_LOAD_REQUESTS=20
export RAG_LOAD_CONCURRENCY=5
./scripts/load_smoke.py | tee load-result.json
```

驗收至少要確認 20 位在線、5 個同時生成時沒有跨租戶內容、無未受控 5xx，且超量請求會排隊或收到明確錯誤，不會拖垮 Worker。

## 9. 備份與還原

冷備份會短暫停止寫入路徑，封存 PostgreSQL、Redis、Qdrant 與 Object Storage 四個 Volume，產生 SHA-256 清單後重啟服務。

```bash
./scripts/cold-backup.sh /mnt/encrypted-backups/$(date -u +%Y%m%dT%H%M%SZ)
```

還原會刪除目前四個 Volume 的內容，必須明確確認：

```bash
./scripts/cold-restore.sh --confirm-destroy-existing /mnt/encrypted-backups/<timestamp>
```

兩支腳本預設使用專案根目錄的 `.env.production`，包含停止與重新啟動服務的命令。使用其他環境檔時，先設定 `RAG_COMPOSE_ENV_FILE=/absolute/path/to/deploy.env`；自訂專案名稱時設定與部署一致的 `COMPOSE_PROJECT_NAME`。腳本重啟既有容器，不會順便執行新版本 Migration。

還原後依序執行：

1. `docker compose ps`
2. `/health/ready`
3. `scripts/smoke_production.py`
4. 隨機抽查文件下載、引用內容與 Active Index。

備份不是完成的證據；Staging 必須實際還原到空白主機並記錄 RPO/RTO。

## 10. 發版與回滾

發版前：

1. CI 全綠。
2. 建立冷備份。
3. 記錄目前 Git Commit、Image Tag、Alembic Head 與每個知識庫的 Active Index ID。
4. 在 Staging 跑 Smoke、格式回歸、跨租戶與負載測試。

部署：

```bash
docker compose --env-file .env.production build
docker compose --env-file .env.production up -d
./scripts/smoke_production.py
```

應用回滾時切回上一個已驗證 Tag，再執行 `docker compose up -d --build`。只有向後相容的 Migration 才能直接回滾應用；破壞性 Schema 變更必須採 Expand/Contract，不能臨時執行 Alembic Downgrade。

索引發布採不可變版本與原子切換。若新索引造成品質退化，先停止新索引工作並保留現行流量；目前尚未提供公開的「重新啟用舊索引」管理 API，正式 Production GO 前必須完成並演練這個操作，或從一致備份還原。

## 11. 事故處理

| 現象 | 第一動作 | 後續 |
|---|---|---|
| 跨租戶疑慮 | 立即下線 Caddy 或封鎖 `/v1/*` | 保存 Audit/Request ID，輪替 Session/Token，啟動資料外洩調查 |
| 模型逾時 | 限制新 Ask 流量 | 檢查 `generation_waiting/active`、推論端點與模型資源 |
| 匯入大量失敗 | 暫停 Worker | 檢查 ClamAV、Object Storage、Parser 與 Job Error Code |
| Qdrant 不可用 | 保持 Readiness 失敗 | 不可改用未隔離的本機 fallback；修復或還原 Qdrant |
| PostgreSQL 不可用 | 停止寫入與問答 | 從一致備份還原，核對 Migration Head |
| Secret 洩漏 | 立即撤銷及輪替 | 重建 Web Session、OIDC Client Secret、DB/Redis/Qdrant/S3 Credentials |

## 12. Production GO Gate

程式與單元回歸通過不等於正式上線。以下證據都存在才可標記 `GO`：

- CI、Container Build、Migration 與 Compose Rendering 通過。
- Production 檔案已納入版本控制，乾淨 Clone 指定 Tag 可重現相同結果。
- 真實 OIDC 登入、登出、Session 過期與跨租戶測試通過。
- 每種支援格式至少一份真實文件完成上傳、解析、索引、問答、引用、下載與刪除。
- Staging 併發、Queue Backpressure、模型逾時及依賴故障測試通過。
- 空白主機的備份還原演練通過，RPO/RTO 有實測數字。
- 舊應用版本與舊索引版本的回滾演練通過。
- Retrieval/Answer Gold Set 達到目標門檻，且跨租戶洩漏為 0。
- 監控、告警、值班聯絡與 Secret Rotation 已落地。

未取得上述環境證據前，本版本應稱為 `Staging Candidate`，不是已完成 Production GO。
