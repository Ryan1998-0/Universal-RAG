# 正式部署與維運手冊

日期：2026-07-26
適用版本：`0.2.x`
部署模式：單一 Linux VM + Docker Compose + 外部 OIDC + 外部推論服務

## 1. 上線前提

- 一台可執行 Docker Engine、Docker Compose v2、Python 3、`timeout` 與 `flock` 的 Linux 主機。
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

## 3. 第一次部署

```bash
git clone <repository-url> Universal-RAG
cd Universal-RAG
cp .env.production.example .env.production
chmod 600 .env.production
```

由同一個部署帳號執行所有會變更 Compose 專案或資料 Volume 的操作。先建立位於 checkout 之外、主機本機檔案系統上、只讓該帳號寫入且不會在發版或回滾時刪除的鎖檔目錄；備份排程與每個部署 shell 都要設定相同的絕對路徑。以下 `$HOME` 必須位於主機本機檔案系統，不能是網路掛載：

```bash
install -d -m 700 "$HOME/.local/state/universal-rag"
export RAG_MAINTENANCE_LOCK_FILE="$HOME/.local/state/universal-rag/maintenance.lock"
```

不要刪除或重建鎖檔；若改由另一個部署帳號操作，先安排停機並讓所有操作者改用同一個受保護的主機路徑。未設定鎖檔時，冷備份、還原與持鎖命令會拒絕執行。

編輯 `.env.production`：

1. 設定正式網域與 ACME Email。
2. 將 PostgreSQL、Redis、Qdrant、Object Storage 密鑰換成獨立的高強度隨機值。
3. 填入 OIDC Issuer、Audience、JWKS、Authorization、Token URL 與 Client。
4. 令 `RAG_CORS_ORIGINS` 與 `https://<APP_DOMAIN>` 完全一致。
5. 填入容器可連線的 `RAG_OLLAMA_URL`。
6. 預設 Embedding 為 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`（384 維），與 `compose.yaml`、`.env.production.example` 及正式服務程式一致。更換模型前，確認 FastEmbed 能載入、實際維度正確，並為新設定建立索引；不可直接修改維度後沿用舊索引。
7. `RAG_MULTI_QUERY_ENABLED` 預設開啟，`RAG_MULTI_QUERY_MAX_VARIANTS` 預設為 4。每個查詢變體都會執行 Dense 與 Sparse 檢索；上線前需比較召回與延遲。

Staging/Production 不接受執行期 `PUT`/`DELETE /v1/models/{node}` 全域覆寫；模型變更須更新部署設定並走發版驗收，避免單一租戶管理 Token 影響其他租戶。

檢查設定與建置：

```bash
docker compose --env-file .env.production config --quiet
./scripts/with-maintenance-lock.sh bash -euc '
  docker compose --env-file .env.production build
  docker compose --env-file .env.production up --wait --wait-timeout 1800
'
docker compose --env-file .env.production ps
```

`migrate` 會先執行 Alembic，`bootstrap` 會建立 Object Storage Bucket 與 Qdrant Collection；兩者成功後 API、Worker、Beat 與 Caddy 才會啟動。

若既有 Qdrant collection 的 Dense 維度或 Cosine distance、`bm25` Sparse IDF 設定不同，`bootstrap` 會失敗，避免在不相容的 collection 中寫入。`/v1/ask` 也會比對 Active Index 記錄的 Embedding 模型、維度、Chunk schema 與 Qdrant collection；不一致時回 `503 INDEX_CONFIGURATION_MISMATCH`。變更這些設定時，先在 staging 以新設定完成建索引及驗收，再安排服務設定和 Active Index 一起切換。若切換中斷，恢復原設定或完成新索引啟用後再提供問答；不要忽略 503 繼續用舊索引。

## 4. 建立第一個租戶與使用者

先從 OIDC Token 確認實際 `sub` 與租戶 Claim，將值填入 `.env.production` 的 `RAG_PROVISION_*`。禁止猜測或使用顯示名稱代替 `sub`。

```bash
./scripts/with-maintenance-lock.sh docker compose --env-file .env.production \
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

目前 `smoke_production.py` 未設定 `RAG_SMOKE_KNOWLEDGE_BASE_ID` 時只檢查健康狀態與受保護的查詢 API；設定後的預設日期題也可能不經檢索。這份腳本回報 `passed` 不能單獨作為業務流程或 Production GO 的證據。發版與還原驗收仍須在對外 HTTPS 端點，以已知內容的測試知識庫和明確需要檢索的問題實際呼叫 `/v1/ask`，人工核對答案、`retrieval.needed`、引用來源與文件下載，並確認憑證、OIDC 身分及跨租戶拒絕行為。記錄測試時間、版本、`run_id` 與結果；證據中不得保存 Token 或文件全文。

## 6. 正式文件流程

1. 使用者在右側面板選擇資料夾並上傳文件。
2. API 建立 Upload Session，檢查大小、SHA-256、MIME 與完成狀態。
3. Worker 執行 ClamAV、Prompt Injection 掃描、解析/OCR、Canonical JSON 與 Chunking。
4. 文件到達可索引狀態後，使用者勾選要納入知識庫的文件。
5. Index Worker 建立 Dense + Sparse 向量，驗證 PostgreSQL/Qdrant/Manifest 指紋與數量。
6. 驗證成功後以 CAS 原子切換 Active Index；失敗候選不會取代現行索引。

支援：PDF、PNG/JPEG/WebP/TIFF/BMP/HEIC、DOCX、TXT、MD/Markdown、JSON。

Staging 與 Production 目前只允許 `RAG_UPLOAD_MODE=proxy`。Presigned 直傳仍缺儲存端內容校驗及經 HTTPS/CORS/CSP 驗證的瀏覽器路徑，設定為 presigned 時服務會拒絕啟動。待隔離 staging 完成這些驗收後才能重新開放。

## 7. 監控與診斷

外部 Caddy 故意封鎖 `/metrics`。Prometheus 應在私有 Compose 網路抓取 `http://api:8080/metrics`。

預設 API 容器使用單一 Uvicorn worker，讓 `/metrics` 的程序內計數器與直方圖涵蓋該容器全部請求。不要只把 worker 數改成多個：多程序部署須先設定 Prometheus Python client 的 multiprocess 收集、啟動時清理資料目錄及 worker 結束時清理 Gauge，並在每個 API 實例驗證抓取結果。單 worker 的容量須通過第 8 節 staging 負載驗證後才可正式上線。

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

備份與還原腳本預設使用專案根目錄的 `.env.production`，並從 Compose 解析實際 project name，停機前檢查設定及目標 Volume。若環境檔另存他處，先設定 `RAG_COMPOSE_ENV_FILE` 為該檔路徑。備份失敗時腳本仍會嘗試重啟服務；重啟失敗會以非零狀態結束，操作人員須立即處理。還原會先要求四份封存的完整 SHA-256 清單及相符的 project metadata，並確認每包可由還原映像解開，再停止服務。若解包或重啟失敗，服務應保持關閉，從已驗證的備份重新復原後才能開放流量。

四份封存只包含上述資料 Volume。空白主機復原還需要另外以加密方式保存 `.env.production` 與相關密鑰、部署用的 Git Commit 與映像版本，以及重新取得模型與 TLS 憑證的操作資料；`caddy-data`、`caddy-config` 和 `model-cache` 不在這四包內。不得把明文密鑰放進此備份目錄或版本控制。冷備份與還原會在設定檢查前取得主機維護鎖，一直持有到重啟及就緒檢查結束；發版與 Provisioning 也須使用同一鎖檔。腳本仍會拒絕開始時正在執行的 migration/bootstrap/provision 容器，以防人工命令繞過鎖；直接執行未持鎖的寫入命令仍可繞過此協定。執行冷備份或還原前仍須安排維護時段。重啟後腳本以預設 600 秒為就緒期限，每次 Docker 查詢另有 10 秒上限；API `/health/ready`（含復機後的新 Beat-to-Worker 心跳）必須為 healthy，且 Caddy、Worker、Beat 容器皆在執行。可用 `RAG_RESTART_READY_TIMEOUT_SECONDS` 將期限設為 1–600 秒。這不是對外 TLS 或完整業務 Smoke 的替代。

```bash
./scripts/cold-backup.sh /mnt/encrypted-backups/$(date -u +%Y%m%dT%H%M%SZ)
```

空白主機還原的前置順序：先取得與備份相容的專案 Commit、容器映像或可重建映像的來源、`.env.production` 與密鑰，接好備份目錄及推論服務，建立第 3 節的主機維護鎖路徑，然後檢查 Compose 設定及備份 `metadata.json` 的 `compose_project` 是否一致。`cold-restore.sh` 會在停止服務前檢查四個資料 Volume 已存在；全新主機可先建立空 Volume，再執行下方的還原命令，不要先啟動會寫入這些 Volume 的服務：

```bash
docker compose --env-file .env.production config --quiet
rag_compose_project=$(docker compose --env-file .env.production config --format json \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["name"])')
for volume in postgres-data redis-data qdrant-data object-data; do
  docker volume create "${rag_compose_project}_${volume}" >/dev/null
done
```

以上只建立空 Volume，不代表備份可用。還原腳本會在清空目標 Volume 前檢查四份封存、雜湊、專案名稱及封存可讀性；空白主機仍須完成實際還原與 RPO/RTO 演練。

還原會刪除目前四個 Volume 的內容，必須明確確認：

```bash
./scripts/cold-restore.sh --confirm-destroy-existing /mnt/encrypted-backups/<timestamp>
```

還原後依序執行：

1. `docker compose --env-file .env.production ps`
2. `/health/ready`
3. `scripts/smoke_production.py`，並依第 5 節完成對外 HTTPS 的實際檢索問答與引用人工驗收。
4. 隨機抽查文件下載、引用內容與 Active Index。

備份不是完成的證據；Staging 必須實際還原到空白主機並記錄 RPO/RTO。

## 10. 發版與回滾

發版前：

1. CI 全綠。
2. 建立冷備份。
3. 記錄目前 Git Commit、Image Tag、Alembic Head 與每個知識庫的 Active Index ID。
4. 在 Staging 跑 Smoke、格式回歸、跨租戶與負載測試。
5. 將 `evals/production_release_gate/fixtures/` 匯入 Staging，填好本機 manifest，執行 `scripts/run_production_release_gate.py`；只有退出碼為 0 且逐題 artifact 經檢查後才能繼續發版。

部署：

```bash
./scripts/with-maintenance-lock.sh bash -euc '
  docker compose --env-file .env.production build
  docker compose --env-file .env.production up --wait --wait-timeout 1800
'
./scripts/smoke_production.py
```

此腳本的 `passed` 僅證明它實際執行的檢查項目；發版仍須完成第 5 節的業務 Smoke，並保存可審查的檢索、引用、下載與權限驗收結果。

應用回滾時，在已完成冷備份後指定上一個已驗證 Tag；checkout 切換、建置和啟動必須由同一把鎖涵蓋：

```bash
export RAG_ROLLBACK_TAG='<verified-tag>'
./scripts/with-maintenance-lock.sh bash -euc '
  git switch --detach "$RAG_ROLLBACK_TAG"
  docker compose --env-file .env.production build
  docker compose --env-file .env.production up --wait --wait-timeout 1800
'
```

手動執行 migration/bootstrap、容器 stop/down、Provisioning 或直接寫入資料 Volume 時也必須先使用同一個 wrapper；唯讀 `ps`、`logs`、`config` 不需要鎖。完整冷備份先獨立結束，再開始持鎖部署，不要在已持有鎖的 wrapper 裡呼叫會自行取鎖的備份腳本。只有向後相容的 Migration 才能直接回滾應用；破壞性 Schema 變更必須採 Expand/Contract，不能臨時執行 Alembic Downgrade。

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
