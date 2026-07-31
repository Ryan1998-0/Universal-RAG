# 正式部署準備度複核

日期：2026-07-26
範圍：`Universal-RAG` 最新本機工作樹
結論：`STAGING CANDIDATE / PRODUCTION NO-GO`

## 結論

原始稽核的程式層 P0 已大幅完成：正式 API 不再使用舊 `ThreadingHTTPServer`，登入與租戶隔離、非同步匯入、外部持久層、不可變索引、可信引用、安全上傳、前端 Session、依賴鎖定、Container 與 CI 都已有實作及本機回歸。

現在的主要阻擋已從「缺少架構」變成「缺少真實環境證據」。這台 Mac 沒有可用的 Docker/Podman，因此本次不能驗證十二個 Compose Service 能否在乾淨 Linux 主機完整啟動，也不能假裝 OIDC、負載、故障注入與備份還原已通過。

## 已完成的 Production Surface

| 領域 | 已完成 |
|---|---|
| API | FastAPI/Uvicorn、穩定錯誤碼、Request ID、大小限制、Rate Limit |
| Auth | OIDC/JWT、Authorization Code + PKCE、Redis Server-side Session、Secure HttpOnly Cookie |
| 隔離 | PostgreSQL 與 Qdrant 強制 Tenant/KB Filter；下載與 Answer Run 重新授權 |
| Metadata | PostgreSQL + SQLAlchemy + Alembic，明確 Tenant/User/Membership/KB/Folder/Document/Version/Run 模型 |
| Storage | 私有 S3-compatible Object Storage，Proxy/Presigned Upload Contract |
| 匯入 | Celery、Lease、Heartbeat、Retry、ClamAV、MIME/SHA、OCR、Prompt Injection Quarantine |
| Retrieval | Qdrant Dense + Sparse、RRF、Reranker、Evidence Gate、Server-side Source Scope |
| Index | Immutable Version/Manifest、內容指紋、PG/Qdrant 雙重驗證、CAS 原子發布與延遲 GC |
| 對話 | PostgreSQL 對話紀錄、建立/開啟/刪除、Answer Run 與引用追溯 |
| Web | OIDC Session、左右可收合面板、文件/資料夾、上傳進度、問答進度、引用詳情 |
| 維運 | Liveness/Readiness、Prometheus、JSON Log、Audit Event、Cold Backup/Restore Script |
| Release | 固定依賴、API/Web Image、Compose、GitHub Actions、部署 Smoke/Load Script |

## 本機驗證

- Python 完整回歸：`162 passed`，另有 `4 subtests passed`。
- JavaScript 完整回歸：`21 passed`。
- `pip check`：通過。
- Alembic：單一 Head，Migration 測試通過。
- Shell：所有 `scripts/*.sh` 通過 `bash -n`。
- 前端實際瀏覽器操作：桌面與 `390x844` 手機尺寸通過；無頁面水平/垂直溢出。
- 前端實際流程：左右面板、建立新對話、問答、引用、檢索詳情均可操作。
- Compose YAML：可由 YAML Parser 讀取，共十二個 Service。

最終重新執行回歸後，若測試數量變動，應以 CI Artifact 與最後一次命令輸出為準。

## 已修復的實際缺陷

1. 文件 Row Renderer 的參數名稱遮蔽瀏覽器 `document`，導致文件面板卡在載入中；已更名並以瀏覽器實測三份文件正常顯示。
2. Production Shell 高度受舊 Demo CSS 影響，多出 9px 並產生捲軸；已固定 Shell 高度與 Grid 約束。
3. 左右面板同時開啟時登出文字被擠成兩行；已改成穩定的兩欄 Topbar。
4. 空白 Transcript 仍保留 Margin，手機首頁出現假捲軸；已在無訊息時隱藏 Transcript。
5. Answer Run 已限制 Tenant/User，但 Citation 展開使用主鍵直接取資料；已補 Chunk/Version/Document 的 Tenant/KB 條件。

## Production GO 阻擋

| 優先級 | 尚缺證據或功能 | 驗收方式 |
|---|---|---|
| P0 | 乾淨 Linux 主機的 Compose E2E | Build、Migrate、Bootstrap、API、Worker、Beat、Caddy 全部 Healthy |
| P0 | 乾淨 Clone/Tag 可重現 | Production 檔案納入版本控制；指定 Tag 可建置並得到相同 Migration/Test 結果 |
| P0 | 真實 OIDC 登入與租戶 Claim | 登入/登出/過期/重播/錯誤 Audience/跨租戶全部通過 |
| P0 | 支援格式真實 E2E | 每種格式完成上傳、掃描、OCR/解析、索引、問答、引用、下載、刪除 |
| P0 | 備份還原演練 | 空白主機還原 PostgreSQL/Redis/Qdrant/Object，Smoke 通過並記錄 RPO/RTO |
| P0 | 舊索引回滾 | 提供受控管理操作並實際從壞索引切回已驗證索引 |
| P1 | 受控 Load/Backpressure | 20 位在線、5 個生成與逾時情境通過 |
| P1 | 故障注入 | PostgreSQL/Qdrant/S3/Redis/模型/Worker 逐一中斷，Readiness 與錯誤行為正確 |
| P1 | OpenTelemetry 與告警落地 | Trace Export、Dashboard、Alert Route 與值班通知通過 |
| P1 | 最新 Gold Set | Routing、Recall@K、Citation Precision、Faithfulness、拒答率達 Gate |
| P1 | Secret/映像治理 | Secret Store、Image Scan/SBOM、Rotation Drill、固定 Image Digest |

## 不可誤解的邊界

- 舊 `rag_demo/web_app.py` 與 GitHub Pages Demo 仍可保留研究展示，但不是 Production 入口。
- 本機 Qwen 7B 可作 Demo；正式併發與回答品質取決於實際推論硬體，不能從 Unit Test 推導。
- Cold Backup Script 已存在不代表 Recovery 成功；只有實際還原才算證據。
- Dockerfile 與 Compose 存在不代表映像能在目標主機建置；必須由 CI 或 Staging 實跑。
- Prometheus 與 Correlation ID 已有，但尚未等同完整 OpenTelemetry Trace 與告警值班。
- 目前前端顯示的是伺服器等待期間的進度階段，不是 SSE Token Streaming。

## 下一個 Gate

在一台乾淨 Linux Staging 主機依 [Production Runbook](production-runbook.md) 完成：

1. Compose Build/Up。
2. OIDC Provision/Login。
3. 全格式資料流程。
4. Smoke + Load + Cross-tenant。
5. Cold Backup/Restore。
6. App/Index Rollback。

這六項都有保存的 Log/JSON/截圖後，才重新評估是否從 `STAGING CANDIDATE` 提升為 Production `GO`。
