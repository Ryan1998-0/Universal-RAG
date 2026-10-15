# 部署、API 與安全升級紀錄

日期：2026-10-15。來源：`Ryan1998-0/Universal-RAG`，基準 Commit `8fb192029b917f1acce059f8f4ce42e1e58a5cb0`。

本階段完成可檢查的程式與文件修改，狀態維持 **Staging Candidate**。本機沒有 Docker Engine，也沒有真實 OIDC、Staging 主機及推論服務憑證，因此未宣稱已完成正式上線。

## 修復與補強

| 項目 | 原問題 | 本次處理 |
|---|---|---|
| 登入瀏覽器綁定 | Redis State 為一次性資料，但沒有綁定發起登入的瀏覽器 | 獨立短效 HttpOnly Cookie、State 內儲存 Binding Hash、Callback 比對後才能建立 Session |
| Cookie 寫入請求 | 同源檢查只有登出入口，其他寫入依靠 Cookie 行為 | 所有使用 Cookie 驗證的寫入請求統一檢查 Origin／Fetch Metadata；Bearer 串接保留 |
| OIDC Issuer | 移除結尾 `/`，不符合 Provider 的精確 `iss` 比對 | 保留完整 Issuer，測試合法與錯誤 Issuer |
| JWKS 金鑰更新 | 簽章金鑰獨立 LRU Cache 可長期保留已被移除的金鑰 | 停用永久單鍵快取，JWKS Set 使用 300 秒生命週期及 5 秒網路 Timeout |
| 身分服務 URL | Issuer／JWKS 未強制 HTTPS；開發 Loopback 只檢查字串前綴 | 解析 URL、限制正式 HTTPS、精確 Loopback Host、拒絕帳密及 Fragment |
| Request Body | 只以宣告的 Content-Length 阻擋超大請求 | ASGI 逐次統計實際位元組，無長度／偽造長度均受限制 |
| API 錯誤 | 驗證錯誤可能回傳原始輸入，404／405／未處理例外格式不一致 | 共用 Error Schema、穩定錯誤碼、Request ID、安全 Header、No-store；不回傳敏感輸入或例外訊息 |
| 正式模型管理 | 租戶 Owner 可改動 Process 全域模型，影響其他租戶且多 Worker 不一致 | Staging／Production 停用 PUT／DELETE 覆寫，改由環境設定部署；開發與單次 Allowlist 模型選擇保留 |
| 健康探測 | urllib 可處理額外 URL Scheme，且預設可跟隨 Redirect | 僅接受 HTTP(S)，使用 httpx、拒絕 Redirect 並設定 Timeout |
| 依賴漏洞 | 6 個套件合計 47 筆掃描告警（38 個不重複公告 ID） | 更新修補版本、重建鎖定檔、加入 CI 掃描 |
| 跨平台建置 | Windows 產生的 Lock 含無條件 pywin32，Linux 缺必要平台依賴 | 使用 `--universal` 鎖定，保留 Windows／Linux Marker |
| Compose 設定 | 部分限制及模型節點變數未傳入容器 | 補齊模型節點、Request Body 與 Rate Limit 設定，應用容器移除 Linux Capabilities 並禁止提高權限 |
| 部署預檢 | 只有 Compose 語法檢查，示例值仍可能被使用 | 新增唯讀 CLI，檢查示例值、短密碼、重複密鑰、URI-safe 密碼、正式 API Settings；輸出遮罩 |
| 備份還原 | 腳本沒有沿用 `.env.production`，重啟會重新執行整個 Compose 啟動流程 | 全程使用相同環境檔及 Project Name，重啟既有容器，補上假 Docker 整合測試 |
| API 串接文件 | 首次部署、驗證、輪詢及重試資訊分散 | 新增串接指南、離線 OpenAPI 匯出與 CI 同步檢查、安全政策及更新維運手冊 |

## 套件修補

| 套件 | 修改前 | 修改後 |
|---|---|---|
| PyJWT | 2.13.0 | 2.15.1 |
| pypdf | 6.14.2 | 6.19.0 |
| python-multipart | 0.0.22 | 0.0.31 |
| cryptography | 49.0.0 | 50.0.2 |
| h2 | 4.4.0 | 4.4.1 |
| urllib3 | 2.7.0 | 2.8.0 |

修補版本依掃描結果及上游公告選擇。PyJWT 2.15.1 另修復部分 Provider 所用 Base64URL Padding 的相容性問題，參考 [PyJWT Changelog](https://pyjwt.readthedocs.io/en/stable/changelog.html)；上傳解析器公告見 [python-multipart Security](https://github.com/Kludex/python-multipart/security)。掃描告警不等同每項漏洞均可從本專案的預設設定利用。

## 本機驗證證據

環境：Windows、Python 3.12.13、Node.js 24.19.0、Git Bash。測試使用隔離的虛擬環境及測試資料，未變更真實部署或資料。

| 驗證 | 結果 |
|---|---|
| 完整 Python 回歸 | `262 passed`、`4 subtests passed`；1 筆既有 LangChain 棄用警告 |
| 前端 JavaScript 回歸 | `28 passed` |
| 依賴相容性 | `uv pip check` 通過 |
| 更新後套件公告掃描 | `pip-audit 2.10.1`：0 筆已知漏洞；另對 Lock 的 163 筆跨平台套件條目去除 Marker 後僅作公告查詢，亦為 0 筆 |
| 靜態程式安全掃描 | Bandit 掃描 `rag_demo/production`：中／高嚴重度 0 筆 |
| OpenAPI | 離線匯出及 `--check` 通過 |
| Linux 依賴解析 | `uv pip install --dry-run --python-platform x86_64-unknown-linux-gnu --python-version 3.12 -r requirements.lock` 通過；沒有執行 Linux 安裝 |
| Compose | YAML 可解析，12 個 Service，僅 Caddy 有主機 Port；本機沒有 Docker，尚未實跑 Compose 渲染／Build／Up |
| Migration | 單一 Alembic Head：`f8d9e0a1b2c3`；相關回歸通過，沒有執行正式資料庫 Migration |
| 備份腳本 | Bash 語法通過；假 Docker 驗證環境檔／Project Name 一致、啟停指令及還原確認參數；未操作真實 Volume |
| 示範設定預檢 | `.env.production.example` 被正確拒絕，輸出不包含密鑰內容 |
| 差異檢查 | `git diff --check` 通過 |

已知漏洞掃描結果只反映查詢當下的套件公告資料，不是「系統沒有漏洞」的保證。

## 相容性改變

- `422` 不再回傳 Pydantic 原始輸入；用戶端依 `error.code`、`request_id` 與 OpenAPI 處理。
- Cookie 寫入需同源；沒有 Origin 的非瀏覽器串接應改用 Bearer Token。
- 正式環境全域模型覆寫回 `409 MODEL_BINDINGS_READ_ONLY`，改由 `RAG_MODEL_*` 部署環境設定。
- 登入必須在發起登入的瀏覽器完成，登入 Cookie 有效期與 State 相同。
- 備份／還原預設讀取專案根目錄 `.env.production`；自訂環境檔透過 `RAG_COMPOSE_ENV_FILE` 指定。

## 下一個正式上線 Gate

下列項目需要真實環境證據，尚未完成：

1. Linux 主機的 Compose Build／Migration／Bootstrap／全服務啟動，保存 GitHub CI 與 Staging Log。
2. 真實 OIDC 登入／登出／到期、錯誤 Audience／Issuer、跨租戶隔離及金鑰輪替。
3. 所有支援文件格式完成上傳、掃描、OCR、索引、問答、引用與刪除。
4. 真實 Volume 冷備份還原、RPO／RTO 量測與舊應用／舊索引回滾演練。現有舊索引重新啟用管理介面仍未提供。
5. 併發負載、依賴故障注入、容器／作業系統映像掃描、告警路由及人工安全測試。

首次部署及驗收命令請見 [維運手冊](production-runbook.md)；串接流程見 [API 指南](api-integration.md)。登入瀏覽器綁定設計參考 [OAuth Security BCP RFC 9700](https://www.rfc-editor.org/rfc/rfc9700.html)，Issuer 精確比對參考 [OpenID Connect Discovery](https://openid.net/specs/openid-connect-discovery-1_0.html)。
