# 安全政策

本專案的正式服務入口為 `rag_demo.production.api:create_app`；舊 Demo HTTP Server 僅供本機研究。對外部署請使用受維護版本、HTTPS、OIDC 與私有資料服務網路。

## 通報

安全問題請透過 GitHub Repository 的 **Security → Report a vulnerability** 私下回報（需 Repository 啟用 Private Vulnerability Reporting）。若尚未啟用，先透過既有私人管道聯絡維護者。不要在公開 Issue 貼 Token、密鑰、使用者文件或可直接利用的跨租戶資料。

回報請包含版本／Commit、受影響入口、重現條件、預期與實際結果、已遮罩的 Request ID。不要附上真實憑證。

## 部署責任與邊界

- Staging／Production 使用 PostgreSQL、Redis、Qdrant、私有 S3 與 ClamAV，禁止開發用 HS256 認證。
- OIDC Issuer 與 JWKS 必須由可信任管理者設定；不接受使用者自行指定 Provider URL。
- Session Cookie 綁定登入瀏覽器；Cookie 寫入操作檢查同源。後端串接使用短效 Bearer Access Token。
- 文件、對話、引用及工作存取受 Tenant／KB 授權。RAG 證據掃描與隔離能降低 Prompt Injection 風險，無法保證模型永不受惡意內容影響。
- `.env.production`、備份、文件內容與 Log 不可公開。請使用獨立隨機密鑰、定期輪替及受限制的檔案權限。
- 請求大小限制依實際收到的位元組計算；TLS 代理與防火牆仍須提供連線、速率與容量保護。

## 驗證

CI 包含權限／登入／上傳回歸與依賴漏洞掃描。`pip-audit` 只檢查套件公告，不能取代容器映像、作業系統、OIDC 設定或人工滲透測試。部署前依 [維運手冊](docs/production-runbook.md) 完成真實環境驗收。

依賴鎖定檔需以 `uv pip compile --all-extras pyproject.toml -o requirements.lock --python-version 3.12 --universal` 重建，保留平台 Marker 與 `pyproject.toml` 內的安全版本下限。不要在 Windows 產生沒有 Linux／Windows Marker 的正式鎖定檔。
