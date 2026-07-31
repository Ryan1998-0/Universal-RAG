# 本機 RAG 完整度稽核報告

- 稽核日期：2026-07-31 14:33（Asia/Taipei）
- 專案：`IFRS17-RAG`
- 稽核範圍：本機 Demo、Canonical RAGPipeline、正式版程式面、測試、資料匯入、介面、部署與維運準備度
- 目前定位：`STAGING CANDIDATE / PRODUCTION NO-GO`

> 發布前修正：本報告找到的「短縮寫定義題誤分流」與「桌面左欄撐高頁面」已於同日修正，並補上回歸測試與桌面/行動版瀏覽器驗證。其餘正式環境與引用語意風險仍維持原判定。

## 一、結論

目前不是只有畫面的 Demo。單人本機使用所需要的對話、文件選擇、多格式上傳、OCR、資料夾、Self-RAG 路由、一般問答、日期工具、混合檢索、引用與本機 Qwen 都已經有實際可運作路徑。

但若以「可直接部署到公司」作為完成標準，仍不能判定完成。正式架構的大部分元件已寫進程式，尚缺真實環境、品質與復原證據；而且今天實測找到兩個會直接影響答案可信度的缺陷：短術語題可能被 Router 錯誤分流，引用的 `verified=true` 也尚未代表來源真的支持該主張。

### 完整度判定

| 目標層級 | 完整度 | 判定 |
|---|---:|---|
| 單人本機 Demo | `84%` | 可用，但仍有 Router、引用與首次啟動延遲問題 |
| 正式版程式實作面 | `80%` | 主要元件已存在，真實整合證據不足 |
| 第一版 100 人部署目標 | `64%` | Staging Candidate，不可直接 Production GO |
| 公司內部 1,000 人使用 | `38%` | 尚未做容量設計與真實壓測，不可估為已完成 |

本報告的主評分為 `64/100`。這是工程完整度評估，不是 RAG 正確率或公開 Benchmark 分數。

## 二、今天實際驗證結果

| 驗證項目 | 結果 | 說明 |
|---|---|---|
| Python 回歸測試 | PASS | `164 passed`，另有 `4 subtests passed`；有 1 個 Starlette TestClient 棄用警告 |
| JavaScript 回歸測試 | PASS | `21 passed` |
| Python 依賴檢查 | PASS | 修復環境後 `pip check` 無破損依賴 |
| Python 編譯檢查 | PASS | `rag_demo` 與 `scripts` 通過 `compileall` |
| Alembic | PASS | 單一 Head：`f8d9e0a1b2c3` |
| Shell 語法 | PASS | 所有 `scripts/*.sh` 通過 `bash -n` |
| Git whitespace | PASS | `git diff --check` 通過 |
| Compose 結構 | PASS（僅解析） | YAML 可解析，共 12 個 Services、8 個 Volumes |
| 真實 Compose 啟動 | BLOCKED | 本機沒有 Docker 或 Podman，未驗證 Build/Up/Health |
| 日期工具 | PASS | `今天是星期幾？` 正確回答 2026-07-31 星期五，約 `1-3 ms` |
| 一般問題 | PASS | `2+2` 跳過檢索並正確回答 |
| 多格式匯入 | PASS | 重新實測 `10/10`，包含 TXT、MD、JSON、DOCX、PNG、文字 PDF、掃描 PDF |
| 資料夾與重啟持久化 | PASS | 重新實測 `17/17`，包含分類、移動、改名、刪除與重啟後檢索 |
| 本機文件 RAG | PASS WITH ISSUES | CSM 文件題可完成 Hybrid Retrieval、生成與引用，但有排序及引用語意問題 |
| 桌面介面 | PARTIAL | 無 Console Error、無水平溢出；大量對話時頁面高度被左欄撐至 1,943 px |
| 行動版介面 | PASS | `390 x 844` 無水平或垂直溢出 |

### 本機依賴故障與修復

稽核開始時，頁面與一般問題可用，但任何需要 Dense Embedding 的文件題都回傳 `HTTP 500`。直接執行 Canonical CLI 後確認原因為本機 `.venv` 缺少：

- `sentence-transformers`
- `torch`
- `transformers`

這些套件已列在 `requirements.lock`，但當時虛擬環境沒有依 Lockfile 完整安裝。今天已執行：

```bash
.venv/bin/python -m pip install -r requirements.lock
```

修復後文件檢索、多格式匯入及資料夾重啟測試均通過。這個事件也證明只看 Unit Test 不足以判斷本機服務可用，CI 需要新增「乾淨環境安裝後實際文件題」Smoke Test。

## 三、目前已完成的功能

### 本機使用層

- ChatGPT 類型的聊天介面。
- 左側對話紀錄，可建立、載入與刪除對話。
- SQLite 對話歷史與明確指令式記憶。
- 右側知識庫、文件、資料夾與檔案勾選。
- 支援 PDF、圖片、DOCX、TXT、MD、JSON；圖片另支援 JPG、JPEG、WEBP、TIFF、BMP、HEIC。
- PDF 文字抽取、掃描 PDF OCR、圖片 OCR、DOCX/XML、Markdown Heading 與 JSON Path Flatten。
- 上傳後切塊、Embedding、持久化與來源隔離。
- 日期與星期使用系統時鐘，不交給 LLM 猜測。
- 基本算術、寒暄及一般問題可跳過 RAG。
- 文件題走 BM25 + Dense Embedding + RRF + 加權 Rerank + Evidence Gate。
- 回答附來源編號，可查看檢索片段與執行時間。

### 正式版程式面

- FastAPI/Uvicorn API。
- OIDC/JWT、Authorization Code + PKCE、Server-side Session。
- PostgreSQL Metadata、Alembic Migration。
- Tenant/User/Knowledge Base 權限模型與 Qdrant Scope Filter。
- S3-compatible 私有 Object Storage。
- Redis、Celery Worker、Beat、Lease、Heartbeat、Retry 與 Backpressure Gate。
- MIME、SHA-256、壓縮炸彈、ClamAV 與 Prompt Injection 掃描。
- Qdrant Dense + Sparse、RRF 與 Cross-Encoder Reranker 程式。
- 不可變 Index Version、Manifest 驗證與 CAS 發布。
- Liveness、Readiness、Prometheus Metric、JSON Log、Audit Event。
- Dockerfile、Caddy、Compose、CI、Smoke/Load、Cold Backup/Restore Script。

## 四、主要缺陷與風險

### P0：正式上線前必須處理

#### 1. 正式版程式尚未納入 Git

目前分支與 `origin/main` 都停在 2026-07-06 的 commit `0ff5fc5`。工作樹有：

- 20 個已追蹤檔案被修改。
- 74 個未追蹤路徑項目，共 130 個實際未追蹤檔案。
- `pyproject.toml`、`requirements.lock`、Dockerfile、Compose、CI、FastAPI Production API、Canonical Pipeline、文件 Pipeline 與 Production Frontend 全部尚未追蹤。

因此目前無法從乾淨 Clone 或指定 Tag 重現這套正式版，也沒有遠端 CI 真正驗證這批程式。

#### 2. Router 會把短專業術語題誤判成一般常識

今天對選定 IFRS17 Knowledge Base 實測：

```text
問題：CSM 的定義是什麼？
Router：不需檢索
回答：CSM 是 Customer Success Management
```

這是錯誤答案。相同問題強制進入檢索時，IFRS 17 的精確定義可排在 Top 1，暖機後 Retrieval 約 `103 ms`。因此瓶頸在 Router，而不是知識庫或 Hybrid Retriever。

目前測試只 Mock 模型輸出，沒有以真實 Qwen 驗證 Router 的錯誤率。需要建立至少 100 題 Route Gold Set，尤其涵蓋縮寫、短術語、專業定義、上下文省略與一般常識。

#### 3. 引用尚未做語意級驗證

目前 `verified=true` 的實際意義是「模型輸出的 `[n]` 可對應到存在的 Context Rank」，不是「該段文字支持回答中的主張」。

今天 CSM 實測中，模型說明未賺得利潤，但引用的 `[1] [5] [8]` 沒有直接包含最精確定義；真正含 `unearned profit` 的片段曾排在第 7 名。系統仍將這些引用標記為 verified。

若目標是 Citation Precision 95%，必須加入 Claim Extraction、Claim-to-Evidence Entailment 與不支持主張的刪除/降級流程。

#### 4. 正式 12-Service Stack 尚未真正啟動

本機沒有 Docker/Podman，因此以下僅有程式或 Mock 測試，沒有真實服務整合證據：

- PostgreSQL
- Redis/Celery Worker/Beat
- Qdrant Server
- SeaweedFS S3
- ClamAV
- Caddy/TLS
- 真實 OIDC Provider

#### 5. 正式版 E2E 與復原 Gate 尚未通過

- 六類主要格式尚未在 Production API + Worker + Object Storage + Qdrant 路徑完整 E2E。
- Cold Backup/Restore Script 存在，但沒有空白主機還原紀錄與 RPO/RTO。
- 不可變索引可發布，但沒有供管理者切回舊索引的受控操作與演練。
- 沒有真實跨租戶攻擊、OIDC 過期/重播/錯誤 Audience 的整套測試證據。

### P1：試營運前應完成

#### 1. 品質評測仍不足

- 最新 Canonical Pipeline Eval 只有日期題 `1` 題。
- 公開資料問答 Smoke Test 是 `7` 題。
- IFRS17 Mixed100 是 Retrieval-only 歷史實驗，不代表現在正式 `/v1/ask` 的端到端正確率。
- 尚無最新版 Route Recall、Recall@K、MRR、nDCG、Faithfulness、Claim/Citation Precision、拒答錯誤率。

歷史 Retrieval-only 結果顯示 BM25 + Dense 最好為 `89.6%`，Full project stack 僅 `49.0%`。新增節點不一定會變好，正式 Pipeline 必須重新建立基線。

#### 2. 延遲與模型容量未達企業證據

- 第一次 CSM 文件題：Router 約 `3.7 s`、Retrieval 約 `31.7 s`、Generation 約 `20.5 s`，總計約 `55.9 s`。
- 暖機後單次 Retrieval 可降至約 `62-278 ms`。
- `2+2` 雖不檢索，仍花約 `13.7 s`，主要是 7B 模型 Route/Generation。
- 尚未驗證 20 位在線、5 個同時生成，更沒有 1,000 人容量測試。

正式使用需要模型常駐、Warmup、Router 降成本、併發 Queue、GPU 推論服務與 P50/P95 壓測。

#### 3. 文件解析仍是平面文字型

目前多格式都能抽到文字，但沒有真正導入 Docling 類型的版面、表格、閱讀順序、BBox、公式與 Parent-Child 結構化解析。對一般文字檔有效，不代表複雜理賠表格、跨頁表格與掃描文件都能可靠問答。

#### 4. 本機長期記憶仍很初階

- 只在使用者以「記住」類前綴時保存。
- 回答時直接帶入最近 12 筆記憶，沒有先做相關性檢索、TTL、版本或衝突處理。
- 正式 Production Pipeline 目前主要保存對話與 Answer Run，尚未具備同等的多租戶長期記憶服務。

#### 5. 介面與即時性仍有殘缺

- 桌面版大量對話時，Collapsed Conversation Rail 仍由內容高度撐開整個 App Shell；今天 1440 x 900 實測頁面高度為 1,943 px。
- 目前顯示的是階段進度，不是 SSE/WebSocket Token Streaming。
- 首次模型載入期間缺少更明確的 Warmup 狀態。

#### 6. 模型介面只有部分可替換

- Ollama、OpenAI、Anthropic 有基本 Adapter。
- Readiness Probe 與主要設定仍使用 `RAG_OLLAMA_URL` 及 Ollama `/api/tags`、`/api/generate` Contract。
- OpenAI endpoint 固定為官方網址，尚未形成通用 OpenAI-compatible vLLM Gateway 設定。

#### 7. 維運與供應鏈驗證未完成

- 有 Prometheus Metric 與 JSON Log，但沒有完整 OpenTelemetry Trace、Dashboard、Alert Route 與值班通知。
- 沒有 SBOM、Container/Image Vulnerability Scan、Secret Rotation Drill。
- Worker 與 Beat 沒有 Container Healthcheck，主要依 Heartbeat 監控。
- 測試套件仍有 Starlette `TestClient` 搭配 `httpx` 的棄用警告，雖不影響本次通過結果，後續依賴升級前應先處理。

## 五、分項評分

| 項目 | 權重 | 得分 | 判斷 |
|---|---:|---:|---|
| 核心功能與介面 | 15 | 13 | 主要工作流可用，桌面高度與真串流尚缺 |
| 多格式文件匯入 | 15 | 13 | 10/10 實測通過，複雜版面與表格仍不足 |
| Retrieval 與 Self-RAG | 20 | 13 | Hybrid 核心可用，但 Router 與 Query Rewrite 會傷害結果 |
| 回答與引用可靠度 | 15 | 8 | 有 Evidence Gate 與 Citation，但沒有 Claim-level 驗證 |
| 安全與租戶隔離 | 15 | 10 | 程式面完整度高，真實 OIDC/跨租戶/惡意檔案證據不足 |
| 部署與擴充性 | 10 | 5 | Compose 與 Queue 已設計，尚未在 Linux 啟動與壓測 |
| 維運與可重現性 | 10 | 2 | 有工具與文件，但 Production 程式未進 Git，無還原與回滾演練 |
| **總分** | **100** | **64** | **Staging Candidate / Production No-Go** |

## 六、建議的下一輪順序

### EN Agent - Environment

1. 將完整工作樹整理成可審查 commit，從乾淨 Clone 建立 Tag。
2. 在 Linux Staging 跑 Compose Build/Up、真實 OIDC、六格式 E2E。
3. 跑 20 在線/5 生成的 Load Test，再設計 1,000 人容量模型。
4. 完成 Backup/Restore、App Rollback、Index Rollback 與故障注入。

### P Agent - Policy Improvement

1. 修改 Router Policy：短縮寫、專業名詞與「定義」類問題預設檢索，除非有高信心 Direct Route。
2. A/B 測試「不改寫原問題」與「LLM Query Rewrite」，避免改寫反而拉低排名。
3. 加入 Claim/Citation Verifier、動態 Top-K、去重與一次 Corrective Retrieval。

### R Agent - Rollout

1. 建立 100-200 題 Canonical Gold Set。
2. 分別量測 Route、Retrieval、Generation、Citation、Refusal 與 Latency。
3. 保留今天兩個 Regression Case：`CSM 的定義是什麼？` 與 CSM Claim-to-Citation 對齊。

### E Agent - Experience / Evolution

1. 將本次「測試全綠但環境缺件」記為失敗案例，新增乾淨環境 Smoke Gate。
2. 將「原查詢 Top 1、錯誤改寫後掉排名」保存為 Query Rewrite 反例。
3. 版本化資料集、Prompt、Router、Embedding、Reranker、Index 與評測結果，禁止只保留最佳結果。

## 七、最短可行上線路徑

若目標是先提供小範圍內部試用，最短路徑是：

```text
納入 Git 並通過遠端 CI
-> Linux Staging 真實 Compose
-> 修正 Router 與 Claim/Citation Verifier
-> 100 題 Canonical Gold Set 達標
-> 真實 OIDC + 六格式 E2E
-> Load / Backup Restore / Index Rollback
-> 10-20 人受控 Pilot
-> 再擴至 100 人與 1,000 人容量設計
```

在以上 Gate 完成前，可以把它描述為「功能完整的本機 RAG 與正式架構候選版」，不應描述為「已可供 1,000 人正式上線」。
