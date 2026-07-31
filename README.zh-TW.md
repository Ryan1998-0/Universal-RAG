# IFRS17-RAG

以 Qwen 2.5 7B 為核心的本機多模態 RAG 應用。整合自適應路由、混合檢索、Rerank、引用、對話紀錄與文件管理，介面採 ChatGPT 類型設計。

[English](README.md) | [靜態網頁 Demo](https://ryan1998-0.github.io/IFRS17-RAG/ifrs17-demo/)

> GitHub Pages 是靜態檢索展示。若要使用 Qwen、文件上傳、OCR、對話與長期記憶，請啟動本機服務。

## 核心功能

- Self-RAG 路由：簡單問題直接回答，需要證據時才進行檢索。
- 混合檢索：BM25 + Dense Embedding + RRF 融合 + Rerank。
- 多模態匯入：PDF、圖片、DOCX、TXT、Markdown、JSON。
- 圖片與掃描 PDF OCR、分塊、Embedding 與索引持久化。
- 以資料夾分類文件，並能逐次選擇要使用的 RAG 資料。
- SQLite 對話歷史、明確指令式長期記憶與來源引用。
- 透過 Ollama 在本機執行 `qwen2.5:7b`。
- 已建立 FastAPI、PostgreSQL、Qdrant、物件儲存、背景 Worker、OIDC、監控、備份與 CI 的正式版程式骨架。

## 處理流程

```text
使用者問題
  -> 自適應路由
     -> 一般回答 / 日期時間工具
     -> 查詢改寫 -> BM25 + Embedding -> 融合 -> Rerank
        -> 證據閘門 -> Qwen 生成附來源回答

上傳文件
  -> 驗證 -> 解析或 OCR -> 標準化
  -> 分塊 -> Embedding -> 持久化索引 -> 加入可選知識庫
```

## 本機啟動

需求：Python 3.12 與 [Ollama](https://ollama.com/)。

```bash
git clone https://github.com/Ryan1998-0/IFRS17-RAG.git
cd IFRS17-RAG

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock

ollama pull qwen2.5:7b
python -m rag_demo.web_app
```

開啟 [http://127.0.0.1:8765/ifrs17-demo/index.html](http://127.0.0.1:8765/ifrs17-demo/index.html)，再上傳並勾選本次問答要使用的文件。

## 驗證方式

```bash
.venv/bin/python -m pytest -q
node --test tests/ifrs17-demo/*.test.mjs tests/frontend/*.test.mjs
```

目前本機發布檢查：

| 項目 | 結果 |
| --- | ---: |
| Python 回歸測試 | 164 passed + 4 subtests |
| 瀏覽器 JavaScript 測試 | 21 passed |
| 多模態匯入 | 10/10 |
| 資料夾與重啟持久化 | 17/17 |
| IFRS 17 Retrieval-only 最佳結果 | 89.6% |

## 上線狀態

本機版可實際使用；正式版目前仍是 Staging Candidate，不等於已通過公司正式上線。Linux、真實 OIDC、負載、跨租戶安全、備份還原與回滾演練仍需在正式環境驗證。

- [正式版目標與發布門檻](docs/production-ready-rag-target.md)
- [部署與維運手冊](docs/production-runbook.md)
- [多模態文件 Pipeline](docs/multimodal-ingestion-pipeline.md)
- [目前完整度稽核](docs/local-rag-completeness-report-2026-07-31.md)

## 資料聲明

IFRS 資料的權利屬於原著作權人。本專案、Demo 資料與回答均不構成會計建議或 IFRS 合規證明；重新散布文件或擷取內容前，請先確認來源授權條款。
