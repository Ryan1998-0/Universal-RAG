# 泛用 RAG 工作台

可在本機執行的多格式 RAG 系統。知識庫與模型皆可替換，不綁定特定領域；IFRS 17 只是內附的其中一組範例資料。

[線上介面展示](https://ryan1998-0.github.io/Universal-RAG/rag-demo/)

> 線上版本用來展示介面。若要使用文件上傳、OCR、本機模型、對話紀錄與持久化索引，請啟動本機服務。

## 核心功能

- 自適應路由：簡單問題直接回答，需要文件證據時才進行檢索。
- 混合檢索：BM25、Embedding、RRF 融合與 Rerank。
- 多格式匯入：PDF、圖片、DOCX、TXT、Markdown、JSON。
- OCR、父子分塊、來源引用與證據不足拒答。
- 知識庫、資料夾與文件選取，不同領域可分開管理。
- SQLite 對話紀錄與長期記憶。
- 預設透過 Ollama 執行 `qwen2.5:7b`，模型可由設定替換。

## 處理流程

```text
使用者問題
  -> 判斷是否需要檢索
     -> 直接回答或日期時間工具
     -> 查詢改寫 -> BM25 + Embedding -> RRF -> Rerank
        -> 證據檢查 -> 生成附來源回答

上傳文件
  -> 格式驗證 -> 文字解析或 OCR -> 標準化
  -> 分塊 -> Embedding -> 持久化索引 -> 加入可選知識庫
```

## 本機啟動

需要 Python 3.12 與 [Ollama](https://ollama.com/)。

```bash
git clone https://github.com/Ryan1998-0/Universal-RAG.git
cd Universal-RAG

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock

ollama pull qwen2.5:7b
python -m rag_demo.web_app
```

開啟 [http://127.0.0.1:8765/](http://127.0.0.1:8765/)，上傳文件並勾選本次問答要使用的資料。

## 驗證

```bash
.venv/bin/python -m pytest -q
node --test tests/rag-demo/*.test.mjs tests/frontend/*.test.mjs
```

目前已驗證多格式匯入、資料夾與重啟持久化、混合檢索、對話紀錄、介面操作及容器建置。正式部署前仍需在目標環境完成真實 OIDC、負載、跨租戶安全、備份還原與回滾演練。

- [多格式文件處理流程](docs/multimodal-ingestion-pipeline.md)
- [完整架構流程](docs/hybrid_rag_architecture_flow.md)
- [正式部署目標](docs/production-ready-rag-target.md)
- [硬編碼檢查報告](docs/runtime-hardcoding-audit-2026-07-23.md)

## 範例資料

`profiles/ifrs17` 是可選的領域 Profile，用來展示術語別名與領域查詢擴展。核心 RAG 流程預設使用 `default`，不依賴任何特定資料集。
