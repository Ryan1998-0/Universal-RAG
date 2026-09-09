# 泛用 RAG 工作台

可在本機執行的多格式 RAG 系統。知識庫與模型皆可替換，不綁定特定領域；IFRS 17 只是內附的其中一組範例資料。

[線上介面展示](https://ryan1998-0.github.io/Universal-RAG/rag-demo/)

> 線上版本用來展示介面。若要使用文件上傳、OCR、本機模型、對話紀錄與持久化索引，請啟動本機服務。

> 想先掌握目前範圍、驗證結果與入口，請看精簡版[專案總覽](docs/project_overview_zh.md)。

## 核心功能

- 自適應路由：簡單問題直接回答，需要文件證據時先分析、拆成子問題與多個互補查詢。
- 多查詢混合檢索：每個 query 分別執行 BM25、Embedding，再以 RRF 融合與 Rerank。
- 可選的第二階段證據聚焦：對初次召回的 chunk 切句／雙句視窗，以 lexical 80% + embedding 20% 縮小送給模型的證據範圍。
- 多格式匯入：PDF、圖片、DOCX、TXT、Markdown、JSON。
- 圖片與掃描 PDF OCR、DOCX 內嵌圖片 OCR／本機 VLM、父子分塊、來源引用與證據不足拒答。
- 知識庫、資料夾與文件選取，不同領域可分開管理。
- 文件查詢 ACL：角色對文件來源授權，後端檢索前強制過濾，未授權證據不會交給模型。
- SQLite 對話紀錄與長期記憶。
- 預設透過 Ollama 執行 `qwen2.5:7b`，模型可由設定替換。

## 處理流程

```text
使用者問題
  -> 判斷是否需要檢索
     -> 直接回答或日期時間工具
     -> 問題分析與拆解 -> 多個查詢 -> 各自執行 BM25 + Embedding
        -> 跨查詢 RRF -> Rerank -> 證據檢查
        -> 證據優先排序 -> 僅根據證據生成附來源回答

若初次召回的 chunk 雜訊較多，可設定 `RAG_EVIDENCE_FOCUS_ENABLED=1` 啟用第二階段證據聚焦；
`RAG_EVIDENCE_FOCUS_TOP_K`、`RAG_EVIDENCE_FOCUS_MAX_CHARS`、
`RAG_EVIDENCE_FOCUS_KEYWORD_WEIGHT` 與 `RAG_EVIDENCE_FOCUS_EMBEDDING_WEIGHT` 可調整聚焦範圍與權重。

若要測試更細的語意證據流程，可改用 `RAG_FINE_EVIDENCE_ENABLED=1`：系統會把初次召回的 parent chunk 再切成相對於各 parent chunk 約 `RAG_FINE_EVIDENCE_CHUNK_FRACTION`（預設 1/3）的細片段（至少 80 字元），重新計算 embedding，以 lexical coverage + 絕對 cosine 門檻篩選，最後只合併同一條文或相鄰的相關片段。`RAG_FINE_EVIDENCE_CHUNK_CHARS` 仍是未指定比例時的固定長度相容設定。這是實驗性流程，預設關閉。

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
ollama pull qwen3-vl:4b-instruct
RAG_VLM_MODEL=qwen3-vl:4b-instruct python -m rag_demo.web_app
```

若要把同一份最終 RAG prompt 交給 Claude 建立 100 分相對基準，並自動評估 Qwen，請先確定 Claude Code CLI 已登入，再設定 `RAG_CLAUDE_REFERENCE_EVALUATION=1`。固定 rubric、重大錯誤分數上限與資料傳輸邊界見 [Qwen / Claude RAG 回答品質評分](docs/qwen-claude-quality-evaluation.md)。

`RAG_VLM_MODEL` 可省略；省略時 DOCX 內嵌圖片仍會執行 OCR，但不產生畫面語意說明。

開啟 [http://127.0.0.1:8765/](http://127.0.0.1:8765/)，上傳文件並勾選本次問答要使用的資料。

### RAG 專案管理器

架構與任務管理入口為 [http://127.0.0.1:8765/architecture.html](http://127.0.0.1:8765/architecture.html)。macOS 可直接雙擊專案根目錄的 `開啟_RAG_專案管理器.command`；它會在需要時啟動本機服務並開啟工作台。詳細說明見 [RAG 專案管理器](RAG_專案管理器.md)。

OCR 品質 Gate、人工校正、版本化金標資料、候選模型評測與 Shadow／灰度發布的受控自我訓練閉環，見 [OCR 受控自我訓練架構](docs/ocr_self_training_architecture.md)。

本機角色與文件查詢 ACL 的設定方式、後端強制點與正式 OIDC/JWT 邊界，見 [文件查詢權限](docs/query_access_control.md)。

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
