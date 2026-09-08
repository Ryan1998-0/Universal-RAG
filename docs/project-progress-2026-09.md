# Universal-RAG 專案進度整理

更新日期：2026-09-08

## 目前定位

這是一個可替換模型與知識庫的本機多格式 RAG 工作台，核心目標是驗證「文件召回、證據約束、答案正確性」三個階段，而不是只測試單一模型。

## 已完成

- 多格式匯入：PDF、掃描 PDF、圖片、DOCX、TXT、Markdown、JSON。
- OCR 與 DOCX 內嵌圖片處理，支援本機 VLM 延伸描述。
- 父子 chunk、Embedding、BM25、RRF、Rerank、來源引用。
- 多查詢問題拆解與查詢擴展。
- 第二階段 evidence focus 與更細粒度 fine evidence 流程。
- Evidence Gate 與 fail-closed 證據不足拒答。
- 乾淨的 RAG 對話介面與專案管理器：歷史對話、檔案上傳、架構粒度縮放、節點任務勾選。
- 39 個核心測試通過（hybrid retrieval、pipeline、fine evidence、評分器）。

## 評估結果

以 50 題數值污染文件 benchmark 測試 GPT-5.6-Luna：

| 指標 | 結果 |
|---|---:|
| Relaxed Evidence Gate | 50/50 通過 |
| 舊版固定字串評分 | 35/50、平均 80.7 |
| 新版問題相關事實＋原始文件支持 | 49/50、平均 98.0 |
| 新版問題相關事實＋實際 final evidence | 48/50、平均 96.0 |

新版評分器只要求回答問題真正詢問的核心事實，並檢查該事實是否存在於文件；不再因為沒有重複 Oracle 全文、同義詞或標點不同而扣分。仍會拒絕明確矛盾、拒答，以及文件中沒有的數值。

目前唯一明顯的模型回答缺漏是第 10 題，漏掉「不得因請假而為不利處分」。第 39 題的事實存在於原始文件，但沒有完整進入 final evidence，屬於 evidence packaging 問題。

## 目前可切換設定

```bash
RAG_EVIDENCE_GATE_MODE=strict   # 預設，fail-closed
RAG_EVIDENCE_GATE_MODE=relaxed  # 檢查前 8 個片段的原始相關性後放行
```

Relaxed 模式不是完全關閉安全機制，仍要求至少 2 個有意義命中詞，且 BM25 ≥ 0.4 或 Embedding ≥ 0.40。

## 重要檔案

- `rag_demo/hybrid_retrieval.py`：混合檢索與 Evidence Gate。
- `rag_demo/rag_pipeline.py`：路由、檢索、證據檢查與生成流程。
- `rag_demo/fine_evidence.py`：第二階段細粒度證據聚焦。
- `scripts/run_closed_book_oracle_eval.py`：A/B/C 評分器與 grounded fact 評分。
- `scripts/build_relaxed_c_50_eval.py`：Relaxed C 50 題重算報告。
- `evals/leave_rules_corrupted/runs/20260908-155747-relaxed-c-50-results.json`：最新完整結果。
- `RAG_專案管理器.md`、`docs/architecture_by_granularity.md`：專案管理器與架構說明。

## 下一步建議

1. 修正第 39 題的 fine evidence 組合，確保完整列舉型條文不被截斷。
2. 將「回答完整性」提示詞加入生成流程，避免第 10 題類似漏答。
3. 以同一份問題相關評分器重新跑 B 與 C，建立可直接比較的基準。
4. 完成私人 repository 推送後，再進行部署、備份與回滾演練。

