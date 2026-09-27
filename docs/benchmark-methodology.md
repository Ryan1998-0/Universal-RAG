# Benchmark 方法與解讀範圍

README 的五組表格是各次歷史**檢索階段**實驗，不是目前 `/v1/ask` 端到端驗收。逐題 JSON、摘要和同目錄的報告是原始紀錄；這次只修正文字標籤與解讀，沒有重跑或改寫既有數值。

| 資料集 | 範圍與分母 | 主指標 | 可得結論 |
| --- | --- | --- | --- |
| MultiHop-RAG | 2,556 題、609 份文件；只以有 Gold fact 的題目計算召回 | Gold fact 完整字串或 BM25 詞項覆蓋至少 45% 的加權召回@5；另計任一 Gold fact 命中 | 衡量啟發式證據召回；題庫沒有字元 span，不能稱為字元召回 |
| LegalBench-RAG | 710 題、714 份文件；每題已知 `document_path`，只在該文件內搜尋 | Gold span 字元覆蓋率@5，以及 Hit@5 | 衡量已知文件內找段落；不衡量全知識庫找文件 |
| EnterpriseRAG-Bench | 500 題、511,962 份文件；只有標註 `expected_doc_ids` 的題目進入召回分母 | Document recall@30，以及至少一個 Gold 文件命中 | 只比較 BM25 文件檢索與父子 Chunk 展開 |
| Open RAG Benchmark | 3,045 題、1,000 份 PDF；索引文字與表格，圖片沒有 OCR | Gold 文件 recall@30，以及 Gold 文件加 Gold section 同時命中 | 只比較 BM25 文字檢索 |
| Fujitsu RAG Hard | 100 題、34 份 PDF；有可抽取文字的 1,775／1,794 頁進入索引 | Gold 文件 recall@30，以及 Gold 文件加頁碼命中 | 只比較 BM25 文字頁面檢索；圖片頁沒有 OCR |

每個資料集的候選數、Chunk、檢索元件、模型、設備和耗時口徑以對應的 `retrieval-report.md` 為準。LegalBench 主表的平均耗時去掉各版本最快與最慢各 20 題；MultiHop 與其他表格未宣稱採用相同裁切方式。平均耗時無法代替 p95 或端到端延遲。歷史 artifact 沒有完整的語料雜湊、模型 revision/digest、設備設定與延遲分布，不能補寫成已驗證的來源資訊。這些缺口要在新一輪驗收中紀錄。

## 新的正式 API 驗收紀錄

[`run_production_release_gate.py`](../scripts/run_production_release_gate.py) 使用固定五份 staging fixture，呼叫正式 `/v1/ask`，並在新 artifact 中紀錄：

- Git commit 與 dirty 狀態、設定檔 SHA-256、各 fixture SHA-256、檔案至 source ID 對照。
- 操作者聲明的部署 image digest、生成／Embedding／Reranker 模型 revision、active index ID、parser／chunk schema 版本、硬體設定。
- 服務回報的預設模型、Embedding／Reranker 名稱和 active index ID。active index 與聲明不同時，驗收會在送問題前失敗。
- 各題回答、引用、驗證狀態、HTTP 狀態與 client wall time；並給出 min、p50、p95、p99、max、mean。百分位數使用 nearest-rank；分母僅為正常驗收題，跨租戶拒絕請求另列，HTTP error rate 也是正常驗收題的非 200 比例。

模型 revision、image digest、parser／chunk 版本與硬體設定由操作者聲明；API 目前只回報模型名稱與 active index ID。腳本可核對 active index，但無法從 API 證實其他聲明，亦無法證實 staging 真正匯入的 bytes 與本機 fixture 完全相同。因此上線前要保存匯入操作紀錄並人工核對；驗收報告把 `staging_content_verified` 保持為 `false`，不把本機 SHA-256 當成遠端驗證結果。七題驗收是固定樣本與規則判定，仍需逐題人工審查引用支持的推論。
