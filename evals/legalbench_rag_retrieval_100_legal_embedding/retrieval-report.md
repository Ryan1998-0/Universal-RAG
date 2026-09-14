# LegalBench-RAG 檢索評測：無優化版 vs 優化版

本報告只執行到證據檢索，沒有建立回答 prompt，也沒有呼叫 Qwen、GPT-5.5 或其他生成模型。
Gold snippets 只在檢索完成後用於計算字元級召回與命中指標；輸出 JSON 不保存檢索文件全文。

- 官方論文題數：`6858` 題
- 下載檔題數：`6889` 題
- 本次評測：`100` 題
- ContractNLI 排除：`31` 題（`nda-10` 類別，對齊論文的 946 題）
- 本次取樣：前 `100` 題（依壓縮檔順序；完整對齊題集為 `6858` 題）
- 本次未測試的對齊題目：`6758` 題
- Corpus：`714` 份文件、`80,026,615` 字元
- Embedding：`bugBug04S/legal-embed-modernbert-v2`
- Cross-Encoder：`BAAI/bge-reranker-base`（只在優化版複雜題啟用）
- 執行時間：`2026-09-14T10:39:17+08:00`

## 執行環境備註

- 文件與查詢 Embedding 使用 NVIDIA RTX 3060 Ti 的 CUDA 推論。
- 本次 `BAAI/bge-reranker-base` 的 ONNX Runtime CUDA provider 因 CUDA 版本相依性未載入，實際以 CPU 執行；因此優化版的重排延遲包含 CPU Cross-Encoder 成本，檢索指標不受此執行環境差異影響。

## 主要結果

| 版本 | Character recall@5 | 任一證據命中 |
| --- | ---: | ---: |
| 無優化版 | 7.74% | 16.00% |
| 優化版 | 14.89% | 14.00% |

Character recall@5 是 Top 5 回傳段落覆蓋標註證據字元的比例；任一證據命中表示 Top 5 至少涵蓋一個標註證據區段。

## 題型分組

| 版本 | 題型 | 題數 | Character recall@5 | 任一證據命中 |
| --- | --- | ---: | ---: | ---: |
| 無優化版 | `contractnli` | 100 | 7.74% | 16.00% |
| 優化版 | `contractnli` | 100 | 14.89% | 14.00% |

## 實作設定

- 無優化版：hard 600-character chunks, no overlap；raw 50/50 BM25+dense weighted addition；always lexical/semantic reranker to Top 5。
- 優化版：dynamic 1024-token parent / 256-token child, no overlap；RRF, k=60；simple direct Top 5; complex Cross-Encoder Top 5；命中的子 Chunk 會展開為父 Chunk 證據。
- 兩個版本使用同一份 714 文件 corpus、同一個 Embedding 模型、同一批 100 題與同一個 Top 5 證據預算。
- 本次沒有回答模型、沒有對話記憶、沒有外部搜尋、沒有把 Gold answer 或 Gold snippets 注入檢索查詢。
- 合法文件原始檔保留在下載壓縮檔中；評測輸出只保存檔名、字元區間、排名與分數欄位。
