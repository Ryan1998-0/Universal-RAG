# MultiHop-RAG 全量檢索評測：無優化版 vs 全優化版

本報告只執行到證據檢索，沒有建立回答 prompt，也沒有呼叫 Qwen、GPT-5.5 或其他生成模型。
完整題庫的 gold evidence 只在檢索完成後用於計算召回指標；輸出 JSON 僅保留排名與指標，不保存檢索文件全文。

- 題庫：`2556` 題 MultiHop-RAG；本次評測 `2556` 題
- Corpus：`609` 份文件
- 題型分布：`{'comparison_query': 856, 'inference_query': 816, 'null_query': 301, 'temporal_query': 583}`
- Embedding：`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Cross-Encoder：`jinaai/jina-reranker-v1-tiny-en`（只在全優化版複雜題啟用）
- 執行時間：`2026-09-13T06:34:09+08:00`

## 主要結果

| 版本 | Character recall@5（MultiHop：加權 Gold fact recall） | 任一證據命中 | 平均耗時（秒／題） |
| --- | ---: | ---: | ---: |
| 無優化版 | 57.6% | 92.9% | 0.482 秒 |
| 全優化版 | 94.4% | 100.0% | 2.970 秒 |

指標分母中的證據題只包含有 gold fact 的題目；null_query 或沒有 gold evidence 的題目另行統計，不把不存在的證據誤算成召回失敗。

## 差異（全優化版 − 無優化版）

- Character recall@5（加權 Gold fact recall）：`+36.8%`
- 任一證據命中率：`+7.1%`
- 平均耗時差異：`+2.488 秒`

## 題型分組

| 版本 | 題型 | 題數 | Character recall@5（加權 Gold fact recall） | 任一證據命中 | 平均耗時（秒／題） |
| --- | --- | ---: | ---: | ---: | ---: |
| 無優化版 | `comparison_query` | 856 | 66.0% | 95.0% | 0.472 秒 |
| 無優化版 | `inference_query` | 816 | 50.9% | 91.4% | 0.496 秒 |
| 無優化版 | `null_query` | 301 | — | — | 0.474 秒 |
| 無優化版 | `temporal_query` | 583 | 59.3% | 91.8% | 0.481 秒 |
| 全優化版 | `comparison_query` | 856 | 91.8% | 100.0% | 2.932 秒 |
| 全優化版 | `inference_query` | 816 | 96.0% | 100.0% | 2.921 秒 |
| 全優化版 | `null_query` | 301 | — | — | 3.232 秒 |
| 全優化版 | `temporal_query` | 583 | 94.8% | 99.8% | 2.959 秒 |

## 實作設定

- 無優化版：hard 600-character chunks, no overlap；raw 50/50 BM25+dense weighted addition；always basic reranker top 5。
- 全優化版：dynamic 1024-token parent / 256-token child, no overlap；RRF, k=60；simple direct top 5; complex Cross-Encoder top 5；命中的子 Chunk 會展開為 parent evidence。
- 兩個版本都使用同一份 corpus、同一個 embedding 模型、同一個 top-5 證據預算與同一批 2,556 題。
- 本次沒有回答模型、沒有對話記憶、沒有外部搜尋、沒有把 gold answer 或 gold evidence 注入檢索查詢。
- Gold fact 命中採完整字串或 BM25 token 覆蓋率至少 45% 的 deterministic heuristic；它衡量檢索召回，不等同於回答正確率或幻覺率。
- MultiHop-RAG 題庫沒有 LegalBench 使用的字元 span 標註，因此本報告的 Character recall@5 欄位以加權 Gold fact recall 對應；LegalBench-RAG 的同名欄位則是字元覆蓋率。
