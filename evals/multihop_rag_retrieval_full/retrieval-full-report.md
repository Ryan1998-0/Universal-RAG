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

| 版本 | 加權 Gold fact recall | 任一證據命中 | 完整證據命中 | 首個命中率 | 平均檢索 ms | P95 檢索 ms | 複雜題數 | 重排題數 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 無優化版 | 57.6% | 92.9% | 28.7% | 79.6% | 481.6 | 574.4 | 0 | 2556 |
| 全優化版 | 94.4% | 100.0% | 86.5% | 99.1% | 2969.9 | 4264.0 | 2556 | 2556 |

指標分母中的證據題只包含有 gold fact 的題目；null_query 或沒有 gold evidence 的題目另行統計，不把不存在的證據誤算成召回失敗。

## 差異（全優化版 − 無優化版）

- 加權 Gold fact recall：`+36.8%`
- 任一證據命中率：`+7.1%`
- 完整證據命中率：`+57.8%`
- 平均檢索延遲：`+2488.3 ms`
- P95 檢索延遲：`+3689.6 ms`

## 題型分組

| 版本 | 題型 | 題數 | 有證據題數 | 加權 fact recall | 完整證據命中 | 任一證據命中 | 平均 ms | P95 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 無優化版 | `comparison_query` | 856 | 856 | 66.0% | 39.5% | 95.0% | 471.7 | 571.8 |
| 無優化版 | `inference_query` | 816 | 816 | 50.9% | 14.6% | 91.4% | 495.8 | 573.5 |
| 無優化版 | `null_query` | 301 | 0 | — | — | — | 473.7 | 571.6 |
| 無優化版 | `temporal_query` | 583 | 583 | 59.3% | 32.6% | 91.8% | 480.5 | 580.8 |
| 全優化版 | `comparison_query` | 856 | 856 | 91.8% | 83.1% | 100.0% | 2932.1 | 4200.3 |
| 全優化版 | `inference_query` | 816 | 816 | 96.0% | 89.0% | 100.0% | 2921.1 | 4106.4 |
| 全優化版 | `null_query` | 301 | 0 | — | — | — | 3231.5 | 4579.5 |
| 全優化版 | `temporal_query` | 583 | 583 | 94.8% | 88.0% | 99.8% | 2958.9 | 4188.6 |

## 實作設定

- 無優化版：hard 600-character chunks, no overlap；raw 50/50 BM25+dense weighted addition；always basic reranker top 5。
- 全優化版：dynamic 1024-token parent / 256-token child, no overlap；RRF, k=60；simple direct top 5; complex Cross-Encoder top 5；命中的子 Chunk 會展開為 parent evidence。
- 兩個版本都使用同一份 corpus、同一個 embedding 模型、同一個 top-5 證據預算與同一批 2,556 題。
- 本次沒有回答模型、沒有對話記憶、沒有外部搜尋、沒有把 gold answer 或 gold evidence 注入檢索查詢。
- Gold fact 命中採完整字串或 BM25 token 覆蓋率至少 45% 的 deterministic heuristic；它衡量檢索召回，不等同於回答正確率或幻覺率。
