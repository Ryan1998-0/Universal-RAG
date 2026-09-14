# LegalBench-RAG 公開 710 題：三種檢索方法比較

本報告使用公開 held-out 710 題完整測試集，三個版本共用同一批題目與同一份 corpus，只執行證據檢索，不呼叫回答模型。

## 評測條件

- 文件範圍：`specified`；每題只在公開測試紀錄提供的 `document_path` 內檢索。這是已知文件的 benchmark 條件。
- 公開題目：710 題；題型分布：{"contractnli": 100, "cuad": 420, "maud": 177, "privacy_qa": 13}
- Corpus：714 份文件、80,026,615 字元。
- Embedding：`bugBug04S/legal-embed-modernbert-v2`。
- Cross-Encoder：`lxyuan/LegalBenchRAG-Ettin-150M-Reranker`（優化版與公開 Ettin 版）。
- 公開版數據來源：[LegalBenchRAG-Ettin-150M-Reranker 模型卡](https://huggingface.co/lxyuan/LegalBenchRAG-Ettin-150M-Reranker)。
- Reranker device：`cuda`。
- 執行時間：`2026-09-14T14:47:26+08:00`。
- Gold snippets 只在檢索完成後計算指標，沒有放入查詢；沒有回答模型、對話記憶或外部搜尋。

## 結果

自有版本耗時各自移除最高 20 題與最低 20 題，使用剩餘 670 題重新計算；主表保留兩個命中欄位與去除極端值後平均延遲。公開版只引用官方聚合數據，沒有逐題耗時。

| 版本 | Character recall@5 | 任一證據命中／Hit@5 | 去除最高／最低20題後平均 ms/題 |
| --- | ---: | ---: | ---: |
| 無優化版 | 15.16% | 37.46% | 3.6 |
| 優化版 | 81.55% | 93.10% | 954.4 |
| 公開 Ettin 版（官方數據） | 80.41% | 86.76% | — |

## 指標定義

- 任一證據命中：Top 5 中至少涵蓋一個標註證據區段。
- Character recall@5：Top 5 回傳段落覆蓋標註證據字元的比例；自有版本使用本機 exact-span 計算，公開版使用官方模型卡數據。
- 任一證據命中／Hit@5：Top 5 中至少涵蓋一個標註證據區段；公開版沿用官方模型卡的 Hit@5。

## 題型分組

| 版本 | 題型 | 題數 | Character recall@5 | 任一證據命中 |
| --- | --- | ---: | ---: | ---: |
| 無優化版 | `contractnli` | 100 | 63.87% | 87.00% |
| 無優化版 | `cuad` | 420 | 19.80% | 37.38% |
| 無優化版 | `maud` | 177 | 3.44% | 9.60% |
| 無優化版 | `privacy_qa` | 13 | 9.26% | 38.46% |
| 優化版 | `contractnli` | 100 | 99.47% | 100.00% |
| 優化版 | `cuad` | 420 | 94.56% | 99.05% |
| 優化版 | `maud` | 177 | 64.31% | 74.58% |
| 優化版 | `privacy_qa` | 13 | 90.25% | 100.00% |

## 架構設定

- 無優化版：硬切 600 字元、無問題改寫、BM25/向量 50/50 直接加權、固定重排 Top 5。
- 優化版：動態 1024-token Parent / 256-token Child、RRF（k=60）、複雜度路由、複雜題 Cross-Encoder 重排 Top 5，命中的 Child 展開為 Parent 證據。
- 公開 Ettin 版：Ettin tokenizer 對齊 384-token 段落、96-token 重疊、BM25 Top 32，再以 LegalBench-RAG Ettin Cross-Encoder 重排 Top 5；本報告直接引用官方 710 題聚合數據。
- 三個版本均限制在同一題的 `document_path`，因此本結果不包含跨文件自動找文件的難度。
- 三個版本都使用同一個 `document_path` 文件範圍；公開 Ettin 版沿用其發表的 BM25 + Cross-Encoder 流程。

## 檔案

- 完整逐題結果：[retrieval-results.json](retrieval-results.json)
- 摘要：[retrieval-summary.json](retrieval-summary.json)
- 可續跑 checkpoint：`checkpoints/`
