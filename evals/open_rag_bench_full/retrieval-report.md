# Open RAG Benchmark 全量檢索比較報告

本次使用 3,045 題、1,000 份 PDF 文件與 18,840 個 section，只執行證據檢索，不呼叫回答模型。
資料集：[vectara/open_ragbench](https://huggingface.co/datasets/vectara/open_ragbench)；原始專案：[vectara/open-rag-bench](https://github.com/vectara/open-rag-bench)。

## 結果

| 版本 | Document recall@30 | 任一證據命中 | 平均耗時（秒／題） |
| --- | ---: | ---: | ---: |
| 無優化版 BM25 Top 30 | 99.15% | 96.32% | 0.075 |
| 優化版 BM25 Top 30 | 98.33% | 93.40% | 0.246 |

指標定義：Document recall@30 代表 Gold 文件出現在 Top 30 證據；任一證據命中代表 Gold 文件與 Gold section 同時出現在 Top 30 證據。

測試設定：無優化版直接以 200-token parent chunk 做 BM25 Top 30；優化版以 40-token child chunk（overlap 10）檢索，再展開至 200-token parent evidence。兩者均不使用 Query Rewrite、Embedding、RRF、重排或回答模型。

## 公開參考結果

[Linkence-Benchmarks 公開 full 結果](https://github.com/Linkence-AI/Linkence-Benchmarks/blob/main/README.md)；[metrics.json](https://github.com/Linkence-AI/Linkence-Benchmarks/blob/main/results/open_ragbench_full/metrics.json) 同樣評測 3,045 題，使用 hybrid hashed-TF-IDF + `text-embedding-3-small`、section 單位、Top 20、無重排；其公開 relaxed document hit@20 為 99.77%、strict section hit@20 為 96.91%、p50 latency 為 0.419 秒。這些指標與本次 Top 30 平均耗時不同，因此只作外部參考。

資料集含文字、表格與圖片標記；圖片內容以 base64 儲存且沒有 OCR／caption，本次 BM25 只索引文字與表格內容。
