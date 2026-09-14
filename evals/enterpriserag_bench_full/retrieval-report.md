# EnterpriseRAG-Bench 全量題目檢索比較報告

本次使用 EnterpriseRAG-Bench 題庫全部 500 題，只執行證據檢索，不呼叫回答模型。
Corpus：511,962 份企業文件。

## 結果

| 版本 | Document recall@30 | 任一證據命中 | 平均耗時（秒／題） |
| --- | ---: | ---: | ---: |
| 無優化版 BM25 Top 30 | 78.01% | 81.06% | 2.492 |
| 優化版 BM25 Top 30 | 78.01% | 81.06% | 3.025 |

指標只對有 `expected_doc_ids` 的題目計算；高階與找不到資訊題沒有指定 Gold 文件，因此列為 n/a，不列入召回率分母。

測試設定：兩個版本均使用原始問題與文件級 BM25 Top 30，不使用 Query Rewrite、Embedding、RRF、重排或回答模型；優化版額外以 40／200 父子 Chunk（子 Chunk overlap 10）展開證據。
