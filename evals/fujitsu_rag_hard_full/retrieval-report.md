# Fujitsu RAG Hard Benchmark 全量檢索比較報告

本次使用公開 benchmark 的 100 題、34 份參考 PDF（共 1,794 頁），只執行證據檢索，不呼叫回答模型。
資料集：[Fujitsu RAG Hard Benchmark](https://github.com/FujitsuResearch/Fujitsu-RAG-Hard-Benchmark)；背景介紹：[Fujitsu Research 技術文章](https://blog-en.fltech.dev/entry/2026/03/11/RAG-Hard-Benchmark-en)。

## 結果

| 版本 | Document recall@30 | 任一證據命中 | 平均耗時（秒／題） |
| --- | ---: | ---: | ---: |
| 無優化版 BM25 Top 30 | 79.50% | 80.00% | 0.020 |
| 優化版 BM25 Top 30 | 78.50% | 74.00% | 0.043 |

指標定義：Document recall@30 是每題 Gold 文件出現在 Top 30 證據的比例；若一題有多份 Gold 文件，先計算該題命中的 Gold 文件比例，再對 100 題取平均。任一證據命中代表至少一個 Top 30 證據同時符合 Gold 文件與 Gold 頁碼。

測試設定：無優化版直接以 page-local 200-token parent chunk 做 BM25 Top 30；優化版以 page-local 40-token child chunk（overlap 10）檢索，再將命中的 child 展開至 200-token parent evidence。兩者均不使用問題改寫、Embedding、RRF、重排或回答模型。

文件處理：本次本機索引 1,775/1,794 頁有可抽取文字；圖片型頁面未加入 OCR，因此若 Gold 只存在於圖片，BM25 文字檢索可能無法命中。PDF 依 benchmark 與各原始發布者條款留在本機暫存，未放入本專案結果。

截至本次檢查，官方公開 repository 與技術文章提供資料集、標註與評測腳本，但沒有可直接對齊本報告 Document recall@30／任一證據命中／平均耗時三欄的聚合基準數據，因此本報告不填入外部數值。
