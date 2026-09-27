# RAG架構

本機優先的多格式 RAG 系統。文件、知識庫、Embedding、重排模型與回答模型都可以替換，適合用來做企業文件問答與檢索實驗。

目前示範預設使用勞動部《勞動基準法》官方最新頁面（民國 113 年 7 月 31 日修正），資料源設定為 `RAG_PROFILE=labor_standards_act`；引擎程式以 GitHub `main` 分支為準，所有問題皆啟用 Cross-Encoder 重排。

## 核心功能

- 多格式匯入：PDF、掃描 PDF、圖片、DOCX、TXT、Markdown、JSON，支援 OCR。
- 父子 Chunk：用小型子 Chunk 做精準檢索，再回傳較完整的父 Chunk 作為證據。
- 混合檢索：BM25 關鍵字檢索與 Embedding 向量檢索，以 RRF 融合排名。
- 問題理解：問題改寫、語意相似度校驗，以及簡單／複雜問題路由。
- 智慧重排：前 100 個候選皆使用 Cross-Encoder 重排，最後保留 Top 5。
- 證據約束：Evidence Gate、來源引用、證據不足拒答，降低模型幻覺。
- 可替換 Embedding：本機腳本與 Compose／正式服務預設均為 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`（384 維）。正式服務以 FastEmbed 的 `query_embed`／`passage_embed` 分別編碼查詢和文件；本機腳本使用 SentenceTransformers。兩種索引不可直接互換。
- 企業功能：知識庫與資料夾管理、文件 ACL、SQLite 對話記憶與可替換模型後端。
- API 與觀測：資料輸入／輸出、模型節點替換、逐節點耗時與 JSONL 除錯記錄，詳見 [API 與可觀測性契約](docs/api-contract.md)。

## RAG 架構圖

![RAG 架構圖](docs/rag-architecture-zh-TW.png)

目前優化版的關鍵設定如下：

| 項目 | 設定 |
| --- | --- |
| Chunk | 父 1024 tokens、子 256 tokens |
| Embedding | `RAG_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`，`RAG_EMBEDDING_DIMENSIONS=384`；與 Compose、正式服務程式及 `.env.production.example` 一致 |
| 混合檢索 | BM25 0.6 + Embedding 0.4，RRF `k=60` |
| 候選與證據 | 前 100 候選，最終 Top 5 |
| 重排 | 所有問題使用 Cross-Encoder；`RAG_COMPLEXITY_ROUTING_ENABLED=0` |
| 改寫校驗 | 原始問題與改寫問題 cosine similarity 至少 `0.60` |

以下為 Compose／正式服務預設值；替換模型時，先確認 FastEmbed 支援該模型及實際向量維度，再以相同設定建立並啟用新索引。啟用中的索引若與服務的模型、維度、Chunk schema 或 Qdrant collection 不一致，`/v1/ask` 會回傳 `INDEX_CONFIGURATION_MISMATCH`，避免用錯誤向量查詢。完整切換流程見[正式環境操作手冊](docs/production-runbook.md)。

```env
RAG_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RAG_EMBEDDING_DIMENSIONS=384
```

## 測試報告

以下為各資料集當時的**檢索階段**實驗，沒有評估生成答案的正確率、引用充分性或正式 `/v1/ask` 服務。不同資料集的指標定義、搜尋範圍、硬體與耗時統計方式不同，數值不可直接橫向比較；也不能視為目前部署版本的保證。各表的「平均耗時」僅對應該次實驗的檢索流程，不含文件匯入及回答生成。詳見[評測方法與可重現性](docs/benchmark-methodology.md)。

### MultiHop-RAG 新聞資料

無優化版  chunk: 600/0<br>
檢索: BM25 & Embedding 直接相加<br>
證據: 直接取 TOP5

全優化版  chunk: 1024/256<br>
檢索: BM25 & Embedding RRF 加權(k=60)<br>
證據: 簡單問題直接取 TOP5；複雜問題使用 Cross-Encoder Top 5；命中子 Chunk 展開父 Chunk 作為證據 Prompt

| 版本 | 加權 Gold fact recall@5 | 任一證據命中 | 平均耗時（秒／題） |
| --- | ---: | ---: | ---: |
| 無優化版 | 57.59% | 92.86% | 0.482 |
| 全優化版 | 94.40% | 99.96% | 2.970 |

完整結果：[逐題結果](evals/multihop_rag_retrieval_full/retrieval-full-results.json)｜[彙整報告](evals/multihop_rag_retrieval_full/retrieval-full-report.md)｜[摘要](evals/multihop_rag_retrieval_full/retrieval-full-summary.json)

本表為 2,556 題、609 份新聞文件的檢索實驗。題庫沒有字元 span 標註，94.40% 是 Gold fact 字串／詞項覆蓋啟發式的加權召回，**不是**字元召回或回答正確率；null 題不計入召回分母。

------------------------------------------------------------

### LegalBench-RAG 法律資料

無優化版  chunk: 600/0<br>
檢索: BM25 & Embedding 直接相加<br>
證據: 直接取 TOP5

優化版  chunk: 1024/256<br>
檢索: BM25 & Embedding RRF 加權(k=60)<br>
證據: 簡單問題直接取 TOP5；複雜問題使用 Cross-Encoder Top 5；命中子 Chunk 展開父 Chunk 作為證據 Prompt

公開 Ettin 版  chunk: 384/96<br>
檢索: BM25 Top 32<br>
證據: Ettin Cross-Encoder Top 5

| 版本 | Character recall@5 | 任一證據命中 | 平均耗時（秒／題） |
| --- | ---: | ---: | ---: |
| 無優化版 | 15.16% | 37.46% | 0.004 |
| 優化版 | 81.55% | 93.10% | 0.954 |

完整結果：[逐題結果](evals/legalbench_public710_full/retrieval-results.json)｜[彙整報告](evals/legalbench_public710_full/retrieval-report.md)｜[摘要](evals/legalbench_public710_full/retrieval-summary.json)

本表為 710 題的**已知文件**檢索：每題只搜尋標註的 `document_path`，沒有衡量從整個知識庫找對文件的能力。Character recall 是 Gold span 字元覆蓋率；耗時是 CUDA 環境下去掉最快與最慢各 20 題後的平均值。

------------------------------------------------------------

### EnterpriseRAG-Bench 企業資料

無優化版  chunk: 文件級<br>
檢索: BM25<br>
證據: 直接取 TOP30

優化版  chunk: 200/40（子 Chunk overlap 10）<br>
檢索: BM25<br>
證據: 取 TOP30；命中子 Chunk 展開父 Chunk

| 版本 | Document recall@30 | 任一證據命中 | 平均耗時（秒／題） |
| --- | ---: | ---: | ---: |
| 無優化版 BM25 Top 30 | 78.01% | 81.06% | 2.492 |
| 優化版 BM25 Top 30 | 78.01% | 81.06% | 3.025 |

完整結果：[逐題結果](evals/enterpriserag_bench_full/retrieval-results.json)｜[彙整報告](evals/enterpriserag_bench_full/retrieval-report.md)｜[摘要](evals/enterpriserag_bench_full/retrieval-summary.json)

本表只使用 BM25；父子 Chunk 展開後，文件召回沒有改善，平均耗時由 2.492 秒增至 3.025 秒。

------------------------------------------------------------

### Open RAG Benchmark PDF資料

無優化版  chunk: 200/0<br>
檢索: BM25<br>
證據: 直接取 TOP30

優化版  chunk: 200/40（子 Chunk overlap 10）<br>
檢索: BM25<br>
證據: 取 TOP30；命中子 Chunk 展開父 Chunk

| 版本 | Document recall@30 | 任一證據命中 | 平均耗時（秒／題） |
| --- | ---: | ---: | ---: |
| 無優化版 BM25 Top 30 | 99.15% | 96.32% | 0.075 |
| 優化版 BM25 Top 30 | 98.33% | 93.40% | 0.246 |

完整結果：[逐題結果](evals/open_rag_bench_full/retrieval-results.json)｜[彙整報告](evals/open_rag_bench_full/retrieval-report.md)｜[摘要](evals/open_rag_bench_full/retrieval-summary.json)

本表只使用文字／表格的 BM25，沒有對 PDF 圖片做 OCR；父子 Chunk 版本的兩項命中率較低，耗時較高。

------------------------------------------------------------

### Fujitsu RAG Hard Benchmark 困難題型資料

無優化版  chunk: 1024/0<br>
檢索: BM25<br>
證據: 直接取 TOP30

優化版  chunk: 1024/256（子 Chunk overlap 10）<br>
檢索: BM25<br>
證據: 取 TOP30；命中子 Chunk 展開父 Chunk

| 版本 | Document recall@30 | 任一證據命中 | 平均耗時（秒／題） |
| --- | ---: | ---: | ---: |
| 無優化版 BM25 Top 30 | 79.50% | 79.00% | 0.020 |
| 優化版 BM25 Top 30 | 79.50% | 78.00% | 0.024 |

完整結果：[逐題結果](evals/fujitsu_rag_hard_full_1024_256/retrieval-results.json)｜[彙整報告](evals/fujitsu_rag_hard_full_1024_256/retrieval-report.md)｜[摘要](evals/fujitsu_rag_hard_full_1024_256/retrieval-summary.json)

本表只使用 BM25；1,794 頁中 1,775 頁有可抽取文字，圖片頁沒有 OCR。父子 Chunk 版本的文件召回持平、頁面命中略降。

------------------------------------------------------------

## RAG 資料庫來源

本專案的知識庫可替換；目前測試與示範資料來源如下：

| 用途 | 公開來源 | 本機用途 |
| --- | --- | --- |
| MultiHop-RAG 全量題庫 | [yixuantt/MultiHop-RAG](https://github.com/yixuantt/MultiHop-RAG) | 新聞文件 RAG benchmark |
| LegalBench-RAG | [ZeroEntropy-AI/legalbenchrag](https://github.com/ZeroEntropy-AI/legalbenchrag) | 法律文件 RAG benchmark |
| EnterpriseRAG-Bench | [onyx-dot-app/EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench) | 企業文件 RAG benchmark |
| Open RAG Benchmark | [vectara/open-rag-bench](https://github.com/vectara/open-rag-bench) | PDF 文件 RAG benchmark |
| Fujitsu RAG Hard Benchmark | [FujitsuResearch/Fujitsu-RAG-Hard-Benchmark](https://github.com/FujitsuResearch/Fujitsu-RAG-Hard-Benchmark) | 多跳與困難題型 benchmark |
