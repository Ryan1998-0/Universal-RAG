# RAG架構

本機優先的多格式 RAG 系統。文件、知識庫、Embedding、重排模型與回答模型都可以替換，適合用來做企業文件問答與檢索實驗。

[線上介面展示](https://ryan1998-0.github.io/Universal-RAG/rag-demo/)｜本機問答介面：`http://127.0.0.1:8765/`

## 核心功能

- 多格式匯入：PDF、掃描 PDF、圖片、DOCX、TXT、Markdown、JSON，支援 OCR。
- 父子 Chunk：用小型子 Chunk 做精準檢索，再回傳較完整的父 Chunk 作為證據。
- 混合檢索：BM25 關鍵字檢索與 Embedding 向量檢索，以 RRF 融合排名。
- 問題理解：問題改寫、語意相似度校驗，以及簡單／複雜問題路由。
- 智慧重排：簡單問題直接取 Top 5；複雜問題才使用 Cross-Encoder 重排。
- 證據約束：Evidence Gate、來源引用、證據不足拒答，降低模型幻覺。
- 可替換 Embedding：透過 `RAG_EMBEDDING_MODEL` 與 `RAG_EMBEDDING_DIMENSIONS` 設定模型與向量維度；預設使用 [bugBug04S/legal-embed-modernbert-v2](https://huggingface.co/bugBug04S/legal-embed-modernbert-v2)，並分別套用查詢與文件前綴。
- 企業功能：知識庫與資料夾管理、文件 ACL、SQLite 對話記憶與可替換模型後端。
- 模型選擇：預設可使用 Ollama Qwen；也支援 OpenAI、Anthropic 與 GPT-5.5 子代理。

## RAG 架構圖

![RAG 架構圖](docs/rag-architecture-zh-TW.png)

目前優化版的關鍵設定如下：

| 項目 | 設定 |
| --- | --- |
| Chunk | 父 1024 tokens、子 256 tokens |
| Embedding | 可替換；`RAG_EMBEDDING_MODEL` 設定模型、`RAG_EMBEDDING_DIMENSIONS` 設定維度，預設為 `bugBug04S/legal-embed-modernbert-v2`（768 維） |
| 混合檢索 | BM25 + Embedding，RRF `k=60` |
| 候選與證據 | 前 100 候選，最終 Top 5 |
| 重排 | 複雜問題使用 Cross-Encoder |
| 改寫校驗 | 原始問題與改寫問題 cosine similarity 至少 `0.60` |

替換 Embedding 時，先在 `.env` 或部署環境設定模型與維度，再重新建立索引，避免不同模型或向量維度混用：

```env
RAG_EMBEDDING_MODEL=your-org/your-embedding-model
RAG_EMBEDDING_DIMENSIONS=768
```

## 測試報告

### MultiHop-RAG

本次使用 MultiHop-RAG 全量資料集，包含 2,556 題測試題與 609 份新聞文件，用來比較無優化版與全優化版的證據檢索表現。

| 版本 | Character recall@5 | 任一證據命中 | 平均耗時（秒／題） |
| --- | ---: | ---: | ---: |
| 無優化版 | 57.59% | 92.86% | 0.482 秒 |
| 全優化版 | 94.40% | 99.96% | 2.970 秒 |

完整結果：[逐題結果](evals/multihop_rag_retrieval_full/retrieval-full-results.json)｜[彙整報告](evals/multihop_rag_retrieval_full/retrieval-full-report.md)｜[摘要](evals/multihop_rag_retrieval_full/retrieval-full-summary.json)

### LegalBench-RAG

使用公開 held-out 710 題，依每題 `document_path` 限定文件範圍：

| 版本 | Character recall@5 | 任一證據命中 | 平均耗時（秒／題） |
| --- | ---: | ---: | ---: |
| 無優化版 | 15.16% | 37.46% | 0.004 秒 |
| 優化版 | 81.55% | 93.10% | 0.954 秒 |
| 公開 Ettin 版 | 80.41% | 86.76% | — |

公開 Ettin 數據來源：[LegalBenchRAG-Ettin-150M-Reranker 模型卡](https://huggingface.co/lxyuan/LegalBenchRAG-Ettin-150M-Reranker)。完整結果：[逐題結果](https://github.com/Ryan1998-0/Universal-RAG/blob/main/evals/legalbench_public710_full/retrieval-results.json)｜[彙整報告](https://github.com/Ryan1998-0/Universal-RAG/blob/main/evals/legalbench_public710_full/retrieval-report.md)｜[摘要](https://github.com/Ryan1998-0/Universal-RAG/blob/main/evals/legalbench_public710_full/retrieval-summary.json)

### EnterpriseRAG-Bench

使用完整 500 題核心題庫與 511,962 份企業文件；無優化版與優化版採相同 BM25 Top 30、父／子 Chunk 200／40、overlap 10 設定，差異為優化版增加父子 Chunk 證據展開。兩個版本均不進行問題改寫、Embedding 或重排。

| 版本 | Document recall@30 | 任一證據命中 | 平均耗時（秒／題） |
| --- | ---: | ---: | ---: |
| 無優化版 BM25 Top 30 | 78.01% | 81.06% | 2.492 |
| 優化版 BM25 Top 30 | 78.01% | 81.06% | 3.025 |

完整結果：[逐題結果](evals/enterpriserag_bench_full/retrieval-results.json)｜[彙整報告](evals/enterpriserag_bench_full/retrieval-report.md)｜[摘要](evals/enterpriserag_bench_full/retrieval-summary.json)

## RAG 資料庫來源

本專案的知識庫可替換；目前測試與示範資料來源如下：

| 用途 | 公開來源 | 本機用途 |
| --- | --- | --- |
| MultiHop-RAG 全量題庫 | [yixuantt/MultiHop-RAG](https://github.com/yixuantt/MultiHop-RAG) | 2,556 題、609 份新聞文件，用於本次全量檢索評測 |
| LegalBench-RAG | [ZeroEntropy-AI/legalbenchrag](https://github.com/ZeroEntropy-AI/legalbenchrag) | 法律文件 RAG benchmark |
| EnterpriseRAG-Bench | [onyx-dot-app/EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench) | 企業文件 RAG benchmark |
| Open RAG Benchmark | [vectara/open-rag-bench](https://github.com/vectara/open-rag-bench) | PDF 文件 RAG benchmark |
| Fujitsu RAG Hard Benchmark | [FujitsuResearch/Fujitsu-RAG-Hard-Benchmark](https://github.com/FujitsuResearch/Fujitsu-RAG-Hard-Benchmark) | 多跳與困難題型 benchmark |

MultiHop-RAG 的原始檔位於本機評測目錄 `RAG測試題庫/01_MultiHop-RAG/`；原始資料的授權條件依各公開來源為準。
