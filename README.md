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
- 法律領域 Embedding：預設使用 [bugBug04S/legal-embed-modernbert-v2](https://huggingface.co/bugBug04S/legal-embed-modernbert-v2)，並分別套用查詢與文件前綴。
- 企業功能：知識庫與資料夾管理、文件 ACL、SQLite 對話記憶與可替換模型後端。
- 模型選擇：預設可使用 Ollama Qwen；也支援 LangChain、OpenAI、Anthropic 與 GPT-5.5 子代理。

## RAG 架構圖

![RAG 架構圖](docs/rag-architecture-zh-TW.png)

目前優化版的關鍵設定如下：

| 項目 | 設定 |
| --- | --- |
| Chunk | 父 1024 tokens、子 256 tokens |
| Embedding | `bugBug04S/legal-embed-modernbert-v2`（法律檢索微調、768 維） |
| 混合檢索 | BM25 + Embedding，RRF `k=60` |
| 候選與證據 | 前 100 候選，最終 Top 5 |
| 重排 | 複雜問題使用 Cross-Encoder |
| 改寫校驗 | 原始問題與改寫問題 cosine similarity 至少 `0.60` |

## 測試報告

### MultiHop-RAG 全量檢索

使用 2,556 題與 609 份文件，只執行到證據檢索，不呼叫回答模型。結果分為無優化版與全優化版：

| 版本 | Gold fact recall | 任一證據命中 | 完整證據命中 | 首個證據命中 | P95 延遲 | 模型呼叫 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 無優化版 | 57.59% | 92.86% | 28.69% | 79.65% | 574.4 ms | 0 |
| 全優化版 | 94.40% | 99.96% | 86.47% | 99.07% | 4,264.0 ms | 0 |

指標只計算 2,255 題有 Gold evidence 的題目，301 題 `null_query` 不列入分母：

- 任一證據命中：Top 5 合併後包含至少一個 Gold fact。
- 完整證據命中：Top 5 合併後包含該題全部 Gold facts。
- 首個證據命中：Top 5 中至少有一個單獨 Chunk 包含 Gold fact；不是只檢查 Rank 1。

完整結果：[逐題結果](evals/multihop_rag_retrieval_full/retrieval-full-results.json)｜[彙整報告](evals/multihop_rag_retrieval_full/retrieval-full-report.md)｜[摘要](evals/multihop_rag_retrieval_full/retrieval-full-summary.json)

### LegalBench-RAG 公開 710 題完整測試

使用公開 held-out 710 題，依每題 `document_path` 限定文件範圍；只執行證據檢索，不呼叫回答模型。Character recall@5 與任一證據命中是本報告的主要比較欄位：

| 版本 | Character recall@5 | 任一證據命中 |
| --- | ---: | ---: |
| 無優化版 | 15.16% | 37.46% |
| 優化版 | 81.55% | 93.10% |
| 公開 Ettin 版（官方 710 題數據） | 80.41% | 86.76% |

公開 Ettin 數據來源：[LegalBenchRAG-Ettin-150M-Reranker 模型卡](https://huggingface.co/lxyuan/LegalBenchRAG-Ettin-150M-Reranker)。完整結果：[逐題結果](evals/legalbench_public710_full/retrieval-results.json)｜[彙整報告](evals/legalbench_public710_full/retrieval-report.md)｜[摘要](evals/legalbench_public710_full/retrieval-summary.json)

### LegalBench-RAG 100 題（法律 Embedding）

使用 `bugBug04S/legal-embed-modernbert-v2`，只執行證據檢索；兩個版本共用同一批題目與 Top 5 預算：

| 版本 | Character recall@5 | 任一證據命中 |
| --- | ---: | ---: |
| 無優化版 | 7.74% | 16.00% |
| 優化版 | 14.89% | 14.00% |

本次取壓縮檔順序前 100 題，皆為 ContractNLI；此結果適合驗證流程，不代表四個子題型的完整分布。完整結果：[逐題結果](evals/legalbench_rag_retrieval_100_legal_embedding/retrieval-results.json)｜[彙整報告](evals/legalbench_rag_retrieval_100_legal_embedding/retrieval-report.md)｜[摘要](evals/legalbench_rag_retrieval_100_legal_embedding/retrieval-summary.json)

本次文件與查詢 Embedding 使用 GPU；Cross-Encoder 因 ONNX Runtime CUDA 相依版本未載入而以 CPU 執行，優化版平均耗時因此較高。

### LegalBench-RAG 發表方法 100 題（Ettin）

依照公開方法重跑同一批題目：Ettin tokenizer 切 384 tokens、重疊 96 tokens，BM25 取 Top 32；優化版使用 `lxyuan/LegalBenchRAG-Ettin-150M-Reranker` 重排後取 Top 5。兩個版本都只跑檢索，不呼叫回答模型。

| 版本 | Character recall@5 | 任一證據命中 |
| --- | ---: | ---: |
| 無優化版（BM25 直接 Top 5） | 29.96% | 39.00% |
| 發表方法（BM25 Top 32 + Ettin Top 5） | 8.32% | 9.00% |

本次取壓縮檔順序前 100 題，皆為 ContractNLI；不是完整四個子題型的統計。公開模型卡的 710 題 held-out 結果為 Hit@5 86.76%、Character recall@5 80.41%，與本機 100 題樣本不可直接互相比較。完整結果：[逐題結果](evals/legalbench_rag_retrieval_100_ettin/retrieval-results.json)｜[彙整報告](evals/legalbench_rag_retrieval_100_ettin/retrieval-report.md)｜[摘要](evals/legalbench_rag_retrieval_100_ettin/retrieval-summary.json)

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
