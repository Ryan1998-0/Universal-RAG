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
- 企業功能：知識庫與資料夾管理、文件 ACL、SQLite 對話記憶與可替換模型後端。
- 模型選擇：預設可使用 Ollama Qwen；也支援 LangChain、OpenAI、Anthropic 與 GPT-5.5 子代理。

## RAG 架構圖

![RAG 架構圖](docs/rag-architecture-zh-TW.png)

目前優化版的關鍵設定如下：

| 項目 | 設定 |
| --- | --- |
| Chunk | 父 1024 tokens、子 256 tokens |
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
