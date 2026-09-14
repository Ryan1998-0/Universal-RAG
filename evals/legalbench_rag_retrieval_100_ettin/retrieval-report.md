# LegalBench-RAG 發表方法：100 題檢索重跑

本次將優化版改為公開 LegalBench-RAG 方法：以 Ettin tokenizer 切 384-token 段落、96-token 重疊、BM25 Top 32，再以 LegalBench-RAG 微調的 Ettin 150M Cross-Encoder 重排並取 Top 5。

參考：[LegalBenchRAG-Ettin-150M-Reranker 模型卡](https://huggingface.co/lxyuan/LegalBenchRAG-Ettin-150M-Reranker)｜[LegalBench-RAG 論文](https://arxiv.org/abs/2408.10343)

## 評測範圍

- 論文對齊題集：6,858 題；本次取壓縮檔順序前 100 題。
- 本次 100 題皆屬 ContractNLI；另外 6,758 題未納入本次樣本。
- Corpus：714 份文件、80,026,615 字元；建立 59,297 個 passages。
- 只執行證據檢索；沒有回答模型、對話記憶、外部搜尋或 Gold evidence 注入。

## 主要結果

| 版本 | Character recall@5 | 任一證據命中 |
| --- | ---: | ---: |
| 無優化版（BM25） | 29.96% | 39.00% |
| LegalBench-RAG（Ettin） | 8.32% | 9.00% |

Character recall@5 是 Top 5 回傳段落覆蓋標註證據字元的比例；任一證據命中表示 Top 5 至少涵蓋一個標註證據區段。

## 題型分組

| 版本 | 題型 | 題數 | Character recall@5 | 任一證據命中 |
| --- | --- | ---: | ---: | ---: |
| 無優化版（BM25） | `contractnli` | 100 | 29.96% | 39.00% |
| LegalBench-RAG（Ettin） | `contractnli` | 100 | 8.32% | 9.00% |

## 可比性說明

- 無優化版是同一段落切分與 BM25 Top 32 的直接 Top 5 基線；優化版只增加發表的 Ettin Cross-Encoder 重排，便於隔離重排器的影響。
- 本次評測直接使用 Ettin tokenizer 的 offset mapping 建立段落，讓段落邊界與 Cross-Encoder 的 384-token 設定一致。
- 模型卡的 86.76% Hit@5 與 80.41% Character recall@5 是 710 題 held-out 評測；本報告是本機 archive 前 100 題，不能直接視為同一統計結果。

## 輸出

- 逐題結果 JSON 同時保存排名、來源與字元區間，不保存 passage 全文。
- checkpoint 可用 `--resume` 接續；本報告沒有呼叫生成模型。
