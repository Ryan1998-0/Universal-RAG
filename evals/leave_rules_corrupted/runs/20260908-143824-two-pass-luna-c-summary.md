# 數值污染副本：二次 RAG + GPT-5.6-Luna 10 題重跑

- Benchmark：`taiwan-leave-rules-corrupted-numeric-v1`
- 模型：`gpt-5.6-luna`（A/B 沿用同一批隔離測試；C 使用二次 RAG）
- 二次 RAG：先取 parent chunks，再將每個 parent 細切至原長度約 1/3，最後取 3 個 fine evidence chunks
- 檢索紀錄：`evals/leave_rules_corrupted/runs/20260908-143537-two-pass-retrieval-results.json`

| 模式 | 完整通過 | 平均分 |
|---|---:|---:|
| A Closed-book | 0/10 | 20.0 |
| B Oracle-context | 10/10 | 100.0 |
| C 二次 RAG + Luna | 9/10 | 90.0 |
| C 一次 RAG 基線 | 9/10 | 90.0 |

| # | 題目 ID | A | B | C 二次 | 一次 C | Gate |
|---:|---|---:|---:|---:|---:|---|
| 1 | corrupt-marriage-leave | 50.0 | 100.0 | 100.0 | 100.0 | True |
| 2 | corrupt-parent-bereavement | 50.0 | 100.0 | 100.0 | 100.0 | True |
| 3 | corrupt-spouse-adoptive-bereavement | 50.0 | 100.0 | 100.0 | 100.0 | True |
| 4 | corrupt-sibling-bereavement | 50.0 | 100.0 | 0.0 | 0.0 | False |
| 5 | corrupt-outpatient-sick-limit | 0.0 | 100.0 | 100.0 | 100.0 | True |
| 6 | corrupt-inpatient-sick-limit | 0.0 | 100.0 | 100.0 | 100.0 | True |
| 7 | corrupt-combined-sick-limit | 0.0 | 100.0 | 100.0 | 100.0 | True |
| 8 | corrupt-sick-pay-ratio | 0.0 | 100.0 | 100.0 | 100.0 | True |
| 9 | corrupt-personal-leave-limit | 0.0 | 100.0 | 100.0 | 100.0 | True |
| 10 | corrupt-adverse-action-threshold | 0.0 | 100.0 | 100.0 | 100.0 | True |

## 結果解讀

- 二次切細後，10 題的每題檢索 trace 都是 2 個 parent、3 個 fine evidence，chunk fraction 為 0.333333；證據上下文更集中。
- C 分數與一次 RAG 基線相同（預期 9/10、90.0 分）。第 4 題仍為 fail-closed，因 evidence gate 沒有把分散在「兄弟姊妹」詞組中的主體與數值判定為 answer-bearing evidence；不是二次切細後完全沒抓到原始證據。
- A/B/C 的分數針對合成污染數值，僅衡量是否遵循測試副本，不代表現行法規正確性。

## 第 4 題證據狀態

二次 RAG 的 fine evidence 仍包含：`曾祖父母、兄弟姊妹、配偶之祖父母喪亡者，給予喪假三十三日，工資照給。`；但 gate 為 false，因此 Luna 正確地拒答。
