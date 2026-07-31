# 公開資料 RAG 實測報告

- 測試日期：2026-07-23
- 本機模型：Ollama Qwen 2.5 7B
- 檢索：BM25 + Embedding + RRF + Rerank + Evidence Relevance Gate
- 結果：7/7 通過

## 測試資料

### 台灣勞動基準法

- 檔名：`台灣勞動基準法_公開資料測試.docx`
- 官方來源：https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=N0030001
- source_id：`upload-26cd116c7f756b98716a`
- chunks：5
- duplicate：false

### NASA Mars 2020

- 檔名：`NASA_Mars_2020_Public_Data_Test.docx`
- 官方來源：https://www.jpl.nasa.gov/news/press_kits/mars_2020/landing/mission/
- source_id：`upload-746d35a395bed875e898`
- chunks：5
- duplicate：false

## 測試結果

| 結果 | 案例 | Router | 證據判斷 | Top 1 來源 | 回答摘要 |
|---|---|---|---|---|---|
| PASS | 中文法律精確問答 | RAG | relevant / high | `upload-26cd116c7f756b98716a` | 依照勞動基準法，繼續工作一年以上但未滿三年，終止契約應提前二十天預告。  來源：[3] |
| PASS | 中文法律多欄位問答 | RAG | relevant / high | `upload-26cd116c7f756b98716a` | 勞工正常工作時間每日上限為八小時，每週上限為四十小時。  來源：[1] |
| PASS | 中文問英文文件 | RAG | relevant / high | `upload-746d35a395bed875e898` | 毅力號從地球到火星的巡航約花了203天，距離約471百萬公里。  來源：[2] |
| PASS | 英文技術名稱檢索 | RAG | relevant / high | `upload-746d35a395bed875e898` | 毅力號為了安全著陸在 Jezero Crater，使用了兩項新的著陸技術：範圍觸發（Range Trigger）和地形相對導航（Terrain-Relative Navigation）。  來源：[3], [1] |
| PASS | 錯選文件拒答：法律問 NASA | RAG | irrelevant / low | `upload-746d35a395bed875e898` | 我已嘗試檢索，但目前可用資料與問題的相關性不足，無法根據知識庫可靠回答。請改寫問題或選擇其他文件後再試。 |
| PASS | 錯選文件拒答：NASA 問法律 | RAG | irrelevant / low | `upload-26cd116c7f756b98716a` | 我已嘗試檢索，但目前可用資料與問題的相關性不足，無法根據知識庫可靠回答。請改寫問題或選擇其他文件後再試。 |
| PASS | 一般常識直接回答 | Direct | n/a | `none` | 2 + 2 等於 4。 |

## 案例明細

### PASS - 中文法律精確問答

- 問題：依勞動基準法，繼續工作一年以上但未滿三年，終止契約應提前幾天預告？
- Router：`needs_retrieval=true`；問題明確依賴可核對的法規、文件或來源，應先檢索外部證據。
- 檢索詞：依勞動基準法，繼續工作一年以上但未滿三年，終止契約應提前幾天預告？
- 證據：`relevant` / `high`；檢索片段具有明確的詞彙或語意相關性，可進入 grounded generation。
- 回答：依照勞動基準法，繼續工作一年以上但未滿三年，終止契約應提前二十天預告。

來源：[3]
- 引用數：4

### PASS - 中文法律多欄位問答

- 問題：勞工正常工作時間每日與每週上限各是多少？
- Router：`needs_retrieval=true`；問題要求可核對的精確事實或數值，為避免模型憑記憶誤答，應先檢索外部證據。
- 檢索詞：勞工正常工作時間每日與每週上限各是多少？
- 證據：`relevant` / `high`；檢索片段具有明確的詞彙或語意相關性，可進入 grounded generation。
- 回答：勞工正常工作時間每日上限為八小時，每週上限為四十小時。

來源：[1]
- 引用數：4

### PASS - 中文問英文文件

- 問題：毅力號從地球到火星的巡航約花多少天，距離約多少公里？
- Router：`needs_retrieval=true`；問題要求可核對的精確事實或數值，為避免模型憑記憶誤答，應先檢索外部證據。
- 檢索詞：毅力號從地球到火星的巡航約花多少天，距離約多少公里？
- 證據：`relevant` / `high`；檢索片段具有明確的詞彙或語意相關性，可進入 grounded generation。
- 回答：毅力號從地球到火星的巡航約花了203天，距離約471百萬公里。

來源：[2]
- 引用數：4

### PASS - 英文技術名稱檢索

- 問題：毅力號為了安全著陸在 Jezero Crater，使用了哪兩項新的著陸技術？
- Router：`needs_retrieval=true`；問題要求可核對的精確事實或數值，為避免模型憑記憶誤答，應先檢索外部證據。
- 檢索詞：毅力號為了安全著陸在 Jezero Crater，使用了哪兩項新的著陸技術？
- 證據：`relevant` / `high`；檢索片段具有明確的詞彙或語意相關性，可進入 grounded generation。
- 回答：毅力號為了安全著陸在 Jezero Crater，使用了兩項新的著陸技術：範圍觸發（Range Trigger）和地形相對導航（Terrain-Relative Navigation）。

來源：[3], [1]
- 引用數：4

### PASS - 錯選文件拒答：法律問 NASA

- 問題：依勞動基準法，工作三年以上的終止契約預告期是幾天？
- Router：`needs_retrieval=true`；問題明確依賴可核對的法規、文件或來源，應先檢索外部證據。
- 檢索詞：依勞動基準法，工作三年以上的終止契約預告期是幾天？
- 證據：`irrelevant` / `low`；最高排名片段缺乏足夠的詞彙與語意相關性，不應交給模型引用。
- 回答：我已嘗試檢索，但目前可用資料與問題的相關性不足，無法根據知識庫可靠回答。請改寫問題或選擇其他文件後再試。
- 引用數：0

### PASS - 錯選文件拒答：NASA 問法律

- 問題：Perseverance 的火星巡航花了多少天？
- Router：`needs_retrieval=true`；問題要求可核對的精確事實或數值，為避免模型憑記憶誤答，應先檢索外部證據。
- 檢索詞：Perseverance 的火星巡航花了多少天？
- 證據：`irrelevant` / `low`；最高排名片段缺乏足夠的詞彙與語意相關性，不應交給模型引用。
- 回答：我已嘗試檢索，但目前可用資料與問題的相關性不足，無法根據知識庫可靠回答。請改寫問題或選擇其他文件後再試。
- 引用數：0

### PASS - 一般常識直接回答

- 問題：2 + 2 等於多少？
- Router：`needs_retrieval=false`；基本算術可由一般問答模型直接回答，不需要查詢 knowledge base。
- 檢索詞：2 + 2 等於多少？
- 證據：`n/a` / `n/a`；n/a
- 回答：2 + 2 等於 4。
- 引用數：0

## 結論

這次公開資料測試顯示目前流程已能處理非 IFRS 17 的任意 Word 文件，並可在中文法律文件、英文技術文件、中文問英文文件及錯選文件時運作。

### 第一輪發現與修復

- 第一輪並非文件解析或 Embedding 失敗，而是 Router 把部分精確法律與 NASA 問題誤判為穩定一般常識，直接讓 Qwen 憑記憶回答。
- 已新增 query-only 風險保護：法規、來源、特定任務及精確數值、期限、距離等問題先檢索；規則不讀文件清單，也未硬編碼本次測試文件名稱。
- 修正後以完全相同的 7 個案例重跑，最終 7/7 通過。

### 目前限制

- 這是 7 題 smoke test，能證明完整工作流可用，但不能替代數十到數百題的正式 benchmark。
- 本次 Word 是依官方公開頁面整理的可追溯測試文件，不是對所有網頁格式、掃描 PDF 或複雜表格的泛用解析測試。
- Qwen 2.5 7B 的中文措辭偶爾不自然，例如把 471 million kilometers 表達為「471 百萬公里」；數值有依據，但生成品質仍可再加格式正規化。
- API 目前回傳最多 4 個候選 citation metadata；模型答案中的實際 rank 標記可能只使用其中一部分，後續可再做精確引用對齊。

完整 API 原始結果另存於 `public-data-rag-validation-results.json`。
