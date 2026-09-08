# 多模態 RAG 架構比較資料集

這組資料以 NIST《Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile》（NIST AI 600-1）作為第一份正式測試文件。它有 64 頁，混合封面視覺、目錄、雙欄／跨頁文字與大量三欄治理表格，適合先測企業知識庫常見的 PDF 擷取、版面理解、表格檢索與來源引用。

## 檔案

- `corpus/NIST.AI.600-1.pdf`：原始 PDF，供一般 PDF 解析、版面解析及多模態模型使用。
- `corpus/NIST.AI.600-1.page-18-table.png`：原始 PDF 第 18 頁的影像版，測試 OCR 與表格理解。
- `corpus/NIST.AI.600-1.cover.png`：封面影像，測試 OCR 與基本視覺欄位擷取。
- `questions.json`：人工整理的標準答案、頁碼與評測類型。
- `manifest.json`：來源、校驗值與檔案角色。
- `rendered/`：人工視覺檢查用頁面，不必全部匯入正式索引。

來源頁面：https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence

## 建議依序比較的架構

1. **固定切塊 + Dense Search**：建立最低基準，觀察表格列被切斷與專有編號搜尋失敗的情況。
2. **Dense + BM25 + RRF**：測試 `GV-1.5-003` 這類精確代碼和語意問題能否同時改善。
3. **Hybrid + Reranker**：觀察多個相似 GOVERN 表格同時命中時，正確頁面能否排到前面。
4. **Parent-child / Section-aware**：小塊負責召回，父章節或完整表格負責回答；重點看跨頁段落和表格列。
5. **Layout-aware / Multimodal**：保留頁碼、標題、欄位、列關係，並直接處理 PNG；重點看表格欄位對齊與影像 OCR。

目前專案的圖片處理屬於 OCR 路線，不等同於完整的視覺語意理解。因此 PNG 題若答得出來，代表 OCR／版面流程有效；不能用它宣稱模型理解圖像含義。

## 評測方式

每個架構使用相同的文件、題目、生成模型與提示詞，只改索引／檢索方法。至少記錄：

- `Recall@5`：標準頁是否出現在前 5 個命中結果。
- `MRR`：第一個正確頁面的排名。
- `answer_accuracy`：答案是否涵蓋 `expected_answer` 的必要事實。
- `citation_accuracy`：引用頁碼是否命中 `source_pages_pdf`。
- `table_row_accuracy`：Action ID、Suggested Action、GAI Risks 是否仍屬同一列。
- `abstention_accuracy`：文件沒有答案時是否明確拒答，而非猜測。

`source_pages_pdf` 是 PDF 檔案的實體頁序；`source_pages_printed` 是頁面底部印刷頁碼。兩者不可混用。

## 預期能看出的差異

- Dense baseline 通常能處理概念題，但對 Action ID、縮寫和表格精確欄位較不穩。
- Hybrid + RRF 通常優先改善代碼、專有名詞與原句查找；它本身不會修復被錯切的表格。
- Reranker 可改善多個相似段落的排序，但第一階段完全沒召回時無法補救。
- Parent-child／section-aware 能提供更完整上下文，代價是送入模型的 token 增加。
- Layout-aware／multimodal 對表格與純影像頁最有價值，但索引時間、模型需求與實作複雜度最高。

這是第一批「深度樣本」，適合驗證架構行為，不代表 1000 人公司所需的最終資料量。正式壓力測試仍應逐步加入多版本政策、權限文件、掃描件、試算表、簡報、FAQ 與相互矛盾的舊文件。
