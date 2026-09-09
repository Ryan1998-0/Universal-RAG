# OCR 受控自我訓練架構

## 目標

讓 OCR 隨著真實文件與人工校正逐步改善，同時避免模型把自己的錯誤輸出當成真實標籤而產生回授污染。此設計適用於繁中 PDF、掃描件、表格、圖表與文件內嵌圖片。

「自我訓練」在本專案指的是**受控的資料與模型改進閉環**，不是線上直接以 OCR 輸出回訓 OCR 模型。

## 端到端閉環

```text
原始文件
  -> 文字層抽取 / OCR / VLM
  -> OCR 品質 Gate
       -> 高信心：正規化、索引、RAG
       -> 低信心 / 數字衝突 / OCR 失敗：主動抽樣與人工校正
  -> 已驗證標註
  -> 版本化金標資料集
  -> 候選 OCR / 版面模型離線評測
  -> Shadow / 灰度發布
  -> 生產品質與漂移監控
  -> 回到 OCR 品質 Gate
```

## 1. OCR 品質 Gate

每個 OCR 頁面或區塊都需留下不可變的 trace：

- `document_sha256`、`page_number`、`region_bbox`
- OCR 引擎、模型版本、語言、DPI、前處理版本
- 原始 OCR 文字、正規化文字、平均 confidence
- 字元數、文字密度、數字／日期／單位 token
- VLM 或規則比對結果，以及 issue code

Gate 的初始規則：

| 狀態 | 條件範例 | 去向 |
|---|---|---|
| `accepted` | 原生文字層完整，或 OCR 高信心且無數字衝突 | 正規化、索引 |
| `review_required` | 低 confidence、空頁、語言異常、關鍵數字不一致 | 人工複核佇列 |
| `rerun_required` | OCR 引擎失敗、渲染失敗、語言包缺失 | 改 DPI／引擎後重跑 |
| `blocked` | 文件損毀或無法取得可用內容 | 顯示原因，不進索引 |

涉及法規、金額、日期、統計數字時，必須檢查「標籤／主體／數值／單位」是否在同一區塊或相鄰區塊；不符合時不能當成可直接引用的 RAG 證據。

## 2. 主動抽樣與人工校正

不是所有頁面都人工標註。優先順序：

1. OCR 低 confidence、OCR 失敗或語言包缺失。
2. 數字、日期、比例、表格欄位與圖表讀值不一致。
3. 新文件家族、新版面、掃描品質明顯改變的漂移資料。
4. 高查詢量、Evidence Gate 經常拒答、或商業風險高的頁面。
5. 每個文件家族固定比例的隨機抽樣，用來發現「高 confidence 但其實錯」的案例。

人工標註至少校正：完整文字、數字與單位、表格 cell、圖表標籤與數字對應、以及頁面／區域座標。每筆標註保留標註人、校對人、標註規範版本與時間。

## 3. 金標資料集與防污染規則

資料集版本至少包含：

- `ocr_dataset_version`
- 原始檔雜湊、頁碼、區域座標與文件類型
- 原始 OCR、人工正解、標註規範版本
- train / validation / holdout split
- OCR 引擎、模型、DPI、語言與前處理設定

防污染規則：

- 未經人工確認的 pseudo-label 不得進入金標訓練資料。
- 同一份文件、同一份報告的不同頁、或高度相似模板，不可同時進 train 與 holdout。
- 評測 holdout 一旦用於人工修正，就必須淘汰並建立新 holdout 版本。
- RAG 回答被使用者按讚不等於 OCR 正確；只有可回到原始頁面的校正才可成為標籤。

## 4. 候選模型與評測門檻

候選可以是 OCR 引擎、語言包、DPI／前處理策略、表格模型、圖表解析器，或 VLM 輔助策略。不要在沒有足夠金標資料時直接微調現行 OCR 模型；先以策略切換或候選模型比較取得基準。

凍結 holdout 至少量測：

| 指標 | 用途 |
|---|---|
| Character Error Rate (CER) | 一般文字辨識品質 |
| 數字 exact match | 金額、日期、比例、統計數字 |
| 單位與日期 exact match | 防止 `259 人`、`113 人`、`43.6%` 混淆 |
| 表格 cell F1 | 表格欄位與值的對應 |
| 圖表 label-value accuracy | 圖表標籤、數值、單位綁定 |
| RAG page recall / evidence sufficiency | 解析品質是否真的改善檢索與回答 |
| P50 / P95 latency 與每頁成本 | 防止品質提升造成不可接受的成本 |

建議初始 promotion policy：候選必須提升或持平 CER、提升數字 exact match，且任何高風險文件家族不得出現未核准退步；通過離線評測後，仍要先 Shadow 再灰度發布。

## 5. Shadow 與灰度發布

候選 OCR 在 Shadow 階段只處理副本，不覆蓋原始檔、現有 index 或使用者可見答案。比對內容：

- 現行與候選 OCR 的文字差異、數字差異與表格差異。
- 相同 benchmark 的 retrieval、Evidence Gate、回答正確率與延遲。
- 新 issue 類型、漂移率、人工複核量與單頁成本。

若候選失敗，保留 raw prediction 與原因，但不影響生產索引；若通過，再採文件類型／租戶／資料夾分批灰度，並提供一鍵回滾至先前 policy version。

## 6. 目前專案的落點

目前 `rag_demo/document_pipeline.py` 已保存部分 OCR trace，例如 OCR engine、confidence、頁碼與 extraction method；PDF 有文字層與 OCR 分流，DOCX 內嵌圖片可使用 OCR 與 VLM。下一步的實作邊界為：

1. 建立 OCR issue 與人工校正資料表／檔案格式。
2. 在 ingest metadata 中加入 OCR policy version、DPI、語言與頁面雜湊。
3. 建立資料集匯出器與 frozen holdout 評測指令。
4. 將 OCR candidate 產物接到既有多模態 RAG benchmark。
5. 實作 Shadow index 與 promotion / rollback gate。

這些步驟完成後，才適合討論特定 OCR 模型的微調、LoRA 或訓練資源配置。
