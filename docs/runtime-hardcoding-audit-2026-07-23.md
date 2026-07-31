# RAG 執行路徑硬編碼檢查報告

更新日期：2026-07-31

## 結論

核心問答、文件匯入與正式服務已改為泛用架構。IFRS 17 只存在於可選的範例 Profile、測試案例與歷史評估報告，不再作為服務名稱、預設知識庫、前端路徑或回答規則。

## 已移除的硬編碼

1. 專案、API、容器、Worker、OIDC audience 與備份預設名稱統一改為 `universal-rag`。
2. 前端與測試目錄由 `ifrs17-demo` 改為 `rag-demo`。
3. 新增 `profiles/default`，預設啟動不再依賴任何特定資料集。
4. 評估腳本改由 `--profile` 或 `RAG_PROFILE` 接收知識庫 Profile，不再固定傳入 `ifrs17`。
5. 刪除未被正式介面使用、內含領域固定答案的舊版瀏覽器檢索模組。
6. IFRS 17 靜態資料移至 `profiles/ifrs17/data`，只在選擇該範例 Profile 時載入。
7. 前端 Profile、模型、文件與資料夾仍由執行期 API 動態載入，不預先寫死資料名稱。
8. 模型、時區、檢索數量、BM25、RRF、Rerank、Evidence Gate 與儲存服務皆由環境設定或 Profile 設定控制。
9. 文件解析器在未指定 `RAG_PROFILE` 時統一使用 `profiles/default`，不再回退到不存在的舊 `data/` 目錄。
10. GitHub Pages 無後端時會切換成泛用靜態展示設定；本機或正式環境仍以 API 回傳的 Profile 與模型為準。

## 保留但不屬於領域硬編碼

- 日期時間與基本算術使用確定性路由，目的是避免小模型猜測。
- 檔案大小、Profile 名稱格式與 API schema 是安全及介面合約。
- `profiles/ifrs17` 是範例資料設定，和 `profiles/default` 使用同一套通用 Pipeline。
- 測試中保留不同領域的問題，用來確認系統不會只對單一資料集有效。

## 執行路徑

```text
Profile 與文件選擇
  -> 通用查詢路由
  -> 通用 BM25 + Embedding 檢索
  -> RRF + Rerank
  -> 證據品質檢查
  -> 通用回答提示與來源引用
```

領域 Profile 只能提供別名、查詢擴展與範例問題，不能繞過文件範圍、證據檢查或回答規則。

## 驗證結果

- 執行期設定預設回傳 `default` 與「泛用知識庫」。
- 核心後端、正式前端與展示介面已加入自動掃描測試，不允許出現 `ifrs17` 執行期綁定。
- `2+2等於多少？` 正確走直接回答，不進行文件檢索。
- 限定一份 TXT 文件詢問其政策代碼時，系統以 BM25 + Embedding + Rerank 找到正確片段，回答 `TXT-731` 並附上已驗證來源。
