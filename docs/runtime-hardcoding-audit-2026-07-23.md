# RAG 執行路徑硬編碼稽核

日期：2026-07-23

## 稽核範圍

- Self-RAG 路由與 query rewrite
- BM25、Embedding、RRF、Rerank 與 evidence gate
- Qwen 生成路徑
- profile、文件、模型與前端載入流程
- Word 上傳文件的索引與來源篩選

## 已移除的行為硬編碼

1. Python 內的 IFRS17 關鍵詞擴展規則已移至 `profiles/<profile>/retrieval.json`。
2. 後端不再只接受指定 profile，而是從 `profiles/*/retrieval.json` 自動發現可用 profile。
3. 前端不再預先寫死 IFRS17、三體、範例問題、資料網址或模型選項。
4. 前端改由 `/api/config` 取得 profile 與模型，再由 `/api/profiles/<profile>` 取得文件清單與統計。
5. 上傳文件不再依賴 `upload-` 字首判斷來源類型，而是使用索引中的來源 provenance。
6. `/api/ask` 不再回退到舊的領域專用回答器；缺少前端 contexts 時會自行執行通用路由、混合檢索與證據檢查。
7. 時區、模型、profile、靜態介面路徑、Top-K、BM25、RRF、Rerank 權重與 evidence threshold 均可由環境設定調整。

## 保留但不屬於答案硬編碼的項目

- 日期時間與簡單系統能力使用 deterministic tool route，避免小模型猜測即時資訊。
- 安全上限、預設演算法參數與 API schema 是系統設定，不是文件內容或固定答案。
- `docs/ifrs17-demo/retrieval-core.js` 是舊的靜態檢索實驗檔；目前頁面沒有 import，也不在實際問答路徑。測試會阻止它再次被接回正式前端。

## 驗證

- Python：52 tests passed。
- JavaScript：執行期設定、profile 載入、Agent API、混合檢索與舊實驗隔離測試皆通過。
- 真實 API：`/api/config` 與 `/api/profiles/ifrs17` 可動態返回設定、1192 chunks、10 sources。
- Browser：動態載入 1 個 profile、1 個模型、10 份文件；輸入框可用且 console 無錯誤。
