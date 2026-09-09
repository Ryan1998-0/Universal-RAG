# 文件查詢權限

本機工作台已加入「角色 → 文件來源 ACL → 檢索過濾」的權限流程。它的目的不是把文件名稱從畫面隱藏，而是確保未授權的片段不會被取回、更不會被送進回答模型的 prompt。

## 操作方式

1. 在右側面板的「查詢身分」切換要模擬的角色。
2. 以「管理員（Demo）」身分，在任一文件右邊點選「權限」。
3. 選擇 `全員公開` 或一個以上的部門角色（人資、財務、資訊、一般同仁），儲存。
4. 切換成另一個身分後，未獲授權的文件不會出現在清單；即使 API 請求刻意附上它的 `source_id`，後端也會在檢索前剔除。

初始狀態不改變既有資料的可見性：尚未分類權限的舊文件預設為「全員公開」。要保護資料，請由管理員逐一設定成部門角色。設定會保存到 `.local/access_policy.json`，此路徑可用 `RAG_ACCESS_POLICY_PATH` 改寫。

## 預設角色

| 角色 | 可做的事 |
| --- | --- |
| `admin` | 管理文件、資料夾、文件 ACL，並可讀取所有來源 |
| `hr` | 查詢被授權給人資的文件 |
| `finance` | 查詢被授權給財務的文件 |
| `it` | 查詢被授權給資訊的文件 |
| `staff` | 查詢被標示為全員公開或一般同仁的文件 |

## 強制點與邊界

請求的 `source_ids` 只是「想搜尋哪些文件」，不是授權憑證。`/api/retrieve` 和 `/api/ask` 都會先計算：

```text
effective_source_ids = requested_source_ids ∩ sources_allowed_for_principal
```

因此前端勾選、直接呼叫 API、或模型端的提示詞都不能繞過 ACL。文件下載、上傳、移動資料夾與變更 ACL 也都要求管理員角色。

目前的「查詢身分」是單機 Demo 的測試工具，讓你可驗證權限矩陣與檢索邊界；它不是登入機制，不能直接當成公司正式權限。正式環境應把驗證過的 OIDC/JWT `Principal`（使用者、tenant、roles）接到同一個 ACL 決策點，並讓向量資料庫的 metadata filter 同步套用 tenant、knowledge base 和文件 ACL。現有正式環境的 JWT/OIDC 基礎可見 `rag_demo/production/auth.py`。
