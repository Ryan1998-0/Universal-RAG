# 索引重試的 Qdrant attempt 隔離設計

狀態：設計提案，尚未實作或完成正式驗收。範圍為 production 索引建置、Qdrant 查詢與清理、索引 manifest；不變更既有建立索引 API 的公開識別值。

## 問題與目標

目前 `index_build_jobs.attempt` 會在每次 `claim()` 遞增，lease heartbeat 與 PostgreSQL 寫入檢查可阻止失去 lease 的 worker 發布索引。然而每次重試仍使用同一個 `index_version_id`。Qdrant 的刪除篩選和 point UUID 只包含 tenant、KB、索引版本；manifest 也寫入同一個版本路徑。因此舊 worker 已送出的 Qdrant delete/upsert 或 manifest put 即使在 lease 失效後才完成，仍可能刪除、覆寫新 attempt 的資料。`wait=True` 與寫入前檢查 lease 無法取消已送出的外部操作。

本設計把「邏輯索引版本」與「實體建置 attempt」分開：`index_version_id` 在一個 job 的所有重試中保持不變，Qdrant point、篩選及 manifest 則依 attempt 隔離。正式查詢只讀取成功發布的 attempt。

## 必須維持的契約與不變條件

1. `POST /v1/knowledge-bases/{knowledge_base_id}/index-builds` 回傳的 `job_id`、`index_version_id`，以及同一 `Idempotency-Key` 的重播結果，在重試前後保持相同。`GET /v1/index-build-jobs/{job_id}` 仍回報同一個版本 ID；attempt 是內部實體世代，不取代版本 ID。
2. 一個 Qdrant point 的 tenant、KB、版本、attempt 均由伺服器信任的 scope 決定，不採用用戶傳入的 payload。不同 attempt 的 point ID 必須不同；僅增加 payload 篩選不足以防止舊 upsert 以相同 point ID 覆寫新點。
3. 已發布版本的查詢、計數、scroll 與 fingerprint 只包含其已發布 attempt。過期 worker 的外部寫入只能影響其舊 attempt。
4. PostgreSQL `chunks` 仍以版本為邏輯單位：重試時清空未發布候選的舊 rows，新的 batch 寫入繼續由現有 job owner／lease 檢查保護。此設計不要求在 `chunks` 加 attempt 欄位。
5. 文件刪除必須清除該文件版本在所有 attempts 的 Qdrant 點；候選 attempt 的 GC 才使用精確 attempt 篩選。兩者不可共用一個含糊的 `delete_index()` 語義。
6. 現有 active 索引保持可讀。沒有 attempt 欄位的舊 Qdrant 點屬於 legacy attempt `0`；新建置從 `1` 開始。

## 資料模型與遷移

在 `rag_demo/production/database.py` 的 `IndexVersionRecord` 與新 Alembic revision 增加 `artifact_attempt INTEGER NOT NULL DEFAULT 0`，並約束值不得為負。舊資料回填為 `0`。建置中此欄記錄目前 attempt；發布後它是正式查詢使用的固定值。`IndexBuildJobRecord.attempt` 已存在，無須另增 counter；`claim()` 在鎖住 job 的同一交易中遞增 job attempt 並設定版本的 `artifact_attempt`。

`IndexBuildWorkItem.attempt` 已存在，應一路傳至 Qdrant scope、manifest 寫入、驗證和發布。`IndexVersionRecord` 的 `manifest_object_key` 繼續保存實際發布的物件 key，不需回填舊 manifest。舊 key 格式只用於 legacy 索引；新 key 包含 attempt。`index_manifest.py` 目前的 fingerprint 包含 Qdrant point ID；point ID 納入 attempt 後，既有欄位即可區分兩次建置。新 manifest body 應明載 attempt 與新的 schema version，方便稽核。

若使用既有 `deletion_outbox` 管理索引 artifact GC，無須新增 GC 表。它已有 `resource_type`、物件 key、vector scopes、重試、lease 與 `next_attempt_at`。內部 GC row 以 job 的 `created_by_user_id` 填必填欄位，採唯一的內部冪等 key；公開文件刪除狀態端點只允許查 `resource_type='document'`，避免把內部 GC 工作呈現為用戶的文件刪除。若實作時發現兩種工作需要不同重試或保留政策，再以獨立 outbox 表取代，不能使用只有記憶體狀態的清理佇列。

## 寫入、讀取與驗證修改點

### Scope 與 Qdrant adapter

在 `rag_demo/retrieval_scope.py` 為 `RetrievalScope` 增加非負的 `artifact_attempt`，預設 `0` 供既有呼叫者使用；`matches()` 在新格式時也驗證 attempt。`rag_demo/production/vector_repository.py` 應：

- 在新建置 payload 寫入整數 `artifact_attempt`，並於 `ensure_collection()` 建立相容的整數 payload index。`_chunk_payload()` 拒絕與可信 scope 不符的 supplied attempt。
- 新格式的 dense、sparse、count、scroll、fingerprint、候選刪除一律用 tenant + KB + index version + **精確 attempt** 篩選。legacy `0` 讀取應明確限定缺少 attempt 的舊點；若安裝的 Qdrant client 無法可靠表達此條件，必須先完成 legacy 索引遷移或 fail closed，不能在混合資料下退回只看版本的篩選。
- `_point_id()` 對新格式把 attempt 納入 UUID 名稱。legacy `0` 保留原 point ID 規則。`upsert_chunks()` 必須使用目前 worker 的 scope；不可信 chunk 不能覆寫 attempt。
- 將建置開始時的刪除命名並限制為 `delete_attempt(scope)`，要求 attempt 大於 `0`。全版本刪除僅供已證明不再 active 的版本 GC；文件刪除使用 tenant + KB + version + document version 的篩選，故意涵蓋所有 attempts。

### 建置流程與 manifest

在 `rag_demo/production/indexing.py`，`claim()` 遞增 attempt 後，`IndexingService.process()` 以該 attempt 建立 scope。`reset_candidate()` 保持 PostgreSQL 清理與 lease guard，並清空所有前次 manifest、PG/Qdrant fingerprint 與 validated evidence 欄位；Qdrant 只清本次 attempt namespace，不再刪除整個版本。前次 attempt 的 Qdrant 點交由 GC 處理。

`rag_demo/production/object_storage.py` 的新 manifest key 應為 `.../indexes/{index_version_id}/attempts/{attempt}/manifest.json`；新格式不得覆寫原本的 `.../indexes/{index_version_id}/manifest.json`。manifest body 明載 `index_version_id` 與 `artifact_attempt`。object storage 的 metadata 目前寫有 `immutable=true`，但同 key 的一般 put 仍可能覆寫；不同 attempt 必須使用不同 key。

`rag_demo/production/repository.py` 的 `validate_index_version()` 應接收 job ID、worker owner、attempt，在同一資料庫交易中先鎖 job，再鎖目標版本，核對 running、未過期 lease、job／version attempt 相同，然後比較只屬於本 attempt 的 Qdrant count／fingerprint 與 PostgreSQL／manifest。`publish_and_complete()` 再核對相同條件，與 active index 更新維持同一交易。過期 worker 即使把外部 Qdrant 操作送出，也不能改寫 validation evidence 或發布。

### Active 查詢與文件刪除

`authorize_knowledge_base()` 從 active `IndexVersionRecord` 帶回 `artifact_attempt`；`AuthorizedKnowledgeBase` 與 `rag_demo/production/api.py` 的 `/v1/ask` scope 同步傳遞。retriever 保持以 scope 查詢，不能自行猜測最新 attempt。active 版本若聲稱新格式卻沒有有效 attempt metadata，應 fail closed。

`rag_demo/production/deletion.py` 的文件 tombstone 已使來源立即不可被授權查詢，後續 outbox 清理應維持**跨 attempts 的版本範圍**，並在取消建置 job 時安排該候選的 artifact GC。只刪 active attempt 會留下舊建置資料，不能視為完成實體刪除。文件刪除與已送出的舊 Qdrant 寫入仍可能交錯，因此需要後續對帳重掃。

## 持久回收與失敗候選

重試覆蓋 attempt、worker 回報失敗或建置取消時，於相關 PostgreSQL 交易內建立／保留 `deletion_outbox` 的內部 `index_attempt` 工作，冪等 key 由 tenant、版本及 attempt 決定。內容包含精確 Qdrant scope、可推導的 per-attempt manifest key，`next_attempt_at` 設在 lease／外部請求可能完成的隔離期之後。清理 worker 先重新查 job、版本與 KB active 指標：不得刪除 active 版本的已發布 attempt，也不得刪除仍由有效 lease 建置的當前 attempt；其餘 attempt 用精確 filter 刪除，確認 count 為零後刪 manifest。

最終 `dead`／`cancelled` 且從未發布的版本使用既有 `gc_after` 安排 `index_version` 工作，清除此版本的 legacy 點、所有 attempts 與 manifest；保留 job 與版本中繼資料供狀態、冪等及稽核。已退役 active 版本的全版本 GC 也必須先核對 grace period、目前 active 指標與答案歷史的保留政策。清理順序應先清外部 artifact，再移除會提供文件版本清單的資料庫 membership；部分失敗可重試，不可先刪掉唯一的回收線索。

一次 `count=0` 無法證明不會再有已送出的舊 Qdrant upsert 落地。完成狀態需保留可定期複查的 cleanup row；若出現新 orphan point，重新排入清理並告警。這提供最終收斂，不能宣稱文件 tombstone 當下已完成物理清除。

## 分階段升級與退版下限

1. 先盤點 active 索引、running/retry_wait 建置與既有 Qdrant payload；套用向後相容的資料庫欄位遷移。舊 active rows 保持 `artifact_attempt=0`，不重寫其 point 或 manifest。
2. 先部署支援 attempt 精確讀取、legacy 讀取與 fail-closed guard 的 API／adapter，等待所有讀取進程更新。此階段尚不產生新格式索引。
3. 暫停建立索引，排空舊 indexing workers，並取得舊版已送出 Qdrant 寫入都已完成的可查驗證據，再啟用新 writer 與 GC。若無法確認仍在途的版本範圍 delete 已結束，就暫停切換；此類舊 delete 可破壞新 attempt，混合 writer 不可直接運行。
4. 以隔離 staging 的候選索引啟用新格式，檢查公開 ID 不變、active pointer、查詢與回收；通過正式驗收後再開放 production 建置。監看 orphan 點數、GC dead rows、validation 失敗與索引建置耗時。

在第一個新格式索引發布前，停用新 writer 並退回舊版本較容易。發布後，**退版下限必須保留 attempt-aware reader**：舊 reader 只按版本搜尋，會混入其他 attempts。若一定要退回更舊 binary，須先停止建置，對新格式索引完成受控物理轉換／重建及清除所有其他 attempts，並驗證舊 reader 的篩選結果；不能只回退程式碼。退版時同樣先排空 worker。新增的資料庫欄位可以留存，不能把欄位 downgrade 誤當作安全的資料格式退版。

## 正式驗收條件（尚未執行）

- 在真實 PostgreSQL + Qdrant 的隔離 staging，讓 attempt 1 的 delete、upsert、manifest put 延遲到 lease 失效後；由 attempt 2 成功發布，再放行舊操作。active dense／sparse 查詢、count 與 fingerprint 必須只看到 attempt 2，manifest key／digest 不變。
- 對同一 `Idempotency-Key` 比對初始、重播與完成後狀態：`job_id`、`index_version_id` 均一致。不同 tenant／KB 的 point 永不跨 scope 可見。
- 驗證 legacy active 索引仍可讀，新建索引需要有效的 attempt metadata，錯誤或缺失時拒絕查詢／發布；讀寫混合部署期間不得有舊 reader 或舊 writer 服務新格式資料。
- 驗證文件 tombstone 後來源立即不可供答案使用；outbox 清除該文件所有 attempts，對遲到寫入的定期對帳會重新清理。最終失敗候選的 Qdrant 點與 manifest 可回收，active attempt 不會被 GC 刪除。
- 演練 compatible-reader 退版、worker 排空與 GC 故障重試。保留可查驗的請求、job、attempt、manifest digest、Qdrant count、清理紀錄及失敗告警。既有 CI 或靜態檢查不能替代此併發與答案驗收。

此設計不要求付費服務或新增雲端資源；正式驗收需要既有隔離 staging、fixture KB、權杖與可用模型。文件內容僅描述待實作與待驗收工作。
