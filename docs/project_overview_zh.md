# Universal RAG｜專案總覽

更新：2026-09-09

## 目前定位

本機優先的文件 RAG 工作台：以「高召回 → 證據約束 → 可替換回答模型」為核心。當前產品範圍是文字型公司文件的內部試行；圖表／表格數值與完整多模態品質不列入基線承諾。

## 可直接使用

- 問答工作台：<http://127.0.0.1:8765/>
- 架構與任務管理器：<http://127.0.0.1:8765/architecture.html>
- 啟動、安裝與完整功能：[README](../README.md)

## 已完成

- 混合檢索：以 128 token 子 Chunk 執行 BM25 + Embedding，透過 RRF 融合並保留前 100 個候選；簡單問題直接取 Top-5，複雜問題使用 Cross-Encoder 重排後取 Top-5，再展開對應的 512 token 父 Chunk。
- 證據約束：Evidence Gate；證據不足時拒答，不以模型猜測補全。
- 文件處理：PDF、DOCX、TXT、Markdown、JSON、圖片的匯入與索引；目前產品驗收以文字內容為主。
- 權限查詢：角色 → 文件 ACL → 後端檢索過濾；未授權片段不會送入模型。
- 專案管理器：七個粗粒度領域，可進入詳細架構並勾選節點目標。

## 已驗證結果

- 法規版本比對：2016／2026 勞工請假規則已確認改動題，條文召回 20/20、Luna 回答 20/20。詳見[版本差異評測](../evals/taiwan_law_versions_changed/runs/20260908-165039-summary.md)。
- 多模態診斷已完成，但圖表數值召回仍不穩定；不納入文字型產品基線。詳見[多模態測試報告](../evals/multimodal_osha_2025/20260909-final-report.md)。
- 最新回歸：194 個 Python 測試、28 個前端測試通過。

## 使用與管理文件

- [權限查詢設定](query_access_control.md)
- [RAG 專案管理器操作](../RAG_專案管理器.md)
- [核心檢索架構](hybrid_rag_architecture_flow.md)
- [OCR 受控學習架構](ocr_self_training_architecture.md)

## 下一步

1. 接上真實 OIDC/JWT、租戶與向量 metadata filter，取代本機 Demo 身分切換。
2. 以公司文字 SOP／規章建立固定測試集與版本化評分基準。
3. 完成部署、備份、回滾與負載驗證後，再擴大使用範圍。

正式部署要求與邊界見[正式環境目標](production-ready-rag-target.md)；歷史實驗細節見[專案進度紀錄](project-progress-2026-09.md)。
