# RAG 專案管理器

這是 Universal RAG 的永久專案管理入口，用來分層查看架構、勾選各節點任務目標並追蹤整體進度。

## 開啟方式

1. 在 Finder 開啟本專案資料夾。
2. 雙擊 `開啟_RAG_專案管理器.command`。
3. 啟動檔會在需要時啟動本機服務，接著開啟 `http://127.0.0.1:8765/architecture.html`。

也可以在專案根目錄執行：

```bash
./開啟_RAG_專案管理器.command
```

## 已保存內容

- 第一層：7 個 RAG 核心領域。
- 第二層：每個領域各自一張詳細架構圖。
- 操作：游標定點滾輪縮放、拖曳平移、上一層與麵包屑導覽。
- 管理：每個節點的任務目標、勾選狀態與整體完成率。
- 勾選狀態保存在目前瀏覽器的 localStorage；清除網站資料會重設進度。

## 主要檔案

- 頁面：`docs/rag-demo/architecture.html`
- 互動：`docs/rag-demo/assets/architecture-workbench.js`
- 樣式：`docs/rag-demo/assets/architecture-workbench.css`
- 測試：`tests/rag-demo/architecture-workbench.test.mjs`
- 啟動紀錄：`.local/rag-project-manager.log`
