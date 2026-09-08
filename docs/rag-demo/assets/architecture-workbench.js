const NODE_WIDTH = 230;
const NODE_HEIGHT = 142;
const STORAGE_KEY = "universal-rag.architecture-goals.v2";

function node(id, code, title, purpose, input, output, x, y, goals, extra = {}) {
  return { id, code, title, purpose, input, output, x, y, goals, ...extra };
}

export const architectureViews = {
  root: {
    id: "root", title: "Universal RAG 整體架構", eyebrow: "LEVEL 1 · 7 個核心領域",
    width: 1640, height: 920, groups: [],
    nodes: [
      node("domain-experience", "01 · EXPERIENCE", "使用者與前端", "承接提問、文件選擇、進度與引用呈現。", "使用者操作", "可信的問答請求與結果", 70, 210,
        ["使用者能完成提問與文件選擇", "處理進度與錯誤狀態清楚", "架構與問答入口一致"], { drilldown: "experience" }),
      node("domain-ingestion", "02 · INGESTION", "文件匯入與解析", "把多格式文件轉成可追溯、可搜尋的 chunks。", "PDF、圖片、DOCX、文字", "索引與原始資料關聯", 70, 530,
        ["所有支援格式可穩定匯入", "文字與圖片內容可追溯到頁面", "失敗階段與原因可記錄"], { drilldown: "ingestion" }),
      node("domain-query", "03 · QUERY", "問題理解與規劃", "判斷是否需要檢索，拆解問題並建立查詢。", "原始問題與對話指涉", "檢索決策與多查詢計畫", 390, 210,
        ["法規與公司資料問題強制檢索", "多查詢保留原始問題語意", "移除前後端重複路由"], { drilldown: "query" }),
      node("domain-retrieval", "04 · RETRIEVAL", "混合檢索", "融合關鍵字、語意與關係型召回結果。", "查詢計畫與文件範圍", "排序後的候選證據", 710, 370,
        ["BM25 與 Dense 都受相同範圍限制", "RRF 融合多個查詢結果", "直接答案片段優先排序"], { drilldown: "retrieval" }),
      node("domain-evidence", "05 · EVIDENCE", "證據驗證", "確認片段能否直接支撐數值、期間、程序或結論。", "候選證據與問題", "充分／不足判定", 1030, 370,
        ["精確值問題檢查主體與數值共現", "證據不足時阻止模型補答", "保留判定理由與 signals"], { drilldown: "evidence" }),
      node("domain-answer", "06 · ANSWER", "模型回答與引用", "只把通過驗證的證據交給模型並檢查引用。", "可信證據包", "附引用回答或安全拒答", 1350, 210,
        ["Prompt 明確標示唯一可信證據", "每個實質主張附來源 rank", "無有效引用時 fail closed"], { drilldown: "answer" }),
      node("domain-platform", "07 · PLATFORM", "資料、模型與部署", "支撐本機 Demo 與正式環境的運算、儲存和安全。", "文件、模型與服務事件", "可維運的 RAG 平台", 1350, 530,
        ["本機 Demo 可單機啟動", "正式環境具租戶與權限邊界", "資料、模型與工作狀態可監控"], { drilldown: "platform" }),
    ],
    edges: [
      ["domain-experience", "domain-query"], ["domain-ingestion", "domain-retrieval"],
      ["domain-query", "domain-retrieval"], ["domain-retrieval", "domain-evidence"],
      ["domain-evidence", "domain-answer"], ["domain-platform", "domain-answer"],
      ["domain-answer", "domain-experience"],
    ],
  },

  experience: {
    id: "experience", title: "使用者與前端", eyebrow: "LEVEL 2 · EXPERIENCE FLOW",
    width: 1660, height: 900,
    groups: [
      { label: "操作入口", x: 40, y: 130, width: 520, height: 590 },
      { label: "可信 API 邊界", x: 620, y: 130, width: 450, height: 590 },
      { label: "結果呈現", x: 1130, y: 130, width: 470, height: 590 },
    ],
    nodes: [
      node("exp-user", "INPUT", "使用者", "提出問題、選擇來源並確認答案。", "工作問題", "問題與回饋", 80, 250,
        ["整理常見使用情境", "定義可接受答案與拒答條件"]),
      node("exp-chat", "UI", "對話介面", "管理輸入、歷史對話與目前回覆。", "問題與歷史", "標準問答請求", 320, 250,
        ["對話新增、切換與刪除可用", "送出期間避免重複請求"]),
      node("exp-docs", "UI", "文件範圍控制", "上傳文件並指定本次允許檢索的來源。", "文件與勾選狀態", "source IDs", 320, 500,
        ["上傳與資料夾選擇可用", "未選文件時清楚提示"]),
      node("exp-client", "CLIENT", "API Client", "建立受控 payload，不接受瀏覽器偽造證據。", "question、model、source IDs", "API request", 670, 250,
        ["不轉送 client contexts", "路由與檢索只執行一次", "逾時與錯誤可恢復"]),
      node("exp-server", "SERVER", "單一 RAG Pipeline", "由伺服器完成路由、檢索、驗證與回答。", "API request", "canonical response", 810, 500,
        ["/api/ask 成為唯一問答入口", "回傳完整 stage timings 與 retrieval trace"]),
      node("exp-result", "OUTPUT", "回答與引用", "呈現回答、信心、引用和證據不足原因。", "canonical response", "可理解的回覆", 1190, 250,
        ["引用可打開來源文件", "證據不足與原因可見"]),
      node("exp-progress", "OUTPUT", "處理進度", "讓使用者知道目前正在路由、檢索或生成。", "stage events", "即時狀態", 1330, 500,
        ["各階段狀態與耗時一致", "不為顯示進度重跑管線"]),
    ],
    edges: [
      ["exp-user", "exp-chat"], ["exp-docs", "exp-client"], ["exp-chat", "exp-client"],
      ["exp-client", "exp-server"], ["exp-server", "exp-result"], ["exp-server", "exp-progress"],
      ["exp-result", "exp-user"],
    ],
  },

  ingestion: {
    id: "ingestion", title: "文件匯入與解析", eyebrow: "LEVEL 2 · ASYNC INGESTION PIPELINE",
    width: 2100, height: 720,
    groups: [{ label: "非同步文件處理", x: 35, y: 130, width: 2010, height: 380 }],
    nodes: [
      node("ing-file", "01", "多格式文件", "接收 PDF、圖片、DOCX 與文字資料。", "uploaded bytes", "detected document", 70, 250,
        ["驗證副檔名與 MIME", "保存原始檔雜湊與來源"]),
      node("ing-validate", "02", "格式與大小驗證", "拒絕空白、過大或不支援的輸入。", "detected document", "validated document", 320, 250,
        ["覆蓋所有支援格式測試", "錯誤訊息指出失敗原因"]),
      node("ing-security", "03", "安全檢查", "執行掃毒與文件 prompt injection 政策。", "validated document", "trusted input", 570, 250,
        ["串接 ClamAV", "定義內容隔離與警告規則"]),
      node("ing-parse", "04", "解析／OCR／VLM", "抽取文字、表格及圖片資訊。", "trusted document", "page elements", 820, 250,
        ["文字層與 OCR 自動分流", "圖片描述保留頁碼與區域", "抽樣檢查多模態品質"]),
      node("ing-normalize", "05", "內容正規化", "統一文字與文件元素結構。", "page elements", "normalized elements", 1070, 250,
        ["保留條文與表格語意", "保留標題層級與定位"]),
      node("ing-chunk", "06", "父子分塊", "建立可召回小塊與可回答父層上下文。", "normalized elements", "parent／child chunks", 1320, 250,
        ["依文件結構切分", "避免答案與條件分離", "保存 parent 關聯"]),
      node("ing-embed", "07", "Embedding", "將 chunks 轉成語意向量。", "child chunks", "vectors", 1570, 250,
        ["記錄模型版本", "驗證批次處理與重試"]),
      node("ing-index", "08", "持久化索引", "保存 BM25、向量與 metadata 關聯。", "chunks、vectors、metadata", "searchable KB", 1820, 250,
        ["權限資訊寫入所有索引", "索引可由原始資料重建"]),
    ],
    edges: [
      ["ing-file", "ing-validate"], ["ing-validate", "ing-security"], ["ing-security", "ing-parse"],
      ["ing-parse", "ing-normalize"], ["ing-normalize", "ing-chunk"], ["ing-chunk", "ing-embed"],
      ["ing-embed", "ing-index"],
    ],
  },

  query: {
    id: "query", title: "問題理解與規劃", eyebrow: "LEVEL 2 · QUERY PLANNING",
    width: 1650, height: 980,
    groups: [
      { label: "自適應路由", x: 40, y: 130, width: 720, height: 700 },
      { label: "檢索規劃", x: 820, y: 130, width: 770, height: 700 },
    ],
    nodes: [
      node("qry-question", "INPUT", "原始問題", "保留問題、對話指涉與文件範圍。", "question、history、sources", "canonical request", 80, 310,
        ["驗證 request contract", "限制問題與來源數量"]),
      node("qry-route", "ROUTE", "自適應路由", "判斷直接回答、時間工具或文件檢索。", "canonical request", "retrieval decision", 340, 310,
        ["公司與法規題強制檢索", "記錄 routeMs", "只由後端執行一次"]),
      node("qry-direct", "BRANCH", "一般回答", "處理不依賴知識庫的穩定問題。", "general question", "direct prompt", 500, 570,
        ["明確定義跳過檢索範圍", "不處理公司內部事實"]),
      node("qry-time", "TOOL", "日期時間工具", "以系統時區直接回答時間問題。", "datetime question", "deterministic answer", 210, 620,
        ["使用設定時區", "不呼叫 LLM 與檢索器"]),
      node("qry-decompose", "PLAN", "問題拆解", "拆出實體、條件、數值、期間與程序。", "retrieval question", "sub-questions", 860, 240,
        ["保留每個必要證據欄位", "避免加入問題不存在的假設"]),
      node("qry-multi", "PLAN", "多查詢產生", "建立語意、精確詞與答案承載詞查詢。", "sub-questions", "query variants", 1110, 430,
        ["保留原始問題查詢", "限制查詢數與總字數", "每個查詢可獨立召回"]),
      node("qry-scope", "FILTER", "Metadata／權限範圍", "先限制知識庫、文件、租戶與角色。", "queries、source scope", "allowed indices", 1360, 240,
        ["BM25 與 Dense 共用範圍", "空選擇回傳空結果"]),
    ],
    edges: [
      ["qry-question", "qry-route"], ["qry-route", "qry-direct"], ["qry-route", "qry-time"],
      ["qry-route", "qry-decompose"], ["qry-decompose", "qry-multi"], ["qry-multi", "qry-scope"],
    ],
  },

  retrieval: {
    id: "retrieval", title: "混合檢索", eyebrow: "LEVEL 2 · HYBRID RETRIEVAL",
    width: 1900, height: 980,
    groups: [
      { label: "召回分支", x: 300, y: 110, width: 580, height: 740 },
      { label: "融合與精排", x: 940, y: 110, width: 890, height: 740 },
    ],
    nodes: [
      node("ret-scope", "INPUT", "查詢與範圍", "接收多查詢與允許搜尋的 chunks。", "queries、allowed indices", "bounded input", 50, 360,
        ["確認來源範圍不為空", "保留 evidence query"]),
      node("ret-bm25", "BRANCH A", "BM25", "召回條文號、專名與完全相符關鍵字。", "query variants", "lexical lists", 350, 190,
        ["每個查詢獨立召回", "保留 BM25 分數"]),
      node("ret-dense", "BRANCH B", "Dense Embedding", "召回語意相近與同義改寫內容。", "queries、vectors", "semantic lists", 350, 390,
        ["模型與索引版本一致", "記錄 cosine 分數與延遲"]),
      node("ret-graph", "BRANCH C", "Graph Retrieval", "補強實體關係與多跳問題。", "entities、relations", "graph context", 350, 590,
        ["只在關係型問題啟用", "抑制泛用 hub entity"]),
      node("ret-rrf", "FUSION", "RRF 融合", "融合多個 ranked lists 並去重。", "branch lists", "fused candidates", 960, 360,
        ["驗證多列表排序", "保留各分支診斷分數"]),
      node("ret-rerank", "PRECISION", "Reranker", "依相關性與答案承載力重排。", "fused candidates", "reranked evidence", 1210, 360,
        ["直接答案片段優先", "避免只因語意相似而高分"]),
      node("ret-expand", "CONTEXT", "Parent Chunk Expansion", "補回必要前後文與父層內容。", "reranked evidence", "expanded contexts", 1460, 240,
        ["補足條件與例外", "控制擴張 token 數"]),
      node("ret-top", "OUTPUT", "Top Evidence Package", "整理 chunks、引用、分數與信心。", "expanded contexts", "evidence package", 1460, 570,
        ["來源定位完整", "只保留 top-k 可用證據"]),
    ],
    edges: [
      ["ret-scope", "ret-bm25"], ["ret-scope", "ret-dense"], ["ret-scope", "ret-graph"],
      ["ret-bm25", "ret-rrf"], ["ret-dense", "ret-rrf"], ["ret-graph", "ret-rrf"],
      ["ret-rrf", "ret-rerank"], ["ret-rerank", "ret-expand"], ["ret-expand", "ret-top"],
    ],
  },

  evidence: {
    id: "evidence", title: "證據驗證", eyebrow: "LEVEL 2 · EVIDENCE GATE",
    width: 1650, height: 940,
    groups: [
      { label: "判定訊號", x: 40, y: 130, width: 700, height: 650 },
      { label: "安全決策", x: 800, y: 130, width: 790, height: 650 },
    ],
    nodes: [
      node("ev-package", "INPUT", "候選證據包", "接收問題、片段、來源與檢索分數。", "question、contexts、scores", "evidence input", 80, 330,
        ["contexts 可追溯來源", "保留各檢索 branch 分數"]),
      node("ev-relevance", "SIGNAL", "主題相關性", "確認片段確實談論問題主體。", "question、contexts", "relevance signals", 360, 210,
        ["忽略只有泛用詞的片段", "記錄 meaningful matched terms"]),
      node("ev-threshold", "SIGNAL", "數值／期間證據", "檢查主體與答案值是否在合理範圍共現。", "question、contexts", "answer-bearing signal", 360, 500,
        ["辨識數字、日期與單位", "區分相鄰制度的數值"]),
      node("ev-gate", "VERIFY", "Evidence Quality Gate", "綜合訊號決定證據是否足夠回答。", "relevance、answer-bearing", "sufficient status", 820, 330,
        ["低信心時 fail closed", "回傳 reason、confidence、signals"]),
      node("ev-sufficient", "PASS", "證據充分", "允許建立 evidence-only prompt。", "sufficient evidence", "trusted contexts", 1100, 210,
        ["只傳遞通過驗證的片段", "保留引用 rank"]),
      node("ev-contract", "CHECK", "回答契約檢查", "生成後再次確認引用與證據標記。", "candidate answer、contexts", "accepted／rejected", 1340, 210,
        ["引用 rank 必須存在", "無來源答案不得輸出"]),
      node("ev-insufficient", "STOP", "證據不足", "停止模型生成並保留缺少內容。", "insufficient evidence", "refusal decision", 1100, 540,
        ["不使用模型記憶補答", "generateMs 接近零"]),
      node("ev-refusal", "OUTPUT", "缺口說明", "告訴使用者缺少哪份文件或直接證據。", "refusal decision", "actionable refusal", 1340, 540,
        ["原因簡短且具體", "提示可補充的資料類型"]),
    ],
    edges: [
      ["ev-package", "ev-relevance"], ["ev-package", "ev-threshold"], ["ev-relevance", "ev-gate"],
      ["ev-threshold", "ev-gate"], ["ev-gate", "ev-sufficient"], ["ev-sufficient", "ev-contract"],
      ["ev-gate", "ev-insufficient"], ["ev-insufficient", "ev-refusal"],
    ],
  },

  answer: {
    id: "answer", title: "模型回答與引用", eyebrow: "LEVEL 2 · GROUNDED ANSWER",
    width: 1880, height: 880,
    groups: [{ label: "受證據約束的生成", x: 40, y: 130, width: 1780, height: 560 }],
    nodes: [
      node("ans-evidence", "INPUT", "可信證據", "只接收通過 Evidence Gate 的 contexts。", "trusted contexts", "ordered evidence", 70, 300,
        ["拒絕未驗證 contexts", "保留來源 rank 與頁碼"]),
      node("ans-order", "ATTENTION", "Evidence Ordering", "把強證據安排在 prompt 前後緣。", "ranked contexts", "attention-aware order", 320, 300,
        ["高分證據靠近上下文邊緣", "避免重複或無關內容"]),
      node("ans-prompt", "PROMPT", "Evidence-only Prompt", "標示唯一可信證據與回答規則。", "question、ordered contexts", "grounded prompt", 570, 300,
        ["使用 trusted_evidence 邊界", "要求實質主張附 rank"]),
      node("ans-llm", "MODEL", "QA Agent LLM", "根據 grounded prompt 生成候選回答。", "grounded prompt", "candidate answer", 870, 300,
        ["provider 可替換", "記錄模型版本與 generateMs"]),
      node("ans-tool", "OPTIONAL", "Tool／API Layer", "在允許時呼叫外部函式或後端工具。", "tool request", "tool evidence", 870, 540,
        ["工具輸出與模型文字分開", "限制可呼叫工具與參數"]),
      node("ans-citation", "GUARD", "引用驗證", "確認答案中的 rank 存在且直接支持主張。", "candidate answer、contexts", "validated answer", 1190, 300,
        ["不存在的 rank 立即拒絕", "每個事實句都有支持"]),
      node("ans-output", "OUTPUT", "附引用回答", "輸出答案、信心、警告、引用與 timings。", "validated answer", "RAG response", 1510, 300,
        ["回答可追溯到原始文件", "呈現 grounding warnings"]),
    ],
    edges: [
      ["ans-evidence", "ans-order"], ["ans-order", "ans-prompt"], ["ans-prompt", "ans-llm"],
      ["ans-llm", "ans-tool"], ["ans-tool", "ans-llm"], ["ans-llm", "ans-citation"],
      ["ans-citation", "ans-output"],
    ],
  },

  platform: {
    id: "platform", title: "資料、模型與部署", eyebrow: "LEVEL 2 · LOCAL & PRODUCTION PLATFORM",
    width: 1980, height: 1120,
    groups: [
      { label: "MODE A · 單機 Demo", x: 40, y: 110, width: 700, height: 420 },
      { label: "MODE B · 正式部署", x: 800, y: 110, width: 1120, height: 900 },
    ],
    nodes: [
      node("plt-webapp", "LOCAL", "rag_demo.web_app", "本機靜態頁面、文件匯入與 RAG API。", "browser request", "local response", 80, 230,
        ["單一指令可啟動", "健康檢查回報模型與檢索器"]),
      node("plt-files", "LOCAL DATA", "本機文件與索引", "保存原始文件、chunks 與 embeddings。", "ingestion output", "local index", 340, 170,
        ["原始檔與 chunk 可對照", "重啟後可重載索引"]),
      node("plt-sqlite", "LOCAL DATA", "SQLite 對話紀錄", "保存 Demo 對話與訊息。", "conversation events", "history", 340, 370,
        ["對話建立與刪除可用", "不同會話互不污染"]),
      node("plt-caddy", "EDGE", "Caddy", "處理 HTTPS、反向代理與入口限制。", "external HTTP/S", "controlled traffic", 840, 210,
        ["啟用 TLS 與安全標頭", "限制上傳大小與逾時"]),
      node("plt-api", "API", "FastAPI", "正式同步 API 與背景工作派發入口。", "authenticated request", "response／job", 1100, 210,
        ["統一 API contract", "產生可追蹤 run ID"]),
      node("plt-auth", "SECURITY", "OIDC／Tenant／Role", "在資料與檢索前套用企業權限。", "identity、resource", "retrieval scope", 840, 450,
        ["禁止跨租戶引用", "權限套用到所有 retrieval branches"]),
      node("plt-model", "MODEL", "Ollama／模型 API", "提供 Embedding、VLM、路由與回答。", "prompt、image、text", "inference", 1100, 450,
        ["設定模型保活", "記錄品質、延遲與失敗"]),
      node("plt-redis", "QUEUE", "Redis", "承載任務佇列、session 與 heartbeat。", "jobs、temporary state", "queued work", 1360, 210,
        ["定義重試與 dead-letter", "監測 worker heartbeat"]),
      node("plt-worker", "ASYNC", "Celery Worker", "執行掃毒、解析、OCR、Embedding 與索引。", "queued work", "indexed result", 1620, 210,
        ["工作具 idempotency", "保存失敗階段與原因"]),
      node("plt-pg", "DATA", "PostgreSQL", "保存 metadata、job、workflow 與稽核資料。", "service records", "transactional state", 1100, 720,
        ["migration 與備份可驗證", "重要寫入具唯一約束"]),
      node("plt-qdrant", "DATA", "Qdrant", "保存向量與 chunk payload。", "vectors、metadata", "dense candidates", 1360, 720,
        ["collection schema 可版本化", "搜尋前強制 tenant filter"]),
      node("plt-s3", "DATA", "SeaweedFS S3", "保存原始檔與擷取產物。", "files、derived assets", "versioned objects", 1620, 720,
        ["原始文件不可被索引覆寫", "設定版本與保留政策"]),
    ],
    edges: [
      ["plt-webapp", "plt-files"], ["plt-webapp", "plt-sqlite"], ["plt-webapp", "plt-model"],
      ["plt-caddy", "plt-api"], ["plt-auth", "plt-api"], ["plt-api", "plt-model"],
      ["plt-api", "plt-redis"], ["plt-redis", "plt-worker"], ["plt-api", "plt-pg"],
      ["plt-api", "plt-qdrant"], ["plt-api", "plt-s3"], ["plt-worker", "plt-pg"],
      ["plt-worker", "plt-qdrant"], ["plt-worker", "plt-s3"],
    ],
  },
};

export function clampZoom(value) {
  return Math.max(30, Math.min(160, Number(value) || 75));
}

export function calculateProgress(views, checkedGoals) {
  const nodes = Object.values(views).flatMap((view) => view.nodes);
  const uniqueNodes = [...new Map(nodes.map((item) => [item.id, item])).values()];
  const total = uniqueNodes.reduce((sum, item) => sum + item.goals.length, 0);
  const done = uniqueNodes.reduce(
    (sum, item) => sum + item.goals.filter((_, index) => checkedGoals[`${item.id}:${index}`]).length,
    0,
  );
  return { done, total, percent: total ? Math.round((done / total) * 100) : 0 };
}

export function calculateAnchoredScroll({ scrollLeft, scrollTop, pointerX, pointerY, oldZoom, newZoom }) {
  const oldScale = clampZoom(oldZoom) / 100;
  const newScale = clampZoom(newZoom) / 100;
  const contentX = (scrollLeft + pointerX) / oldScale;
  const contentY = (scrollTop + pointerY) / oldScale;
  return { left: contentX * newScale - pointerX, top: contentY * newScale - pointerY };
}

function loadCheckedGoals() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"); }
  catch { return {}; }
}

const state = {
  viewId: "root", trail: [], zoom: 75, selectedNodeId: "",
  checkedGoals: typeof localStorage === "undefined" ? {} : loadCheckedGoals(),
};

function allNodes() { return Object.values(architectureViews).flatMap((view) => view.nodes); }
function findNode(nodeId) { return allNodes().find((item) => item.id === nodeId); }
function currentView() { return architectureViews[state.viewId]; }

function goalProgress(nodes) {
  const total = nodes.reduce((sum, item) => sum + item.goals.length, 0);
  const done = nodes.reduce(
    (sum, item) => sum + item.goals.filter((_, index) => state.checkedGoals[`${item.id}:${index}`]).length,
    0,
  );
  return { done, total };
}

function nodeProgress(item) {
  const ownedNodes = [item];
  if (item.drilldown && architectureViews[item.drilldown]) ownedNodes.push(...architectureViews[item.drilldown].nodes);
  return goalProgress(ownedNodes);
}

function renderGroups(view) {
  const layer = document.querySelector("#groupLayer");
  layer.innerHTML = "";
  view.groups.forEach((group) => {
    const element = document.createElement("div");
    element.className = "architecture-group";
    Object.assign(element.style, { left: `${group.x}px`, top: `${group.y}px`, width: `${group.width}px`, height: `${group.height}px` });
    element.innerHTML = `<span>${group.label}</span>`;
    layer.append(element);
  });
}

function renderConnections(view) {
  const svg = document.querySelector("#connectionLayer");
  const nodeMap = new Map(view.nodes.map((item) => [item.id, item]));
  svg.innerHTML = `<defs><marker id="arrowhead" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#9aabc1"></path></marker></defs>`;
  view.edges.forEach(([fromId, toId]) => {
    const from = nodeMap.get(fromId);
    const to = nodeMap.get(toId);
    if (!from || !to) return;
    const fromCenter = { x: from.x + NODE_WIDTH / 2, y: from.y + NODE_HEIGHT / 2 };
    const toCenter = { x: to.x + NODE_WIDTH / 2, y: to.y + NODE_HEIGHT / 2 };
    const dx = toCenter.x - fromCenter.x;
    const dy = toCenter.y - fromCenter.y;
    let x1; let y1; let x2; let y2; let controls;
    if (Math.abs(dx) >= Math.abs(dy)) {
      const direction = dx >= 0 ? 1 : -1;
      x1 = fromCenter.x + direction * NODE_WIDTH / 2; y1 = fromCenter.y;
      x2 = toCenter.x - direction * NODE_WIDTH / 2; y2 = toCenter.y;
      const bend = Math.max(42, Math.abs(x2 - x1) * 0.42);
      controls = `${x1 + direction * bend} ${y1}, ${x2 - direction * bend} ${y2}`;
    } else {
      const direction = dy >= 0 ? 1 : -1;
      x1 = fromCenter.x; y1 = fromCenter.y + direction * NODE_HEIGHT / 2;
      x2 = toCenter.x; y2 = toCenter.y - direction * NODE_HEIGHT / 2;
      const bend = Math.max(42, Math.abs(y2 - y1) * 0.42);
      controls = `${x1} ${y1 + direction * bend}, ${x2} ${y2 - direction * bend}`;
    }
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("class", "connection-line");
    path.setAttribute("d", `M ${x1} ${y1} C ${controls}, ${x2} ${y2}`);
    svg.append(path);
  });
}

function renderNodes(view) {
  const layer = document.querySelector("#nodeLayer");
  layer.innerHTML = "";
  view.nodes.forEach((item) => {
    const progress = nodeProgress(item);
    const targetView = item.drilldown ? architectureViews[item.drilldown] : null;
    const button = document.createElement("button");
    button.type = "button";
    button.className = `architecture-node${item.id === state.selectedNodeId ? " is-selected" : ""}${progress.done === progress.total ? " is-complete" : ""}${targetView ? " is-drillable" : ""}`;
    button.style.left = `${item.x}px`; button.style.top = `${item.y}px`; button.dataset.nodeId = item.id;
    button.setAttribute("aria-label", targetView ? `${item.title}，進入包含 ${targetView.nodes.length} 個節點的子架構` : `${item.title}，完成 ${progress.done}／${progress.total}`);
    button.innerHTML = `
      <div class="node-head"><div><span>${item.code}</span><strong>${item.title}</strong></div><span class="node-progress">${progress.done}/${progress.total}</span></div>
      <p class="node-purpose">${item.purpose}</p>
      <span class="node-task-preview">${targetView ? `進入子架構 · ${targetView.nodes.length} 節點 →` : progress.done === progress.total ? "任務已完成" : "查看並勾選任務目標"}</span>`;
    button.addEventListener("click", () => targetView ? enterView(item.drilldown, item.id) : selectNode(item.id));
    layer.append(button);
  });
}

function renderInspector() {
  const item = findNode(state.selectedNodeId);
  const empty = document.querySelector("#inspectorEmpty");
  const content = document.querySelector("#inspectorContent");
  if (!item) { empty.hidden = false; content.hidden = true; return; }
  const ownProgress = goalProgress([item]);
  const targetView = item.drilldown ? architectureViews[item.drilldown] : null;
  empty.hidden = true; content.hidden = false;
  content.innerHTML = `
    <p class="inspector-kicker">${item.code}</p><h2 class="inspector-title">${item.title}</h2>
    <p class="inspector-purpose">${item.purpose}</p>
    ${targetView ? `<p class="inspector-domain-note">目前正在檢視此領域的 ${targetView.nodes.length} 個子節點；點選畫布節點可切換任務。</p>` : ""}
    <div class="inspector-meta">${item.module ? `<div class="meta-row"><span>模組</span><strong>${item.module}</strong></div>` : ""}
      <div class="meta-row"><span>輸入</span><strong>${item.input}</strong></div><div class="meta-row"><span>輸出</span><strong>${item.output}</strong></div></div>
    <section class="task-section"><div class="task-section-head"><h3>任務目標</h3><span>${ownProgress.done} / ${ownProgress.total} 完成</span></div>
      <div class="goal-list">${item.goals.map((goal, index) => `<label class="goal-item"><input type="checkbox" data-goal-index="${index}" ${state.checkedGoals[`${item.id}:${index}`] ? "checked" : ""} /><span>${goal}</span></label>`).join("")}</div>
    </section>`;
  content.querySelectorAll("[data-goal-index]").forEach((input) => {
    input.addEventListener("change", () => toggleGoal(item.id, Number(input.dataset.goalIndex), input.checked));
  });
}

function renderOverallProgress() {
  const progress = calculateProgress(architectureViews, state.checkedGoals);
  document.querySelector("#overallProgressText").textContent = `${progress.percent}%`;
  document.querySelector("#overallProgressBar").style.width = `${progress.percent}%`;
}

function renderBreadcrumb() {
  const breadcrumb = document.querySelector("#architectureBreadcrumb");
  const view = currentView();
  if (state.viewId === "root") {
    breadcrumb.innerHTML = `<span class="depth-badge">L1</span><button type="button" disabled>整體架構</button>`;
    return;
  }
  breadcrumb.innerHTML = `<span class="depth-badge">L2</span><button type="button" data-root-link>整體架構</button><span class="breadcrumb-separator">›</span><button type="button" disabled>${view.title}</button>`;
  breadcrumb.querySelector("[data-root-link]").addEventListener("click", goToRoot);
}

function renderView() {
  const view = currentView();
  document.querySelector("#levelTitle").textContent = view.title;
  document.querySelector("#levelEyebrow").textContent = view.eyebrow;
  document.querySelector("#upLevel").hidden = state.viewId === "root";
  const canvas = document.querySelector("#architectureCanvas");
  canvas.style.width = `${view.width}px`; canvas.style.height = `${view.height}px`;
  renderBreadcrumb(); renderGroups(view); renderConnections(view); renderNodes(view); renderInspector(); renderOverallProgress();
}

function selectNode(nodeId) { state.selectedNodeId = nodeId; renderNodes(currentView()); renderInspector(); }

function toggleGoal(nodeId, index, checked) {
  state.checkedGoals[`${nodeId}:${index}`] = checked;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state.checkedGoals));
  renderNodes(currentView()); renderInspector(); renderOverallProgress();
}

function enterView(viewId, parentNodeId) {
  if (!architectureViews[viewId]) return;
  state.trail.push({ viewId: state.viewId, selectedNodeId: parentNodeId });
  state.viewId = viewId; state.selectedNodeId = parentNodeId;
  renderView(); requestAnimationFrame(fitCanvas);
}

function goUp() {
  const previous = state.trail.pop();
  if (!previous) return;
  state.viewId = previous.viewId; state.selectedNodeId = previous.selectedNodeId;
  renderView(); requestAnimationFrame(fitCanvas);
}

function goToRoot() {
  const rootSelection = state.trail[0]?.selectedNodeId || "";
  state.trail = []; state.viewId = "root"; state.selectedNodeId = rootSelection;
  renderView(); requestAnimationFrame(fitCanvas);
}

function applyZoom(value) {
  state.zoom = clampZoom(value);
  const scale = state.zoom / 100;
  const view = currentView();
  document.querySelector("#architectureCanvas").style.transform = `scale(${scale})`;
  document.querySelector("#canvasSizer").style.width = `${view.width * scale}px`;
  document.querySelector("#canvasSizer").style.height = `${view.height * scale}px`;
  document.querySelector("#zoomRange").value = String(Math.round(state.zoom));
  document.querySelector("#zoomValue").textContent = `${Math.round(state.zoom)}%`;
}

function fitCanvas() {
  const viewport = document.querySelector("#canvasViewport");
  const view = currentView();
  const availableWidth = Math.max(320, viewport.clientWidth - 28);
  const availableHeight = Math.max(320, viewport.clientHeight - 28);
  applyZoom(Math.floor(Math.min(availableWidth / view.width, availableHeight / view.height) * 100 / 5) * 5);
  viewport.scrollTo({ top: 0, left: 0 });
}

function zoomAtPointer(event) {
  event.preventDefault();
  const viewport = document.querySelector("#canvasViewport");
  const rect = viewport.getBoundingClientRect();
  const pointerX = event.clientX - rect.left;
  const pointerY = event.clientY - rect.top;
  const normalizedDelta = event.deltaY * (event.deltaMode === 1 ? 16 : 1);
  const amount = Math.max(-8, Math.min(8, -normalizedDelta * 0.08));
  const nextZoom = clampZoom(state.zoom + amount);
  if (nextZoom === state.zoom) return;
  const nextScroll = calculateAnchoredScroll({
    scrollLeft: viewport.scrollLeft, scrollTop: viewport.scrollTop, pointerX, pointerY,
    oldZoom: state.zoom, newZoom: nextZoom,
  });
  applyZoom(nextZoom);
  viewport.scrollLeft = Math.max(0, nextScroll.left);
  viewport.scrollTop = Math.max(0, nextScroll.top);
}

function bindPan(viewport) {
  let drag = null;
  viewport.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || event.target.closest(".architecture-node")) return;
    drag = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, left: viewport.scrollLeft, top: viewport.scrollTop };
    viewport.setPointerCapture(event.pointerId);
    viewport.classList.add("is-panning");
  });
  viewport.addEventListener("pointermove", (event) => {
    if (!drag || drag.pointerId !== event.pointerId) return;
    viewport.scrollLeft = drag.left - (event.clientX - drag.x);
    viewport.scrollTop = drag.top - (event.clientY - drag.y);
  });
  const finish = (event) => {
    if (!drag || drag.pointerId !== event.pointerId) return;
    drag = null; viewport.classList.remove("is-panning");
  };
  viewport.addEventListener("pointerup", finish);
  viewport.addEventListener("pointercancel", finish);
}

function zoomAtCenter(delta) {
  const viewport = document.querySelector("#canvasViewport");
  const rect = viewport.getBoundingClientRect();
  const pointerX = rect.width / 2;
  const pointerY = rect.height / 2;
  const nextZoom = clampZoom(state.zoom + delta);
  const nextScroll = calculateAnchoredScroll({
    scrollLeft: viewport.scrollLeft, scrollTop: viewport.scrollTop, pointerX, pointerY,
    oldZoom: state.zoom, newZoom: nextZoom,
  });
  applyZoom(nextZoom);
  viewport.scrollLeft = Math.max(0, nextScroll.left);
  viewport.scrollTop = Math.max(0, nextScroll.top);
}

function bindControls() {
  const range = document.querySelector("#zoomRange");
  const viewport = document.querySelector("#canvasViewport");
  range.addEventListener("input", () => applyZoom(range.value));
  document.querySelector("#zoomOut").addEventListener("click", () => zoomAtCenter(-10));
  document.querySelector("#zoomIn").addEventListener("click", () => zoomAtCenter(10));
  document.querySelector("#fitCanvas").addEventListener("click", fitCanvas);
  document.querySelector("#upLevel").addEventListener("click", goUp);
  viewport.addEventListener("wheel", zoomAtPointer, { passive: false });
  bindPan(viewport);
  document.addEventListener("keydown", (event) => {
    if (event.target instanceof Element && event.target.matches("input, textarea, select")) return;
    if (event.key === "Backspace" && state.viewId !== "root") { event.preventDefault(); goUp(); }
    if (event.key === "0") fitCanvas();
  });
}

function init() { bindControls(); renderView(); requestAnimationFrame(fitCanvas); }
if (typeof document !== "undefined") init();
