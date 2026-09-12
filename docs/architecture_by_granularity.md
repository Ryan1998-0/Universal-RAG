# Universal-RAG 分層架構圖

本文以三種顆粒度描述系統：L1 適合對外簡報，L2 適合系統設計與部署，L3 適合開發、追蹤資料流與除錯。

## L1：系統全景圖

```mermaid
flowchart LR
    U[使用者] --> FE[Web 前端]

    FE --> LOCAL[單機 Demo]
    FE --> PROD[正式部署 API]

    LOCAL --> CORE[核心 RAG 引擎]
    PROD --> CORE

    DOCS[多格式文件] --> CORE
    CORE --> DATA[(文件、索引與對話資料)]
    CORE --> MODEL[Ollama／可替換模型]

    CORE --> ANSWER[附引用回答<br/>或證據不足拒答]
    ANSWER --> U
```

這一層只回答使用者、RAG 系統、文件、資料與模型之間如何互動。

## L2：服務與基礎設施圖

```mermaid
flowchart TB
    USER[使用者瀏覽器]

    subgraph ENTRY[入口層]
        FE[前端靜態頁面]
        CADDY[Caddy<br/>HTTPS／Reverse Proxy]
    end

    subgraph LOCAL[模式 A：單機 Demo]
        WEBAPP[rag_demo.web_app]
        LOCAL_FILES[(本機文件與索引)]
        SQLITE[(SQLite 對話紀錄)]
    end

    subgraph PROD[模式 B：正式部署]
        API[FastAPI<br/>rag_demo.production.api]
        AUTH[OIDC／Tenant／Role]
        GATE[Rate Limit<br/>Generation Gate]
        WORKER[Celery Worker]
        BEAT[Celery Beat]
    end

    subgraph STORAGE[正式資料層]
        PG[(PostgreSQL<br/>Metadata／Job／Workflow)]
        QD[(Qdrant<br/>Vector／Chunk)]
        S3[(SeaweedFS S3<br/>原始與擷取文件)]
        REDIS[(Redis<br/>Queue／Session／Heartbeat)]
    end

    subgraph EXTERNAL[外部或可替換服務]
        CLAM[ClamAV]
        MODEL[Ollama／外部模型 API]
        IDP[OIDC Identity Provider]
    end

    USER --> FE
    FE --> WEBAPP
    FE --> CADDY --> API

    WEBAPP --> LOCAL_FILES
    WEBAPP --> SQLITE
    WEBAPP --> MODEL

    IDP --> AUTH --> API
    API --> GATE
    API --> PG
    API --> QD
    API --> S3
    API --> MODEL

    API -->|派發非同步任務| REDIS
    REDIS --> WORKER
    BEAT --> REDIS

    WORKER --> CLAM
    WORKER --> PG
    WORKER --> QD
    WORKER --> S3
```

這一層呈現單機與正式部署的差異，以及服務、背景任務與資料儲存之間的依賴關係。

## L3：RAG 處理管線與模組圖

```mermaid
flowchart TB
    subgraph INGEST[文件匯入管線｜非同步]
        FILE[PDF／圖片／DOCX<br/>TXT／Markdown／JSON]
        VALIDATE[格式、大小與檔案驗證]
        SECURITY[ClamAV 掃毒<br/>Prompt Injection Policy]
        PARSE[文件解析／OCR]
        NORMALIZE[內容正規化]
        CHUNK[父子分塊]
        EMBED[Embedding]
        INDEX[建立持久化索引]

        FILE --> VALIDATE --> SECURITY --> PARSE
        PARSE --> NORMALIZE --> CHUNK --> EMBED --> INDEX
    end

    subgraph QUERY[問答管線｜同步]
        QUESTION[使用者問題]
        ROUTE{自適應路由}

        DIRECT[一般回答]
        TIME[日期時間工具]
        REWRITE[查詢改寫]
        SCOPE[KB／文件／權限範圍過濾]

        subgraph RETRIEVAL[混合檢索]
            BM25[BM25]
            DENSE[Dense Embedding]
            GRAPH[Graph Retrieval<br/>可選]
            RRF[候選集合合併]
        end

        FUSION[LambdaMART 分數映射與融合]
        COMPLEXITY{問題複雜度}
        SIMPLE[簡單問題 Top-5]
        RERANK[複雜問題 Rerank]
        EXPAND[Parent Chunk Expansion]
        VERIFY{Evidence Quality Gate}
        PROMPT[Prompt Builder]
        LLM[Ollama／可替換模型]
        GROUNDED[附來源引用回答]
        REFUSE[證據不足拒答]

        QUESTION --> ROUTE
        ROUTE -->|不需檢索| DIRECT
        ROUTE -->|日期時間| TIME
        ROUTE -->|需要文件證據| REWRITE

        REWRITE --> SCOPE
        SCOPE --> BM25
        SCOPE --> DENSE
        SCOPE -.-> GRAPH

        BM25 --> RRF
        DENSE --> RRF
        GRAPH -.-> RRF

        RRF --> FUSION --> COMPLEXITY
        COMPLEXITY -->|簡單| SIMPLE --> EXPAND
        COMPLEXITY -->|複雜| RERANK --> EXPAND
        EXPAND --> VERIFY
        VERIFY -->|證據充分| PROMPT --> LLM --> GROUNDED
        VERIFY -->|證據不足| REFUSE
    end

    subgraph MODULES[主要程式模組]
        M1[document_pipeline.py]
        M2[rag_pipeline.py]
        M3[query_rewriter.py]
        M4[hybrid_retrieval.py]
        M5[retrieval_planner.py]
        M6[retrieval_verifier.py]
        M7[self_rag_reflection.py]
        M8[model_providers.py]
    end

    M1 -.-> INGEST
    M2 -.-> QUERY
    M3 -.-> REWRITE
    M4 -.-> RETRIEVAL
    M5 -.-> ROUTE
    M6 -.-> VERIFY
    M7 -.-> VERIFY
    M8 -.-> LLM

    INDEX -.-> BM25
    INDEX -.-> DENSE
    INDEX -.-> GRAPH
```

## 使用方式

- L1：對外介紹專案定位與核心價值。
- L2：說明部署架構、服務依賴與資料儲存。
- L3：追蹤文件匯入與問答過程，並對應實際 Python 模組。
