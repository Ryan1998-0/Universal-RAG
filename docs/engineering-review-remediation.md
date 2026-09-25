# Engineering review remediation

This checklist tracks work prompted by the static review of commit
`8fb192029b917f1acce059f8f4ce42e1e58a5cb0` on 2026-09-24. Each item
should be changed and reviewed separately. The review's findings are hypotheses
until the relevant runtime behavior or answer quality has been checked.

| Order | Finding | Status | Evidence / next check |
| --- | --- | --- | --- |
| 1 | Non-root container log directory | Code changed | Image now creates `/var/lib/rag/logs` for UID 10001. Existing observability unit tests pass; container smoke is in CI and awaits a CI run. |
| 2 | Answer evidence validation | Code changed | Mixed refusals, uncited sentences, unrelated cited passages and numbers absent from cited passages now fail closed. The historical 10-case leave-rules artifact retains 5 accepted answers and 3 valid refusals; 2 malformed answers fail. This deterministic lexical check cannot prove semantic entailment, so item 4 must measure real answer support. |
| 3 | Production query variants and evidence query | Code changed | Production now runs bounded Dense/Sparse searches for distinct planned variants and the focused evidence query, then fuses their ranks. Scope and source filters are reused on every search. The focused retriever tests pass; live corpus recall/latency still need item 4's gate. |
| 4 | Production `/v1/ask` end-to-end release gate | Gate implemented; staging run pending | A frozen staging corpus and CLI gate cover Traditional Chinese, conflicting versions, refusal, OCR, multi-hop, citation support, cross-tenant denial, p95 wall time and error rate. The gate exits nonzero and saves a local artifact on failure. A real staging run still requires an indexed fixture KB and bearer token. |
| 5 | Benchmark interpretation and provenance | Code changed; new staging record pending | README and MultiHop report now distinguish Gold-fact recall from true character recall, and state each benchmark's scope and limits. The staging gate requires declared revisions, hashes local fixtures, checks the active index against the server and records a latency distribution. The historic runs lack complete revision/hardware metadata, so their missing provenance remains unknown; a configured staging run is still pending. |
| 6 | Embedding defaults across README and production | Code changed | README now states the MiniLM/384 default shared by code, Compose and production example. Bootstrap checks an existing Qdrant collection's Dense dimensions; `/v1/ask` refuses an active index with a different Embedding model/dimension, Chunk schema or collection. Focused tests pass; a live model migration still needs staging. |
| 7 | Prompt injection validation | Static and ingestion checks pass; live answer probe pending | A frozen 14-case corpus yields 11/11 detected attacks and 3/3 benign controls in the static scanner. English and Traditional Chinese attacks are quarantined before canonical artifacts are written. Evidence delimiters are escaped and production cannot use `flag` mode. A separate staging `/v1/ask` answer probe records safe answers, refusals and attack-following; it awaits an isolated flagged index and credentials. This corpus does not establish general model robustness. |
| 8 | Demo duplicate route/retrieve and ask | Pending | Make preview and final answer use one evidence path. |

The original checkout contains unrelated uncommitted changes. This work is on
the `codex/universal-rag-improvements` branch in a separate Git worktree.
