# Engineering review remediation

This checklist tracks work prompted by the static review of commit
`8fb192029b917f1acce059f8f4ce42e1e58a5cb0` on 2026-09-24. Each item
should be changed and reviewed separately. The review's findings are hypotheses
until the relevant runtime behavior or answer quality has been checked.

| Order | Finding | Status | Evidence / next check |
| --- | --- | --- | --- |
| 1 | Non-root container log directory | Code changed | Image now creates `/var/lib/rag/logs` for UID 10001. Existing observability unit tests pass; container smoke is in CI and awaits a CI run. |
| 2 | Answer evidence validation | Code changed | Mixed refusals, uncited sentences, unrelated cited passages and numbers absent from cited passages now fail closed. The historical 10-case leave-rules artifact retains 5 accepted answers and 3 valid refusals; 2 malformed answers fail. This deterministic lexical check cannot prove semantic entailment, so item 4 must measure real answer support. |
| 3 | Production query variants and evidence query | Pending | Trace and align production retrieval with its public contract. |
| 4 | Production `/api/ask` end-to-end release gate | Pending | Define a representative, reproducible gate. |
| 5 | Benchmark interpretation and provenance | Pending | Record exact scope, metric definitions, model versions and latency distribution. |
| 6 | Embedding defaults across README and production | Pending | Clarify effective profiles and prevent incompatible index/query configuration. |
| 7 | Prompt injection validation | Pending | Add an attack corpus and measure quarantine/answer behavior. |
| 8 | Demo duplicate route/retrieve and ask | Pending | Make preview and final answer use one evidence path. |

The original checkout contains unrelated uncommitted changes. This work is on
the `codex/universal-rag-improvements` branch in a separate Git worktree.
