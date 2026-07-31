# IFRS17-RAG

A local-first, multimodal RAG application built around Qwen 2.5 7B. It combines adaptive routing, hybrid retrieval, reranking, citations, conversation history, and document management in a ChatGPT-style interface.

[Traditional Chinese](README.zh-TW.md) | [Static browser demo](https://ryan1998-0.github.io/IFRS17-RAG/ifrs17-demo/)

> The GitHub Pages demo is a static retrieval showcase. Run the local service to use Qwen, upload documents, OCR, conversations, and persistent memory.

## Highlights

- Self-RAG routing: direct answers for simple questions, retrieval for source-dependent questions.
- Hybrid search: BM25 + dense embeddings + reciprocal-rank fusion + reranking.
- Multimodal ingestion: PDF, images, DOCX, TXT, Markdown, and JSON.
- OCR for images and scanned PDFs, with chunking and persistent indexes.
- Folder-based knowledge management and per-query document selection.
- SQLite conversation history, explicit long-term memory, and source citations.
- Local inference through Ollama with `qwen2.5:7b`.
- Production-oriented FastAPI, PostgreSQL, Qdrant, object storage, workers, OIDC, metrics, backup, and CI scaffolding.

## Flow

```text
Question
  -> adaptive router
     -> direct answer / date-time tool
     -> query rewrite -> BM25 + embeddings -> fusion -> rerank
        -> evidence gate -> Qwen answer with citations

Document
  -> validation -> parser or OCR -> normalized units
  -> chunks -> embeddings -> persistent index -> selectable knowledge base
```

## Run Locally

Requirements: Python 3.12 and [Ollama](https://ollama.com/).

```bash
git clone https://github.com/Ryan1998-0/IFRS17-RAG.git
cd IFRS17-RAG

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock

ollama pull qwen2.5:7b
python -m rag_demo.web_app
```

Open [http://127.0.0.1:8765/ifrs17-demo/index.html](http://127.0.0.1:8765/ifrs17-demo/index.html), then upload and select the documents to use for RAG.

## Verification

```bash
.venv/bin/python -m pytest -q
node --test tests/ifrs17-demo/*.test.mjs tests/frontend/*.test.mjs
```

Latest local release checks:

| Check | Result |
| --- | ---: |
| Python regression suite | 164 passed + 4 subtests |
| Browser JavaScript suite | 21 passed |
| Multimodal ingestion | 10/10 |
| Folder and restart persistence | 17/17 |
| IFRS 17 retrieval-only benchmark, best variant | 89.6% |

## Production Status

The local application is functional. The production service is a staging candidate, not a production-approved deployment. Real Linux, OIDC, load, cross-tenant security, backup/restore, and rollback drills are still required.

- [Production target and release gates](docs/production-ready-rag-target.md)
- [Deployment runbook](docs/production-runbook.md)
- [Multimodal ingestion pipeline](docs/multimodal-ingestion-pipeline.md)
- [Current completeness audit](docs/local-rag-completeness-report-2026-07-31.md)

## Data Notice

IFRS materials remain the property of their respective copyright holders. Do not treat this project, its demo data, or its answers as accounting advice or IFRS compliance evidence. Check source terms before redistributing documents or extracted content.
