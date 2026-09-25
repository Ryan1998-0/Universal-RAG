# Production API release gate

This gate calls the production `POST /v1/ask` endpoint against a staging
knowledge base. It checks actual generated answers, citations, refusals,
cross-tenant denial, wall latency and request errors. A nonzero exit code blocks
release. The gate does not ingest files or create a service.

## Prepare the frozen staging corpus

Upload each file in `fixtures/` as a **separate source** into a staging
knowledge base and wait until its immutable index is active:

- `policy-2024.txt` and `policy-2026.txt` exercise conflicting versions.
- `workflow-request.txt` and `workflow-approval.txt` must remain separate
  sources so the multi-hop case needs two citations.
- `ocr-notice.png` is an image containing Traditional Chinese and `BLUE-17`;
  its content must go through the OCR ingestion path.

Use a separate knowledge base owned by another staging tenant for the access
denial check. Copy `manifest.template.json` to a local manifest and replace
the knowledge base IDs and source IDs with the IDs returned by staging. The
template intentionally cannot run with its placeholders. Review the expected
facts and latency threshold for the staging hardware before using the result
as a release decision.

## Run

Set `RAG_RELEASE_BASE_URL` and `RAG_RELEASE_BEARER_TOKEN` in the local
environment, then run:

```bash
python scripts/run_production_release_gate.py \
  --manifest path/to/configured-manifest.json
```

The script writes a detailed JSON artifact under `runs/` and prints a short
summary. `runs/` is ignored by Git because answers and deployment details may
be sensitive. Keep the configured manifest with the release record, along with
the commit, image tag, model revision, active index ID and fixture hashes.

The checks are deterministic: they verify expected fact strings and the
server's citation diagnostics. They cannot prove that every natural-language
inference is correct. A human should inspect the saved answers before
declaring a new model or prompt suitable for production.
