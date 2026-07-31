import assert from "node:assert/strict";
import test from "node:test";

import {
  ApiClientError,
  buildIndex,
  listKnowledgeBases,
  requestJson,
  uploadDocument,
} from "../../frontend/api-client.js";

function response(status, body = null) {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: new Headers(),
    async json() {
      if (body === null) throw new Error("no body");
      return body;
    },
  };
}

test("requestJson preserves structured API errors", async () => {
  const fetchImpl = async () => response(409, {
    error: { code: "INDEX_BUILD_CONFLICT", message: "conflict" },
    request_id: "request-1",
  });
  await assert.rejects(
    requestJson("/v1/test", {}, fetchImpl),
    (error) => error instanceof ApiClientError
      && error.status === 409
      && error.code === "INDEX_BUILD_CONFLICT"
      && error.requestId === "request-1",
  );
});

test("listKnowledgeBases normalizes the production collection envelope", async () => {
  const fetchImpl = async (path, options) => {
    assert.equal(path, "/v1/knowledge-bases");
    assert.equal(options.credentials, "same-origin");
    return response(200, { items: [{ id: "kb-1", name: "Policies" }] });
  };
  assert.deepEqual(await listKnowledgeBases(fetchImpl), [{ id: "kb-1", name: "Policies" }]);
});

test("uploadDocument follows reserve, content, complete, and ingestion status", async () => {
  const calls = [];
  const fetchImpl = async (path, options) => {
    calls.push({ path: String(path), options });
    if (calls.length === 1) {
      return response(201, {
        upload_id: "upload-1",
        upload: {
          mode: "proxy",
          method: "PUT",
          url: "/v1/uploads/upload-1/content",
          headers: { "Content-Type": "text/plain" },
        },
      });
    }
    if (calls.length === 2) return response(201, { state: "uploaded" });
    if (calls.length === 3) {
      return response(202, {
        document_id: "document-1",
        ingestion_job_id: "ingestion-1",
      });
    }
    return response(200, { id: "ingestion-1", status: "succeeded", stage: "complete" });
  };
  const payload = new TextEncoder().encode("hello");
  const file = {
    name: "notes.txt",
    type: "text/plain",
    size: payload.byteLength,
    async arrayBuffer() {
      return payload.buffer;
    },
  };
  const stages = [];
  const result = await uploadDocument({
    knowledgeBaseId: "kb-1",
    file,
    onStage: (stage) => stages.push(stage),
    fetchImpl,
  });
  assert.equal(result.document_id, "document-1");
  assert.deepEqual(stages, ["hashing", "reserving", "uploading", "queuing", "ingesting", "ready"]);
  assert.equal(calls[1].options.method, "PUT");
  assert.equal(calls[2].path, "/v1/uploads/upload-1/complete");
});

test("buildIndex waits until the immutable index is published", async () => {
  let call = 0;
  const stages = [];
  const fetchImpl = async () => {
    call += 1;
    if (call === 1) {
      return response(202, { job_id: "index-job-1", index_version_id: "index-2" });
    }
    return response(200, { id: "index-job-1", status: "succeeded", stage: "published" });
  };
  const result = await buildIndex({
    knowledgeBaseId: "kb-1",
    documentIds: ["document-1"],
    onStage: (stage) => stages.push(stage),
    fetchImpl,
  });
  assert.equal(result.job.stage, "published");
  assert.deepEqual(stages, ["queuing", "building", "published"]);
});
