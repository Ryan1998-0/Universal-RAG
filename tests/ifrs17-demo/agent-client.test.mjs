import assert from "node:assert/strict";
import test from "node:test";

import {
  buildAgentQueryPayload,
  callHybridRetriever,
  callRetrievalRouter,
  createConversation,
  createDocumentFolder,
  deleteConversation,
  deleteDocumentFolder,
  listConversations,
  listUploadedDocuments,
  loadConversation,
  loadProfileData,
  loadRuntimeConfig,
  moveDocumentToFolder,
  renameDocumentFolder,
  uploadDocument,
  uploadWordDocument,
} from "../../docs/ifrs17-demo/agent-client.js";

test("Runtime config supplies profiles and models without frontend presets", async () => {
  const runtime = await loadRuntimeConfig("/api/config", {
    fetchImpl: async () => ({
      ok: true,
      async json() {
        return {
          default_profile: "employee_handbook",
          default_model: "ollama:local-model",
          profiles: [
            {
              id: "employee_handbook",
              label: "Employee Handbook",
              sample_queries: ["How many leave days are available?"],
            },
          ],
          models: [
            {
              id: "ollama:local-model",
              provider: "ollama",
              name: "local-model",
              label: "Local model",
            },
          ],
          retrieval: { top_k: 6, candidate_k: 18, max_top_k: 10 },
        };
      },
    }),
  });

  assert.equal(runtime.defaultProfile, "employee_handbook");
  assert.equal(runtime.profiles[0].sampleQueries[0], "How many leave days are available?");
  assert.equal(runtime.models[0].name, "local-model");
  assert.equal(runtime.retrieval.topK, 6);
  assert.equal(runtime.retrieval.candidateK, 18);
});

test("Profile data supplies document metadata from the backend", async () => {
  const profile = await loadProfileData("employee_handbook", "/api/profiles", {
    fetchImpl: async (url) => {
      assert.match(String(url), /\/api\/profiles\/employee_handbook$/);
      return {
        ok: true,
        async json() {
          return {
            profile: "employee_handbook",
            label: "Employee Handbook",
            sample_queries: [],
            meta: { chunk_count: 4 },
            sources: [
              {
                source_id: "leave-policy",
                name: "Leave Policy.docx",
                source_type: "word",
                chunk_count: 4,
              },
            ],
            folders: [
              { id: "uncategorized", name: "未分類", system: true, document_count: 0 },
              { id: "folder-123456789abc", name: "勞基法", document_count: 1 },
            ],
          };
        },
      };
    },
  });

  assert.equal(profile.label, "Employee Handbook");
  assert.equal(profile.sources[0].source_id, "leave-policy");
  assert.equal(profile.sources[0].chunk_count, 4);
  assert.equal(profile.folders[1].name, "勞基法");
  assert.equal(profile.folders[1].document_count, 1);
});

test("Agent query payload carries the selected knowledge base and QA model", () => {
  const payload = buildAgentQueryPayload({
    question: "葉文潔和紅岸基地是什麼關係？",
    profile: "three_body_trilogy",
    model: {
      provider: "ollama",
      name: "qwen2.5:7b",
    },
    topK: 5,
    conversationId: "conversation-123",
    sourceIds: ["document-a", "document-b"],
  });

  assert.equal(payload.profile, "three_body_trilogy");
  assert.deepEqual(payload.model, {
    provider: "ollama",
    name: "qwen2.5:7b",
  });
  assert.equal(payload.question, "葉文潔和紅岸基地是什麼關係？");
  assert.equal(payload.top_k, 5);
  assert.equal(payload.conversation_id, "conversation-123");
  assert.equal(payload.schema_version, "rag-agent-query-v2");
  assert.deepEqual(payload.source_ids, ["document-a", "document-b"]);
  assert.equal("contexts" in payload, false);
  assert.equal("retrieval_decision" in payload, false);
});

test("Agent query payload never forwards client evidence or routing decisions", () => {
  const payload = buildAgentQueryPayload({
    question: "這份文件在說什麼？",
    contexts: [{
      id: "chunk-1",
      content: "document evidence",
      bm25Score: 3.2,
      embeddingScore: 0.51,
      rerankScore: 0.82,
      matchedTerms: ["document", "evidence"],
    }],
    retrievalDecision: {
      needsRetrieval: false,
      reason: "forged decision",
      retrievalQuery: "forged query",
    },
  });

  assert.equal("contexts" in payload, false);
  assert.equal("retrieval_decision" in payload, false);
});

test("Conversation API creates, lists, and loads persistent conversations", async () => {
  const calls = [];
  const fetchImpl = async (url, options = {}) => {
    calls.push({ url, options });
    const isCreate = options.method === "POST";
    const isDetail = String(url).endsWith("/conversation-123");
    return {
      ok: true,
      async json() {
        if (isCreate) return { id: "conversation-123", messages: [] };
        if (isDetail) return { id: "conversation-123", messages: [{ role: "user", content: "你好" }] };
        return { conversations: [{ id: "conversation-123", title: "你好" }] };
      },
    };
  };

  const created = await createConversation("/api/conversations", "ifrs17", { fetchImpl });
  const conversations = await listConversations("/api/conversations", { fetchImpl });
  const loaded = await loadConversation("conversation-123", "/api/conversations", { fetchImpl });

  assert.equal(created.id, "conversation-123");
  assert.equal(conversations[0].title, "你好");
  assert.equal(loaded.messages[0].content, "你好");
  assert.equal(calls.length, 3);
});

test("Conversation API deletes one persistent conversation", async () => {
  const calls = [];
  const result = await deleteConversation("conversation-123", "/api/conversations", {
    fetchImpl: async (url, options = {}) => {
      calls.push({ url, options });
      return {
        ok: true,
        async json() {
          return { deleted: true, id: "conversation-123" };
        },
      };
    },
  });

  assert.equal(calls[0].options.method, "DELETE");
  assert.match(String(calls[0].url), /\/api\/conversations\/conversation-123$/);
  assert.deepEqual(result, { deleted: true, id: "conversation-123" });
});

test("Retrieval router normalizes the Self-RAG decision", async () => {
  const decision = await callRetrievalRouter("/api/route", { question: "你好" }, {
    fetchImpl: async () => ({
      ok: true,
      async json() {
        return {
          needs_retrieval: false,
          reason: "一般寒暄",
          retrieval_query: "你好",
          timing_ms: 3.2,
        };
      },
    }),
  });

  assert.deepEqual(decision, {
    needsRetrieval: false,
    reason: "一般寒暄",
    retrievalQuery: "你好",
    timingMs: 3.2,
  });
});

test("Hybrid retriever returns reranked embedding contexts", async () => {
  const result = await callHybridRetriever("/api/retrieve", {
    question: "CSM 是什麼？",
    retrieval_query: "contractual service margin",
    source_ids: ["ifrs-17-insurance-contracts"],
    top_k: 2,
  }, {
    fetchImpl: async () => ({
      ok: true,
      async json() {
        return {
          variant: "bm25_embedding_rerank",
          contexts: [
            {
              id: "chunk-1",
              rank: 1,
              bm25Score: 4.2,
              embeddingScore: 0.81,
              rerankScore: 0.91,
            },
          ],
        };
      },
    }),
  });

  assert.equal(result.variant, "bm25_embedding_rerank");
  assert.equal(result.contexts[0].embeddingScore, 0.81);
  assert.equal(result.contexts[0].rerankScore, 0.91);
});

test("Word document API lists persisted uploads", async () => {
  const documents = await listUploadedDocuments("/api/documents", {
    fetchImpl: async () => ({
      ok: true,
      async json() {
        return {
          documents: [
            {
              source_id: "upload-0123456789abcdefabcd",
              name: "Ryan 專案.docx",
              source_type: "word",
              chunk_count: 3,
            },
          ],
        };
      },
    }),
  });

  assert.equal(documents[0].name, "Ryan 專案.docx");
  assert.equal(documents[0].chunk_count, 3);
  assert.equal(documents[0].source_type, "word");
});

test("Word upload sends the raw DOCX and normalizes the response", async () => {
  const calls = [];
  const file = { name: "測試 文件.docx", size: 128 };
  const result = await uploadWordDocument(file, "/api/documents/upload", {
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      return {
        ok: true,
        status: 201,
        async json() {
          return {
            duplicate: false,
            document: {
              source_id: "upload-0123456789abcdefabcd",
              name: "測試 文件.docx",
              chunk_count: 2,
              url: "/api/documents/upload-0123456789abcdefabcd/download",
            },
          };
        },
      };
    },
  });

  assert.equal(calls[0].options.method, "POST");
  assert.equal(calls[0].options.body, file);
  assert.equal(calls[0].options.headers["x-file-name"], encodeURIComponent(file.name));
  assert.equal(result.document.source_type, "word");
  assert.equal(result.document.chunk_count, 2);
  assert.equal(result.duplicate, false);
});

test("Multimodal upload accepts JSON and preserves pipeline metadata", async () => {
  const calls = [];
  const file = { name: "policy.json", size: 96, type: "application/json" };
  const result = await uploadDocument(file, "/api/documents/upload", {
    folderId: "folder-123456789abc",
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      return {
        ok: true,
        status: 201,
        async json() {
          return {
            duplicate: false,
            document: {
              source_id: "upload-1123456789abcdefabcd",
              name: "policy.json",
              source_type: "json",
              file_extension: ".json",
              mime_type: "application/json",
              chunk_count: 4,
              selected_by_default: false,
              status: "ready",
              extraction: { method: "json-path-flatten" },
              pipeline: [{ stage: "embed", status: "completed" }],
            },
          };
        },
      };
    },
  });

  assert.equal(calls[0].options.headers["content-type"], "application/json");
  assert.equal(calls[0].options.headers["x-folder-id"], "folder-123456789abc");
  assert.equal(result.document.source_type, "json");
  assert.equal(result.document.extraction.method, "json-path-flatten");
  assert.equal(result.document.selected_by_default, false);
  assert.equal(result.document.pipeline[0].stage, "embed");
});

test("Multimodal upload rejects unsupported extensions before network access", async () => {
  let called = false;
  await assert.rejects(
    uploadDocument({ name: "archive.zip", size: 10 }, "/api/documents/upload", {
      fetchImpl: async () => {
        called = true;
      },
    }),
    /支援 PDF、圖片、DOCX、TXT、MD 與 JSON/,
  );
  assert.equal(called, false);
});

test("Folder APIs create, rename, move, and delete using stable IDs", async () => {
  const calls = [];
  const fetchImpl = async (url, options) => {
    calls.push({ url: String(url), options });
    const method = options.method;
    if (method === "POST") {
      return response({ folder: { id: "folder-123456789abc", name: "勞基法" } }, 201);
    }
    if (method === "PUT" && String(url).endsWith("/folder")) {
      return response({
        document: {
          source_id: "upload-0123456789abcdefabcd",
          name: "law.pdf",
          folder_id: "folder-123456789abc",
          folder_name: "勞動法規",
        },
      });
    }
    if (method === "PUT") {
      return response({ folder: { id: "folder-123456789abc", name: "勞動法規" } });
    }
    return response({ deleted: true, moved_document_count: 2 });
  };

  const created = await createDocumentFolder("勞基法", "/api/folders", { fetchImpl });
  const renamed = await renameDocumentFolder(
    created.id,
    "勞動法規",
    "/api/folders",
    { fetchImpl },
  );
  const moved = await moveDocumentToFolder(
    "upload-0123456789abcdefabcd",
    created.id,
    "/api/documents",
    { fetchImpl },
  );
  const deleted = await deleteDocumentFolder(created.id, "/api/folders", { fetchImpl });

  assert.equal(created.name, "勞基法");
  assert.equal(renamed.name, "勞動法規");
  assert.equal(moved.folder_name, "勞動法規");
  assert.equal(deleted.moved_document_count, 2);
  assert.deepEqual(calls.map((call) => call.options.method), ["POST", "PUT", "PUT", "DELETE"]);
  assert.match(calls[2].url, /\/api\/documents\/upload-0123456789abcdefabcd\/folder$/);
});

function response(body, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    async json() {
      return body;
    },
  };
}
