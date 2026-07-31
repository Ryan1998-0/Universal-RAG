const AGENT_QUERY_SCHEMA = "rag-agent-query-v2";
const AGENT_RESPONSE_SCHEMA = "rag-agent-response-v1";
const DEFAULT_ENDPOINT_KEY = "rag.agentEndpoint";
const DOCUMENT_MIME_TYPES = {
  docx: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  pdf: "application/pdf",
  txt: "text/plain",
  md: "text/markdown",
  markdown: "text/markdown",
  json: "application/json",
  png: "image/png",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  webp: "image/webp",
  tif: "image/tiff",
  tiff: "image/tiff",
  bmp: "image/bmp",
  heic: "image/heic",
};

export function buildAgentQueryPayload({
  question,
  profile = "default",
  model = { provider: "ollama", name: "qwen2.5:7b" },
  topK = null,
  conversationId = "",
  sourceIds = [],
} = {}) {
  const cleanQuestion = String(question || "").trim();
  if (!cleanQuestion) throw new Error("Agent question is required.");
  const normalizedTopK = Number(topK) > 0 ? Number(topK) : null;

  const payload = {
    schema_version: AGENT_QUERY_SCHEMA,
    profile: String(profile || "default"),
    model: normalizeModel(model),
    question: cleanQuestion,
    conversation_id: String(conversationId || ""),
    source_ids: Array.isArray(sourceIds) ? sourceIds.map(String) : [],
  };
  if (normalizedTopK) payload.top_k = normalizedTopK;
  return payload;
}

function normalizeModel(model) {
  if (!model || typeof model !== "object") {
    return { provider: "ollama", name: "qwen2.5:7b" };
  }
  return {
    provider: String(model.provider || "ollama"),
    name: String(model.name || "qwen2.5:7b"),
  };
}

export function validateAgentResponse(rawResponse) {
  if (!rawResponse || typeof rawResponse !== "object") {
    throw new Error("Agent response must be an object.");
  }
  if (rawResponse.schema_version !== AGENT_RESPONSE_SCHEMA) {
    throw new Error(`Unsupported Agent response schema: ${rawResponse.schema_version || "missing"}`);
  }
  if (!String(rawResponse.answer || "").trim()) {
    throw new Error("Agent response answer is required.");
  }

  const contexts = rawResponse.retrieval?.contexts || [];
  const contextRanks = new Set(contexts.map((context) => Number(context.rank)));
  const contextIds = new Set(contexts.map((context) => String(context.id)));
  const citations = Array.isArray(rawResponse.citations) ? rawResponse.citations : [];

  for (const citation of citations) {
    const rank = Number(citation.rank);
    const id = String(citation.id || "");
    if (!contextRanks.has(rank) && !contextIds.has(id)) {
      throw new Error(`Agent citation does not match returned contexts: ${rank || id}`);
    }
  }

  return {
    schemaVersion: rawResponse.schema_version,
    runId: String(rawResponse.run_id || ""),
    profile: rawResponse.profile || "default",
    conversationId: String(rawResponse.conversation_id || ""),
    answer: String(rawResponse.answer),
    confidence: rawResponse.confidence || "medium",
    citations,
    groundingWarnings: rawResponse.grounding_warnings || [],
    retrieval: rawResponse.retrieval || { contexts: [] },
    model: rawResponse.model || { provider: "unknown", name: "unknown" },
    timings: rawResponse.timings || {},
  };
}

export async function loadRuntimeConfig(endpoint = "/api/config", options = {}) {
  const body = await requestJson(endpoint, { method: "GET" }, options);
  const profiles = Array.isArray(body.profiles)
    ? body.profiles.map(normalizeProfileConfig).filter((profile) => profile.id)
    : [];
  const models = Array.isArray(body.models)
    ? body.models.map(normalizeRuntimeModel).filter((model) => model.id && model.name)
    : [];
  if (!profiles.length) throw new Error("Runtime config does not expose any knowledge base profile.");
  if (!models.length) throw new Error("Runtime config does not expose any model.");
  return {
    defaultProfile: String(body.default_profile || profiles[0].id),
    defaultModel: String(body.default_model || models[0].id),
    profiles,
    models,
    retrieval: {
      topK: Number(body.retrieval?.top_k || 0) || null,
      candidateK: Number(body.retrieval?.candidate_k || 0) || null,
      maxTopK: Number(body.retrieval?.max_top_k || 0) || null,
    },
  };
}

export async function loadProfileData(profile, endpoint = "/api/profiles", options = {}) {
  const cleanProfile = String(profile || "").trim();
  if (!cleanProfile) throw new Error("Knowledge base profile is required.");
  const baseEndpoint = String(endpoint).replace(/\/$/, "");
  const body = await requestJson(
    `${baseEndpoint}/${encodeURIComponent(cleanProfile)}`,
    { method: "GET" },
    options,
  );
  return {
    profile: String(body.profile || cleanProfile),
    label: String(body.label || cleanProfile),
    sampleQueries: Array.isArray(body.sample_queries) ? body.sample_queries.map(String) : [],
    meta: body.meta && typeof body.meta === "object" ? body.meta : {},
    sources: Array.isArray(body.sources) ? body.sources.map(normalizeSource) : [],
    folders: Array.isArray(body.folders) ? body.folders.map(normalizeFolder) : [],
  };
}

function normalizeProfileConfig(profile) {
  return {
    id: String(profile?.id || ""),
    label: String(profile?.label || profile?.id || ""),
    sampleQueries: Array.isArray(profile?.sample_queries) ? profile.sample_queries.map(String) : [],
  };
}

function normalizeRuntimeModel(model) {
  return {
    id: String(model?.id || ""),
    provider: String(model?.provider || ""),
    name: String(model?.name || ""),
    label: String(model?.label || model?.id || ""),
  };
}

function normalizeSource(source) {
  return {
    ...source,
    source_id: String(source?.source_id || ""),
    name: String(source?.name || source?.source_id || "Untitled"),
    source_type: String(source?.source_type || "document"),
    chunk_count: Number(source?.chunk_count || 0),
    folder_id: String(source?.folder_id || ""),
    folder_name: String(source?.folder_name || ""),
  };
}

function normalizeFolder(folder) {
  return {
    id: String(folder?.id || ""),
    name: String(folder?.name || ""),
    system: Boolean(folder?.system),
    document_count: Number(folder?.document_count || 0),
    created_at: String(folder?.created_at || ""),
    updated_at: String(folder?.updated_at || ""),
  };
}

export async function listConversations(endpoint = "/api/conversations", options = {}) {
  const body = await requestJson(endpoint, { method: "GET" }, options);
  return Array.isArray(body.conversations) ? body.conversations : [];
}

export async function createConversation(endpoint = "/api/conversations", profile = "default", options = {}) {
  return requestJson(endpoint, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ profile }),
  }, options);
}

export async function loadConversation(conversationId, endpoint = "/api/conversations", options = {}) {
  const cleanId = String(conversationId || "").trim();
  if (!cleanId) throw new Error("Conversation ID is required.");
  return requestJson(`${String(endpoint).replace(/\/$/, "")}/${encodeURIComponent(cleanId)}`, {
    method: "GET",
  }, options);
}

export async function deleteConversation(conversationId, endpoint = "/api/conversations", options = {}) {
  const cleanId = String(conversationId || "").trim();
  if (!cleanId) throw new Error("Conversation ID is required.");
  return requestJson(`${String(endpoint).replace(/\/$/, "")}/${encodeURIComponent(cleanId)}`, {
    method: "DELETE",
  }, options);
}

export async function listUploadedDocuments(endpoint = "/api/documents", options = {}) {
  const body = await requestJson(endpoint, { method: "GET" }, options);
  return Array.isArray(body.documents) ? body.documents.map(normalizeUploadedDocument) : [];
}

export async function listDocumentFolders(endpoint = "/api/folders", options = {}) {
  const body = await requestJson(endpoint, { method: "GET" }, options);
  return Array.isArray(body.folders) ? body.folders.map(normalizeFolder) : [];
}

export async function createDocumentFolder(name, endpoint = "/api/folders", options = {}) {
  const body = await requestJson(endpoint, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ name: String(name || "") }),
  }, options);
  if (!body.folder) throw new Error("Folder API response is missing folder metadata.");
  return normalizeFolder(body.folder);
}

export async function renameDocumentFolder(folderId, name, endpoint = "/api/folders", options = {}) {
  const cleanFolderId = String(folderId || "").trim();
  if (!cleanFolderId) throw new Error("Folder ID is required.");
  const body = await requestJson(
    `${String(endpoint).replace(/\/$/, "")}/${encodeURIComponent(cleanFolderId)}`,
    {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ name: String(name || "") }),
    },
    options,
  );
  if (!body.folder) throw new Error("Folder API response is missing folder metadata.");
  return normalizeFolder(body.folder);
}

export async function deleteDocumentFolder(folderId, endpoint = "/api/folders", options = {}) {
  const cleanFolderId = String(folderId || "").trim();
  if (!cleanFolderId) throw new Error("Folder ID is required.");
  return requestJson(
    `${String(endpoint).replace(/\/$/, "")}/${encodeURIComponent(cleanFolderId)}`,
    { method: "DELETE" },
    options,
  );
}

export async function moveDocumentToFolder(
  sourceId,
  folderId,
  endpoint = "/api/documents",
  options = {},
) {
  const cleanSourceId = String(sourceId || "").trim();
  if (!cleanSourceId) throw new Error("Document source ID is required.");
  const body = await requestJson(
    `${String(endpoint).replace(/\/$/, "")}/${encodeURIComponent(cleanSourceId)}/folder`,
    {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ folder_id: String(folderId || "") }),
    },
    options,
  );
  if (!body.document) throw new Error("Document API response is missing document metadata.");
  return normalizeUploadedDocument(body.document);
}

export async function uploadDocument(file, endpoint = "/api/documents/upload", options = {}) {
  const filename = String(file?.name || "").trim();
  if (!filename) throw new Error("請先選擇檔案。");
  const extension = filename.toLowerCase().split(".").pop();
  if (extension === "doc") {
    throw new Error("舊式 .doc 請先在 Word 另存為 .docx。");
  }
  if (!DOCUMENT_MIME_TYPES[extension]) {
    throw new Error("支援 PDF、圖片、DOCX、TXT、MD 與 JSON 檔案。");
  }
  const maximumBytes = ["txt", "md", "markdown", "json"].includes(extension)
    ? 10 * 1024 * 1024
    : 50 * 1024 * 1024;
  if (Number(file?.size || 0) > maximumBytes) {
    throw new Error(`檔案超過 ${maximumBytes / (1024 * 1024)} MB 上限。`);
  }

  const endpointUrl = parseEndpointUrl(endpoint);
  const fetchImpl = options.fetchImpl || fetch;
  const response = await fetchImpl(endpointUrl.href, {
    method: "POST",
    headers: {
      "content-type": String(file?.type || DOCUMENT_MIME_TYPES[extension]),
      "x-file-name": encodeURIComponent(filename),
      ...(options.folderId ? { "x-folder-id": encodeURIComponent(String(options.folderId)) } : {}),
    },
    body: file,
  });
  let body = {};
  try {
    body = await response.json();
  } catch {
    body = {};
  }
  if (!response.ok) {
    throw new Error(String(body.error || `Document upload returned HTTP ${response.status}`));
  }
  if (!body.document) throw new Error("Document upload response is missing document metadata.");
  return {
    document: normalizeUploadedDocument(body.document),
    duplicate: Boolean(body.duplicate),
  };
}

export function uploadWordDocument(file, endpoint = "/api/documents/upload", options = {}) {
  return uploadDocument(file, endpoint, options);
}

function normalizeUploadedDocument(document) {
  const name = String(document?.name || "Untitled");
  return {
    source_id: String(document?.source_id || ""),
    name,
    source_type: String(document?.source_type || sourceTypeFromFilename(name)),
    file_extension: String(document?.file_extension || ""),
    mime_type: String(document?.mime_type || "application/octet-stream"),
    sha256: String(document?.sha256 || ""),
    chunk_count: Number(document?.chunk_count || 0),
    uploaded_at: String(document?.uploaded_at || ""),
    selected_by_default: Boolean(document?.selected_by_default),
    status: String(document?.status || "ready"),
    extraction: document?.extraction && typeof document.extraction === "object"
      ? document.extraction
      : {},
    pipeline: Array.isArray(document?.pipeline) ? document.pipeline : [],
    processing_ms: Number(document?.processing_ms || 0),
    url: String(document?.url || ""),
    folder_id: String(document?.folder_id || "uncategorized"),
    folder_name: String(document?.folder_name || "未分類"),
  };
}

function sourceTypeFromFilename(filename) {
  const extension = String(filename || "").toLowerCase().split(".").pop();
  if (extension === "docx") return "word";
  if (extension === "pdf") return "pdf";
  if (["png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp", "heic"].includes(extension)) {
    return "image";
  }
  if (["md", "markdown"].includes(extension)) return "markdown";
  if (extension === "txt") return "text";
  if (extension === "json") return "json";
  return "document";
}

async function requestJson(endpoint, requestOptions, options = {}) {
  const endpointUrl = parseEndpointUrl(endpoint);
  const fetchImpl = options.fetchImpl || fetch;
  const response = await fetchImpl(endpointUrl.href, requestOptions);
  let body = {};
  try {
    body = await response.json();
  } catch {
    body = {};
  }
  if (!response.ok) {
    throw new Error(String(body.error || `Conversation API returned HTTP ${response.status}`));
  }
  return body;
}

export async function callAgentEndpoint(endpoint, payload, options = {}) {
  const cleanEndpoint = String(endpoint || "").trim();
  if (!cleanEndpoint) throw new Error("Agent endpoint is not configured.");
  const endpointUrl = parseEndpointUrl(cleanEndpoint);
  const pageProtocol = options.pageProtocol ?? globalThis.location?.protocol ?? "";
  if (pageProtocol === "https:" && endpointUrl.protocol !== "https:") {
    throw new Error("HTTPS GitHub Pages requires an HTTPS Agent endpoint.");
  }

  const fetchImpl = options.fetchImpl || fetch;
  const timeoutMs = Number(options.timeoutMs || 90000);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetchImpl(endpointUrl.href, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    if (!response.ok) {
      throw new Error(`Agent endpoint returned HTTP ${response.status}`);
    }
    const body = await response.json();
    return validateAgentResponse(body);
  } finally {
    clearTimeout(timeout);
  }
}

export async function callRetrievalRouter(endpoint, payload, options = {}) {
  const cleanEndpoint = String(endpoint || "/api/route").trim();
  const endpointUrl = parseEndpointUrl(cleanEndpoint);
  const fetchImpl = options.fetchImpl || fetch;
  const timeoutMs = Number(options.timeoutMs || 90000);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetchImpl(endpointUrl.href, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`Retrieval router returned HTTP ${response.status}`);
    const body = await response.json();
    return {
      needsRetrieval: Boolean(body.needs_retrieval),
      reason: String(body.reason || ""),
      retrievalQuery: String(body.retrieval_query || payload.question || ""),
      timingMs: Number(body.timing_ms || 0),
    };
  } finally {
    clearTimeout(timeout);
  }
}

export async function callHybridRetriever(endpoint, payload, options = {}) {
  const cleanEndpoint = String(endpoint || "/api/retrieve").trim();
  const endpointUrl = parseEndpointUrl(cleanEndpoint);
  const fetchImpl = options.fetchImpl || fetch;
  const timeoutMs = Number(options.timeoutMs || 120000);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetchImpl(endpointUrl.href, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`Hybrid retriever returned HTTP ${response.status}`);
    const body = await response.json();
    if (!Array.isArray(body.contexts)) {
      throw new Error("Hybrid retriever response is missing contexts.");
    }
    return body;
  } finally {
    clearTimeout(timeout);
  }
}

function parseEndpointUrl(endpoint) {
  try {
    return new URL(endpoint, globalThis.location?.origin || "http://127.0.0.1");
  } catch {
    throw new Error("Agent endpoint must be a valid URL.");
  }
}

export function readAgentEndpoint({
  location = globalThis.location,
  storage = globalThis.localStorage,
  defaultEndpoint = "",
} = {}) {
  const fromUrl = new URLSearchParams(location?.search || "").get("agent");
  if (fromUrl) return fromUrl.trim();
  return storage?.getItem(DEFAULT_ENDPOINT_KEY)?.trim() || defaultEndpoint;
}

export function saveAgentEndpoint(endpoint, storage = globalThis.localStorage) {
  const cleanEndpoint = String(endpoint || "").trim();
  if (!storage) return cleanEndpoint;
  if (cleanEndpoint) storage.setItem(DEFAULT_ENDPOINT_KEY, cleanEndpoint);
  else storage.removeItem(DEFAULT_ENDPOINT_KEY);
  return cleanEndpoint;
}
