export class ApiClientError extends Error {
  constructor(message, { status = 0, code = "REQUEST_FAILED", requestId = "" } = {}) {
    super(message);
    this.name = "ApiClientError";
    this.status = status;
    this.code = code;
    this.requestId = requestId;
  }
}

export async function requestJson(path, options = {}, fetchImpl = globalThis.fetch) {
  const headers = new Headers(options.headers || {});
  const response = await fetchImpl(path, {
    credentials: "same-origin",
    cache: "no-store",
    ...options,
    headers,
  });
  if (response.status === 204) return null;
  let body = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  if (!response.ok) {
    const error = body?.error || {};
    throw new ApiClientError(
      String(error.message || `請求回傳 HTTP ${response.status}`),
      {
        status: response.status,
        code: String(error.code || "REQUEST_FAILED"),
        requestId: String(body?.request_id || response.headers.get("x-request-id") || ""),
      },
    );
  }
  return body;
}

export const authStatus = (fetchImpl) => requestJson("/auth/status", {}, fetchImpl);

export const logout = (fetchImpl) => requestJson(
  "/auth/logout",
  { method: "POST", headers: { Origin: globalThis.location?.origin || "" } },
  fetchImpl,
);

export const loadRuntime = (fetchImpl) => requestJson("/v1/runtime", {}, fetchImpl);

export async function listKnowledgeBases(fetchImpl) {
  const body = await requestJson("/v1/knowledge-bases", {}, fetchImpl);
  return Array.isArray(body?.items) ? body.items : [];
}

export async function createKnowledgeBase(payload, fetchImpl) {
  return requestJson("/v1/knowledge-bases", jsonRequest("POST", payload), fetchImpl);
}

export async function listFolders(knowledgeBaseId, fetchImpl) {
  const body = await requestJson(
    `/v1/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/folders`,
    {},
    fetchImpl,
  );
  return Array.isArray(body?.items) ? body.items : [];
}

export const createFolder = (knowledgeBaseId, name, fetchImpl) => requestJson(
  `/v1/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/folders`,
  jsonRequest("POST", { name }),
  fetchImpl,
);

export const renameFolder = (knowledgeBaseId, folderId, name, fetchImpl) => requestJson(
  `/v1/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/folders/${encodeURIComponent(folderId)}`,
  jsonRequest("PUT", { name }),
  fetchImpl,
);

export const deleteFolder = (knowledgeBaseId, folderId, fetchImpl) => requestJson(
  `/v1/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/folders/${encodeURIComponent(folderId)}`,
  { method: "DELETE" },
  fetchImpl,
);

export async function listDocuments(knowledgeBaseId, fetchImpl) {
  const body = await requestJson(
    `/v1/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/documents`,
    {},
    fetchImpl,
  );
  return Array.isArray(body?.items) ? body.items : [];
}

export const moveDocument = (
  knowledgeBaseId,
  documentId,
  folderId,
  fetchImpl,
) => requestJson(
  `/v1/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/documents/${encodeURIComponent(documentId)}/folder`,
  jsonRequest("PUT", { folder_id: folderId || null }),
  fetchImpl,
);

export async function listConversations(fetchImpl) {
  const body = await requestJson("/v1/conversations", {}, fetchImpl);
  return Array.isArray(body?.items) ? body.items : [];
}

export const loadConversation = (conversationId, fetchImpl) => requestJson(
  `/v1/conversations/${encodeURIComponent(conversationId)}`,
  {},
  fetchImpl,
);

export const deleteConversation = (conversationId, fetchImpl) => requestJson(
  `/v1/conversations/${encodeURIComponent(conversationId)}`,
  { method: "DELETE" },
  fetchImpl,
);

export const askQuestion = (payload, fetchImpl) => requestJson(
  "/v1/ask",
  jsonRequest("POST", payload),
  fetchImpl,
);

export const loadAnswerRun = (runId, fetchImpl) => requestJson(
  `/v1/answer-runs/${encodeURIComponent(runId)}`,
  {},
  fetchImpl,
);

export async function uploadDocument({
  knowledgeBaseId,
  file,
  folderId = null,
  onStage = () => {},
  fetchImpl = globalThis.fetch,
}) {
  onStage("hashing");
  const sha256 = await sha256File(file);
  const idempotencyKey = `upload:${crypto.randomUUID()}`;
  onStage("reserving");
  const reservation = await requestJson(
    `/v1/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/uploads`,
    jsonRequest(
      "POST",
      {
        filename: file.name,
        content_type: normalizedContentType(file),
        size_bytes: file.size,
        sha256,
        folder_id: folderId || null,
      },
      { "Idempotency-Key": idempotencyKey },
    ),
    fetchImpl,
  );
  onStage("uploading");
  await uploadReservedContent(reservation.upload, file, fetchImpl);
  onStage("queuing");
  const completed = await requestJson(
    `/v1/uploads/${encodeURIComponent(reservation.upload_id)}/complete`,
    { method: "POST" },
    fetchImpl,
  );
  onStage("ingesting");
  const ingestion = await pollJob(
    `/v1/ingestion-jobs/${encodeURIComponent(completed.ingestion_job_id)}`,
    { success: ["succeeded"], failure: ["dead", "cancelled"] },
    fetchImpl,
  );
  onStage("ready");
  return { ...completed, ingestion };
}

export async function buildIndex({
  knowledgeBaseId,
  documentIds,
  onStage = () => {},
  fetchImpl = globalThis.fetch,
}) {
  onStage("queuing");
  const reservation = await requestJson(
    `/v1/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/index-builds`,
    jsonRequest(
      "POST",
      { document_ids: documentIds },
      { "Idempotency-Key": `index:${crypto.randomUUID()}` },
    ),
    fetchImpl,
  );
  onStage("building");
  const job = await pollJob(
    `/v1/index-build-jobs/${encodeURIComponent(reservation.job_id)}`,
    { success: ["succeeded"], failure: ["dead", "cancelled"] },
    fetchImpl,
  );
  onStage("published");
  return { ...reservation, job };
}

export async function pollJob(
  path,
  { success, failure, timeoutMs = 15 * 60 * 1000, intervalMs = 1500 },
  fetchImpl = globalThis.fetch,
) {
  const deadline = Date.now() + timeoutMs;
  const timeoutError = new ApiClientError("背景工作未在期限內完成。", {
    code: "BACKGROUND_JOB_TIMEOUT",
  });
  while (Date.now() < deadline) {
    const controller = new AbortController();
    const remainingMs = deadline - Date.now();
    let timeoutId;
    const requestTimeout = new Promise((_, reject) => {
      timeoutId = setTimeout(() => {
        reject(timeoutError);
        controller.abort();
      }, remainingMs);
    });
    let job;
    try {
      job = await Promise.race([
        requestJson(path, { signal: controller.signal }, fetchImpl),
        requestTimeout,
      ]);
    } finally {
      clearTimeout(timeoutId);
    }
    if (success.includes(job.status)) return job;
    if (failure.includes(job.status)) {
      throw new ApiClientError(
        `背景工作失敗${job.error_code ? `：${job.error_code}` : ""}`,
        { code: String(job.error_code || "BACKGROUND_JOB_FAILED") },
      );
    }
    await delay(Math.min(intervalMs, Math.max(0, deadline - Date.now())));
  }
  throw timeoutError;
}

export async function sha256File(file) {
  const payload = await file.arrayBuffer();
  const digest = await crypto.subtle.digest("SHA-256", payload);
  return [...new Uint8Array(digest)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

async function uploadReservedContent(target, file, fetchImpl) {
  const method = String(target?.method || "PUT");
  const url = String(target?.url || "");
  if (!url) throw new ApiClientError("缺少上傳目的地。");
  const sameOrigin = new URL(url, globalThis.location?.origin || "http://127.0.0.1").origin
    === (globalThis.location?.origin || "http://127.0.0.1");
  const response = await fetchImpl(url, {
    method,
    headers: target.headers || {},
    body: file,
    credentials: sameOrigin ? "same-origin" : "omit",
  });
  if (!response.ok) {
    throw new ApiClientError(`上傳回傳 HTTP ${response.status}`, {
      status: response.status,
      code: "UPLOAD_FAILED",
    });
  }
}

function normalizedContentType(file) {
  const extension = String(file.name || "").toLowerCase().split(".").pop();
  const byExtension = {
    pdf: "application/pdf",
    docx: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
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
  return byExtension[extension] || String(file.type || "application/octet-stream");
}

function jsonRequest(method, body, extraHeaders = {}) {
  return {
    method,
    headers: { "Content-Type": "application/json", ...extraHeaders },
    body: JSON.stringify(body),
  };
}

const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));
