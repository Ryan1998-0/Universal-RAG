import {
  ApiClientError,
  askQuestion,
  authStatus,
  buildIndex,
  createFolder,
  createKnowledgeBase,
  deleteConversation,
  deleteFolder,
  listConversations,
  listDocuments,
  listFolders,
  listKnowledgeBases,
  loadAnswerRun,
  loadConversation,
  loadRuntime,
  logout,
  moveDocument,
  renameFolder,
  uploadDocument,
} from "../api-client.js";

const state = {
  runtime: null,
  knowledgeBases: [],
  knowledgeBaseId: "",
  folders: [],
  documents: [],
  selectedSourceIds: new Set(),
  conversations: [],
  conversationId: "",
  messages: [],
  language: "zh-Hant",
  fileSearch: "",
  busy: false,
  loadId: 0,
};

const copy = {
  "zh-Hant": {
    loading: "正在載入",
    ready: "可以開始提問",
    noKnowledgeBase: "尚無知識庫，請先建立一個",
    newConversation: "新對話",
    noConversation: "尚無對話",
    uncategorized: "未分類",
    indexed: "已在索引",
    pendingIndex: "待加入索引",
    processing: "處理中",
    selected: "已選擇 {count} 份文件",
    uploadDone: "文件解析完成，勾選後按「更新知識庫」",
    indexDone: "新索引已驗證並發布",
    noReadyDocuments: "請至少勾選一份已完成解析的文件",
    noQuestion: "請先輸入問題",
    noKnowledgeForAsk: "請先建立或選擇知識庫",
    askFailed: "回答失敗",
    sourceDetails: "查看檢索過程與來源",
    noEvidence: "本次沒有使用文件檢索",
  },
};

const elements = Object.fromEntries(
  [
    "authGate",
    "appShell",
    "dataStatus",
    "logoutButton",
    "conversationRail",
    "historyToggle",
    "historyClose",
    "newChatButton",
    "conversationList",
    "knowledgePanel",
    "knowledgeToggle",
    "knowledgeClose",
    "knowledgeBaseSelect",
    "languageSelect",
    "fileSearchInput",
    "uploadFolderSelect",
    "newFolderButton",
    "renameFolderButton",
    "deleteFolderButton",
    "documentUploadStatus",
    "fileListSection",
    "documentInput",
    "uploadDocument",
    "selectAllDocuments",
    "clearDocuments",
    "documentList",
    "selectedDocCount",
    "indexSummary",
    "publishIndexButton",
    "homePanel",
    "homeHint",
    "createKnowledgeBaseButton",
    "conversationTranscript",
    "searchProgress",
    "searchProgressSteps",
    "searchForm",
    "queryInput",
    "qaModelSelect",
    "sendButton",
    "chatScroll",
    "ragDetailsDialog",
    "ragDetailsTitle",
    "ragDetailsBody",
    "closeRagDetails",
    "appToast",
  ].map((id) => [id, document.getElementById(id)]),
);

bindEvents();
init();

async function init() {
  setBusy(true);
  try {
    const session = await authStatus();
    if (!session.authenticated) {
      elements.authGate.hidden = false;
      return;
    }
    elements.appShell.hidden = false;
    [state.runtime, state.knowledgeBases, state.conversations] = await Promise.all([
      loadRuntime(),
      listKnowledgeBases(),
      listConversations(),
    ]);
    renderModels();
    renderKnowledgeBases();
    renderConversationList();
    if (state.knowledgeBases.length) {
      state.knowledgeBaseId = state.knowledgeBases[0].id;
      elements.knowledgeBaseSelect.value = state.knowledgeBaseId;
      await loadCurrentKnowledgeBase({ resetSelection: true });
    } else {
      renderEmptyKnowledgeBase();
    }
  } catch (error) {
    handleError(error);
  } finally {
    setBusy(false);
  }
}

function bindEvents() {
  elements.historyToggle.addEventListener("click", () => {
    elements.conversationRail.classList.toggle("is-collapsed");
  });
  elements.historyClose.addEventListener("click", () => {
    elements.conversationRail.classList.add("is-collapsed");
  });
  elements.knowledgeToggle.addEventListener("click", () => {
    elements.knowledgePanel.classList.toggle("is-collapsed");
  });
  elements.knowledgeClose.addEventListener("click", () => {
    elements.knowledgePanel.classList.add("is-collapsed");
  });
  elements.newChatButton.addEventListener("click", startNewConversation);
  elements.logoutButton.addEventListener("click", async () => {
    try {
      await logout();
      globalThis.location.assign("/");
    } catch (error) {
      handleError(error);
    }
  });
  elements.searchForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    await submitQuestion();
  });
  elements.queryInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      elements.searchForm.requestSubmit();
    }
  });
  elements.knowledgeBaseSelect.addEventListener("change", async () => {
    state.knowledgeBaseId = elements.knowledgeBaseSelect.value;
    startNewConversation();
    await loadCurrentKnowledgeBase({ resetSelection: true });
  });
  elements.languageSelect.addEventListener("change", () => {
    state.language = elements.languageSelect.value;
    renderStatus();
    renderDocuments();
    renderConversationList();
  });
  elements.fileSearchInput.addEventListener("input", () => {
    state.fileSearch = elements.fileSearchInput.value.trim().toLowerCase();
    renderDocuments();
  });
  elements.uploadDocument.addEventListener("click", () => elements.documentInput.click());
  elements.documentInput.addEventListener("change", async () => {
    const files = [...(elements.documentInput.files || [])];
    elements.documentInput.value = "";
    await uploadFiles(files);
  });
  for (const eventName of ["dragenter", "dragover"]) {
    elements.fileListSection.addEventListener(eventName, (event) => {
      event.preventDefault();
      elements.fileListSection.classList.add("is-dragover");
    });
  }
  elements.fileListSection.addEventListener("dragleave", () => {
    elements.fileListSection.classList.remove("is-dragover");
  });
  elements.fileListSection.addEventListener("drop", async (event) => {
    event.preventDefault();
    elements.fileListSection.classList.remove("is-dragover");
    await uploadFiles([...(event.dataTransfer?.files || [])]);
  });
  elements.selectAllDocuments.addEventListener("click", () => {
    state.documents.forEach((document) => state.selectedSourceIds.add(document.source_id));
    renderDocuments();
  });
  elements.clearDocuments.addEventListener("click", () => {
    state.selectedSourceIds.clear();
    renderDocuments();
  });
  elements.documentList.addEventListener("change", handleDocumentListChange);
  elements.publishIndexButton.addEventListener("click", publishSelectedIndex);
  elements.newFolderButton.addEventListener("click", createNewFolder);
  elements.renameFolderButton.addEventListener("click", renameSelectedFolder);
  elements.deleteFolderButton.addEventListener("click", deleteSelectedFolder);
  elements.createKnowledgeBaseButton.addEventListener("click", createNewKnowledgeBase);
  elements.conversationList.addEventListener("click", handleConversationClick);
  elements.conversationTranscript.addEventListener("click", handleTranscriptClick);
  elements.closeRagDetails.addEventListener("click", () => elements.ragDetailsDialog.close());
}

function text(key, values = {}) {
  let value = copy[state.language]?.[key] || copy["zh-Hant"][key] || key;
  for (const [name, replacement] of Object.entries(values)) {
    value = value.replace(`{${name}}`, String(replacement));
  }
  return value;
}

function renderModels() {
  elements.qaModelSelect.replaceChildren();
  for (const model of state.runtime?.allowed_models || []) {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model;
    option.selected = model === state.runtime.default_model;
    elements.qaModelSelect.append(option);
  }
}

function renderKnowledgeBases() {
  elements.knowledgeBaseSelect.replaceChildren();
  for (const knowledgeBase of state.knowledgeBases) {
    const option = document.createElement("option");
    option.value = knowledgeBase.id;
    option.textContent = knowledgeBase.name;
    elements.knowledgeBaseSelect.append(option);
  }
}

function renderEmptyKnowledgeBase() {
  state.knowledgeBaseId = "";
  state.documents = [];
  state.folders = [];
  elements.homeHint.textContent = text("noKnowledgeBase");
  elements.createKnowledgeBaseButton.hidden = false;
  renderFolders();
  renderDocuments();
  renderStatus();
}

async function loadCurrentKnowledgeBase({ resetSelection = false } = {}) {
  if (!state.knowledgeBaseId) return;
  const loadId = ++state.loadId;
  elements.dataStatus.textContent = text("loading");
  try {
    const [folders, documents] = await Promise.all([
      listFolders(state.knowledgeBaseId),
      listDocuments(state.knowledgeBaseId),
    ]);
    if (loadId !== state.loadId) return;
    state.folders = folders;
    state.documents = documents;
    if (resetSelection) {
      state.selectedSourceIds = new Set(
        documents.filter((item) => item.in_active_index).map((item) => item.source_id),
      );
    } else {
      const currentIds = new Set(documents.map((item) => item.source_id));
      state.selectedSourceIds = new Set(
        [...state.selectedSourceIds].filter((sourceId) => currentIds.has(sourceId)),
      );
    }
    renderFolders();
    renderDocuments();
    renderStatus();
    elements.createKnowledgeBaseButton.hidden = true;
    elements.homeHint.textContent = text("ready");
  } catch (error) {
    handleError(error);
  }
}

function currentKnowledgeBase() {
  return state.knowledgeBases.find((item) => item.id === state.knowledgeBaseId) || null;
}

function renderStatus() {
  const knowledgeBase = currentKnowledgeBase();
  if (!knowledgeBase) {
    elements.dataStatus.textContent = text("noKnowledgeBase");
    return;
  }
  const indexed = state.documents.filter((item) => item.in_active_index).length;
  elements.dataStatus.textContent = `${knowledgeBase.name} · ${indexed}/${state.documents.length}`;
}

function renderFolders() {
  const previous = elements.uploadFolderSelect.value;
  elements.uploadFolderSelect.replaceChildren();
  elements.uploadFolderSelect.append(new Option(text("uncategorized"), ""));
  for (const folder of state.folders) {
    elements.uploadFolderSelect.append(new Option(folder.name, folder.id));
  }
  if ([...elements.uploadFolderSelect.options].some((option) => option.value === previous)) {
    elements.uploadFolderSelect.value = previous;
  }
}

function renderDocuments() {
  elements.documentList.replaceChildren();
  const visible = state.documents.filter((document) => {
    if (!state.fileSearch) return true;
    return document.name.toLowerCase().includes(state.fileSearch);
  });
  const groups = [
    { id: "", name: text("uncategorized") },
    ...state.folders.map((folder) => ({ id: folder.id, name: folder.name })),
  ];
  for (const group of groups) {
    const groupDocuments = visible.filter((item) => (item.folder_id || "") === group.id);
    if (!groupDocuments.length) continue;
    const section = document.createElement("section");
    section.className = "document-group";
    const title = document.createElement("div");
    title.className = "document-group-title";
    title.append(document.createTextNode(group.name));
    const count = document.createElement("span");
    count.textContent = String(groupDocuments.length);
    title.append(count);
    section.append(title);
    for (const item of groupDocuments) section.append(renderDocumentRow(item));
    elements.documentList.append(section);
  }
  if (!visible.length) {
    const empty = document.createElement("p");
    empty.className = "conversation-empty";
    empty.textContent = state.documents.length ? "找不到符合的文件" : "尚未上傳文件";
    elements.documentList.append(empty);
  }
  renderSelectionSummary();
}

function renderDocumentRow(documentRecord) {
  const row = document.createElement("div");
  row.className = "production-document-row";
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.dataset.sourceId = documentRecord.source_id;
  checkbox.checked = state.selectedSourceIds.has(documentRecord.source_id);
  const label = document.createElement("label");
  label.addEventListener("click", () => checkbox.click());
  const name = document.createElement("strong");
  name.textContent = documentRecord.name;
  const status = document.createElement("small");
  status.textContent = documentRecord.in_active_index
    ? text("indexed")
    : documentRecord.status === "ready"
      ? text("pendingIndex")
      : text("processing");
  label.append(name, status);
  const folderSelect = document.createElement("select");
  folderSelect.className = "document-folder-select";
  folderSelect.dataset.documentId = documentRecord.id;
  folderSelect.append(new Option(text("uncategorized"), ""));
  for (const folder of state.folders) folderSelect.append(new Option(folder.name, folder.id));
  folderSelect.value = documentRecord.folder_id || "";
  folderSelect.disabled = !currentKnowledgeBase()?.can_write;
  row.append(checkbox, label, folderSelect);
  return row;
}

async function handleDocumentListChange(event) {
  const sourceId = event.target.dataset.sourceId;
  if (sourceId) {
    if (event.target.checked) state.selectedSourceIds.add(sourceId);
    else state.selectedSourceIds.delete(sourceId);
    renderSelectionSummary();
    return;
  }
  const documentId = event.target.dataset.documentId;
  if (!documentId) return;
  event.target.disabled = true;
  try {
    await moveDocument(state.knowledgeBaseId, documentId, event.target.value || null);
    await loadCurrentKnowledgeBase();
  } catch (error) {
    handleError(error);
    await loadCurrentKnowledgeBase();
  }
}

function renderSelectionSummary() {
  const count = state.selectedSourceIds.size;
  elements.selectedDocCount.textContent = `(${count})`;
  elements.indexSummary.textContent = text("selected", { count });
  const readySelected = selectedDocumentsForIndex();
  elements.publishIndexButton.disabled = state.busy
    || !currentKnowledgeBase()?.can_write
    || !readySelected.length;
}

function selectedDocumentsForIndex() {
  return state.documents.filter((document) => {
    const versionStatus = document.latest_version?.status || "";
    const parsed = document.status === "ready" || ["parsed", "ready"].includes(versionStatus);
    return state.selectedSourceIds.has(document.source_id) && parsed;
  });
}

async function submitQuestion() {
  const question = elements.queryInput.value.trim();
  if (!question) return showToast(text("noQuestion"), true);
  if (!state.knowledgeBaseId) return showToast(text("noKnowledgeForAsk"), true);
  if (state.busy) return;
  const indexedSources = state.documents
    .filter((document) => document.in_active_index && state.selectedSourceIds.has(document.source_id))
    .map((document) => document.source_id);
  state.messages.push({ role: "user", content: question, created_at: new Date().toISOString() });
  elements.queryInput.value = "";
  renderMessages();
  const progress = startAnswerProgress();
  setBusy(true);
  try {
    const answer = await askQuestion({
      question,
      knowledge_base_id: state.knowledgeBaseId,
      source_ids: indexedSources,
      model: elements.qaModelSelect.value || state.runtime.default_model,
      conversation_id: state.conversationId || null,
    });
    state.conversationId = answer.conversation_id;
    state.messages.push({
      role: "assistant",
      content: answer.answer,
      created_at: new Date().toISOString(),
      answer,
    });
    finishAnswerProgress(progress, answer);
    renderMessages();
    state.conversations = await listConversations();
    renderConversationList();
  } catch (error) {
    stopAnswerProgress(progress);
    handleError(error, text("askFailed"));
  } finally {
    setBusy(false);
    elements.queryInput.focus();
  }
}

function startAnswerProgress() {
  const labels = [
    "判斷是否需要檢索",
    "整理檢索問題",
    "BM25 + Embedding 混合檢索與 Rerank",
    "檢查證據並生成回答",
  ];
  elements.searchProgressSteps.replaceChildren();
  const steps = labels.map((label, index) => {
    const row = document.createElement("div");
    row.className = `progress-step ${index === 0 ? "is-active" : ""}`;
    const indicator = document.createElement("span");
    indicator.className = "progress-step-indicator";
    indicator.textContent = "✓";
    const content = document.createElement("span");
    const strong = document.createElement("strong");
    strong.textContent = label;
    content.append(strong);
    row.append(indicator, content);
    elements.searchProgressSteps.append(row);
    return row;
  });
  elements.searchProgress.hidden = false;
  const timers = steps.slice(1).map((step, index) => setTimeout(() => {
    steps[index].className = "progress-step is-done";
    step.className = "progress-step is-active";
  }, 900 * (index + 1)));
  scrollToBottom();
  return { steps, timers };
}

function finishAnswerProgress(progress, answer) {
  progress.timers.forEach(clearTimeout);
  progress.steps.forEach((step) => { step.className = "progress-step is-done"; });
  const retrievalStep = progress.steps[2].querySelector("strong");
  if (answer.retrieval?.needed === false) retrievalStep.textContent = "本次不需文件檢索";
  setTimeout(() => { elements.searchProgress.hidden = true; }, 350);
}

function stopAnswerProgress(progress) {
  progress.timers.forEach(clearTimeout);
  elements.searchProgress.hidden = true;
}

function renderMessages() {
  elements.conversationTranscript.replaceChildren();
  elements.homePanel.hidden = state.messages.length > 0;
  elements.conversationTranscript.hidden = state.messages.length === 0;
  for (const message of state.messages) {
    const article = document.createElement("article");
    const role = message.role === "assistant" ? "assistant" : "user";
    article.className = `transcript-message transcript-${role}`;
    if (role === "assistant") {
      const avatar = document.createElement("span");
      avatar.className = "transcript-avatar";
      avatar.textContent = "AI";
      const body = document.createElement("div");
      const content = document.createElement("p");
      content.className = "message-text";
      content.textContent = message.content;
      body.append(content);
      if (message.answer) body.append(renderAnswerMeta(message.answer));
      article.append(avatar, body);
    } else {
      const content = document.createElement("p");
      content.className = "message-text";
      content.textContent = message.content;
      article.append(content);
    }
    elements.conversationTranscript.append(article);
  }
  scrollToBottom();
}

function renderAnswerMeta(answer) {
  const fragment = document.createDocumentFragment();
  const citations = document.createElement("div");
  citations.className = "citation-list";
  for (const citation of answer.citations || []) {
    if (!citation.source_url) continue;
    const link = document.createElement("a");
    link.className = "citation-link";
    link.href = citation.source_url;
    link.target = "_blank";
    link.rel = "noopener";
    link.textContent = `[${citation.rank || citations.children.length + 1}] ${citation.title || citation.source || "來源"}`;
    citations.append(link);
  }
  if (citations.children.length) fragment.append(citations);
  if (answer.run_id) {
    const details = document.createElement("button");
    details.className = "details-link";
    details.type = "button";
    details.dataset.runId = answer.run_id;
    details.textContent = text("sourceDetails");
    fragment.append(details);
  }
  const meta = document.createElement("p");
  meta.className = "assistant-meta";
  const model = answer.model?.name || answer.model?.provider || "model";
  const elapsed = Number(answer.timings?.totalMs || answer.timings?.total_ms || 0);
  meta.textContent = `${model}${elapsed ? ` · ${(elapsed / 1000).toFixed(1)}s` : ""}`;
  fragment.append(meta);
  return fragment;
}

async function handleTranscriptClick(event) {
  const runId = event.target.dataset.runId;
  if (!runId) return;
  elements.ragDetailsTitle.textContent = "檢索來源";
  elements.ragDetailsBody.textContent = "正在載入...";
  elements.ragDetailsDialog.showModal();
  try {
    const run = await loadAnswerRun(runId);
    renderRunDetails(run);
  } catch (error) {
    elements.ragDetailsBody.textContent = error instanceof Error ? error.message : String(error);
  }
}

function renderRunDetails(run) {
  elements.ragDetailsTitle.textContent = run.question || "檢索來源";
  elements.ragDetailsBody.replaceChildren();
  if (!run.citations?.length) {
    elements.ragDetailsBody.textContent = text("noEvidence");
    return;
  }
  for (const citation of run.citations) {
    const item = document.createElement("article");
    item.className = "evidence-item";
    const title = document.createElement("h3");
    if (citation.available === false) {
      title.textContent = `[${citation.rank}] 來源已無法使用`;
      item.append(title);
      elements.ragDetailsBody.append(item);
      continue;
    }
    title.textContent = `[${citation.rank}] ${citation.document_name || citation.title || "來源"}${citation.page ? ` · p.${citation.page}` : ""}`;
    const content = document.createElement("p");
    content.textContent = citation.content || "";
    item.append(title, content);
    if (citation.source_url) {
      const source = document.createElement("a");
      source.className = "citation-link";
      source.href = citation.source_url;
      source.target = "_blank";
      source.rel = "noopener";
      source.textContent = "開啟原始文件";
      item.append(source);
    }
    elements.ragDetailsBody.append(item);
  }
}

async function uploadFiles(files) {
  if (!files.length || state.busy || !state.knowledgeBaseId) return;
  setBusy(true);
  const folderId = elements.uploadFolderSelect.value || null;
  try {
    for (const file of files) {
      await uploadDocument({
        knowledgeBaseId: state.knowledgeBaseId,
        file,
        folderId,
        onStage: (stage) => setUploadStatus(file.name, stage),
      });
    }
    setUploadMessage(text("uploadDone"), false);
    await loadCurrentKnowledgeBase();
  } catch (error) {
    setUploadMessage(error instanceof Error ? error.message : String(error), true);
  } finally {
    setBusy(false);
  }
}

function setUploadStatus(filename, stage) {
  const labels = {
    hashing: "計算檔案雜湊",
    reserving: "建立安全上傳工作",
    uploading: "上傳檔案",
    queuing: "排入解析佇列",
    ingesting: "病毒掃描、OCR 與結構化解析",
    ready: "解析完成",
  };
  setUploadMessage(`${filename} · ${labels[stage] || stage}`, false);
}

function setUploadMessage(message, error) {
  elements.documentUploadStatus.hidden = !message;
  elements.documentUploadStatus.textContent = message;
  elements.documentUploadStatus.className = `upload-status${error ? " is-error" : ""}`;
}

async function publishSelectedIndex() {
  const documents = selectedDocumentsForIndex();
  if (!documents.length) return showToast(text("noReadyDocuments"), true);
  setBusy(true);
  try {
    await buildIndex({
      knowledgeBaseId: state.knowledgeBaseId,
      documentIds: documents.map((document) => document.id),
      onStage: (stage) => {
        const message = stage === "building"
          ? "正在建立向量、稀疏索引並驗證內容指紋"
          : stage === "published"
            ? text("indexDone")
            : "索引工作已排入佇列";
        elements.indexSummary.textContent = message;
      },
    });
    showToast(text("indexDone"));
    await loadCurrentKnowledgeBase({ resetSelection: true });
  } catch (error) {
    handleError(error);
  } finally {
    setBusy(false);
  }
}

async function createNewKnowledgeBase() {
  const name = globalThis.prompt("知識庫名稱");
  if (!name?.trim()) return;
  setBusy(true);
  try {
    const created = await createKnowledgeBase({
      name: name.trim(),
      profile: "default",
      visibility: "private",
    });
    state.knowledgeBases = await listKnowledgeBases();
    renderKnowledgeBases();
    state.knowledgeBaseId = created.id;
    elements.knowledgeBaseSelect.value = created.id;
    await loadCurrentKnowledgeBase({ resetSelection: true });
  } catch (error) {
    handleError(error);
  } finally {
    setBusy(false);
  }
}

async function createNewFolder() {
  if (!state.knowledgeBaseId) return;
  const name = globalThis.prompt("資料夾名稱");
  if (!name?.trim()) return;
  try {
    await createFolder(state.knowledgeBaseId, name.trim());
    await loadCurrentKnowledgeBase();
  } catch (error) {
    handleError(error);
  }
}

async function renameSelectedFolder() {
  const folderId = elements.uploadFolderSelect.value;
  if (!folderId) return showToast("請先在「上傳到」選擇資料夾", true);
  const folder = state.folders.find((item) => item.id === folderId);
  const name = globalThis.prompt("新的資料夾名稱", folder?.name || "");
  if (!name?.trim()) return;
  try {
    await renameFolder(state.knowledgeBaseId, folderId, name.trim());
    await loadCurrentKnowledgeBase();
    elements.uploadFolderSelect.value = folderId;
  } catch (error) {
    handleError(error);
  }
}

async function deleteSelectedFolder() {
  const folderId = elements.uploadFolderSelect.value;
  if (!folderId) return showToast("請先在「上傳到」選擇資料夾", true);
  const folder = state.folders.find((item) => item.id === folderId);
  if (!globalThis.confirm(`刪除「${folder?.name || "資料夾"}」？文件會移到未分類。`)) return;
  try {
    await deleteFolder(state.knowledgeBaseId, folderId);
    await loadCurrentKnowledgeBase();
  } catch (error) {
    handleError(error);
  }
}

function startNewConversation() {
  state.conversationId = "";
  state.messages = [];
  elements.queryInput.value = "";
  renderMessages();
  renderConversationList();
  elements.queryInput.focus();
}

function renderConversationList() {
  elements.conversationList.replaceChildren();
  if (!state.conversations.length) {
    const empty = document.createElement("p");
    empty.className = "conversation-empty";
    empty.textContent = text("noConversation");
    elements.conversationList.append(empty);
    return;
  }
  for (const conversation of state.conversations) {
    const row = document.createElement("div");
    row.className = `conversation-row${conversation.id === state.conversationId ? " is-active" : ""}`;
    const open = document.createElement("button");
    open.className = "conversation-item";
    open.type = "button";
    open.dataset.conversationId = conversation.id;
    open.textContent = conversation.title || text("newConversation");
    const remove = document.createElement("button");
    remove.className = "conversation-delete-button";
    remove.type = "button";
    remove.dataset.deleteConversationId = conversation.id;
    remove.setAttribute("aria-label", "刪除對話");
    remove.textContent = "×";
    row.append(open, remove);
    elements.conversationList.append(row);
  }
}

async function handleConversationClick(event) {
  const deleteId = event.target.dataset.deleteConversationId;
  if (deleteId) {
    if (!globalThis.confirm("刪除這筆對話紀錄？")) return;
    try {
      await deleteConversation(deleteId);
      if (state.conversationId === deleteId) startNewConversation();
      state.conversations = await listConversations();
      renderConversationList();
    } catch (error) {
      handleError(error);
    }
    return;
  }
  const conversationId = event.target.dataset.conversationId;
  if (!conversationId) return;
  try {
    const conversation = await loadConversation(conversationId);
    if (conversation.knowledge_base_id !== state.knowledgeBaseId) {
      state.knowledgeBaseId = conversation.knowledge_base_id;
      elements.knowledgeBaseSelect.value = state.knowledgeBaseId;
      await loadCurrentKnowledgeBase({ resetSelection: true });
    }
    state.conversationId = conversation.id;
    state.messages = (conversation.messages || []).map((message) => ({
      ...message,
      answer: message.answer_run || undefined,
    }));
    renderMessages();
    renderConversationList();
    if (matchMedia("(max-width: 760px)").matches) {
      elements.conversationRail.classList.add("is-collapsed");
    }
  } catch (error) {
    handleError(error);
  }
}

function setBusy(busy) {
  state.busy = busy;
  for (const element of [
    elements.sendButton,
    elements.uploadDocument,
    elements.publishIndexButton,
    elements.knowledgeBaseSelect,
  ]) {
    element.disabled = busy;
  }
  renderSelectionSummary();
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    elements.chatScroll.scrollTop = elements.chatScroll.scrollHeight;
  });
}

let toastTimer = null;
function showToast(message, error = false) {
  clearTimeout(toastTimer);
  elements.appToast.textContent = String(message || "");
  elements.appToast.className = `app-toast${error ? " is-error" : ""}`;
  elements.appToast.hidden = false;
  toastTimer = setTimeout(() => { elements.appToast.hidden = true; }, 6000);
}

function handleError(error, prefix = "") {
  if (error instanceof ApiClientError && error.status === 401) {
    elements.appShell.hidden = true;
    elements.authGate.hidden = false;
    return;
  }
  const detail = error instanceof Error ? error.message : String(error);
  showToast(prefix ? `${prefix}：${detail}` : detail, true);
}
