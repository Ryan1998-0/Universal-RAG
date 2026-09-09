import {
  buildAgentQueryPayload,
  callAgentEndpoint,
  callHybridRetriever,
  callRetrievalRouter,
  createDocumentFolder,
  deleteConversation,
  deleteDocumentFolder,
  loadAccessSession,
  listConversations,
  loadSourceAccessPolicy,
  loadConversation,
  loadProfileData,
  loadRuntimeConfig,
  moveDocumentToFolder,
  readAgentEndpoint,
  renameDocumentFolder,
  saveAgentEndpoint,
  saveSourceAccessPolicy,
  switchAccessSession,
  uploadDocument,
} from "../agent-client.js?v=canonical-pipeline-3";

let knowledgeBases = {};
let qaModels = {};

const HISTORY_STORAGE_KEY = "rag-query-history-v1";
const SOURCE_SELECTION_STORAGE_KEY = "rag-selected-sources-v1";
const STATIC_SHOWCASE_RUNTIME = {
  defaultProfile: "default",
  defaultModel: "ollama:qwen2.5:7b",
  profiles: [{ id: "default", label: "泛用知識庫", sampleQueries: [] }],
  models: [{
    id: "ollama:qwen2.5:7b",
    provider: "ollama",
    name: "qwen2.5:7b",
    label: "Qwen 2.5 7B（本機）",
  }],
  retrieval: { topK: 8, candidateK: 24, maxTopK: 12 },
};

const translations = {
  "zh-Hant": {
    loadingProfile: "載入知識庫中...",
    profileLoaded: "知識庫已載入",
    profileSelected: "已選擇知識庫",
    staticBuild: "靜態展示版",
    knowledgeBase: "知識庫",
    model: "模型",
    language: "語言",
    data: "資料",
    answer: "回答",
    retrievedEvidence: "檢索證據",
    retrievedInfo: "檢索到的相關資訊",
    openTrace: "開啟本次 RAG 紀錄",
    runTrace: "執行紀錄",
    traceTitle: "本次 RAG 節點紀錄",
    send: "送出",
    documents: "文件",
    ragDocuments: "RAG 文件",
    selectAll: "全選",
    clear: "清除",
    uploadWorking: "正在偵測格式、抽取內容並建立檢索索引...",
    uploadComplete: "轉換與索引完成，請勾選要用於 RAG 的文件",
    uploadDuplicate: "這份文件已存在，請勾選後使用",
    uploadFailed: "文件上傳或轉換失敗",
    folderCreated: "資料夾已建立",
    folderRenamed: "資料夾已重新命名",
    folderDeleted: "資料夾已刪除，原有文件已移到未分類",
    folderFailed: "資料夾操作失敗",
    documentMoved: "文件分類已更新",
    uncategorized: "未分類",
    liveAgent: "本機模型",
    graph: "圖譜",
    selected: "已選",
    source: "來源",
    noEvidence: "找不到檢索證據，請換一個更明確的問題。",
    evidenceFor: "檢索問題",
    retrievalConfidence: "檢索信心",
    qaLlm: "QA 模型",
    contexts: "片段",
    totalTime: "總耗時",
    translation: "語言處理",
    aliasHits: "別名命中",
    direct: "直接檢索",
    chineseAnswer: "中文回答",
    retrievalTerms: "檢索詞",
    citation: "來源",
    liveAgentReady: "本機模型已設定",
    liveAgentOffline: "本機模型未連線",
    liveAgentPending: "本機模型正在回答",
    liveAgentError: "本機模型呼叫失敗",
    agentReadyMessage: "送出問題後會呼叫本機後端，再由目前選定的模型根據檢索資料生成回答。",
    agentOfflineMessage: "目前找不到本機服務端點；請用 rag_demo.web_app 啟動服務。",
    agentEndpointRequired: "此知識庫需要模型服務端點",
    staticUnavailable: "公開靜態頁未包含此知識庫，請設定模型服務端點後查詢。",
    agentProfile: "模型知識庫",
    matchedEntities: "命中實體",
    relationSupport: "關係支援",
    noGraphEntity: "沒有命中圖譜實體。",
    noRelationSupport: "沒有找到關係支援。",
    query: "問題",
    queryPlaceholder: "輸入問題",
    runIdEmpty: "",
  },
};

const state = {
  data: null,
  profile: "",
  qaModel: "",
  topK: null,
  candidateK: null,
  language: "zh-Hant",
  query: "",
  fileSearch: "",
  variant: "bm25_dense",
  agentEndpoint: "/api/ask",
  latestResult: null,
  selectedSourceIds: new Set(),
  latestTraceUrl: "",
  latestRunId: "",
  agentRequestId: 0,
  pendingProfile: "",
  history: readConversationHistory(),
  conversationId: "",
  messages: [],
  latestRouteDecision: null,
  workflowRequestId: 0,
  uploadingDocuments: false,
  folders: [],
  uploadFolderId: "uncategorized",
  collapsedDocumentGroups: new Set(),
  folderDialogMode: "create",
  editingFolderId: "",
  staticShowcase: false,
  access: { principal: { id: "", label: "", roles: [] }, principals: [], roles: [], canManage: false },
  editingAccessSourceId: "",
};

const elements = {
  dataStatus: document.querySelector("#dataStatus"),
  dataMeta: document.querySelector("#dataMeta"),
  form: document.querySelector("#searchForm"),
  input: document.querySelector("#queryInput"),
  knowledgeBase: document.querySelector("#knowledgeBaseSelect"),
  qaModel: document.querySelector("#qaModelSelect"),
  language: document.querySelector("#languageSelect"),
  chips: document.querySelector("#questionChips"),
  metrics: document.querySelector("#metrics"),
  variantDetails: document.querySelector("#variantDetails"),
  results: document.querySelector("#results"),
  resultTitle: document.querySelector("#resultTitle"),
  confidence: document.querySelector("#confidenceBadge"),
  agentEndpoint: document.querySelector("#agentEndpointInput"),
  saveAgentEndpoint: document.querySelector("#saveAgentEndpoint"),
  askAgent: document.querySelector("#askAgentButton"),
  agentAnswer: document.querySelector("#agentAnswerPanel"),
  answer: document.querySelector("#answerPanel"),
  comparison: document.querySelector("#comparisonPanel"),
  graphPanel: document.querySelector("#graphPanel"),
  pipeline: document.querySelector("#pipeline"),
  documentList: document.querySelector("#documentList"),
  selectedDocCount: document.querySelector("#selectedDocCount"),
  selectAllDocuments: document.querySelector("#selectAllDocuments"),
  clearDocuments: document.querySelector("#clearDocuments"),
  traceDocumentLink: document.querySelector("#traceDocumentLink"),
  traceSummary: document.querySelector("#traceSummary"),
  runId: document.querySelector("#runId"),
  statusMessage: document.querySelector("#statusMessage"),
  answerMessage: document.querySelector("#answerMessage"),
  evidenceMessage: document.querySelector("#evidenceMessage"),
  ragTrace: document.querySelector("#ragTrace"),
  homePanel: document.querySelector("#homePanel"),
  conversationRail: document.querySelector("#conversationRail"),
  knowledgePanel: document.querySelector("#knowledgePanel"),
  historyToggle: document.querySelector("#historyToggle"),
  historyClose: document.querySelector("#historyClose"),
  newChat: document.querySelector("#newChatButton"),
  knowledgeToggle: document.querySelector("#knowledgeToggle"),
  knowledgeClose: document.querySelector("#knowledgeClose"),
  conversationList: document.querySelector("#conversationList"),
  conversationTranscript: document.querySelector("#conversationTranscript"),
  chatScroll: document.querySelector(".chat-scroll"),
  fileSearch: document.querySelector("#fileSearchInput"),
  demoCta: document.querySelector("#demoCta"),
  searchProgress: document.querySelector("#searchProgress"),
  searchProgressSteps: document.querySelector("#searchProgressSteps"),
  userMessage: document.querySelector("#userMessage"),
  userQuestion: document.querySelector("#userQuestion"),
  ragDetailsGuide: document.querySelector("#ragDetailsGuide"),
  ragDetailsToggle: document.querySelector("#ragDetailsToggle"),
  documentInput: document.querySelector("#documentInput"),
  uploadDocument: document.querySelector("#uploadDocument"),
  documentUploadStatus: document.querySelector("#documentUploadStatus"),
  fileListSection: document.querySelector("#fileListSection"),
  uploadFolder: document.querySelector("#uploadFolderSelect"),
  newFolder: document.querySelector("#newFolderButton"),
  renameFolder: document.querySelector("#renameFolderButton"),
  deleteFolder: document.querySelector("#deleteFolderButton"),
  folderDialog: document.querySelector("#folderDialog"),
  folderDialogTitle: document.querySelector("#folderDialogTitle"),
  folderNameInput: document.querySelector("#folderNameInput"),
  folderDialogError: document.querySelector("#folderDialogError"),
  saveFolder: document.querySelector("#saveFolderButton"),
  deleteFolderDialog: document.querySelector("#deleteFolderDialog"),
  deleteFolderMessage: document.querySelector("#deleteFolderMessage"),
  confirmDeleteFolder: document.querySelector("#confirmDeleteFolderButton"),
  accessProfile: document.querySelector("#accessProfileSelect"),
  accessNotice: document.querySelector("#accessNotice"),
  accessPolicyDialog: document.querySelector("#accessPolicyDialog"),
  accessPolicyDocument: document.querySelector("#accessPolicyDocument"),
  accessRoleList: document.querySelector("#accessRoleList"),
  accessPolicyError: document.querySelector("#accessPolicyError"),
  saveAccessPolicy: document.querySelector("#saveAccessPolicyButton"),
  deleteConversationDialog: document.querySelector("#deleteConversationDialog"),
  deleteConversationMessage: document.querySelector("#deleteConversationMessage"),
};

let responsiveLayoutMode = "";

init();

async function init() {
  applyResponsivePanelDefaults();
  elements.input.disabled = true;
  elements.askAgent.disabled = true;
  elements.newChat.disabled = true;
  bindEvents();
  try {
    let runtime;
    try {
      runtime = await loadRuntimeConfig();
    } catch (error) {
      if (!isStaticShowcaseHost()) throw error;
      state.staticShowcase = true;
      runtime = STATIC_SHOWCASE_RUNTIME;
    }
    configureRuntime(runtime);
    elements.knowledgeBase.value = state.profile;
    elements.qaModel.value = state.qaModel;
    elements.language.value = state.language;
    try {
      configureAccessSession(await loadAccessSession());
    } catch (error) {
      if (!state.staticShowcase) throw error;
      elements.accessProfile.closest("label").hidden = true;
    }
    applyLanguage();
    state.agentEndpoint = readAgentEndpoint({ defaultEndpoint: "/api/ask" });
    elements.agentEndpoint.value = state.agentEndpoint;
    await loadKnowledgeBase(state.profile);
    await refreshConversationHistory();
    startControlWatcher();
  } finally {
    elements.input.disabled = false;
    elements.askAgent.disabled = false;
    elements.newChat.disabled = false;
  }
}

function applyResponsivePanelDefaults() {
  const nextMode = window.matchMedia("(max-width: 899px)").matches
    ? "narrow"
    : window.matchMedia("(max-width: 1439px)").matches
      ? "compact"
      : "wide";
  elements.conversationRail.classList.toggle("is-collapsed", nextMode === "narrow");
  elements.knowledgePanel.classList.toggle("is-collapsed", nextMode !== "wide");
  responsiveLayoutMode = nextMode;
  syncPanelToggleState();
}

function syncPanelToggleState() {
  const historyExpanded = !elements.conversationRail.classList.contains("is-collapsed");
  const knowledgeExpanded = !elements.knowledgePanel.classList.contains("is-collapsed");
  elements.historyToggle.setAttribute("aria-expanded", String(historyExpanded));
  elements.historyToggle.setAttribute("aria-label", historyExpanded ? "收合對話紀錄" : "展開對話紀錄");
  elements.knowledgeToggle.setAttribute("aria-expanded", String(knowledgeExpanded));
  elements.knowledgeToggle.setAttribute("aria-label", knowledgeExpanded ? "收合文件選擇" : "展開文件選擇");
}

function configureRuntime(runtime) {
  knowledgeBases = Object.fromEntries(
    runtime.profiles.map((profile) => [
      profile.id,
      {
        label: profile.label,
        profile: profile.id,
        defaultQuery: profile.sampleQueries[0] || "",
        queries: profile.sampleQueries,
      },
    ]),
  );
  qaModels = Object.fromEntries(
    runtime.models.map((model) => [
      model.id,
      {
        provider: model.provider,
        name: model.name,
        label: model.label,
      },
    ]),
  );
  state.profile = knowledgeBases[runtime.defaultProfile]
    ? runtime.defaultProfile
    : Object.keys(knowledgeBases)[0];
  state.qaModel = qaModels[runtime.defaultModel]
    ? runtime.defaultModel
    : Object.keys(qaModels)[0];
  state.topK = runtime.retrieval.topK;
  state.candidateK = runtime.retrieval.candidateK;

  elements.knowledgeBase.replaceChildren(
    ...Object.values(knowledgeBases).map((profile) => {
      const option = document.createElement("option");
      option.value = profile.profile;
      option.textContent = profile.label;
      return option;
    }),
  );
  elements.qaModel.replaceChildren(
    ...Object.entries(qaModels).map(([id, model]) => {
      const option = document.createElement("option");
      option.value = id;
      option.textContent = model.label;
      return option;
    }),
  );
}

function bindEvents() {
  window.addEventListener("resize", () => {
    const nextMode = window.matchMedia("(max-width: 899px)").matches
      ? "narrow"
      : window.matchMedia("(max-width: 1439px)").matches
        ? "compact"
        : "wide";
    if (nextMode !== responsiveLayoutMode) applyResponsivePanelDefaults();
  });
  elements.form.addEventListener("submit", (event) => {
    event.preventDefault();
    handleAdaptiveQuestion(elements.input.value);
  });

  const handleKnowledgeBaseChange = async () => {
    await loadKnowledgeBase(elements.knowledgeBase.value);
  };
  elements.knowledgeBase.addEventListener("change", handleKnowledgeBaseChange);
  elements.knowledgeBase.addEventListener("input", handleKnowledgeBaseChange);

  const handleQaModelChange = () => {
    state.qaModel = elements.qaModel.value;
  };
  elements.qaModel.addEventListener("change", handleQaModelChange);
  elements.qaModel.addEventListener("input", handleQaModelChange);

  elements.language.addEventListener("change", () => {
    state.language = elements.language.value;
    applyLanguage();
    renderDocumentSelector();
    renderChips();
    renderConversationHistory();
    if (state.latestResult) runSearch();
  });

  elements.fileSearch.addEventListener("input", () => {
    state.fileSearch = elements.fileSearch.value.trim().toLowerCase();
    renderDocumentSelector();
  });

  elements.uploadFolder.addEventListener("change", () => {
    state.uploadFolderId = elements.uploadFolder.value || "uncategorized";
    renderFolderControls();
  });
  elements.newFolder.addEventListener("click", () => openFolderDialog("create"));
  elements.renameFolder.addEventListener("click", () => openFolderDialog("rename"));
  elements.deleteFolder.addEventListener("click", openDeleteFolderDialog);
  elements.saveFolder.addEventListener("click", saveFolderFromDialog);
  elements.folderNameInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      saveFolderFromDialog();
    }
  });
  elements.confirmDeleteFolder.addEventListener("click", deleteSelectedFolder);
  elements.accessProfile.addEventListener("change", async () => {
    const principalId = elements.accessProfile.value;
    if (!principalId || principalId === state.access.principal.id) return;
    elements.accessProfile.disabled = true;
    try {
      configureAccessSession(await switchAccessSession(principalId));
      await loadKnowledgeBase(state.profile);
      setDocumentUploadStatus(`已切換為「${state.access.principal.label}」，只顯示可查詢文件。`, "success");
    } catch (error) {
      elements.accessProfile.value = state.access.principal.id;
      setDocumentUploadStatus(`身分切換失敗：${error instanceof Error ? error.message : String(error)}`, "error");
    } finally {
      elements.accessProfile.disabled = false;
    }
  });
  elements.saveAccessPolicy.addEventListener("click", saveAccessPolicyFromDialog);

  elements.uploadDocument.addEventListener("click", () => {
    if (!state.uploadingDocuments) elements.documentInput.click();
  });
  elements.documentInput.addEventListener("change", async () => {
    const files = [...(elements.documentInput.files || [])];
    elements.documentInput.value = "";
    await handleDocumentFiles(files);
  });
  for (const eventName of ["dragenter", "dragover"]) {
    elements.fileListSection.addEventListener(eventName, (event) => {
      event.preventDefault();
      if (!state.uploadingDocuments) elements.fileListSection.classList.add("is-dragover");
    });
  }
  elements.fileListSection.addEventListener("dragleave", (event) => {
    if (!elements.fileListSection.contains(event.relatedTarget)) {
      elements.fileListSection.classList.remove("is-dragover");
    }
  });
  elements.fileListSection.addEventListener("drop", async (event) => {
    event.preventDefault();
    elements.fileListSection.classList.remove("is-dragover");
    await handleDocumentFiles([...(event.dataTransfer?.files || [])]);
  });

  const toggleRail = () => {
    const willOpen = elements.conversationRail.classList.contains("is-collapsed");
    elements.conversationRail.classList.toggle("is-collapsed");
    if (willOpen && window.matchMedia("(max-width: 899px)").matches) {
      elements.knowledgePanel.classList.add("is-collapsed");
    }
    syncPanelToggleState();
  };
  elements.historyToggle.addEventListener("click", toggleRail);
  elements.historyClose.addEventListener("click", () => {
    elements.conversationRail.classList.add("is-collapsed");
    syncPanelToggleState();
  });
  elements.newChat.addEventListener("click", () => startNewConversation());
  elements.ragDetailsToggle.addEventListener("click", toggleRagDetails);

  const toggleKnowledgePanel = () => {
    const willOpen = elements.knowledgePanel.classList.contains("is-collapsed");
    elements.knowledgePanel.classList.toggle("is-collapsed");
    if (willOpen && window.matchMedia("(max-width: 899px)").matches) {
      elements.conversationRail.classList.add("is-collapsed");
    }
    syncPanelToggleState();
  };
  elements.knowledgeToggle.addEventListener("click", toggleKnowledgePanel);
  elements.knowledgeClose.addEventListener("click", () => {
    elements.knowledgePanel.classList.add("is-collapsed");
    syncPanelToggleState();
  });
  window.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    let changed = false;
    if (!elements.knowledgePanel.classList.contains("is-collapsed")) {
      elements.knowledgePanel.classList.add("is-collapsed");
      changed = true;
    }
    if (
      window.matchMedia("(max-width: 899px)").matches
      && !elements.conversationRail.classList.contains("is-collapsed")
    ) {
      elements.conversationRail.classList.add("is-collapsed");
      changed = true;
    }
    if (changed) syncPanelToggleState();
  });

  elements.demoCta.addEventListener("click", () => {
    const query = currentKnowledgeBase().defaultQuery;
    if (!query) {
      elements.input.focus();
      return;
    }
    elements.input.value = query;
    handleAdaptiveQuestion(query);
  });

  elements.saveAgentEndpoint.addEventListener("click", () => {
    state.agentEndpoint = saveAgentEndpoint(elements.agentEndpoint.value);
    if (state.latestResult) renderAgentIdle();
  });

  elements.askAgent.addEventListener("click", () => {
    askLiveAgent(state.latestResult, { retrievalDecision: state.latestRouteDecision });
  });

  elements.selectAllDocuments.addEventListener("click", () => {
    selectAllSources();
    renderDocumentSelector();
    if (state.query && state.latestRouteDecision?.needsRetrieval) rerunCurrentRetrieval();
  });

  elements.clearDocuments.addEventListener("click", () => {
    state.selectedSourceIds = new Set();
    saveSelectedSourceIds();
    renderDocumentSelector();
    if (state.query && state.latestRouteDecision?.needsRetrieval) rerunCurrentRetrieval();
  });
}

function t(key) {
  return translations[state.language]?.[key] || translations["zh-Hant"][key] || key;
}

function applyLanguage() {
  document.documentElement.lang = state.language;
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    const key = node.getAttribute("data-i18n");
    node.textContent = t(key);
  });
  elements.input.placeholder = t("queryPlaceholder");
  elements.traceDocumentLink.textContent = t("openTrace");
  elements.dataStatus.textContent = state.data
    ? `${currentKnowledgeBase().label} ${t("profileLoaded")}`
    : t("loadingProfile");
  renderFolderControls();
}

function renderChips() {
  elements.chips.innerHTML = "";
  const chipLabels = (currentKnowledgeBase().queries || [])
    .slice(0, 4)
    .map((query) => [query, query]);
  elements.chips.hidden = chipLabels.length === 0;
  elements.demoCta.hidden = !currentKnowledgeBase().defaultQuery;
  for (const [label, query] of chipLabels) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.addEventListener("click", () => {
      elements.input.value = query;
      handleAdaptiveQuestion(query);
    });
    elements.chips.append(button);
  }
}

function folderList() {
  if (!state.folders.length) {
    state.folders = [{ id: "uncategorized", name: t("uncategorized"), system: true }];
  }
  return state.folders;
}

function currentUploadFolder() {
  return folderList().find((folder) => folder.id === state.uploadFolderId) || folderList()[0];
}

function renderFolderControls() {
  const counts = new Map(folderList().map((folder) => [folder.id, 0]));
  for (const source of state.data?.sources || []) {
    if (!source.folder_id) continue;
    const folderId = counts.has(source.folder_id) ? source.folder_id : "uncategorized";
    counts.set(folderId, (counts.get(folderId) || 0) + 1);
  }
  if (!folderList().some((folder) => folder.id === state.uploadFolderId)) {
    state.uploadFolderId = "uncategorized";
  }
  elements.uploadFolder.replaceChildren(
    ...folderList().map((folder) => {
      const option = document.createElement("option");
      option.value = folder.id;
      const label = folder.id === "uncategorized" ? t("uncategorized") : folder.name;
      option.textContent = `${label} (${counts.get(folder.id) || 0})`;
      return option;
    }),
  );
  elements.uploadFolder.value = state.uploadFolderId;
  const selectedFolder = currentUploadFolder();
  const protectedFolder = Boolean(selectedFolder.system);
  const canManage = Boolean(state.access.canManage);
  elements.uploadFolder.disabled = !canManage || state.uploadingDocuments;
  elements.uploadDocument.disabled = !canManage || state.uploadingDocuments;
  elements.newFolder.disabled = !canManage;
  elements.renameFolder.disabled = !canManage || protectedFolder;
  elements.deleteFolder.disabled = !canManage || protectedFolder;
}

function openFolderDialog(mode) {
  const folder = currentUploadFolder();
  if (mode === "rename" && folder.system) return;
  state.folderDialogMode = mode;
  state.editingFolderId = mode === "rename" ? folder.id : "";
  elements.folderDialogTitle.textContent = mode === "rename"
    ? "重新命名資料夾"
    : "新增資料夾";
  elements.folderNameInput.value = mode === "rename" ? folder.name : "";
  elements.folderDialogError.textContent = "";
  elements.folderDialogError.hidden = true;
  elements.folderDialog.showModal();
  elements.folderNameInput.focus();
}

async function saveFolderFromDialog() {
  const name = elements.folderNameInput.value.trim();
  elements.saveFolder.disabled = true;
  elements.folderDialogError.hidden = true;
  try {
    if (state.folderDialogMode === "rename") {
      const folder = await renameDocumentFolder(state.editingFolderId, name);
      state.folders = state.folders.map((item) => item.id === folder.id ? folder : item);
      for (const source of state.data?.sources || []) {
        if (source.folder_id === folder.id) source.folder_name = folder.name;
      }
      setDocumentUploadStatus(`${t("folderRenamed")}：${folder.name}`, "success");
    } else {
      const folder = await createDocumentFolder(name);
      state.folders.push(folder);
      state.uploadFolderId = folder.id;
      setDocumentUploadStatus(`${t("folderCreated")}：${folder.name}`, "success");
    }
    elements.folderDialog.close();
    renderFolderControls();
    renderDocumentSelector();
  } catch (error) {
    elements.folderDialogError.textContent = error instanceof Error ? error.message : String(error);
    elements.folderDialogError.hidden = false;
  } finally {
    elements.saveFolder.disabled = false;
  }
}

function openDeleteFolderDialog() {
  const folder = currentUploadFolder();
  if (folder.system) return;
  state.editingFolderId = folder.id;
  elements.deleteFolderMessage.textContent = `刪除「${folder.name}」？`;
  elements.deleteFolderDialog.showModal();
}

async function deleteSelectedFolder() {
  const folderId = state.editingFolderId;
  const folder = state.folders.find((item) => item.id === folderId);
  if (!folder || folder.system) return;
  elements.confirmDeleteFolder.disabled = true;
  try {
    const result = await deleteDocumentFolder(folderId);
    state.folders = state.folders.filter((item) => item.id !== folderId);
    for (const source of state.data?.sources || []) {
      if (source.folder_id !== folderId) continue;
      source.folder_id = "uncategorized";
      source.folder_name = t("uncategorized");
    }
    state.uploadFolderId = "uncategorized";
    elements.deleteFolderDialog.close();
    renderFolderControls();
    renderDocumentSelector();
    setDocumentUploadStatus(
      `${t("folderDeleted")} (${Number(result.moved_document_count || 0)})`,
      "success",
    );
  } catch (error) {
    elements.deleteFolderDialog.close();
    setDocumentUploadStatus(
      `${t("folderFailed")}：${error instanceof Error ? error.message : String(error)}`,
      "error",
    );
  } finally {
    elements.confirmDeleteFolder.disabled = false;
  }
}

function selectAllSources() {
  state.selectedSourceIds = new Set((state.data?.sources || []).map((source) => source.source_id));
  saveSelectedSourceIds();
}

function selectDefaultSources() {
  const availableSourceIds = new Set(
    (state.data?.sources || []).map((source) => String(source.source_id || "")).filter(Boolean),
  );
  const savedSourceIds = readSelectedSourceIds();
  const defaults = (state.data?.sources || [])
    .filter((source) => Boolean(source.selected_by_default))
    .map((source) => source.source_id);
  state.selectedSourceIds = new Set(
    (savedSourceIds === null ? defaults : savedSourceIds)
      .filter((sourceId) => availableSourceIds.has(sourceId)),
  );
}

function sourceSelectionStorageKey() {
  return `${SOURCE_SELECTION_STORAGE_KEY}:${state.profile || "default"}`;
}

function readSelectedSourceIds() {
  try {
    const raw = localStorage.getItem(sourceSelectionStorageKey());
    if (raw === null) return null;
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.map(String) : null;
  } catch {
    return null;
  }
}

function saveSelectedSourceIds() {
  try {
    localStorage.setItem(
      sourceSelectionStorageKey(),
      JSON.stringify([...state.selectedSourceIds]),
    );
  } catch {
    // Selection remains usable for the current page when storage is unavailable.
  }
}

function sourceStats() {
  const counts = new Map();
  for (const chunk of state.data?.chunks || []) {
    const sourceId = chunk.source_id || chunk.source || "";
    counts.set(sourceId, (counts.get(sourceId) || 0) + 1);
  }
  for (const source of state.data?.sources || []) {
    if (!counts.has(source.source_id) && Number(source.chunk_count || 0) > 0) {
      counts.set(source.source_id, Number(source.chunk_count));
    }
  }
  return counts;
}

function renderDocumentSelector() {
  elements.documentList.innerHTML = "";
  const sources = state.data?.sources || [];
  const counts = sourceStats();
  const visibleSources = filterSources(sources);

  if (!sources.length) {
    elements.selectedDocCount.textContent = "(0)";
    elements.documentList.innerHTML =
      `<div class="empty compact">${escapeHtml(t("staticUnavailable"))}</div>`;
    return;
  }

  if (!visibleSources.length) {
    elements.selectedDocCount.textContent = `(${state.selectedSourceIds.size})`;
    elements.documentList.innerHTML = `<div class="empty compact">找不到符合搜尋的文件。</div>`;
    return;
  }

  for (const group of groupSources(visibleSources)) {
    const selectedInGroup = group.sources.filter((source) => state.selectedSourceIds.has(source.source_id));
    const collapsed = state.collapsedDocumentGroups.has(group.key);
    const groupCard = document.createElement("section");
    groupCard.className = `doc-group${collapsed ? " is-collapsed" : ""}`;
    groupCard.innerHTML = `
      <div class="doc-group-head">
        <button class="doc-chevron" type="button" aria-label="${collapsed ? "展開" : "收合"}${escapeHtml(group.title)}" aria-expanded="${collapsed ? "false" : "true"}">⌄</button>
        <input type="checkbox" aria-label="${escapeHtml(group.title)}" ${selectedInGroup.length === group.sources.length ? "checked" : ""} />
        <strong>${escapeHtml(group.title)}</strong>
        <span class="count-pill">${selectedInGroup.length}/${group.sources.length}</span>
      </div>
      <div class="doc-group-items" ${collapsed ? "hidden" : ""}></div>
    `;

    groupCard.querySelector(".doc-chevron").addEventListener("click", () => {
      if (state.collapsedDocumentGroups.has(group.key)) {
        state.collapsedDocumentGroups.delete(group.key);
      } else {
        state.collapsedDocumentGroups.add(group.key);
      }
      renderDocumentSelector();
    });

    groupCard.querySelector(".doc-group-head input").addEventListener("change", (event) => {
      for (const source of group.sources) {
        if (event.currentTarget.checked) state.selectedSourceIds.add(source.source_id);
        else state.selectedSourceIds.delete(source.source_id);
      }
      saveSelectedSourceIds();
      renderDocumentSelector();
      if (state.query && state.latestRouteDecision?.needsRetrieval) rerunCurrentRetrieval();
    });

    const items = groupCard.querySelector(".doc-group-items");
    for (const source of group.sources) {
      const checked = state.selectedSourceIds.has(source.source_id);
      const row = document.createElement("div");
      row.className = "document-item";
      const typeClass = source.source_type || "pdf";
      const checkboxId = `source-${source.source_id}`;
      row.innerHTML = `
        <input id="${escapeHtml(checkboxId)}" type="checkbox" value="${escapeHtml(source.source_id)}" ${checked ? "checked" : ""} />
        <span class="file-type ${escapeHtml(typeClass)}">${escapeHtml(fileTypeLabel(source))}</span>
        <span class="document-item-body">
          <label for="${escapeHtml(checkboxId)}"><strong title="${escapeHtml(source.name)}">${escapeHtml(source.name)}</strong></label>
          <small>${sourceStatusLabel(source, counts.get(source.source_id) || 0)}</small>
        </span>
      `;
      if (state.access.canManage) {
        const accessButton = document.createElement("button");
        accessButton.type = "button";
        accessButton.className = "document-access-button";
        accessButton.textContent = "權限";
        accessButton.title = `設定 ${source.name} 的查詢權限`;
        accessButton.addEventListener("click", () => openAccessPolicyDialog(source));
        row.append(accessButton);
      }
      row.querySelector("input").addEventListener("change", (event) => {
        const sourceId = event.currentTarget.value;
        if (event.currentTarget.checked) state.selectedSourceIds.add(sourceId);
        else state.selectedSourceIds.delete(sourceId);
        saveSelectedSourceIds();
        renderDocumentSelector();
        if (state.query && state.latestRouteDecision?.needsRetrieval) rerunCurrentRetrieval();
      });
      if (source.folder_id) {
        const picker = document.createElement("select");
        picker.className = "document-folder-select";
        picker.setAttribute("aria-label", `${source.name} 的資料夾`);
        picker.replaceChildren(
          ...folderList().map((folder) => {
            const option = document.createElement("option");
            option.value = folder.id;
            option.textContent = folder.id === "uncategorized" ? t("uncategorized") : folder.name;
            return option;
          }),
        );
        picker.value = folderList().some((folder) => folder.id === source.folder_id)
          ? source.folder_id
          : "uncategorized";
        picker.disabled = !state.access.canManage;
        picker.addEventListener("change", (event) => {
          handleDocumentFolderChange(source.source_id, event.currentTarget.value, event.currentTarget);
        });
        row.querySelector(".document-item-body").append(picker);
      }
      items.append(row);
    }
    elements.documentList.append(groupCard);
  }

  elements.selectedDocCount.textContent = `(${state.selectedSourceIds.size})`;
}

function configureAccessSession(session) {
  state.access = session || state.access;
  const principal = state.access.principal || {};
  elements.accessProfile.replaceChildren(
    ...(state.access.principals || []).map((item) => {
      const option = document.createElement("option");
      option.value = item.id;
      option.textContent = item.label;
      return option;
    }),
  );
  elements.accessProfile.value = principal.id || "";
  const roleLabels = (principal.roles || []).map((roleId) => (
    (state.access.roles || []).find((role) => role.id === roleId)?.label || roleId
  ));
  elements.accessNotice.textContent = state.access.notice || "";
  elements.accessNotice.hidden = !state.access.notice;
  elements.accessProfile.closest("label").hidden = !(state.access.principals || []).length;
  elements.accessProfile.title = roleLabels.length ? `角色：${roleLabels.join("、")}` : "";
}

async function openAccessPolicyDialog(source) {
  if (!state.access.canManage || !source?.source_id) return;
  state.editingAccessSourceId = source.source_id;
  elements.accessPolicyDocument.textContent = source.name || source.source_id;
  elements.accessPolicyError.hidden = true;
  elements.accessPolicyError.textContent = "";
  elements.accessRoleList.replaceChildren();
  try {
    const policy = await loadSourceAccessPolicy(source.source_id);
    const choices = [
      { id: "all", label: "全員公開", description: "任何已設定的查詢身分都能檢索" },
      ...(state.access.roles || []).filter((role) => role.id !== "admin"),
    ];
    for (const choice of choices) {
      const label = document.createElement("label");
      label.className = "access-role-option";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.value = choice.id;
      checkbox.checked = policy.allowedRoles.includes(choice.id);
      const copy = document.createElement("span");
      copy.innerHTML = `<strong>${escapeHtml(choice.label)}</strong><small>${escapeHtml(choice.description || "")}</small>`;
      label.append(checkbox, copy);
      elements.accessRoleList.append(label);
    }
    elements.accessPolicyDialog.showModal();
  } catch (error) {
    setDocumentUploadStatus(`讀取權限設定失敗：${error instanceof Error ? error.message : String(error)}`, "error");
  }
}

async function saveAccessPolicyFromDialog() {
  const sourceId = state.editingAccessSourceId;
  const allowedRoles = [...elements.accessRoleList.querySelectorAll('input[type="checkbox"]:checked')]
    .map((input) => input.value);
  if (!sourceId || !allowedRoles.length) {
    elements.accessPolicyError.textContent = "至少選擇一個可查詢角色。";
    elements.accessPolicyError.hidden = false;
    return;
  }
  elements.saveAccessPolicy.disabled = true;
  try {
    await saveSourceAccessPolicy(sourceId, allowedRoles);
    elements.accessPolicyDialog.close();
    setDocumentUploadStatus("文件查詢權限已儲存；下一次檢索會先套用此限制。", "success");
  } catch (error) {
    elements.accessPolicyError.textContent = error instanceof Error ? error.message : String(error);
    elements.accessPolicyError.hidden = false;
  } finally {
    elements.saveAccessPolicy.disabled = false;
  }
}

function sourceStatusLabel(source, chunkCount) {
  const extractionMethod = String(source.extraction?.method || "").trim();
  const status = source.selected_by_default ? "內建" : "已轉換";
  return [status, `${chunkCount} 個片段`, extractionMethod].filter(Boolean).join(" · ");
}

function filterSources(sources) {
  if (!state.fileSearch) return sources;
  return sources.filter((source) =>
    `${source.name} ${source.source_type || ""} ${source.folder_name || ""}`.toLowerCase().includes(state.fileSearch),
  );
}

function groupSources(sources) {
  const groups = new Map();
  for (const source of sources) {
    const isFolder = Boolean(source.folder_id);
    const folder = isFolder
      ? folderList().find((item) => item.id === source.folder_id) || folderList()[0]
      : null;
    const rawKey = isFolder ? folder.id : String(source.group || source.source_type || "other");
    const key = `${isFolder ? "folder" : "builtin"}:${rawKey}`;
    if (!groups.has(key)) {
      const fallbackTitle = rawKey
        .replace(/[_-]+/g, " ")
        .replace(/\b\w/g, (char) => char.toUpperCase());
      groups.set(key, {
        key,
        title: isFolder
          ? (folder.id === "uncategorized" ? t("uncategorized") : folder.name)
          : String(source.group_label || fallbackTitle || "其他"),
        folderId: isFolder ? folder.id : "",
        sources: [],
      });
    }
    groups.get(key).sources.push(source);
  }
  return [...groups.values()];
}

async function handleDocumentFolderChange(sourceId, folderId, picker) {
  const source = (state.data?.sources || []).find((item) => item.source_id === sourceId);
  if (!source || !folderId || source.folder_id === folderId) return;
  const originalFolderId = source.folder_id || "uncategorized";
  picker.disabled = true;
  try {
    const document = await moveDocumentToFolder(sourceId, folderId);
    Object.assign(source, document);
    renderFolderControls();
    renderDocumentSelector();
    setDocumentUploadStatus(`${t("documentMoved")}：${source.name}`, "success");
  } catch (error) {
    picker.value = originalFolderId;
    picker.disabled = false;
    setDocumentUploadStatus(
      `${t("folderFailed")}：${error instanceof Error ? error.message : String(error)}`,
      "error",
    );
  }
}

function fileTypeLabel(source) {
  const name = String(source.name || source.url || "");
  const extension = name.match(/\.([a-z0-9]{1,5})(?:$|[?#])/i)?.[1];
  if (extension) return extension.toUpperCase();
  return String(source.source_type || "DOC").slice(0, 4).toUpperCase();
}

function startControlWatcher() {
  setInterval(() => {
    const selectedProfile = elements.knowledgeBase?.value;
    if (
      selectedProfile &&
      selectedProfile !== state.profile &&
      selectedProfile !== state.pendingProfile
    ) {
      state.pendingProfile = selectedProfile;
      loadKnowledgeBase(selectedProfile).finally(() => {
        state.pendingProfile = "";
      });
      return;
    }

    const selectedQaModel = elements.qaModel?.value;
    if (selectedQaModel && selectedQaModel !== state.qaModel) {
      state.qaModel = selectedQaModel;
    }
  }, 300);
}

async function loadKnowledgeBase(profile) {
  const config = knowledgeBases[profile];
  if (!config) throw new Error(`找不到知識庫設定：${profile}`);
  state.profile = config.profile;
  state.query = "";
  state.latestResult = null;
  state.latestRouteDecision = null;
  state.conversationId = "";
  state.messages = [];
  state.workflowRequestId += 1;
  revokeLatestTraceUrl();
  elements.knowledgeBase.value = state.profile;
  elements.input.value = "";
  resetChatPanels();
  renderConversationTranscript();
  renderChips();

  elements.dataStatus.textContent = `${config.label} ${t("loadingProfile")}`;
  elements.dataMeta.textContent = "";
  let profileData;
  try {
    profileData = await loadProfileData(config.profile);
  } catch (error) {
    if (!state.staticShowcase) throw error;
    profileData = {
      profile: config.profile,
      label: config.label,
      sampleQueries: [],
      meta: { mode: "static-showcase" },
      sources: [],
      folders: [{ id: "uncategorized", name: "未分類", system: true, document_count: 0 }],
    };
  }
  state.data = {
    meta: profileData.meta,
    sources: profileData.sources,
    chunks: [],
    aliases: [],
    graph: {},
  };
  state.folders = profileData.folders || [];
  state.uploadFolderId = "uncategorized";
  state.collapsedDocumentGroups = new Set();
  config.label = profileData.label || config.label;
  config.queries = profileData.sampleQueries || config.queries;
  config.defaultQuery = config.queries[0] || "";
  selectDefaultSources();
  renderFolderControls();
  renderDocumentSelector();
  renderChips();
  elements.dataStatus.textContent = `${config.label} ${t("profileLoaded")}`;
  if (state.staticShowcase) elements.dataStatus.textContent = t("staticBuild");
  updateDataMeta();
}

function isStaticShowcaseHost() {
  return window.location.protocol === "file:" || window.location.hostname.endsWith(".github.io");
}

async function handleDocumentFiles(files) {
  const selectedFiles = Array.from(files || []).filter(Boolean);
  if (!selectedFiles.length || state.uploadingDocuments) return;
  if (!state.data) {
    state.data = { meta: { profile: state.profile }, sources: [], chunks: [], aliases: [], graph: {} };
  }

  state.uploadingDocuments = true;
  const targetFolderId = state.uploadFolderId || "uncategorized";
  elements.uploadDocument.disabled = true;
  elements.uploadDocument.setAttribute("aria-busy", "true");
  elements.uploadFolder.disabled = true;
  const completed = [];
  const failed = [];
  try {
    for (const file of selectedFiles) {
      setDocumentUploadStatus(`${t("uploadWorking")} ${file.name}`, "working");
      try {
        const result = await uploadDocument(file, "/api/documents/upload", {
          folderId: targetFolderId,
        });
        mergeUploadedSources([result.document]);
        completed.push({ name: result.document.name, duplicate: result.duplicate });
      } catch (error) {
        failed.push({
          name: String(file?.name || "document"),
          message: error instanceof Error ? error.message : String(error),
        });
      }
    }

    renderDocumentSelector();
    updateDataMeta();
    if (failed.length) {
      const detail = failed.map((item) => `${item.name}: ${item.message}`).join("；");
      setDocumentUploadStatus(`${t("uploadFailed")}：${detail}`, "error");
    } else if (completed.length === 1) {
      const item = completed[0];
      setDocumentUploadStatus(
        `${item.duplicate ? t("uploadDuplicate") : t("uploadComplete")}：${item.name}`,
        "success",
      );
    } else {
      const duplicateCount = completed.filter((item) => item.duplicate).length;
      const detail = duplicateCount ? `，${duplicateCount} 份原已存在` : "";
      setDocumentUploadStatus(`${t("uploadComplete")}：${completed.length} 份${detail}`, "success");
    }
  } finally {
    state.uploadingDocuments = false;
    elements.uploadDocument.disabled = false;
    elements.uploadDocument.removeAttribute("aria-busy");
    elements.uploadFolder.disabled = false;
    renderFolderControls();
  }
}

function mergeUploadedSources(documents, { select = false } = {}) {
  if (!state.data || !Array.isArray(documents)) return;
  state.data.sources = Array.isArray(state.data.sources) ? state.data.sources : [];
  for (const document of documents) {
    const sourceId = String(document?.source_id || "").trim();
    if (!sourceId) continue;
    const source = {
      ...document,
      source_id: sourceId,
      source_type: String(document?.source_type || "document"),
      selected_by_default: Boolean(document?.selected_by_default),
    };
    const existingIndex = state.data.sources.findIndex((item) => item.source_id === sourceId);
    if (existingIndex >= 0) {
      state.data.sources[existingIndex] = { ...state.data.sources[existingIndex], ...source };
    } else {
      state.data.sources.push(source);
    }
    if (select) state.selectedSourceIds.add(sourceId);
  }
}

function updateDataMeta() {
  if (!state.data) return;
  const totalChunks = (state.data.sources || []).reduce(
    (total, source) => total + Number(source.chunk_count || 0),
    0,
  );
  const aliasCount = Number(state.data.meta?.alias_count || state.data.aliases?.length || 0);
  const graphCount = Number(state.data.meta?.graph_relation_count || state.data.graph?.relations?.length || 0);
  elements.dataMeta.textContent = `${totalChunks} 個片段 · ${aliasCount} 個別名 · ${graphCount} 筆圖譜關係`;
}

function setDocumentUploadStatus(message, status = "") {
  elements.documentUploadStatus.textContent = String(message || "");
  elements.documentUploadStatus.className = `upload-status${status ? ` is-${status}` : ""}`;
  elements.documentUploadStatus.hidden = !message;
}

function currentKnowledgeBase() {
  return knowledgeBases[state.profile] || {
    label: state.profile || "知識庫",
    profile: state.profile || "default",
    defaultQuery: "",
    queries: [],
  };
}

function selectedModel() {
  state.qaModel = elements.qaModel?.value || state.qaModel;
  return qaModels[state.qaModel] || Object.values(qaModels)[0] || {
    provider: "ollama",
    name: state.qaModel || "unknown",
    label: state.qaModel || "模型",
  };
}

function resetChatPanels() {
  elements.homePanel.hidden = false;
  elements.statusMessage.hidden = true;
  elements.answerMessage.hidden = true;
  elements.evidenceMessage.hidden = true;
  elements.ragTrace.hidden = true;
  elements.metrics.innerHTML = "";
  elements.answer.hidden = true;
  elements.answer.innerHTML = "";
  elements.agentAnswer.innerHTML = "";
  elements.comparison.hidden = true;
  elements.comparison.innerHTML = "";
  elements.results.innerHTML = "";
  elements.pipeline.innerHTML = "";
  elements.graphPanel.innerHTML = "";
  elements.traceSummary.innerHTML = "";
  elements.traceDocumentLink.hidden = true;
  elements.runId.textContent = t("runIdEmpty");
  elements.resultTitle.textContent = "";
  elements.confidence.textContent = "";
  elements.userMessage.hidden = true;
  elements.userQuestion.textContent = "";
  elements.ragDetailsGuide.hidden = true;
  elements.ragDetailsToggle.setAttribute("aria-expanded", "false");
  elements.ragDetailsToggle.textContent = "查看檢索過程與來源 →";
  elements.searchProgress.hidden = true;
  elements.searchProgressSteps.innerHTML = "";
}

async function startNewConversation() {
  resetCurrentConversationState();
  renderConversationHistory();
  elements.input.focus();

  if (window.matchMedia("(max-width: 899px)").matches) {
    elements.conversationRail.classList.add("is-collapsed");
    syncPanelToggleState();
  }
}

function resetCurrentConversationState() {
  state.agentRequestId += 1;
  state.workflowRequestId += 1;
  state.query = "";
  state.latestResult = null;
  state.latestRouteDecision = null;
  state.conversationId = "";
  state.messages = [];
  revokeLatestTraceUrl();
  elements.input.value = "";
  elements.askAgent.disabled = false;
  resetChatPanels();
  renderConversationTranscript();
}

function revealRunPanels() {
  elements.homePanel.hidden = true;
  elements.answerMessage.hidden = false;
  elements.statusMessage.hidden = true;
  elements.evidenceMessage.hidden = true;
  elements.ragTrace.hidden = true;
  elements.ragDetailsGuide.hidden = false;
}

function toggleRagDetails() {
  const willOpen = elements.ragDetailsToggle.getAttribute("aria-expanded") !== "true";
  elements.ragDetailsToggle.setAttribute("aria-expanded", String(willOpen));
  elements.ragDetailsToggle.textContent = willOpen
    ? "收合檢索過程與來源 ↑"
    : "查看檢索過程與來源 →";
  elements.statusMessage.hidden = !willOpen;
  elements.evidenceMessage.hidden = !willOpen;
  elements.ragTrace.hidden = !willOpen;
}

const adaptiveProgressSteps = [
  ["route", "判斷是否需要檢索"],
  ["rewrite", "整理檢索問題"],
  ["retrieve", "搜尋可用資料"],
  ["grade", "檢查證據相關性"],
  ["answer", "Qwen 生成回答"],
];

async function handleAdaptiveQuestion(rawQuestion) {
  const question = String(rawQuestion || "").trim();
  if (!question) return;
  if ((state.data?.sources || []).length && state.selectedSourceIds.size === 0) {
    elements.knowledgePanel.classList.remove("is-collapsed");
    syncPanelToggleState();
    setDocumentUploadStatus("請先在右側勾選至少一份文件或一個資料夾，再送出問題。", "error");
    elements.input.focus();
    return;
  }

  const workflowId = ++state.workflowRequestId;
  state.query = question;
  state.latestResult = null;
  state.latestRouteDecision = null;
  elements.input.value = question;
  renderConversationTranscript();
  startSearchProgress();
  elements.input.value = "";

  let decision;
  try {
    decision = await callRetrievalRouter("/api/route", {
      question,
      model: selectedModel(),
      conversation_id: state.conversationId,
    });
  } catch (error) {
    decision = {
      needsRetrieval: true,
      reason: `檢索判斷暫時無法使用，為避免漏掉文件證據，保守改走檢索。${error instanceof Error ? ` ${error.message}` : ""}`,
      retrievalQuery: question,
      timingMs: 0,
    };
  }
  if (workflowId !== state.workflowRequestId) return;

  state.latestRouteDecision = decision;
  updateProgressStep(
    "route",
    "done",
    decision.needsRetrieval ? "需要外部證據，嘗試檢索" : "不需要外部證據，直接回答",
  );

  if (!decision.needsRetrieval) {
    updateProgressStep("rewrite", "skipped", "不需要建立檢索查詢");
    updateProgressStep("retrieve", "skipped", "本次未搜尋知識庫");
    updateProgressStep("grade", "skipped", "沒有檢索結果需要評估");
    prepareGeneralAnswer(decision);
    updateProgressStep("answer", "active", "本機 Qwen 正在生成回答");
    const response = await askLiveAgent(null, { retrievalDecision: decision });
    if (workflowId !== state.workflowRequestId) return;
    updateProgressStep(
      "answer",
      response ? "done" : "error",
      response ? qualityProgressLabel(response) : "回答失敗",
    );
    return;
  }

  updateProgressStep(
    "rewrite",
    "done",
    decision.queryVariants?.length
      ? `已拆解並建立 ${decision.queryVariants.length} 個檢索查詢`
      : decision.retrievalQuery || question,
  );
  updateProgressStep("retrieve", "active", "BM25 與 Embedding 正在取候選，接著進行 Rerank");
  await nextPaint();

  let result = null;
  try {
    result = await runSearch({ retrievalQuery: decision.retrievalQuery });
  } catch (error) {
    updateProgressStep(
      "retrieve",
      "error",
      error instanceof Error ? error.message : "混合檢索失敗",
    );
    updateProgressStep("grade", "skipped", "尚未取得可評估的證據");
    updateProgressStep("answer", "skipped", "尚未取得可用證據");
    renderAgentMessage({
      status: "error",
      title: "混合檢索失敗",
      message: error instanceof Error ? error.message : String(error),
    });
    return;
  }
  updateProgressStep(
    "retrieve",
    "done",
    `BM25 + Embedding 已融合並 Rerank，保留 ${result?.contexts?.length || 0} 個片段`,
  );
  updateEvidenceProgress(result);

  updateProgressStep("answer", "active", "本機 Qwen 正在根據檢索內容生成回答");
  const response = await askLiveAgent(result, { retrievalDecision: decision });
  if (workflowId !== state.workflowRequestId) return;
  if (!result && response?.retrieval?.evidence_evaluation) {
    updateEvidenceProgress({ evidenceEvaluation: response.retrieval.evidence_evaluation });
  }
  updateProgressStep(
    "answer",
    response ? "done" : "error",
    response ? qualityProgressLabel(response) : "回答失敗",
  );
}

function startSearchProgress() {
  resetChatPanels();
  elements.homePanel.hidden = true;
  elements.userQuestion.textContent = state.query;
  elements.userMessage.hidden = false;
  elements.searchProgress.hidden = false;
  elements.searchProgressSteps.innerHTML = adaptiveProgressSteps
    .map(
      ([id, label]) => `
        <div class="progress-step" data-progress-step="${id}">
          <span class="progress-step-indicator">✓</span>
          <span><strong>${label}</strong><small class="progress-step-detail"></small></span>
        </div>
      `,
    )
    .join("");
  updateProgressStep("route", "active", "正在理解問題");
}

function updateProgressStep(id, status, detail = "") {
  const step = elements.searchProgressSteps.querySelector(`[data-progress-step="${id}"]`);
  if (!step) return;
  step.className = `progress-step is-${status}`;
  const detailNode = step.querySelector(".progress-step-detail");
  if (detailNode) detailNode.textContent = detail;
}

function prepareGeneralAnswer(decision, { retrievalPending = false } = {}) {
  elements.homePanel.hidden = true;
  elements.statusMessage.hidden = true;
  elements.answerMessage.hidden = false;
  elements.evidenceMessage.hidden = true;
  elements.ragTrace.hidden = true;
  elements.ragDetailsGuide.hidden = true;
  elements.resultTitle.textContent = state.query;
  elements.confidence.textContent = retrievalPending ? "後端檢索" : "未使用檢索";
  elements.confidence.className = "confidence confidence-medium";
  elements.answer.hidden = true;
  elements.answer.innerHTML = `
    <div class="answer-head">
      <div>
        <p class="eyebrow">自適應檢索路由</p>
        <h3>${retrievalPending ? "需要檢索" : "直接回答"}</h3>
      </div>
    </div>
    <p>${escapeHtml(decision.reason || "")}</p>
  `;
  elements.agentAnswer.innerHTML = "";
}

function nextPaint() {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

async function rerunCurrentRetrieval() {
  if (!state.query || !state.latestRouteDecision?.needsRetrieval) return;
  startSearchProgress();
  updateProgressStep("route", "done", "沿用本次 query-only 判斷：需要外部證據");
  updateProgressStep("rewrite", "done", state.latestRouteDecision.retrievalQuery || state.query);
  updateProgressStep("retrieve", "active", "文件選擇已變更，正在重新搜尋");
  await nextPaint();
  let result;
  try {
    result = await runSearch({ retrievalQuery: state.latestRouteDecision.retrievalQuery });
  } catch (error) {
    updateProgressStep(
      "retrieve",
      "error",
      error instanceof Error ? error.message : "混合檢索失敗",
    );
    updateProgressStep("grade", "skipped", "尚未取得可評估的證據");
    updateProgressStep("answer", "skipped", "尚未取得可用證據");
    return;
  }
  updateProgressStep("retrieve", "done", `找到 ${result?.contexts?.length || 0} 個相關片段`);
  updateEvidenceProgress(result);
  updateProgressStep("answer", "active", "本機 Qwen 正在重新生成回答");
  const response = await askLiveAgent(result, { retrievalDecision: state.latestRouteDecision });
  updateProgressStep(
    "answer",
    response ? "done" : "error",
    response ? qualityProgressLabel(response) : "回答失敗",
  );
}

function updateEvidenceProgress(result) {
  const evaluation = result?.evidenceEvaluation;
  if (!evaluation) {
    updateProgressStep("grade", "done", "檢索來源未提供原始分數，沿用相容模式");
    return;
  }
  const prefix = evaluation.sufficient === false ? "證據不足" : "證據可用";
  updateProgressStep("grade", "done", `${prefix}：${evaluation.reason || "已完成相關性檢查"}`);
}

async function runSearch(options = {}) {
  if (!state.data || !state.query) return;
  revealRunPanels();
  const retrievalQuery = String(
    options.retrievalQuery || state.latestRouteDecision?.retrievalQuery || state.query,
  ).trim();
  const result = await callHybridRetriever(hybridRetrievalEndpoint(), {
    question: state.query,
    retrieval_query: retrievalQuery,
    query_variants: state.latestRouteDecision?.queryVariants || [],
    evidence_query: [
      retrievalQuery,
      ...(state.latestRouteDecision?.subQuestions || []),
    ].filter(Boolean).join(" "),
    profile: state.profile,
    source_ids: [...state.selectedSourceIds],
    top_k: state.topK,
    candidate_k: state.candidateK,
  });
  state.latestResult = result;
  const effectiveRetrievalQuery = result.retrievalQuery || retrievalQuery || state.query;
  const expanded = {
    query: effectiveRetrievalQuery,
    matchedAliases: (result.diagnostics?.denseMatchedAliases || []).map((canonical) => ({
      canonical,
      aliases: [],
    })),
  };
  const graphResult = {
    matchedEntities: (result.diagnostics?.matchedEntities || []).map((name) => ({ name })),
    results: [],
    hubWarning: Boolean(result.diagnostics?.hubWarning),
  };
  state.latestRunId = buildRunId();

  renderMetrics(result, expanded);
  renderVariantDetails(result.variant);
  renderAgentIdle();
  renderComparisonGraph(result.comparisonGraph);
  renderResults(result, expanded);
  renderGraph(graphResult);
  renderPipeline(result);
  renderTrace(result, expanded);
  return result;
}

function hybridRetrievalEndpoint() {
  const endpoint = String(state.agentEndpoint || "/api/ask").trim();
  if (/\/api\/ask\/?$/i.test(endpoint)) {
    return endpoint.replace(/\/api\/ask\/?$/i, "/api/retrieve");
  }
  return "/api/retrieve";
}

function readConversationHistory() {
  try {
    const history = JSON.parse(localStorage.getItem(HISTORY_STORAGE_KEY) || "[]");
    if (!Array.isArray(history)) return [];
    return history
      .filter((item) => item && typeof item.id === "string" && typeof item.title === "string")
      .slice(0, 30);
  } catch {
    return [];
  }
}

function saveConversationHistory() {
  try {
    localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(state.history));
  } catch {
    // The demo remains usable when browser storage is unavailable.
  }
}

function renderConversationHistory() {
  elements.conversationList.innerHTML = "";
  if (!state.history.length) {
    elements.conversationList.innerHTML = '<p class="conversation-empty">尚無對話</p>';
    return;
  }

  for (const item of state.history) {
    const row = document.createElement("div");
    row.className = `conversation-row${item.id === state.conversationId ? " is-active" : ""}`;
    const button = document.createElement("button");
    button.type = "button";
    button.className = `conversation-item${item.id === state.conversationId ? " is-active" : ""}`;
    button.innerHTML = `
      <strong>${escapeHtml(shortTitle(item.title || "新對話"))}</strong>
      <span>${escapeHtml(`${item.message_count || 0} 則訊息`)}</span>
    `;
    button.addEventListener("click", async () => {
      await openSavedConversation(item.id);
    });

    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "conversation-delete-button";
    deleteButton.title = "刪除對話";
    deleteButton.setAttribute("aria-label", `刪除對話：${shortTitle(item.title || "新對話")}`);
    deleteButton.innerHTML = '<span class="trash-icon" aria-hidden="true"></span>';
    deleteButton.addEventListener("click", async (event) => {
      event.stopPropagation();
      await deleteSavedConversation(item, deleteButton);
    });

    row.append(button, deleteButton);
    elements.conversationList.append(row);
  }
}

function confirmConversationDeletion(title) {
  elements.deleteConversationMessage.textContent = `確定刪除「${title}」？刪除後無法復原。`;
  elements.deleteConversationDialog.returnValue = "cancel";
  elements.deleteConversationDialog.showModal();

  return new Promise((resolve) => {
    elements.deleteConversationDialog.addEventListener("close", () => {
      resolve(elements.deleteConversationDialog.returnValue === "confirm");
    }, { once: true });
  });
}

async function deleteSavedConversation(item, deleteButton) {
  const title = shortTitle(item.title || "新對話");
  const confirmed = await confirmConversationDeletion(title);
  if (!confirmed) return;

  deleteButton.disabled = true;
  try {
    await deleteConversation(item.id, "/api/conversations");
  } catch (error) {
    deleteButton.disabled = false;
    window.alert(`無法刪除對話：${error instanceof Error ? error.message : String(error)}`);
    return;
  }

  const deletedCurrentConversation = item.id === state.conversationId;
  state.history = state.history.filter((conversation) => conversation.id !== item.id);
  saveConversationHistory();
  if (!deletedCurrentConversation) {
    renderConversationHistory();
    return;
  }

  resetCurrentConversationState();
  const nextConversation = state.history[0];
  if (nextConversation) {
    try {
      await openSavedConversation(nextConversation.id);
      return;
    } catch {
      resetCurrentConversationState();
    }
  }
  renderConversationHistory();
  elements.input.focus();
}

async function refreshConversationHistory({ loadLatest = false } = {}) {
  try {
    state.history = await listConversations("/api/conversations");
    saveConversationHistory();
    renderConversationHistory();
    if (loadLatest && !state.conversationId && state.history.length) {
      await openSavedConversation(state.history[0].id);
    }
  } catch {
    renderConversationHistory();
  }
}

async function openSavedConversation(conversationId) {
  const conversation = await loadConversation(conversationId, "/api/conversations");
  if (conversation.profile !== state.profile && knowledgeBases[conversation.profile]) {
    await loadKnowledgeBase(conversation.profile);
  }
  state.conversationId = String(conversation.id || "");
  state.messages = Array.isArray(conversation.messages) ? conversation.messages : [];
  state.query = "";
  state.latestResult = null;
  state.latestRouteDecision = null;
  state.agentRequestId += 1;
  state.workflowRequestId += 1;
  elements.input.value = "";
  resetChatPanels();
  renderConversationTranscript();
  if (state.messages.length) elements.homePanel.hidden = true;
  renderConversationHistory();
  scrollChatToBottom();

  if (window.matchMedia("(max-width: 899px)").matches) {
    elements.conversationRail.classList.add("is-collapsed");
    syncPanelToggleState();
  }
}

function renderConversationTranscript() {
  elements.conversationTranscript.innerHTML = "";
  elements.conversationTranscript.hidden = !state.messages.length;
  for (const message of state.messages) {
    const role = message.role === "assistant" ? "assistant" : "user";
    const article = document.createElement("article");
    article.className = `transcript-message transcript-${role}`;
    if (role === "assistant") {
      article.innerHTML = `
        <span class="transcript-avatar" aria-hidden="true">AI</span>
        <div><p>${escapeHtml(String(message.content || ""))}</p></div>
      `;
    } else {
      article.innerHTML = `<p>${escapeHtml(String(message.content || ""))}</p>`;
    }
    elements.conversationTranscript.append(article);
  }
}

function scrollChatToBottom() {
  if (!elements.chatScroll) return;
  elements.chatScroll.scrollTop = elements.chatScroll.scrollHeight;
}

function shortTitle(query) {
  const text = String(query).replace(/\s+/g, " ").trim();
  return text.length > 24 ? `${text.slice(0, 24)}...` : text;
}

async function askLiveAgent(result, { retrievalDecision = null } = {}) {
  if (state.query && retrievalDecision?.needsRetrieval !== false) revealRunPanels();
  if (!state.agentEndpoint) {
    renderAgentMessage({
      status: "offline",
      title: t("liveAgentOffline"),
      message: `${t("agentOfflineMessage")} ${currentKnowledgeBase().label} · ${selectedModel().label}`,
    });
    return null;
  }

  const requestId = ++state.agentRequestId;
  elements.askAgent.disabled = true;
  renderAgentMessage({
    status: "pending",
    title: t("liveAgentPending"),
    message: `模型服務：${currentKnowledgeBase().label} · ${selectedModel().label}`,
  });

  try {
    const payload = buildAgentQueryPayload({
      question: result?.query || state.query,
      profile: state.profile,
      model: selectedModel(),
      topK: state.topK,
      conversationId: state.conversationId,
      sourceIds: [...state.selectedSourceIds],
    });
    const response = await callAgentEndpoint(state.agentEndpoint, payload);
    if (requestId !== state.agentRequestId) return;
    if (response.conversationId) state.conversationId = response.conversationId;
    state.messages.push(
      { role: "user", content: state.query, created_at: new Date().toISOString() },
      { role: "assistant", content: response.answer, created_at: new Date().toISOString() },
    );
    renderAgentResponse(response);
    await refreshConversationHistory();
    return response;
  } catch (error) {
    if (requestId !== state.agentRequestId) return;
    renderAgentMessage({
      status: "error",
      title: t("liveAgentError"),
      message: error instanceof Error ? error.message : String(error),
    });
    return null;
  } finally {
    if (requestId === state.agentRequestId) elements.askAgent.disabled = false;
  }
}

function renderMetrics(result, expanded) {
  elements.resultTitle.textContent = `${t("evidenceFor")}: ${state.query}`;
  elements.confidence.textContent = `${t("retrievalConfidence")}: ${confidenceLabel(result.confidence)}`;
  elements.confidence.className = `confidence confidence-${result.confidence}`;
  elements.metrics.innerHTML = "";
  const metrics = [
    [t("knowledgeBase"), currentKnowledgeBase().label],
    [t("qaLlm"), selectedModel().label],
    [t("documents"), `${state.selectedSourceIds.size}/${state.data.sources.length}`],
    [t("contexts"), result.contexts.length],
    [t("totalTime"), `${result.timings.totalMs.toFixed(2)} ms`],
    [
      t("translation"),
      result.translation.addedTerms?.length ? "查詢擴展" : t("direct"),
    ],
    [t("aliasHits"), expanded.matchedAliases.length],
  ];
  for (const [label, value] of metrics) {
    const item = document.createElement("div");
    item.className = "metric";
    item.innerHTML = `<span>${label}</span><strong>${escapeHtml(String(value))}</strong>`;
    elements.metrics.append(item);
  }
}

function renderVariantDetails(variant) {
  const detail = architectureForVariant(variant);
  elements.variantDetails.innerHTML = `
    <div>
      <p class="eyebrow">檢索架構</p>
      <h3>${escapeHtml(detail.label)}</h3>
    </div>
    <p>${escapeHtml(detail.summary)}</p>
    <div class="architecture-steps">
      ${detail.steps.map((step) => `<span>${escapeHtml(step)}</span>`).join("")}
    </div>
  `;
}

function renderAgentIdle() {
  elements.askAgent.disabled = false;
  renderAgentMessage({
    status: state.agentEndpoint ? "ready" : "offline",
    title: state.agentEndpoint ? t("liveAgentReady") : t("liveAgentOffline"),
    message: state.agentEndpoint
      ? `${t("agentReadyMessage")} ${currentKnowledgeBase().label} · ${selectedModel().label}`
      : t("agentOfflineMessage"),
  });
}

function renderAgentMessage({ status, title, message }) {
  elements.agentAnswer.className = `agent-answer-panel agent-${status}`;
  elements.agentAnswer.innerHTML = `
    <div class="answer-head">
      <div>
        <p class="eyebrow">本機模型</p>
        <h3>${escapeHtml(title)}</h3>
      </div>
      <span>${escapeHtml(agentStatusLabel(status))}</span>
    </div>
    <p>${escapeHtml(message)}</p>
  `;
}

function renderAgentResponse(response) {
  const model = `${response.model.provider || "model"}:${response.model.name || "unknown"}`;
  const warnings = response.groundingWarnings || [];
  const totalMs = response.timings?.total_ms || response.timings?.totalMs;
  elements.agentAnswer.className = "agent-answer-panel agent-success";
  elements.agentAnswer.innerHTML = `
    <div class="answer-head">
      <div>
        <p class="eyebrow">本機模型</p>
        <h3>${escapeHtml(t("answer"))}</h3>
      </div>
      <span>${escapeHtml(model)}</span>
    </div>
    <p>${escapeHtml(response.answer)}</p>
    <p class="answer-terms">
      ${totalMs != null ? `模型總耗時 ${Number(totalMs).toFixed(0)} ms` : "無模型耗時資料"}
      ${warnings.length ? ` · 警告：${warnings.map(escapeHtml).join("、")}` : ""}
    </p>
    ${renderQualityEvaluation(response.qualityEvaluation)}
  `;
}

function qualityProgressLabel(response) {
  const evaluation = response?.qualityEvaluation;
  if (evaluation?.status === "completed") {
    return `回答完成 · Claude 基準 100 · Qwen ${Number(evaluation.candidate?.score || 0)} 分`;
  }
  if (evaluation?.status === "unavailable") return "回答完成 · Claude 評分暫時無法使用";
  return "回答完成";
}

function renderQualityEvaluation(evaluation) {
  if (!evaluation) return "";
  if (evaluation.status !== "completed") {
    return `
      <section class="quality-evaluation-card quality-evaluation-unavailable">
        <div class="quality-score-head">
          <div><span>Claude 品質基準</span><strong>尚未完成評分</strong></div>
          <span class="quality-score-badge">—</span>
        </div>
        <p>${escapeHtml(evaluation.reason || "Claude 參考評測目前無法使用，Qwen 回答仍保留。")}</p>
      </section>
    `;
  }

  const reference = evaluation.reference || {};
  const candidate = evaluation.candidate || {};
  const dimensions = Array.isArray(evaluation.dimensions) ? evaluation.dimensions : [];
  const improvements = Array.isArray(evaluation.improvements) ? evaluation.improvements : [];
  const capReasons = Array.isArray(evaluation.score_cap?.reasons)
    ? evaluation.score_cap.reasons
    : [];
  return `
    <section class="quality-evaluation-card">
      <div class="quality-score-head">
        <div>
          <span>Claude 參考答案 = 100 分</span>
          <strong>Qwen 回答品質</strong>
        </div>
        <span class="quality-score-badge">${Number(candidate.score || 0)}<small>/100</small></span>
      </div>
      ${evaluation.summary ? `<p>${escapeHtml(evaluation.summary)}</p>` : ""}
      <div class="quality-dimensions">
        ${dimensions.map((item) => `
          <div>
            <span>${escapeHtml(item.label || item.id || "評分項目")}</span>
            <strong>${Number(item.score || 0)}/${Number(item.max_score || 0)}</strong>
            <small>${escapeHtml(item.reason || "")}</small>
          </div>
        `).join("")}
      </div>
      ${capReasons.length ? `<p class="quality-cap">分數上限：${capReasons.map(escapeHtml).join("；")}</p>` : ""}
      ${improvements.length ? `
        <div class="quality-improvements">
          <strong>優先改善</strong>
          <ul>${improvements.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
        </div>
      ` : ""}
      <details class="quality-reference-answer">
        <summary>查看 Claude 參考答案（${escapeHtml(reference.model || "Claude")}）</summary>
        <p>${escapeHtml(reference.answer || "")}</p>
      </details>
    </section>
  `;
}

function renderResults(result, expanded) {
  elements.results.innerHTML = "";
  if (!result.contexts.length) {
    elements.results.innerHTML =
      `<div class="empty">${escapeHtml(t("noEvidence"))}</div>`;
    return;
  }

  for (const context of result.contexts) {
    const card = document.createElement("article");
    card.className = "result-card";
    const source = sourceFor(context.source);
    const terms = [
      ...new Set([
        ...(result.translation.addedTerms || []),
        ...expanded.matchedAliases.flatMap((item) => [item.canonical, ...(item.aliases || [])]),
        ...(context.matchedTerms || []),
      ]),
    ].slice(0, 8);
    const excerpt = evidenceSnippet(context.content, terms);
    card.innerHTML = `
      <header>
        <span class="rank">#${context.rank}</span>
        <div>
          <h3>${escapeHtml(context.title)}</h3>
          <p>
            ${source?.url ? `<a href="${escapeHtml(source.url)}" target="_blank" rel="noopener">${escapeHtml(source.name || context.source)}</a>` : escapeHtml(source?.name || context.source)}
            · ${escapeHtml(context.page || "chunk")}
          </p>
        </div>
      </header>
      <div class="tags">
        <span>${escapeHtml(context.branch || "merged")}</span>
        <span>rerank ${formatRetrievalScore(context.rerankScore ?? context.score)}</span>
        ${context.bm25Score != null ? `<span>BM25 ${formatRetrievalScore(context.bm25Score)}</span>` : ""}
        ${context.embeddingScore != null ? `<span>embedding ${formatRetrievalScore(context.embeddingScore)}</span>` : ""}
        ${source?.source_type ? `<span>${escapeHtml(source.source_type)}</span>` : ""}
      </div>
      <p class="excerpt">${highlight(escapeHtml(excerpt), terms)}</p>
        ${terms.length ? `<p class="terms">命中詞：${terms.map(escapeHtml).join("、")}</p>` : ""}
    `;
    elements.results.append(card);
  }
}

function renderComparisonGraph(comparisonGraph) {
  if (!comparisonGraph) {
    elements.comparison.hidden = true;
    elements.comparison.innerHTML = "";
    return;
  }

  elements.comparison.hidden = false;
  elements.comparison.innerHTML = `
    <div class="comparison-head">
      <div>
        <p class="eyebrow">圖譜比較表</p>
        <h3>${escapeHtml(comparisonGraph.title)}</h3>
      </div>
      <span>${comparisonGraph.nodes.length} 個節點 · ${comparisonGraph.edges.length} 條關係</span>
    </div>
    <p class="comparison-summary">${escapeHtml(comparisonGraph.summary)}</p>
    <div class="comparison-table-wrap">
      <table class="comparison-table">
        <thead>
          <tr>
            <th>比較面向</th>
            <th>${escapeHtml(comparisonGraph.oldLabel || "調整前")}</th>
            <th>${escapeHtml(comparisonGraph.newLabel || "調整後")}</th>
            <th>影響</th>
            <th>證據</th>
          </tr>
        </thead>
        <tbody>
          ${comparisonGraph.rows.map(renderComparisonRow).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderComparisonRow(row) {
  const evidence = row.evidence
    .slice(0, 2)
    .map((item) => formatCitation(item))
    .join("；");
  return `
    <tr>
      <th scope="row">${escapeHtml(row.aspect)}</th>
      <td>${escapeHtml(row.oldPolicy)}</td>
      <td>${escapeHtml(row.newPolicy)}</td>
      <td>${escapeHtml(row.impact)}</td>
      <td>${escapeHtml(evidence || "沒有連結的片段")}</td>
    </tr>
  `;
}

function renderGraph(graphResult) {
  const names = graphResult.matchedEntities.map((entity) => entity.name).slice(0, 12);
  const relations = graphResult.results
    .flatMap((item) => item.relations || [])
    .slice(0, 8);
  elements.graphPanel.innerHTML = "";
  if (graphResult.hubWarning) {
    const warning = document.createElement("div");
    warning.className = "warning";
    warning.textContent =
      "偵測到過於寬廣的圖譜命中。中心實體可能帶入無關片段，因此優先採用文字檢索排序。";
    elements.graphPanel.append(warning);
  }
  const entityBlock = document.createElement("div");
  entityBlock.className = "graph-block";
  entityBlock.innerHTML = `
    <h3>${escapeHtml(t("matchedEntities"))}</h3>
    <p>${names.length ? names.map(escapeHtml).join(", ") : escapeHtml(t("noGraphEntity"))}</p>
  `;
  elements.graphPanel.append(entityBlock);

  const relationBlock = document.createElement("div");
  relationBlock.className = "graph-block";
  relationBlock.innerHTML = `
    <h3>${escapeHtml(t("relationSupport"))}</h3>
    <ul>${relations.map((item) => `<li>${escapeHtml(item)}</li>`).join("") || `<li>${escapeHtml(t("noRelationSupport"))}</li>`}</ul>
  `;
  elements.graphPanel.append(relationBlock);
}

function renderPipeline(result) {
  elements.pipeline.innerHTML = "";
  for (const step of result.pipeline) {
    const li = document.createElement("li");
    li.innerHTML = `<strong>${escapeHtml(step.name)}</strong><span>${escapeHtml(step.detail)}</span>`;
    elements.pipeline.append(li);
  }
}

function renderTrace(result, expanded) {
  revokeLatestTraceUrl();
  const selectedSources = (state.data?.sources || []).filter((source) =>
    state.selectedSourceIds.has(source.source_id),
  );
  const traceMarkdown = buildTraceMarkdown(result, expanded, selectedSources);
  const traceBlob = new Blob([traceMarkdown], { type: "text/markdown;charset=utf-8" });
  state.latestTraceUrl = URL.createObjectURL(traceBlob);
  elements.traceDocumentLink.href = state.latestTraceUrl;
  elements.traceDocumentLink.download = `${state.latestRunId}.md`;
  elements.traceDocumentLink.hidden = false;
  elements.runId.textContent = state.latestRunId;
  elements.traceSummary.innerHTML = `
    <div class="trace-grid">
      <div><span>${escapeHtml(t("query"))}</span><strong>${escapeHtml(result.query)}</strong></div>
      <div><span>${escapeHtml(t("model"))}</span><strong>${escapeHtml(selectedModel().label)}</strong></div>
      <div><span>${escapeHtml(t("documents"))}</span><strong>${selectedSources.length}</strong></div>
      <div><span>${escapeHtml(t("totalTime"))}</span><strong>${result.timings.totalMs.toFixed(2)} ms</strong></div>
    </div>
  `;
}

function buildTraceMarkdown(result, expanded, selectedSources) {
  const lines = [
    "# RAG 執行紀錄",
    "",
    `執行編號：${state.latestRunId}`,
    `時間：${new Date().toISOString()}`,
    `知識庫：${currentKnowledgeBase().label}`,
    `模型：${selectedModel().label}`,
    `問題：${result.query}`,
    `檢索查詢：${result.retrievalQuery}`,
    "",
    "## 已選文件",
    "",
    ...selectedSources.map(
      (source) =>
        `- ${source.name} (${source.source_type || "source"})${source.url ? ` - ${source.url}` : ""}`,
    ),
    "",
    "## 處理節點",
    "",
    ...result.pipeline.map((step, index) => `${index + 1}. ${step.name}: ${step.detail}`),
    "",
    "## 各階段耗時",
    "",
    ...Object.entries(result.timings).map(([key, value]) => `- ${key}: ${Number(value).toFixed(2)} ms`),
    "",
    "## 檢索診斷",
    "",
    `- 信心：${confidenceLabel(result.confidence)}`,
    `- 命中別名：${expanded.matchedAliases.map((item) => item.canonical).join("、") || "無"}`,
    `- 命中圖譜實體：${result.diagnostics.matchedEntities.join("、") || "無"}`,
    `- 中心實體警告：${result.diagnostics.hubWarning ? "是" : "否"}`,
    "",
    "## 取回片段",
    "",
  ];

  for (const context of result.contexts) {
    const source = sourceFor(context.source);
    lines.push(
      `### ${context.rank}. ${context.title}`,
      "",
      `- 來源：${source?.name || context.source}`,
      `- 頁面或片段：${context.page || "片段"}`,
      `- 檢索分支：${context.branch || "已融合"}`,
      `- 重排分數：${context.rerankScore ?? context.score}`,
      context.bm25Score != null ? `- BM25 分數：${context.bm25Score}` : "- BM25 分數：無資料",
      context.embeddingScore != null
        ? `- Embedding 餘弦相似度：${context.embeddingScore}`
        : "- Embedding 餘弦相似度：無資料",
      context.fusionScore != null ? `- RRF 融合分數：${context.fusionScore}` : "- RRF 融合分數：無資料",
      source?.url ? `- 網址：${source.url}` : "- 網址：無資料",
      "",
      context.content,
      "",
    );
  }

  return `${lines.join("\n")}\n`;
}

function revokeLatestTraceUrl() {
  if (state.latestTraceUrl) URL.revokeObjectURL(state.latestTraceUrl);
  state.latestTraceUrl = "";
}

function buildRunId() {
  const stamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\..+/, "Z");
  return `rag-run-${stamp}`;
}

function sourceFor(sourceId) {
  return state.data?.sources?.find((source) => source.source_id === sourceId);
}

function labelForVariant(variant) {
  return {
    bm25: "僅使用 BM25",
    dense: "語意檢索代理",
    bm25_dense: "BM25 + Dense",
    bm25_embedding_rerank: "BM25 + Embedding + Rerank",
    bm25_dense_graph: "BM25 + Dense + Graph",
    full: "完整檢索流程",
  }[variant] || variant;
}

function architectureForVariant(variant) {
  const architectures = {
    bm25: {
      label: "僅使用 BM25",
      summary: "以關鍵詞與問題詞彙重疊程度排序片段，作為字面檢索基準。",
      steps: ["BM25", "前 K 筆證據"],
    },
    bm25_dense: {
      label: "BM25 + 語意檢索",
      summary: "結合字面與語意候選，再以 RRF 融合排序。",
      steps: ["BM25", "語意檢索或別名擴展", "RRF 融合", "前 K 筆證據"],
    },
    bm25_embedding_rerank: {
      label: "BM25 + Embedding + Rerank",
      summary: "BM25 與多語 Embedding 產生候選，RRF 融合後再以相關性模型重排證據。",
      steps: ["BM25", "多語 Embedding", "RRF 融合", "混合重排", "前 K 筆證據"],
    },
    bm25_dense_graph: {
      label: "BM25 + 語意檢索 + 圖譜",
      summary: "在字面與語意候選之外加入圖譜關係，再融合三條檢索分支。",
      steps: ["BM25", "語意檢索或別名擴展", "圖譜檢索", "RRF 融合", "前 K 筆證據"],
    },
    full: {
      label: "完整檢索流程",
      summary: "融合字面、語意與圖譜證據，經重排、中心實體抑制及證據品質檢查後輸出。",
      steps: [
        "查詢擴展",
        "Metadata 範圍過濾",
        "BM25",
        "語意檢索或別名擴展",
        "圖譜檢索",
        "RRF 融合",
        "重排",
        "中心實體防護",
        "證據品質檢查",
        "需要時建立比較圖",
      ],
    },
  };

  return architectures[variant] || architectures.full;
}

function confidenceLabel(value) {
  return {
    high: "高",
    medium: "中",
    low: "低",
  }[String(value || "").toLowerCase()] || String(value || "未知");
}

function agentStatusLabel(value) {
  return {
    ready: "就緒",
    offline: "未連線",
    pending: "處理中",
    error: "錯誤",
    success: "完成",
  }[String(value || "").toLowerCase()] || String(value || "未知");
}

function formatRetrievalScore(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(4) : "0.0000";
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char],
  );
}

function highlight(html, terms) {
  let output = html;
  const safeTerms = terms
    .filter((term) => String(term).length > 2)
    .sort((a, b) => String(b).length - String(a).length)
    .slice(0, 8);
  for (const term of safeTerms) {
    const pattern = new RegExp(`(${escapeRegex(escapeHtml(term))})`, "ig");
    output = output.replace(pattern, "<mark>$1</mark>");
  }
  return output;
}

function escapeRegex(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function evidenceSnippet(content, terms, maxLength = 760) {
  const text = String(content || "").replace(/\s+/g, " ").trim();
  const lower = text.toLowerCase();
  const strongTerms = terms
    .map((term) => String(term).toLowerCase())
    .filter((term) => term.length >= 8)
    .sort((a, b) => b.length - a.length);
  const firstHit = strongTerms
    .map((term) => lower.indexOf(term))
    .filter((index) => index >= 0)
    .sort((a, b) => a - b)[0];
  if (firstHit == null || firstHit <= 140) return text.slice(0, maxLength);

  const start = Math.max(0, firstHit - 120);
  const cleanStart = text.indexOf(" ", start);
  return `...${text.slice(cleanStart > 0 ? cleanStart + 1 : start, cleanStart + maxLength)}`;
}

function formatCitation(citation) {
  const title = citation.title || "未命名來源";
  if (!citation.page || title.toLowerCase().includes(String(citation.page).toLowerCase())) {
    return title;
  }
  return `${title} / ${citation.page}`;
}
