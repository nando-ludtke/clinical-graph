"use strict";

(() => {
  // --- Constants & State ---
  const API_KEY_STORAGE_KEY = "openai_api_key";

  const NODE_COLORS = {
    drug: { fill: "#93c5fd", border: "#3b82f6" },
    drug_class: { fill: "#a5b4fc", border: "#6366f1" },
    condition: { fill: "#fca5a5", border: "#ef4444" },
    symptom: { fill: "#fdba74", border: "#f97316" },
    biomarker: { fill: "#d8b4fe", border: "#a855f7" },
    procedure: { fill: "#99f6e4", border: "#14b8a6" },
    population: { fill: "#fde68a", border: "#eab308" },
    entity: { fill: "#cbd5e1", border: "#94a3b8" },
  };

  const CURSOR_REVEAL_RADIUS_PX = 180;

  const state = {
    hasServerApiKey: false,
    guardReachable: null,
    guardPollTimer: null,
    documents: [],
    pendingJobs: new Map(), // job_id -> { job_id, title, current_step, timerId }
    selectedDocId: null,
    relatedDocIds: new Set(),
    graphData: { nodes: [], edges: [] },
    cy: null,
    sessionId:
      typeof crypto !== "undefined" && crypto.randomUUID
        ? crypto.randomUUID()
        : `sess-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    chatHistory: [],
    isQuerying: false,
    activeHighlightMode: null, // { type: "doc"|"element"|"retrieval"|"nodeType", ... }
    activeNodeTypeFilter: null,
    cursorModelPos: null,
    cursorRenderedPos: null,
    lastLabelZoom: null,
    baseFitZoom: 1,
    cyContainerMouseLeaveBound: false,
    rewriteCollapseTimer: null,
    documentCache: new Map(), // doc_id -> DocumentRecord
    docViewerRequestSeq: 0,
    citationModalRequestSeq: 0,
    apiGuideLang: "python",
    activeView: "overview",
    workspaceInitializedFit: false,
    retrievalMode: "auto",
  };

  // --- DOM References ---
  const el = {
    navOverviewBtn: document.getElementById("nav-overview-btn"),
    navWorkspaceBtn: document.getElementById("nav-workspace-btn"),
    overviewView: document.getElementById("overview-view"),
    workspaceView: document.getElementById("workspace-view"),
    overviewApiGuideBtn: document.getElementById("overview-api-guide-btn"),
    openWorkspaceCta: document.getElementById("open-workspace-cta"),
    unitEvalsGrid: document.getElementById("unit-evals-grid"),
    benchmarkSummaryGrid: document.getElementById("benchmark-summary-grid"),
    benchmarkQueriesList: document.getElementById("benchmark-queries-list"),

    nodeLegend: document.getElementById("node-legend"),
    zoomInBtn: document.getElementById("zoom-in-btn"),
    zoomOutBtn: document.getElementById("zoom-out-btn"),
    resetFocusBtn: document.getElementById("reset-focus-btn"),
    apiKeyBtn: document.getElementById("api-key-btn"),
    apiKeyStatusDot: document.getElementById("api-key-status-dot"),
    apiKeyStatusLabel: document.getElementById("api-key-status-label"),
    guardStatusPill: document.getElementById("guard-status-pill"),
    guardStatusDot: document.getElementById("guard-status-dot"),
    guardStatusLabel: document.getElementById("guard-status-label"),
    apiGuideBtn: document.getElementById("api-guide-btn"),

    sidebarLeft: document.getElementById("sidebar-left"),
    toggleDocsBtn: document.getElementById("toggle-docs-btn"),
    docCountBadge: document.getElementById("doc-count-badge"),
    docShowcaseNote: document.getElementById("doc-showcase-note"),
    docList: document.getElementById("doc-list"),
    openUploadBtn: document.getElementById("open-upload-btn"),

    cyContainer: document.getElementById("cy"),
    graphEmptyOverlay: document.getElementById("graph-empty-overlay"),
    graphTooltip: document.getElementById("graph-tooltip"),
    inspectorBar: document.getElementById("inspector-bar"),
    inspectorKind: document.getElementById("inspector-kind"),
    inspectorTitle: document.getElementById("inspector-title"),
    inspectorBody: document.getElementById("inspector-body"),
    closeInspectorBtn: document.getElementById("close-inspector-btn"),

    pipelineBar: document.getElementById("pipeline-bar"),
    closePipelineBtn: document.getElementById("close-pipeline-btn"),
    pipelineStatusText: document.getElementById("pipeline-status-text"),
    stageLatency: document.getElementById("stage-latency"),
    pipelineLatency: document.getElementById("pipeline-latency"),
    pipelineLatencyValue: document.getElementById("pipeline-latency-value"),
    pipelineLatencyTooltip: document.getElementById("pipeline-latency-tooltip"),
    stageRewrite: document.getElementById("stage-rewrite"),
    rewriteWrapper: document.getElementById("rewrite-wrapper"),
    rewriteCompactPill: document.getElementById("rewrite-compact-pill"),
    rewriteChips: document.getElementById("rewrite-chips"),
    stageRoute: document.getElementById("stage-route"),
    modeIconRag: document.getElementById("mode-icon-rag"),
    modeIconGraph: document.getElementById("mode-icon-graph"),
    stageRetrieval: document.getElementById("stage-retrieval"),
    evidenceStack: document.getElementById("evidence-stack"),
    stageWebSearch: document.getElementById("stage-web-search"),
    webEvidenceStack: document.getElementById("web-evidence-stack"),
    stageSynthesis: document.getElementById("stage-synthesis"),
    synthStatusPill: document.getElementById("synth-status-pill"),

    chatMessages: document.getElementById("chat-messages"),
    chatForm: document.getElementById("chat-form"),
    chatGuardNote: document.getElementById("chat-guard-note"),
    chatInput: document.getElementById("chat-input"),
    retrievalModeGroup: document.getElementById("retrieval-mode-group"),
    webSearchSwitch: document.getElementById("web-search-switch"),
    webSearchToggle: document.getElementById("web-search-toggle"),
    webSearchModal: document.getElementById("web-search-modal"),
    closeWebSearchModalBtn: document.getElementById("close-web-search-modal-btn"),
    cancelWebSearchBtn: document.getElementById("cancel-web-search-btn"),
    confirmWebSearchBtn: document.getElementById("confirm-web-search-btn"),
    sendQueryBtn: document.getElementById("send-query-btn"),

    apiKeyModal: document.getElementById("api-key-modal"),
    closeApiKeyModalBtn: document.getElementById("close-api-key-modal-btn"),
    apiKeyForm: document.getElementById("api-key-form"),
    apiKeyInput: document.getElementById("api-key-input"),
    clearApiKeyBtn: document.getElementById("clear-api-key-btn"),

    uploadModal: document.getElementById("upload-modal"),
    closeUploadModalBtn: document.getElementById("close-upload-modal-btn"),
    cancelUploadBtn: document.getElementById("cancel-upload-btn"),
    uploadForm: document.getElementById("upload-form"),
    uploadFile: document.getElementById("upload-file"),
    uploadTitle: document.getElementById("upload-title"),
    uploadRestriction: document.getElementById("upload-restriction"),
    uploadDescription: document.getElementById("upload-description"),
    submitUploadBtn: document.getElementById("submit-upload-btn"),

    citationModal: document.getElementById("citation-modal"),
    citationModalTitle: document.getElementById("citation-modal-title"),
    citationModalSubtitle: document.getElementById("citation-modal-subtitle"),
    citationModalBody: document.getElementById("citation-modal-body"),
    citationDocPane: document.getElementById("citation-doc-pane"),
    citationQuotesPane: document.getElementById("citation-quotes-pane"),
    closeCitationModalBtn: document.getElementById("close-citation-modal-btn"),

    docViewerModal: document.getElementById("doc-viewer-modal"),
    docViewerTitle: document.getElementById("doc-viewer-title"),
    docViewerSubtitle: document.getElementById("doc-viewer-subtitle"),
    docViewerBody: document.getElementById("doc-viewer-body"),
    closeDocViewerBtn: document.getElementById("close-doc-viewer-btn"),

    noticeModal: document.getElementById("notice-modal"),
    noticeModalTitle: document.getElementById("notice-modal-title"),
    noticeModalMessage: document.getElementById("notice-modal-message"),
    closeNoticeModalBtn: document.getElementById("close-notice-modal-btn"),
    okNoticeModalBtn: document.getElementById("ok-notice-modal-btn"),

    guardUploadConfirmModal: document.getElementById("guard-upload-confirm-modal"),
    closeGuardUploadModalBtn: document.getElementById("close-guard-upload-modal-btn"),
    cancelGuardUploadBtn: document.getElementById("cancel-guard-upload-btn"),
    confirmGuardUploadBtn: document.getElementById("confirm-guard-upload-btn"),

    apiGuideModal: document.getElementById("api-guide-modal"),
    closeApiGuideModalBtn: document.getElementById("close-api-guide-modal"),
    snippetLangCurl: document.getElementById("snippet-lang-curl"),
    snippetLangPython: document.getElementById("snippet-lang-python"),
    apiSnippetsContainer: document.getElementById("api-snippets-container"),
  };

  // --- Utilities ---
  function escapeHtml(str) {
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function renderHighlightedChunk(rawHtml) {
    const escaped = escapeHtml(rawHtml);
    return escaped
      .replace(/&lt;mark&gt;/gi, "<mark>")
      .replace(/&lt;\/mark&gt;/gi, "</mark>");
  }

  function renderSafeMarkdown(rawText, { allowHighlightTags = false } = {}) {
    const text = String(rawText ?? "");
    if (
      typeof window !== "undefined" &&
      window.marked &&
      typeof window.marked.parse === "function" &&
      window.DOMPurify &&
      typeof window.DOMPurify.sanitize === "function"
    ) {
      const parsedHtml = window.marked.parse(text, { gfm: true, breaks: true });
      const sanitizeConfig = allowHighlightTags
        ? {
            ADD_TAGS: ["mark", "span"],
            ADD_ATTR: ["id", "class", "title", "data-quote"],
          }
        : undefined;
      return window.DOMPurify.sanitize(parsedHtml, sanitizeConfig);
    }
    return `<div class="markdown-fallback">${escapeHtml(text)}</div>`;
  }

  function parseChunkOrder(chunkId) {
    const id = String(chunkId || "");
    const match = id.match(/^(.*?):(chunk|table):(\d+)$/);
    if (match) {
      return {
        docId: match[1],
        kindRank: match[2] === "chunk" ? 0 : 1,
        index: Number.parseInt(match[3], 10),
        raw: id,
      };
    }
    return { docId: id, kindRank: 2, index: 0, raw: id };
  }

  function compareByChunkOrder(aId, bId) {
    const a = parseChunkOrder(aId);
    const b = parseChunkOrder(bId);
    if (a.docId !== b.docId) return a.docId.localeCompare(b.docId);
    if (a.kindRank !== b.kindRank) return a.kindRank - b.kindRank;
    if (a.index !== b.index) return a.index - b.index;
    return a.raw.localeCompare(b.raw);
  }

  function getSessionApiKey() {
    return (sessionStorage.getItem(API_KEY_STORAGE_KEY) || "").trim();
  }

  async function apiFetch(url, options = {}) {
    const headers = new Headers(options.headers || {});
    const sessionKey = getSessionApiKey();
    if (sessionKey) {
      headers.set("X-OpenAI-API-Key", sessionKey);
    }
    return fetch(url, { ...options, headers });
  }

  async function parseErrorDetail(response) {
    try {
      const data = await response.json();
      if (typeof data?.detail === "string") return data.detail;
      if (data?.detail) return JSON.stringify(data.detail);
      if (data?.error) return String(data.error);
    } catch {
      // Fallback to plain text
    }
    try {
      const text = await response.text();
      if (text) return text;
    } catch {
      // Ignore
    }
    return `HTTP ${response.status} ${response.statusText}`;
  }

  function showNoticeModal(title, message) {
    el.noticeModalTitle.textContent = title;
    el.noticeModalMessage.textContent = message;
    el.noticeModal.classList.remove("hidden");
  }

  function hideNoticeModal() {
    el.noticeModal.classList.add("hidden");
  }

  // --- API Key Gate & Status ---
  function updateApiKeyStatusUI() {
    const sessionKey = getSessionApiKey();
    el.apiKeyStatusDot.classList.remove("ok", "warn", "err");
    if (sessionKey) {
      el.apiKeyStatusDot.classList.add("ok");
      el.apiKeyStatusLabel.textContent = "API Key: Session";
    } else if (state.hasServerApiKey) {
      el.apiKeyStatusDot.classList.add("ok");
      el.apiKeyStatusLabel.textContent = "API Key: Server";
    } else {
      el.apiKeyStatusDot.classList.add("err");
      el.apiKeyStatusLabel.textContent = "API Key: Missing";
    }
  }

  function openApiKeyModal(blocking = false) {
    el.apiKeyInput.value = getSessionApiKey();
    el.closeApiKeyModalBtn.classList.toggle("hidden", blocking);
    el.clearApiKeyBtn.classList.toggle("hidden", !getSessionApiKey());
    el.apiKeyModal.dataset.blocking = blocking ? "true" : "false";
    el.apiKeyModal.classList.remove("hidden");
    el.apiKeyInput.focus();
  }

  function closeApiKeyModal() {
    if (el.apiKeyModal.dataset.blocking === "true") return;
    el.apiKeyModal.classList.add("hidden");
  }

  function updateGuardStatusUI(reachable) {
    state.guardReachable = typeof reachable === "boolean" ? reachable : null;
    const isOffline = reachable === false;

    if (el.guardStatusPill && el.guardStatusLabel) {
      el.guardStatusPill.classList.remove("ok", "unverified");
      if (reachable === true) {
        el.guardStatusPill.classList.add("ok");
        el.guardStatusLabel.textContent = "Prompt Guard: Active";
      } else if (isOffline) {
        el.guardStatusPill.classList.add("unverified");
        el.guardStatusLabel.textContent = "Inputs are not verified";
      }
    }

    if (el.chatGuardNote) {
      el.chatGuardNote.classList.toggle("hidden", !isOffline);
    }
    if (el.chatForm) {
      el.chatForm.classList.toggle("unverified", isOffline);
    }
    if (el.webSearchToggle) {
      el.webSearchToggle.disabled = false;
      el.webSearchToggle.removeAttribute("aria-disabled");
    }
    if (el.webSearchSwitch) {
      el.webSearchSwitch.classList.remove("is-disabled");
      el.webSearchSwitch.title = "Enable web fallback on allowlisted medical websites";
    }
  }

  function syncWebSearchStageState() {
    if (!el.stageWebSearch || !el.webEvidenceStack) return;
    if (
      el.stageWebSearch.classList.contains("active") ||
      el.stageWebSearch.classList.contains("done")
    ) {
      return;
    }
    const webEnabled = Boolean(el.webSearchToggle && el.webSearchToggle.checked);
    el.stageWebSearch.classList.toggle("disabled", !webEnabled);
    el.webEvidenceStack.innerHTML = webEnabled
      ? '<span class="evidence-empty">Waiting</span>'
      : '<span class="evidence-empty">Disabled</span>';
  }

  function openWebSearchModal() {
    if (!el.webSearchModal) return;
    el.webSearchModal.classList.remove("hidden");
    if (el.confirmWebSearchBtn) el.confirmWebSearchBtn.focus();
  }

  function closeWebSearchModal(confirmed = false) {
    if (el.webSearchModal) {
      el.webSearchModal.classList.add("hidden");
    }
    if (el.webSearchToggle) {
      el.webSearchToggle.checked = Boolean(confirmed);
    }
    syncWebSearchStageState();
  }

  const FROZEN_SHOWCASE_NOTE =
    "The German diabetes guideline and Metformin Label are frozen and cannot be removed for showcase purposes.";

  function scheduleGuardHealthPoll(pollSeconds) {
    if (state.guardPollTimer !== null) {
      clearInterval(state.guardPollTimer);
      state.guardPollTimer = null;
    }
    if (pollSeconds > 0) {
      state.guardPollTimer = setInterval(() => {
        checkSystemStatus();
      }, pollSeconds * 1000);
    }
  }

  async function checkSystemStatus() {
    let guardReachable = false;
    let pollSeconds = 30;
    try {
      const res = await apiFetch("/v1/system/status");
      if (res.ok) {
        const data = await res.json();
        state.hasServerApiKey = Boolean(data.has_openai_api_key);
        guardReachable = Boolean(data.prompt_guard_reachable);
        pollSeconds = Number.isFinite(data.prompt_guard_poll_seconds)
          ? Number(data.prompt_guard_poll_seconds)
          : 30;
        if (typeof data.auto_prune_non_frozen_hours === "number" && el.docShowcaseNote) {
          el.docShowcaseNote.textContent =
            data.auto_prune_non_frozen_hours > 0
              ? `${FROZEN_SHOWCASE_NOTE} Newly uploaded documents are automatically removed after ${data.auto_prune_non_frozen_hours} hours.`
              : FROZEN_SHOWCASE_NOTE;
        }
      }
    } catch {
      // If status endpoint is unreachable, fall back to checking sessionStorage
      state.hasServerApiKey = false;
      guardReachable = false;
    }
    updateGuardStatusUI(guardReachable);
    updateApiKeyStatusUI();
    scheduleGuardHealthPoll(pollSeconds);
    if (!state.hasServerApiKey && !getSessionApiKey()) {
      openApiKeyModal(true);
    }
  }

  // --- Documents Sidebar ---
  async function loadDocuments() {
    try {
      const res = await apiFetch("/v1/documents");
      if (!res.ok) throw new Error(await parseErrorDetail(res));
      state.documents = await res.json();
      if (state.selectedDocId) {
        const stillExists = state.documents.some((d) => d.doc_id === state.selectedDocId);
        if (!stillExists) {
          state.selectedDocId = null;
        }
      }
      renderDocumentList();
    } catch (err) {
      el.docList.innerHTML = `<div class="empty-state">Failed to load documents: ${escapeHtml(err.message)}</div>`;
    }
  }

  function buildAccordionBody(doc) {
    const meta = doc.metadata || {};
    const conditions = Array.isArray(meta.conditions) && meta.conditions.length
      ? meta.conditions.join(", ")
      : "—";
    const chunkCount = Array.isArray(doc.chunk_ids) ? doc.chunk_ids.length : 0;
    const tableCount = Array.isArray(doc.tables) ? doc.tables.length : 0;

    const rows = [
      ["doc_id", doc.doc_id],
      ["slug", doc.slug || "—"],
      ["doc_type", meta.doc_type || "—"],
      ["source_type", meta.source_type || "—"],
      ["restriction", meta.local_restriction || "none"],
      ["conditions", conditions],
      ["chunks", String(chunkCount)],
      ["tables", String(tableCount)],
    ];

    const body = document.createElement("div");
    body.className = "doc-accordion-body";
    body.addEventListener("click", (e) => e.stopPropagation());

    const dl = document.createElement("dl");
    dl.className = "metadata-grid";
    dl.innerHTML = rows
      .map(([k, v]) => `<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(v)}</dd>`)
      .join("");
    body.appendChild(dl);

    const actionsWrap = document.createElement("div");
    actionsWrap.className = "doc-accordion-actions";

    const viewBtn = document.createElement("button");
    viewBtn.type = "button";
    viewBtn.className = "btn btn-secondary btn-block doc-view-btn";
    viewBtn.textContent = "View Document";
    viewBtn.title = "Open full document content";
    viewBtn.addEventListener("click", () => {
      openDocumentViewer(doc.doc_id);
    });
    actionsWrap.appendChild(viewBtn);

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "btn btn-danger btn-block doc-remove-btn";
    removeBtn.textContent = "Remove Document";

    if (meta.is_frozen) {
      removeBtn.disabled = true;
      removeBtn.title = "Frozen example document cannot be removed";
    } else {
      removeBtn.title = "Delete this document and its graph entities";
      removeBtn.addEventListener("mouseenter", () => {
        highlightGraphByDocIdRed(doc.doc_id);
      });
      removeBtn.addEventListener("mouseleave", () => {
        restorePreviousGraphFocus();
      });
      removeBtn.addEventListener("click", () => {
        handleRemoveDocument(doc.doc_id, removeBtn);
      });
    }

    actionsWrap.appendChild(removeBtn);
    body.appendChild(actionsWrap);
    return body;
  }

  function renderDocumentList() {
    el.docCountBadge.textContent = String(state.documents.length);
    el.docList.innerHTML = "";

    if (state.documents.length === 0 && state.pendingJobs.size === 0) {
      el.docList.innerHTML = '<div class="empty-state">No documents ingested yet.</div>';
      return;
    }

    for (const doc of state.documents) {
      const item = document.createElement("div");
      item.className = "doc-item";
      item.dataset.docId = doc.doc_id;
      const isSelected = doc.doc_id === state.selectedDocId;
      if (isSelected) item.classList.add("active");
      if (state.relatedDocIds.has(doc.doc_id)) item.classList.add("related");

      const meta = doc.metadata || {};
      const badges = [];
      if (meta.is_frozen) {
        badges.push('<span class="badge badge-frozen">Frozen Example</span>');
      }
      if (meta.doc_type) {
        badges.push(`<span class="badge">${escapeHtml(meta.doc_type)}</span>`);
      }
      if (meta.local_restriction && meta.local_restriction !== "none") {
        badges.push(`<span class="badge badge-region">${escapeHtml(meta.local_restriction)}</span>`);
      }

      const header = document.createElement("div");
      header.className = "doc-item-header";
      header.innerHTML = `
        <div class="doc-item-title">${escapeHtml(meta.title || doc.slug || doc.doc_id)}</div>
        <div class="doc-item-badges">${badges.join("")}</div>
      `;
      header.addEventListener("click", () => {
        if (state.selectedDocId === doc.doc_id) {
          clearSelectedDocument();
          resetGraphFocus();
        } else {
          selectDocument(doc.doc_id);
        }
      });
      item.appendChild(header);

      if (isSelected) {
        item.appendChild(buildAccordionBody(doc));
      }

      el.docList.appendChild(item);
    }

    // Pending upload jobs at bottom
    for (const job of state.pendingJobs.values()) {
      const pendingEl = document.createElement("div");
      pendingEl.className = "doc-item doc-item-pending";
      pendingEl.dataset.jobId = job.job_id;
      pendingEl.innerHTML = `
        <div class="pending-row">
          <span class="spinner" aria-hidden="true"></span>
          <div class="doc-item-title">${escapeHtml(job.title || "Uploading document...")}</div>
        </div>
        <div class="pending-step">${escapeHtml(job.current_step || "Queued...")}</div>
      `;
      el.docList.appendChild(pendingEl);
    }
  }

  function updateDocumentListHighlights() {
    const items = el.docList.querySelectorAll(".doc-item[data-doc-id]");
    items.forEach((item) => {
      const id = item.dataset.docId;
      item.classList.toggle("active", id === state.selectedDocId);
      item.classList.toggle("related", state.relatedDocIds.has(id));
    });
  }

  function setDocsCollapsed(collapsed) {
    const isCollapsed = Boolean(collapsed);
    if (el.sidebarLeft) {
      el.sidebarLeft.classList.toggle("collapsed", isCollapsed);
    }
    if (el.toggleDocsBtn) {
      el.toggleDocsBtn.setAttribute("aria-expanded", String(!isCollapsed));
      const label = isCollapsed ? "Expand document list" : "Collapse document list";
      el.toggleDocsBtn.setAttribute("title", label);
      el.toggleDocsBtn.setAttribute("aria-label", label);
    }
  }

  function scrollDocItemIntoView(docId) {
    if (!el.docList || !docId) return;
    const item = el.docList.querySelector(`.doc-item[data-doc-id="${CSS.escape(docId)}"]`);
    if (item) {
      item.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }

  function selectDocument(docId) {
    const doc = state.documents.find((d) => d.doc_id === docId);
    if (!doc) return;
    clearNodeTypeFilter();
    state.selectedDocId = docId;
    state.relatedDocIds.clear();
    renderDocumentList();
    closeInspector();
    highlightGraphByDocIds([docId], { fit: true });
  }

  function clearSelectedDocument() {
    state.selectedDocId = null;
    renderDocumentList();
  }

  async function handleRemoveDocument(docId, removeBtn) {
    if (!docId) return;
    const doc = state.documents.find((d) => d.doc_id === docId);
    if (doc?.metadata?.is_frozen) return;

    if (removeBtn) {
      removeBtn.disabled = true;
      removeBtn.textContent = "Removing...";
    }
    try {
      const res = await apiFetch(`/v1/documents/${encodeURIComponent(docId)}`, {
        method: "DELETE",
      });
      if (!res.ok) throw new Error(await parseErrorDetail(res));

      state.selectedDocId = null;
      resetGraphFocus();
      await Promise.all([loadDocuments(), loadGraph()]);
    } catch (err) {
      showNoticeModal("Removal Failed", err.message);
      if (removeBtn) {
        removeBtn.disabled = false;
        removeBtn.textContent = "Remove Document";
      }
    }
  }

  // --- Upload & Job Polling ---
  function openUploadModal() {
    el.uploadForm.reset();
    el.uploadModal.classList.remove("hidden");
  }

  function closeUploadModal() {
    el.uploadModal.classList.add("hidden");
  }

  function openGuardUploadConfirmModal() {
    if (!el.guardUploadConfirmModal) return;
    el.guardUploadConfirmModal.classList.remove("hidden");
  }

  function closeGuardUploadConfirmModal() {
    if (!el.guardUploadConfirmModal) return;
    el.guardUploadConfirmModal.classList.add("hidden");
  }

  async function performUpload() {
    const file = el.uploadFile.files?.[0];
    if (!file) return;

    const title = el.uploadTitle.value.trim() || file.name.replace(/\.[^.]+$/, "");
    const restriction = el.uploadRestriction.value || "none";
    const description = el.uploadDescription.value.trim();

    const formData = new FormData();
    formData.append("file", file);
    formData.append("title", title);
    formData.append("local_restriction", restriction);
    if (description) formData.append("description", description);
    formData.append("classify", "true");
    formData.append("extract", "true");
    formData.append("async_extract", "true");

    el.submitUploadBtn.disabled = true;
    el.submitUploadBtn.textContent = "Uploading...";

    try {
      const res = await apiFetch(
        "/v1/documents/upload?classify=true&extract=true&async_extract=true",
        { method: "POST", body: formData }
      );

      if (!res.ok) {
        throw new Error(await parseErrorDetail(res));
      }

      const payload = await res.json();
      closeUploadModal();

      if (payload.job_id) {
        startJobPolling(payload.job_id, title, payload.current_step || "Queued...");
      } else if (payload.doc_id) {
        // Synchronous fallback if backend returned DocumentRecord directly
        showNoticeModal("Upload Complete", `Document "${title}" ingested successfully.`);
        await Promise.all([loadDocuments(), loadGraph()]);
        selectDocument(payload.doc_id);
      }
    } catch (err) {
      showNoticeModal("Upload Failed", err.message);
    } finally {
      el.submitUploadBtn.disabled = false;
      el.submitUploadBtn.textContent = "Upload & Extract";
    }
  }

  async function handleUploadSubmit(event) {
    event.preventDefault();
    const file = el.uploadFile.files?.[0];
    if (!file) return;

    if (state.guardReachable === false) {
      openGuardUploadConfirmModal();
      return;
    }

    await performUpload();
  }

  function startJobPolling(jobId, title, initialStep) {
    const jobState = {
      job_id: jobId,
      title,
      current_step: initialStep,
      timerId: null,
    };
    state.pendingJobs.set(jobId, jobState);
    renderDocumentList();

    const poll = async () => {
      try {
        const res = await apiFetch(`/v1/documents/jobs/${encodeURIComponent(jobId)}`);
        if (!res.ok) throw new Error(await parseErrorDetail(res));
        const job = await res.json();

        if (job.status === "completed") {
          clearInterval(jobState.timerId);
          state.pendingJobs.delete(jobId);
          await Promise.all([loadDocuments(), loadGraph()]);
          if (job.doc_id) {
            selectDocument(job.doc_id);
          }
          showNoticeModal(
            "Ingestion Complete",
            `"${job.title || title}" has been classified, extracted, and added to the knowledge graph.`
          );
          return;
        }

        if (job.status === "failed") {
          clearInterval(jobState.timerId);
          state.pendingJobs.delete(jobId);
          renderDocumentList();
          showNoticeModal("Ingestion Rejected / Failed", job.error || "Document ingestion failed.");
          return;
        }

        jobState.current_step = job.current_step || job.status || "Processing...";
        const stepEl = el.docList.querySelector(
          `.doc-item-pending[data-job-id="${CSS.escape(jobId)}"] .pending-step`
        );
        if (stepEl) {
          stepEl.textContent = jobState.current_step;
        }
      } catch (err) {
        clearInterval(jobState.timerId);
        state.pendingJobs.delete(jobId);
        renderDocumentList();
        showNoticeModal("Job Polling Error", err.message);
      }
    };

    jobState.timerId = setInterval(poll, 1000);
  }

  // --- Cytoscape Graph ---
  function normalizeGraphPayload(raw) {
    const rawNodes = Array.isArray(raw?.nodes) ? raw.nodes : [];
    const rawEdges = Array.isArray(raw?.edges) ? raw.edges : [];

    const nodes = rawNodes.map((item) => {
      const n = item?.data ? item.data : item;
      const nodeId = n.node_id || n.id;
      const nodeType = (n.node_type || "entity").toLowerCase();
      return {
        group: "nodes",
        data: {
          id: nodeId,
          node_id: nodeId,
          canonical_name: n.canonical_name || n.label || nodeId,
          node_type: nodeType,
          color: (NODE_COLORS[nodeType] || NODE_COLORS.entity).fill,
          aliases: Array.isArray(n.aliases) ? n.aliases : [],
          description: n.description || "",
          doc_ids: Array.isArray(n.doc_ids) ? n.doc_ids : [],
        },
      };
    });

    const edges = rawEdges.map((item, idx) => {
      const e = item?.data ? item.data : item;
      const edgeId = e.id || e.edge_id || `${e.source}__${e.relation || "rel"}__${e.target}__${idx}`;
      const quote =
        e.quote ||
        (Array.isArray(e.evidence) && e.evidence.length ? e.evidence[0].quote : "") ||
        e.relationship_description ||
        "";
      return {
        group: "edges",
        data: {
          id: edgeId,
          source: e.source,
          target: e.target,
          relation: e.relation || "RELATED_TO",
          relationship_description: e.relationship_description || "",
          quote,
          doc_ids: Array.isArray(e.doc_ids) ? e.doc_ids : [],
        },
      };
    });

    return { nodes, edges };
  }

  async function loadGraph() {
    try {
      const res = await apiFetch("/v1/graph");
      if (!res.ok) throw new Error(await parseErrorDetail(res));
      const raw = await res.json();
      state.graphData = normalizeGraphPayload(raw);
      renderCytoscape(state.graphData);
    } catch (err) {
      el.graphEmptyOverlay.classList.remove("hidden");
      el.graphEmptyOverlay.innerHTML = `<p>Failed to load graph: ${escapeHtml(err.message)}</p>`;
    }
  }

  function renderCytoscape({ nodes, edges }) {
    const hasElements = nodes.length > 0;
    el.graphEmptyOverlay.classList.toggle("hidden", hasElements);

    if (state.cy) {
      state.cy.destroy();
      state.cy = null;
    }

    if (typeof window.cytoscape !== "function") {
      el.graphEmptyOverlay.classList.remove("hidden");
      el.graphEmptyOverlay.innerHTML = "<p>Cytoscape.js failed to load.</p>";
      return;
    }

    state.cy = window.cytoscape({
      container: el.cyContainer,
      elements: [...nodes, ...edges],
      minZoom: 0.1,
      maxZoom: 8,
      wheelSensitivity: 0.25,
      style: [
        {
          selector: "node",
          style: {
            "background-color": "#94a3b8",
            label: "data(canonical_name)",
            "font-size": 10,
            "font-weight": 600,
            color: "#334155",
            "text-opacity": 0,
            "text-valign": "bottom",
            "text-margin-y": 5,
            "text-wrap": "ellipsis",
            "text-max-width": 110,
            width: 10,
            height: 10,
            "border-width": 1,
            "border-color": "#64748b",
            "transition-property":
              "opacity, border-color, border-width, background-color, text-opacity",
            "transition-duration": "0.15s",
          },
        },
        {
          selector: "edge",
          style: {
            width: 0.8,
            "line-color": "#e2e8f0",
            "target-arrow-color": "#cbd5e1",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            "arrow-scale": 0.6,
            "text-opacity": 0,
            "transition-property":
              "opacity, line-color, target-arrow-color, width, text-opacity",
            "transition-duration": "0.15s",
          },
        },
        {
          selector: ".dimmed",
          style: {
            opacity: 0.2,
          },
        },
        {
          selector: "node.highlighted",
          style: {
            opacity: 1,
            "background-color": "#cbd5e1",
            "border-width": 2,
            "border-color": "#64748b",
          },
        },
        ...Object.entries(NODE_COLORS)
          .filter(([type]) => type !== "entity")
          .map(([type, palette]) => ({
            selector: `node[node_type = "${type}"].highlighted`,
            style: {
              opacity: 1,
              "background-color": palette.fill,
              "border-color": palette.border,
              "border-width": 2,
            },
          })),
        {
          selector: "edge.highlighted",
          style: {
            opacity: 1,
            width: 1.4,
            "line-color": "#64748b",
            "target-arrow-color": "#94a3b8",
          },
        },
        {
          selector: "node.show-label",
          style: {
            "text-opacity": 1,
          },
        },
        {
          selector: "edge.show-label",
          style: {
            label: "data(relation)",
            "text-opacity": 1,
            "font-size": 10,
            color: "#334155",
            "text-background-color": "#ffffff",
            "text-background-opacity": 0.85,
            "text-background-padding": 2,
          },
        },
        {
          selector: "node.cursor-nearby",
          style: {
            "background-color": "#bbf7d0",
            "border-color": "#4ade80",
            "text-opacity": 1,
            "font-size": "9px",
          },
        },
        {
          selector: "edge.cursor-nearby",
          style: {
            "line-color": "#86efac",
            "target-arrow-color": "#4ade80",
          },
        },
        {
          selector: "node.danger-highlight",
          style: {
            opacity: 1,
            "background-color": "#e2e8f0",
            "border-color": "#dc2626",
            "border-width": 2.5,
          },
        },
        {
          selector: "edge.danger-highlight",
          style: {
            opacity: 1,
            width: 2,
            "line-color": "#dc2626",
            "target-arrow-color": "#dc2626",
          },
        },
      ],
      layout: hasElements
        ? {
            name: "cose",
            animate: false,
            padding: 48,
            nodeRepulsion: 55000,
            idealEdgeLength: 155,
            nodeOverlap: 32,
            gravity: 0.2,
            numIter: 1500,
          }
        : { name: "grid" },
    });

    bindCytoscapeEvents(state.cy);
    if (hasElements) {
      fitGraphToVisibleArea(state.cy);
      state.baseFitZoom = state.cy.zoom();
    } else {
      state.baseFitZoom = 1;
    }
  }

  function fitGraphToVisibleArea(cy, eles = null, { animate = false, complete = null } = {}) {
    if (!cy) return;
    const target = eles && eles.length ? eles : cy.elements();
    if (!target || target.length === 0) return;

    const bb = target.boundingBox();
    if (!bb || bb.w <= 0 || bb.h <= 0) {
      if (animate) {
        cy.animate({
          fit: { eles: target, padding: 36 },
          duration: 280,
          ...(complete ? { complete } : {}),
        });
      } else {
        cy.fit(target, 36);
      }
      return;
    }

    const isWide = window.innerWidth > 1024;
    const padLeft = isWide ? 330 : 36;
    const padRight = isWide ? 420 : 36;
    const padTop = isWide ? 105 : 76;
    const padBottom =
      isWide && el.pipelineBar && !el.pipelineBar.classList.contains("hidden") ? 155 : 36;

    const availW = Math.max(80, cy.width() - padLeft - padRight);
    const availH = Math.max(80, cy.height() - padTop - padBottom);
    const rawZoom = Math.min(availW / bb.w, availH / bb.h) * 0.9;
    const zoom = Math.max(cy.minZoom(), Math.min(cy.maxZoom(), rawZoom));

    const bbCenterX = bb.x1 + bb.w / 2;
    const bbCenterY = bb.y1 + bb.h / 2;
    const targetCenterX = padLeft + availW / 2;
    const targetCenterY = padTop + availH / 2;

    const pan = {
      x: targetCenterX - bbCenterX * zoom,
      y: targetCenterY - bbCenterY * zoom,
    };

    if (animate) {
      cy.animate({ zoom, pan, duration: 280, ...(complete ? { complete } : {}) });
    } else {
      cy.viewport({ zoom, pan });
    }
  }

  function getCursorRevealMinZoom(cy) {
    return (state.baseFitZoom || 1) * Math.pow(1.25, 6) * 0.99;
  }

  function updateDynamicLabelScale(cy) {
    if (!cy) return;
    const zoom = Math.max(cy.zoom(), 0.05);
    if (state.lastLabelZoom === zoom) return;
    state.lastLabelZoom = zoom;
    cy.style()
      .selector("node.cursor-nearby, node.show-label")
      .style({
        "font-size": 12 / zoom,
        "text-margin-y": 6 / zoom,
        "text-max-width": 130 / zoom,
      })
      .selector("edge.show-label")
      .style({
        "font-size": 10 / zoom,
        "text-background-padding": 2 / zoom,
      })
      .update();
  }

  function updateCursorNearby(cy) {
    if (!cy) return;
    const zoom = cy.zoom();
    if (zoom < getCursorRevealMinZoom(cy) || !state.cursorModelPos) {
      cy.elements(".cursor-nearby").removeClass("cursor-nearby");
      return;
    }

    updateDynamicLabelScale(cy);

    const { x: cx, y: cyPos } = state.cursorModelPos;
    const radius = CURSOR_REVEAL_RADIUS_PX / zoom;
    const radiusSq = radius * radius;

    const nearbyNodes = cy.nodes().filter((node) => {
      const pos = node.position();
      const dx = pos.x - cx;
      const dy = pos.y - cyPos;
      return dx * dx + dy * dy <= radiusSq;
    });

    const nearbyEdges = nearbyNodes.connectedEdges();
    const nextNearby = nearbyNodes.union(nearbyEdges);

    cy.batch(() => {
      cy.elements(".cursor-nearby").difference(nextNearby).removeClass("cursor-nearby");
      nextNearby.difference(cy.elements(".cursor-nearby")).addClass("cursor-nearby");
    });
  }

  function clearCursorReveal(cy) {
    state.cursorModelPos = null;
    state.cursorRenderedPos = null;
    if (cy) {
      cy.elements(".cursor-nearby").removeClass("cursor-nearby");
    }
  }

  function bindCytoscapeEvents(cy) {
    state.lastLabelZoom = null;
    clearCursorReveal(cy);

    cy.on("mousemove", (evt) => {
      if (evt.renderedPosition) {
        state.cursorRenderedPos = {
          x: evt.renderedPosition.x,
          y: evt.renderedPosition.y,
        };
      }
      if (evt.position && Number.isFinite(evt.position.x) && Number.isFinite(evt.position.y)) {
        state.cursorModelPos = { x: evt.position.x, y: evt.position.y };
        updateCursorNearby(cy);
      }
    });

    cy.on("zoom", () => {
      if (state.cursorRenderedPos) {
        const pan = cy.pan();
        const zoom = cy.zoom();
        state.cursorModelPos = {
          x: (state.cursorRenderedPos.x - pan.x) / zoom,
          y: (state.cursorRenderedPos.y - pan.y) / zoom,
        };
      }
      if (cy.zoom() >= getCursorRevealMinZoom(cy) || cy.elements(".show-label").length > 0) {
        updateDynamicLabelScale(cy);
      }
      updateCursorNearby(cy);
    });

    cy.on("mouseout", (evt) => {
      if (evt.target === cy) {
        clearCursorReveal(cy);
      }
    });

    if (!state.cyContainerMouseLeaveBound && el.cyContainer) {
      state.cyContainerMouseLeaveBound = true;
      el.cyContainer.addEventListener("mouseleave", () => {
        clearCursorReveal(state.cy);
      });
    }

    cy.on("mouseover", "node", (evt) => {
      const target = evt.target;
      el.graphTooltip.textContent = `${target.data("canonical_name")} (${target.data("node_type")})`;
      el.graphTooltip.classList.remove("hidden");
      positionTooltip(evt.originalEvent);
    });

    cy.on("mousemove", "node", (evt) => {
      positionTooltip(evt.originalEvent);
    });

    cy.on("mouseout", "node", () => {
      el.graphTooltip.classList.add("hidden");
    });

    cy.on("tap", "node", (evt) => {
      const node = evt.target;
      focusElementNeighborhood(node);
    });

    cy.on("tap", "edge", (evt) => {
      const edge = evt.target;
      focusElementNeighborhood(edge);
    });

    cy.on("tap", (evt) => {
      if (evt.target === cy) {
        resetGraphFocus({ fit: false });
      }
    });
  }

  function positionTooltip(mouseEvent) {
    if (!mouseEvent) return;
    const rect = el.cyContainer.getBoundingClientRect();
    const x = mouseEvent.clientX - rect.left + 12;
    const y = mouseEvent.clientY - rect.top - 10;
    el.graphTooltip.style.left = `${Math.max(8, Math.min(x, rect.width - 180))}px`;
    el.graphTooltip.style.top = `${Math.max(8, Math.min(y, rect.height - 34))}px`;
  }

  function clearGraphClasses() {
    if (!state.cy) return;
    state.cy.elements().removeClass("dimmed highlighted danger-highlight show-label");
  }

  function clearNodeTypeFilter() {
    state.activeNodeTypeFilter = null;
    if (el.nodeLegend) {
      el.nodeLegend.querySelectorAll(".legend-item.active").forEach((btn) => {
        btn.classList.remove("active");
      });
    }
  }

  function resetGraphFocus({ fit = true } = {}) {
    clearGraphClasses();
    clearNodeTypeFilter();
    state.activeHighlightMode = null;
    state.relatedDocIds.clear();
    updateDocumentListHighlights();
    closeInspector();
    if (fit && state.cy && state.cy.elements().length > 0) {
      fitGraphToVisibleArea(state.cy);
      state.baseFitZoom = state.cy.zoom();
    }
  }

  function toggleNodeTypeFilter(nodeType) {
    if (!nodeType) return;
    if (state.activeNodeTypeFilter === nodeType) {
      resetGraphFocus();
      return;
    }

    state.selectedDocId = null;
    state.relatedDocIds.clear();
    renderDocumentList();
    closeInspector();

    state.activeNodeTypeFilter = nodeType;
    if (el.nodeLegend) {
      el.nodeLegend.querySelectorAll(".legend-item[data-node-type]").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.nodeType === nodeType);
      });
    }
    highlightGraphByNodeType(nodeType);
  }

  function highlightGraphByNodeType(nodeType) {
    if (!state.cy) return;
    clearGraphClasses();

    const targetType = nodeType.toLowerCase();
    const matchingNodes = state.cy.nodes().filter((node) => {
      return (node.data("node_type") || "").toLowerCase() === targetType;
    });

    state.cy.elements().difference(matchingNodes).addClass("dimmed");
    matchingNodes.addClass("highlighted");
    state.activeHighlightMode = { type: "nodeType", nodeType };
  }

  function focusElementNeighborhood(element) {
    if (!state.cy) return;
    clearGraphClasses();
    clearNodeTypeFilter();

    const neighborhood = element.isNode()
      ? element.closedNeighborhood()
      : element.union(element.connectedNodes());

    state.cy.elements().difference(neighborhood).addClass("dimmed");
    neighborhood.addClass("highlighted");

    const docIds = element.data("doc_ids") || [];
    state.relatedDocIds = new Set(docIds);
    updateDocumentListHighlights();
    renderInspector(element);

    state.activeHighlightMode = { type: "element", elementId: element.id() };
  }

  function highlightGraphByDocIds(docIds, { returnMatchedCount = false, fit = false } = {}) {
    if (!state.cy) return 0;
    clearGraphClasses();

    const docSet = new Set(docIds);
    const matching = state.cy.elements().filter((ele) => {
      const ids = ele.data("doc_ids") || [];
      return ids.some((id) => docSet.has(id));
    });

    if (matching.length > 0) {
      state.cy.elements().difference(matching).addClass("dimmed");
      matching.addClass("highlighted");
      state.activeHighlightMode = { type: "doc", docIds };
      if (fit) {
        fitGraphToVisibleArea(state.cy, matching, { animate: true });
      }
    }

    return returnMatchedCount ? matching.length : matching.length;
  }

  function highlightGraphByNodeIds(
    nodeIds,
    { transient = false, fit = false, showLabels = false } = {}
  ) {
    if (!state.cy) return;
    clearGraphClasses();

    const shouldShowLabels = Boolean(showLabels || fit);
    const idSet = new Set(nodeIds || []);
    const matchingNodes = state.cy.nodes().filter((node) => {
      return idSet.has(node.id()) || idSet.has(node.data("node_id"));
    });

    if (matchingNodes.length > 0) {
      const connectedEdges = matchingNodes.edgesWith(matchingNodes);
      const highlightSet = matchingNodes.union(connectedEdges);
      state.cy.elements().difference(highlightSet).addClass("dimmed");
      highlightSet.addClass("highlighted");
      if (shouldShowLabels) {
        highlightSet.addClass("show-label");
        updateDynamicLabelScale(state.cy);
      }
      if (!transient) {
        state.activeHighlightMode = { type: "nodes", nodeIds, showLabels: shouldShowLabels };
      }
      if (fit && highlightSet.length > 0) {
        fitGraphToVisibleArea(state.cy, highlightSet, {
          animate: true,
          complete: () => updateDynamicLabelScale(state.cy),
        });
      }
    }
  }

  function highlightGraphByDocIdRed(docId) {
    if (!state.cy || !docId) return;
    clearGraphClasses();

    const matching = state.cy.elements().filter((ele) => {
      const ids = ele.data("doc_ids") || [];
      return ids.includes(docId);
    });

    state.cy.elements().difference(matching).addClass("dimmed");
    matching.addClass("danger-highlight");
  }

  function restorePreviousGraphFocus() {
    if (state.selectedDocId) {
      highlightGraphByDocIds([state.selectedDocId]);
    } else if (state.activeHighlightMode?.type === "retrieval") {
      highlightRetrievalElements(
        state.activeHighlightMode.seedNodeIds,
        state.activeHighlightMode.highlightedEdgeIds
      );
    } else if (state.activeHighlightMode?.type === "nodes") {
      highlightGraphByNodeIds(state.activeHighlightMode.nodeIds, {
        showLabels: Boolean(state.activeHighlightMode.showLabels),
      });
    } else if (state.activeHighlightMode?.type === "nodeType") {
      highlightGraphByNodeType(state.activeHighlightMode.nodeType);
    } else {
      clearGraphClasses();
    }
  }

  function highlightRetrievalElements(seedNodeIds, highlightedEdgeIds) {
    if (!state.cy) return;
    clearGraphClasses();
    clearNodeTypeFilter();

    const seedSet = new Set(seedNodeIds || []);
    const edgeSet = new Set(highlightedEdgeIds || []);

    const matched = state.cy.elements().filter((ele) => {
      if (ele.isNode()) {
        return seedSet.has(ele.id()) || seedSet.has(ele.data("node_id"));
      }
      if (ele.isEdge()) {
        if (edgeSet.has(ele.id())) return true;
        return false;
      }
      return false;
    });

    if (matched.length > 0) {
      state.cy.elements().difference(matched).addClass("dimmed");
      matched.addClass("highlighted");
      state.activeHighlightMode = {
        type: "retrieval",
        seedNodeIds,
        highlightedEdgeIds,
      };
    }
  }

  // --- Inspector Bar ---
  function renderInspector(element) {
    const isNode = element.isNode();
    el.inspectorKind.textContent = isNode ? element.data("node_type") || "NODE" : "EDGE";
    el.inspectorTitle.textContent = isNode
      ? element.data("canonical_name")
      : `${element.data("source")} —[${element.data("relation")}]→ ${element.data("target")}`;

    const rawDocIds = element.data("doc_ids") || [];
    const docIdsHtml = rawDocIds.length
      ? rawDocIds
          .map(
            (id) =>
              `<button type="button" class="inspector-doc-chip" data-doc-id="${escapeHtml(id)}">${escapeHtml(id)}</button>`
          )
          .join("")
      : "—";
    const fields = [];

    if (isNode) {
      const aliases = (element.data("aliases") || []).join(", ") || "—";
      fields.push(["node_id", element.data("node_id")]);
      fields.push(["doc_ids", docIdsHtml, true]);
      fields.push(["aliases", aliases]);
      fields.push(["description", element.data("description") || "—"]);
    } else {
      fields.push(["edge id", element.id()]);
      fields.push(["doc_ids", docIdsHtml, true]);
      fields.push(["quote", element.data("quote") || element.data("relationship_description") || "—"]);
    }

    el.inspectorBody.innerHTML = fields
      .map(([label, val, isHtml]) => {
        const content = isHtml ? val : escapeHtml(val);
        return `<div class="inspector-field"><strong>${escapeHtml(label)}:</strong><span>${content}</span></div>`;
      })
      .join("");

    el.inspectorBody.querySelectorAll(".inspector-doc-chip").forEach((chip) => {
      const docId = chip.dataset.docId;
      chip.addEventListener("mouseenter", () => {
        setDocsCollapsed(false);
        state.relatedDocIds = new Set([docId]);
        updateDocumentListHighlights();
        scrollDocItemIntoView(docId);
      });
      chip.addEventListener("mouseleave", () => {
        state.relatedDocIds = new Set(element.data("doc_ids") || []);
        updateDocumentListHighlights();
      });
      chip.addEventListener("click", () => {
        setDocsCollapsed(false);
        selectDocument(docId);
        scrollDocItemIntoView(docId);
      });
    });

    el.inspectorBar.classList.remove("hidden");
  }

  function closeInspector() {
    el.inspectorBar.classList.add("hidden");
  }

  // --- Real-Time Pipeline Visualizer ---
  function resetPipelineVisualizer() {
    if (state.rewriteCollapseTimer) {
      clearTimeout(state.rewriteCollapseTimer);
      state.rewriteCollapseTimer = null;
    }

    el.pipelineBar.classList.remove("hidden");
    el.pipelineStatusText.textContent = "Starting...";

    el.stageRewrite.classList.remove("active", "done");
    el.stageRoute.classList.remove("active", "done");
    el.stageRetrieval.classList.remove("active", "done");
    el.stageSynthesis.classList.remove("active", "done");

    el.rewriteWrapper.classList.remove("expanded");
    el.rewriteCompactPill.classList.add("hidden");
    el.rewriteCompactPill.textContent = "0 queries";
    el.rewriteChips.innerHTML = "";

    el.modeIconRag.classList.remove("lit", "done", "pulse-highlight");
    el.modeIconGraph.classList.remove("lit", "done", "pulse-highlight");

    el.evidenceStack.innerHTML = "";

    if (el.stageWebSearch && el.webEvidenceStack) {
      el.stageWebSearch.classList.remove("active", "done");
      const webEnabled = Boolean(el.webSearchToggle && el.webSearchToggle.checked);
      el.stageWebSearch.classList.toggle("disabled", !webEnabled);
      el.webEvidenceStack.innerHTML = webEnabled
        ? '<span class="evidence-empty">Waiting</span>'
        : '<span class="evidence-empty">Disabled</span>';
    }

    el.synthStatusPill.classList.remove("synthesizing", "done");
    el.synthStatusPill.innerHTML = '<span class="synth-idle-text">Waiting</span>';

    if (el.stageLatency) {
      el.stageLatency.classList.add("hidden");
      el.stageLatency.classList.remove("done");
    }
    if (el.pipelineLatencyValue) {
      el.pipelineLatencyValue.textContent = "";
    }
    if (el.pipelineLatencyTooltip) {
      el.pipelineLatencyTooltip.innerHTML = "";
    }
  }

  function buildEvidenceItems(data, seedNodes) {
    if (Array.isArray(data.evidence_items) && data.evidence_items.length > 0) {
      return [...data.evidence_items].sort((a, b) => compareByChunkOrder(a.id, b.id));
    }
    const items = [];
    const chunkCount = data.chunk_count ?? 0;
    const tableCount = data.table_count ?? 0;
    for (let i = 0; i < chunkCount; i++) {
      items.push({ source: "rag", label: `RAG Chunk #${i + 1}`, node_ids: [] });
    }
    for (let i = 0; i < tableCount; i++) {
      items.push({ source: "table", label: `Table #${i + 1}`, node_ids: [] });
    }
    if (items.length === 0 && seedNodes.length > 0) {
      items.push({
        source: "graph",
        label: `Graph (${seedNodes.length} nodes)`,
        node_ids: seedNodes,
      });
    }
    return items;
  }

  function renderEvidenceStack(items) {
    el.evidenceStack.innerHTML = "";
    if (!items.length) {
      el.evidenceStack.innerHTML = '<span class="evidence-empty">None</span>';
      return;
    }

    const fileSvg = `
      <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true">
        <path d="M4.5 2.5h4.8L12.5 5.7V13a1 1 0 0 1-1 1h-7a1 1 0 0 1-1-1V3.5a1 1 0 0 1 1-1z" />
        <path d="M9 2.5V6h3.5" />
      </svg>
    `;

    items.forEach((item) => {
      const source = (item.source || "rag").toLowerCase();
      const label = item.label || `${source.toUpperCase()} evidence`;
      const nodeIds = Array.isArray(item.node_ids) ? item.node_ids : [];

      const iconBtn = document.createElement("button");
      iconBtn.type = "button";
      iconBtn.className = "evidence-file-icon";
      iconBtn.dataset.source = source;
      iconBtn.dataset.tooltip = label;
      iconBtn.setAttribute("aria-label", label);
      iconBtn.innerHTML = fileSvg;

      iconBtn.addEventListener("mouseenter", () => {
        if (source === "graph" && nodeIds.length > 0) {
          highlightGraphByNodeIds(nodeIds, { transient: true });
        } else if (source === "rag" || source === "table") {
          el.modeIconRag.classList.add("pulse-highlight");
        }
      });

      iconBtn.addEventListener("mouseleave", () => {
        el.modeIconRag.classList.remove("pulse-highlight");
        if (source === "graph" && nodeIds.length > 0) {
          restorePreviousGraphFocus();
        }
      });

      if (source === "graph" && nodeIds.length > 0) {
        iconBtn.addEventListener("click", () => {
          clearSelectedDocument();
          clearNodeTypeFilter();
          highlightGraphByNodeIds(nodeIds, { transient: false, fit: true, showLabels: true });
        });
      }

      el.evidenceStack.appendChild(iconBtn);
    });
  }

  function renderWebSearchStage(data) {
    if (!el.stageWebSearch || !el.webEvidenceStack) return;

    const enabled = Boolean(data.enabled);
    el.stageWebSearch.classList.toggle("disabled", !enabled);

    if (!enabled) {
      el.stageWebSearch.classList.remove("active", "done");
      el.webEvidenceStack.innerHTML = '<span class="evidence-empty">Disabled</span>';
      return;
    }

    if (data.status === "evaluating_sufficiency") {
      el.stageWebSearch.classList.add("active");
      el.stageWebSearch.classList.remove("done");
      el.pipelineStatusText.textContent = "Checking web fallback sufficiency...";
      el.webEvidenceStack.innerHTML = '<span class="evidence-empty">Checking...</span>';
      return;
    }

    el.stageWebSearch.classList.remove("active");
    el.stageWebSearch.classList.add("done");

    const sources = Array.isArray(data.web_sources) ? data.web_sources : [];
    el.webEvidenceStack.innerHTML = "";
    if (!sources.length) {
      const msg = data.sufficient_locally ? "Local sufficient" : "None";
      el.webEvidenceStack.innerHTML = `<span class="evidence-empty">${msg}</span>`;
      return;
    }

    const fileSvg = `
      <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true">
        <path d="M4.5 2.5h4.8L12.5 5.7V13a1 1 0 0 1-1 1h-7a1 1 0 0 1-1-1V3.5a1 1 0 0 1 1-1z" />
        <path d="M9 2.5V6h3.5" />
      </svg>
    `;

    sources.forEach((src) => {
      const isPdf = (src.content_type || "").toLowerCase() === "pdf";
      const title = src.title || src.domain || "Web Source";
      const url = src.url || "";
      const tooltip = url ? `${title} — ${url}` : title;

      const iconBtn = document.createElement("button");
      iconBtn.type = "button";
      iconBtn.className = "evidence-file-icon";
      iconBtn.dataset.source = isPdf ? "web-pdf" : "web";
      iconBtn.dataset.tooltip = tooltip;
      iconBtn.title = tooltip;
      iconBtn.setAttribute("aria-label", tooltip);
      iconBtn.innerHTML = fileSvg;

      if (url) {
        iconBtn.addEventListener("click", () => {
          window.open(url, "_blank", "noopener,noreferrer");
        });
      }
      el.webEvidenceStack.appendChild(iconBtn);
    });
  }

  function handlePipelineEvent(eventType, data) {
    if (eventType === "guard") {
      if (typeof data.reachable === "boolean") {
        updateGuardStatusUI(data.reachable);
      }
      return;
    }

    if (eventType === "rewrite") {
      el.pipelineStatusText.textContent = "Rewriting & routing query...";
      el.stageRewrite.classList.remove("active");
      el.stageRewrite.classList.add("done");
      el.stageRoute.classList.add("active");

      const queries = data.search_queries || data.queries || [];
      el.rewriteChips.innerHTML = queries
        .map((q) => `<span class="query-chip">${escapeHtml(q)}</span>`)
        .join("");

      const qCount = queries.length;
      el.rewriteCompactPill.textContent = `${qCount} ${qCount === 1 ? "query" : "queries"}`;
      el.rewriteCompactPill.classList.remove("hidden");

      if (qCount > 0) {
        el.rewriteWrapper.classList.add("expanded");
        if (state.rewriteCollapseTimer) {
          clearTimeout(state.rewriteCollapseTimer);
        }
        state.rewriteCollapseTimer = setTimeout(() => {
          el.rewriteWrapper.classList.remove("expanded");
          state.rewriteCollapseTimer = null;
        }, 2500);
      }

      const mode = (data.chosen_mode || data.mode || "hybrid").toLowerCase();
      el.modeIconRag.classList.add("lit");
      if (mode !== "rag") {
        el.modeIconGraph.classList.add("lit");
      } else {
        el.modeIconGraph.classList.remove("lit");
      }
      return;
    }

    if (eventType === "retrieval") {
      el.stageRoute.classList.remove("active");
      el.stageRoute.classList.add("done");
      if (el.modeIconRag.classList.contains("lit")) el.modeIconRag.classList.add("done");
      if (el.modeIconGraph.classList.contains("lit")) el.modeIconGraph.classList.add("done");

      el.stageRetrieval.classList.remove("active");
      el.stageRetrieval.classList.add("done");
      el.pipelineStatusText.textContent = "Retrieving clinical evidence...";

      const seedNodes = Array.isArray(data.seed_nodes) ? data.seed_nodes : [];
      const highlightedEdges = Array.isArray(data.highlighted_edges)
        ? data.highlighted_edges
        : [];

      const evidenceItems = buildEvidenceItems(data, seedNodes);
      renderEvidenceStack(evidenceItems);

      if (seedNodes.length > 0 || highlightedEdges.length > 0) {
        highlightRetrievalElements(seedNodes, highlightedEdges);
      }
      return;
    }

    if (eventType === "web_search") {
      renderWebSearchStage(data);
      return;
    }

    if (eventType === "synthesizing") {
      el.stageSynthesis.classList.add("active");
      el.pipelineStatusText.textContent = "Synthesizing grounded answer...";

      el.synthStatusPill.classList.remove("done");
      el.synthStatusPill.classList.add("synthesizing");
      el.synthStatusPill.innerHTML = '<span class="spinner" aria-hidden="true"></span><span>Generating...</span>';
      return;
    }

    if (eventType === "done") {
      if (typeof data.guard_reachable === "boolean") {
        updateGuardStatusUI(data.guard_reachable);
      }
      if (Array.isArray(data.web_sources) && (data.web_sources.length > 0 || data.used_web_fallback)) {
        renderWebSearchStage({
          enabled: true,
          triggered: Boolean(data.used_web_fallback),
          web_sources: data.web_sources,
        });
      }
      el.stageSynthesis.classList.remove("active");
      el.stageSynthesis.classList.add("done");
      el.pipelineStatusText.textContent = "Complete";

      el.synthStatusPill.classList.remove("synthesizing");
      el.synthStatusPill.classList.add("done");
      el.synthStatusPill.innerHTML = '<span class="synth-done-check">&#10003;</span><span>Complete</span>';

      renderPipelineLatency(data.metrics && data.metrics.latencies_ms);
    }
  }

  function formatPipelineDuration(ms) {
    if (typeof ms !== "number" || !Number.isFinite(ms)) return "—";
    if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`;
    return `${Math.round(ms)} ms`;
  }

  function renderPipelineLatency(latencies) {
    if (!el.stageLatency || !el.pipelineLatency || !el.pipelineLatencyValue || !el.pipelineLatencyTooltip) {
      return;
    }
    if (!latencies || typeof latencies !== "object") {
      el.stageLatency.classList.add("hidden");
      return;
    }

    const totalMs = typeof latencies.total === "number" ? latencies.total : null;
    if (totalMs === null) {
      el.stageLatency.classList.add("hidden");
      return;
    }

    const hasRetrieval =
      typeof latencies.rag_retrieval === "number" ||
      typeof latencies.seed_search === "number" ||
      typeof latencies.k_hop_traversal === "number";
    const retrievalMs = hasRetrieval
      ? (latencies.rag_retrieval || 0) +
        (latencies.seed_search || 0) +
        (latencies.k_hop_traversal || 0)
      : null;

    const steps = [
      ["Prompt Guard", latencies.guard],
      ["Search Strings", latencies.query_rewrite],
      ["Retrieval", retrievalMs],
      ["Web Search", latencies.web_search],
      ["Synthesis", latencies.answer_synthesis],
    ];

    el.pipelineLatencyValue.textContent = formatPipelineDuration(totalMs);
    el.pipelineLatencyTooltip.innerHTML = steps
      .map(
        ([label, val]) =>
          `<div class="pipeline-latency-row"><span class="pipeline-latency-row-label">${escapeHtml(label)}</span><span class="pipeline-latency-row-val">${escapeHtml(formatPipelineDuration(val))}</span></div>`
      )
      .join("");
    el.stageLatency.classList.add("done");
    el.stageLatency.classList.remove("hidden");
  }

  // --- Chat & Citations ---
  function appendUserMessage(text) {
    const welcome = el.chatMessages.querySelector(".chat-welcome");
    if (welcome) welcome.remove();

    const msgEl = document.createElement("div");
    msgEl.className = "chat-msg chat-msg-user markdown-body";
    msgEl.innerHTML = renderSafeMarkdown(text);
    el.chatMessages.appendChild(msgEl);
    el.chatMessages.scrollTop = el.chatMessages.scrollHeight;
  }

  function appendAssistantMessage(payload) {
    const msgEl = document.createElement("div");
    msgEl.className = "chat-msg chat-msg-assistant";

    const answerEl = document.createElement("div");
    answerEl.className = "assistant-answer markdown-body";
    answerEl.innerHTML = renderSafeMarkdown(payload.answer || "No answer returned.");
    msgEl.appendChild(answerEl);

    const citations = Array.isArray(payload.citations) ? payload.citations : [];
    if (citations.length > 0) {
      const citSection = document.createElement("div");
      citSection.className = "citations-section";
      citSection.innerHTML = '<div class="citations-heading">Citations</div>';

      const pillsRow = document.createElement("div");
      pillsRow.className = "citation-pills-row";

      citations.forEach((cit) => {
        const pill = document.createElement("button");
        pill.type = "button";
        pill.className = "citation-pill";

        const passagesLen = Array.isArray(cit.passages) ? cit.passages.length : 0;
        const excerptsLen = Array.isArray(cit.excerpts) ? cit.excerpts.length : 0;
        const quoteCount = passagesLen || excerptsLen || 1;
        const quoteLabel = `${quoteCount} ${quoteCount === 1 ? "quote" : "quotes"}`;
        const docTitle = cit.title || cit.slug || cit.doc_id || "Document";

        pill.innerHTML = `
          <span class="citation-pill-title">${escapeHtml(docTitle)}</span>
          <span class="citation-pill-sep" aria-hidden="true">&middot;</span>
          <span class="citation-pill-count">${escapeHtml(quoteLabel)}</span>
        `;
        pill.title = `${docTitle} (${quoteLabel}) — click to inspect`;

        pill.addEventListener("mouseenter", () => handleCitationHover(cit, true));
        pill.addEventListener("mouseleave", () => handleCitationHover(cit, false));
        pill.addEventListener("click", () => openCitationModal(citations, cit));

        pillsRow.appendChild(pill);
      });

      citSection.appendChild(pillsRow);
      msgEl.appendChild(citSection);
    }

    const webSources = Array.isArray(payload.web_sources) ? payload.web_sources : [];
    if (webSources.length > 0) {
      const webSection = document.createElement("div");
      webSection.className = "citations-section web-sources-section";
      webSection.innerHTML = '<div class="citations-heading">Web Sources (External)</div>';

      const webRow = document.createElement("div");
      webRow.className = "citation-pills-row";

      webSources.forEach((src) => {
        const title = src.title || src.domain || "Web Source";
        const domain = src.domain || "";
        const excerpts = Array.isArray(src.excerpts) ? src.excerpts : [];
        const quoteCount = excerpts.length;
        const metaParts = [];
        if (domain) metaParts.push(domain);
        if (quoteCount > 0) metaParts.push(`${quoteCount} ${quoteCount === 1 ? "quote" : "quotes"}`);

        const pill = document.createElement("a");
        pill.className = "citation-pill web-source-pill";
        pill.href = src.url || "#";
        pill.target = "_blank";
        pill.rel = "noopener noreferrer";
        pill.innerHTML = `
          <span class="citation-pill-title">${escapeHtml(title)}</span>
          ${metaParts.length ? `<span class="citation-pill-sep" aria-hidden="true">&middot;</span><span class="citation-pill-count">${escapeHtml(metaParts.join(" · "))}</span>` : ""}
        `;
        const tooltipLines = [title];
        if (src.url) tooltipLines.push(src.url);
        if (excerpts.length > 0) tooltipLines.push(`"${excerpts[0]}"`);
        pill.title = tooltipLines.join("\n");

        webRow.appendChild(pill);
      });

      webSection.appendChild(webRow);
      msgEl.appendChild(webSection);
    }

    el.chatMessages.appendChild(msgEl);
    el.chatMessages.scrollTop = el.chatMessages.scrollHeight;
  }

  function handleCitationHover(citation, isEntering) {
    if (isEntering) {
      state.relatedDocIds = new Set([citation.doc_id]);
      updateDocumentListHighlights();

      const matchedCount = highlightGraphByDocIds([citation.doc_id], {
        returnMatchedCount: true,
      });
      if (matchedCount === 0) {
        // Retrieved via RAG chunks/tables or no graph entities for this doc: pulse RAG icon
        el.modeIconRag.classList.add("pulse-highlight");
      }
    } else {
      state.relatedDocIds.clear();
      updateDocumentListHighlights();
      el.modeIconRag.classList.remove("pulse-highlight");
      restorePreviousGraphFocus();
    }
  }

  async function fetchDocumentRecord(docId) {
    let record = state.documentCache.get(docId);
    if (!record) {
      const res = await apiFetch(`/v1/documents/${encodeURIComponent(docId)}`);
      if (!res.ok) {
        throw new Error(await parseErrorDetail(res));
      }
      record = await res.json();
      state.documentCache.set(docId, record);
    }
    return record;
  }

  function buildCitationQuoteItems(citations) {
    const items = [];
    (Array.isArray(citations) ? citations : []).forEach((cit) => {
      const docId = cit.doc_id;
      const docTitle = cit.title || cit.slug || cit.doc_id || "Document";
      const docType = cit.doc_type || "document";
      const localRestriction = cit.local_restriction || "none";

      if (Array.isArray(cit.passages) && cit.passages.length > 0) {
        const sortedPassages = [...cit.passages].sort((a, b) =>
          compareByChunkOrder(a.chunk_id, b.chunk_id)
        );
        sortedPassages.forEach((p) => {
          items.push({
            docId,
            docTitle,
            docType,
            localRestriction,
            chunkId: p.chunk_id || "—",
            hasChunkId: Boolean(p.chunk_id),
            quote: p.quote || "",
            chunkText:
              p.chunk_text ||
              (p.highlighted_chunk ? p.highlighted_chunk.replace(/<\/?mark>/gi, "") : ""),
          });
        });
      } else if (Array.isArray(cit.excerpts) && cit.excerpts.length > 0) {
        cit.excerpts.forEach((excerpt, i) => {
          const cid =
            (cit.chunk_ids && cit.chunk_ids[i]) ||
            (cit.chunk_ids && cit.chunk_ids[0]) ||
            "";
          items.push({
            docId,
            docTitle,
            docType,
            localRestriction,
            chunkId: cid || "—",
            hasChunkId: Boolean(cid),
            quote: excerpt || "",
            chunkText: excerpt || "",
          });
        });
      } else {
        const cid = (cit.chunk_ids && cit.chunk_ids[0]) || "";
        items.push({
          docId,
          docTitle,
          docType,
          localRestriction,
          chunkId: cid || "—",
          hasChunkId: Boolean(cid),
          quote: cit.excerpt || "",
          chunkText: cit.excerpt || "",
        });
      }
    });
    return items;
  }

  function scrollPaneToTarget(paneEl, targetSelector) {
    const target =
      paneEl.querySelector(targetSelector) ||
      paneEl.querySelector(".doc-paragraph-highlight");
    if (target) {
      requestAnimationFrame(() => {
        target.scrollIntoView({ behavior: "smooth", block: "center" });
      });
    } else {
      paneEl.scrollTop = 0;
    }
  }

  function openCitationModal(allCitations, targetCitation = null) {
    const citations = Array.isArray(allCitations) ? allCitations : [allCitations].filter(Boolean);
    const quoteItems = buildCitationQuoteItems(citations);
    if (quoteItems.length === 0) return;

    let initialIdx = 0;
    if (targetCitation && targetCitation.doc_id) {
      const foundIdx = quoteItems.findIndex((item) => item.docId === targetCitation.doc_id);
      if (foundIdx !== -1) initialIdx = foundIdx;
    }

    el.citationQuotesPane.innerHTML = "";
    const paneHeader = document.createElement("div");
    paneHeader.className = "citation-quotes-pane-header";
    paneHeader.textContent = `Cited Quotes (${quoteItems.length})`;
    el.citationQuotesPane.appendChild(paneHeader);

    const cardButtons = quoteItems.map((item, idx) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "citation-quote-card";
      btn.dataset.quoteIdx = String(idx);
      const displayQuote = item.quote || item.chunkText || "No quote preview available.";
      btn.innerHTML = `
        <div class="citation-quote-card-meta">
          <span class="citation-quote-doc">${escapeHtml(item.docTitle)}</span>
          <span class="citation-quote-chunk-badge">${escapeHtml(item.chunkId)}</span>
        </div>
        <div class="citation-quote-card-text">&ldquo;${escapeHtml(displayQuote)}&rdquo;</div>
      `;
      btn.addEventListener("click", () => selectCitationQuote(idx));
      el.citationQuotesPane.appendChild(btn);
      return btn;
    });

    async function selectCitationQuote(idx) {
      const item = quoteItems[idx];
      if (!item) return;

      cardButtons.forEach((btn, i) => {
        const isActive = i === idx;
        btn.classList.toggle("active", isActive);
        if (isActive) {
          btn.setAttribute("aria-current", "true");
          btn.scrollIntoView({ behavior: "smooth", block: "nearest" });
        } else {
          btn.removeAttribute("aria-current");
        }
      });

      el.citationModalTitle.textContent = item.docTitle;
      const subtitleParts = [item.docType, item.localRestriction, item.docId];
      if (item.hasChunkId) subtitleParts.push(item.chunkId);
      el.citationModalSubtitle.textContent = subtitleParts.join(" • ");

      const reqId = ++state.citationModalRequestSeq;
      if (!state.documentCache.has(item.docId)) {
        el.citationDocPane.innerHTML = '<div class="empty-state">Loading document...</div>';
      }

      try {
        const record = await fetchDocumentRecord(item.docId);
        if (reqId !== state.citationModalRequestSeq) return;

        el.citationDocPane.innerHTML = renderDocumentContentWithHighlight(
          record.content || "",
          item.quote || null,
          item.chunkText || null,
          "citation-doc-target"
        );
        scrollPaneToTarget(el.citationDocPane, "#citation-doc-target");
      } catch (err) {
        if (reqId !== state.citationModalRequestSeq) return;
        if (item.chunkText) {
          el.citationDocPane.innerHTML = renderDocumentContentWithHighlight(
            item.chunkText,
            item.quote || null,
            item.chunkText,
            "citation-doc-target"
          );
          scrollPaneToTarget(el.citationDocPane, "#citation-doc-target");
        } else {
          el.citationDocPane.innerHTML = `<div class="empty-state">Failed to load document: ${escapeHtml(err.message)}</div>`;
        }
      }
    }

    el.citationModal.classList.remove("hidden");
    selectCitationQuote(initialIdx);
  }

  function closeCitationModal() {
    state.citationModalRequestSeq += 1;
    el.citationModal.classList.add("hidden");
  }

  function escapeRegex(str) {
    return String(str).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function buildWhitespaceRegex(text, flags = "") {
    const tokens = String(text || "").trim().split(/\s+/).filter(Boolean);
    if (tokens.length === 0) return null;
    const pattern = tokens.map(escapeRegex).join("\\s+");
    return new RegExp(pattern, flags);
  }

  function findRangeInContent(content, needle) {
    if (!content || !needle) return null;
    const trimmed = String(needle).trim();
    if (!trimmed) return null;

    const exactIdx = content.indexOf(trimmed);
    if (exactIdx !== -1) {
      return { start: exactIdx, end: exactIdx + trimmed.length };
    }

    const exactRegex = buildWhitespaceRegex(trimmed);
    if (exactRegex) {
      const m = exactRegex.exec(content);
      if (m) return { start: m.index, end: m.index + m[0].length };
    }

    const ciRegex = buildWhitespaceRegex(trimmed, "i");
    if (ciRegex) {
      const m = ciRegex.exec(content);
      if (m) return { start: m.index, end: m.index + m[0].length };
    }

    return null;
  }

  function expandToParagraphBounds(content, start, end) {
    const MAX_WINDOW = 1600;
    const prevDouble = content.lastIndexOf("\n\n", start);
    let pStart = prevDouble !== -1 && start - prevDouble <= MAX_WINDOW
      ? prevDouble + 2
      : Math.max(0, content.lastIndexOf("\n", start) + 1);

    const nextDouble = content.indexOf("\n\n", end);
    let pEnd;
    if (nextDouble !== -1 && nextDouble - end <= MAX_WINDOW) {
      pEnd = nextDouble;
    } else {
      const nextSingle = content.indexOf("\n", end);
      pEnd = nextSingle !== -1 ? nextSingle : content.length;
    }

    if (pStart > start) pStart = start;
    if (pEnd < end) pEnd = end;
    return { start: pStart, end: pEnd };
  }

  function locateChunkSpan(content, chunkText, quote) {
    if (chunkText) {
      const direct = findRangeInContent(content, chunkText);
      if (direct) return direct;

      // Prefix fallback when chunkText has minor trailing differences
      const words = String(chunkText).trim().split(/\s+/).filter(Boolean);
      if (words.length > 10) {
        const prefixMatch = findRangeInContent(content, words.slice(0, 12).join(" "));
        if (prefixMatch) {
          let approxEnd = Math.min(content.length, prefixMatch.start + chunkText.length + 80);
          if (quote) {
            const quoteMatch = findRangeInContent(content.slice(prefixMatch.start), quote);
            if (quoteMatch && quoteMatch.end <= chunkText.length + 400) {
              approxEnd = Math.max(approxEnd, prefixMatch.start + quoteMatch.end);
            }
          }
          return expandToParagraphBounds(content, prefixMatch.start, Math.max(prefixMatch.end, approxEnd));
        }
      }
    }

    if (quote) {
      const quoteMatch = findRangeInContent(content, quote);
      if (quoteMatch) {
        return expandToParagraphBounds(content, quoteMatch.start, quoteMatch.end);
      }
    }

    return null;
  }

  function renderDocumentContentWithHighlight(
    content,
    quote,
    chunkText,
    targetId = "doc-viewer-target"
  ) {
    const text = String(content || "");
    if (!text) return '<span class="empty-state">Document content is empty.</span>';

    const span = locateChunkSpan(text, chunkText, quote);
    if (!span) {
      return renderSafeMarkdown(text);
    }

    const before = text.slice(0, span.start);
    const paragraph = text.slice(span.start, span.end);
    const after = text.slice(span.end);
    const quoteMatch = quote ? findRangeInContent(paragraph, quote) : null;

    const hasMarkdownLibs =
      typeof window !== "undefined" &&
      window.marked &&
      typeof window.marked.parse === "function" &&
      window.DOMPurify &&
      typeof window.DOMPurify.sanitize === "function";

    if (hasMarkdownLibs) {
      const beforeHtml = before ? renderSafeMarkdown(before) : "";
      const afterHtml = after ? renderSafeMarkdown(after) : "";

      let paragraphHtml;
      if (quoteMatch) {
        const qBefore = paragraph.slice(0, quoteMatch.start);
        const qExact = paragraph.slice(quoteMatch.start, quoteMatch.end);
        const qAfter = paragraph.slice(quoteMatch.end);
        const markedParagraph = `${qBefore}<mark id="${targetId}">${qExact}</mark>${qAfter}`;
        const renderedParagraph = renderSafeMarkdown(markedParagraph, {
          allowHighlightTags: true,
        });
        paragraphHtml = `<span class="doc-paragraph-highlight">${renderedParagraph}</span>`;
      } else {
        const renderedParagraph = renderSafeMarkdown(paragraph, {
          allowHighlightTags: true,
        });
        paragraphHtml = `<span class="doc-paragraph-highlight" id="${targetId}">${renderedParagraph}</span>`;
      }

      return window.DOMPurify.sanitize(`${beforeHtml}${paragraphHtml}${afterHtml}`, {
        ADD_TAGS: ["mark", "span"],
        ADD_ATTR: ["id", "class", "title", "data-quote"],
      });
    }

    // Safe plain-text fallback when marked / DOMPurify are unavailable
    let paragraphHtml;
    if (quoteMatch) {
      const qBefore = paragraph.slice(0, quoteMatch.start);
      const qExact = paragraph.slice(quoteMatch.start, quoteMatch.end);
      const qAfter = paragraph.slice(quoteMatch.end);
      paragraphHtml = `<span class="doc-paragraph-highlight">${escapeHtml(qBefore)}<mark id="${targetId}">${escapeHtml(qExact)}</mark>${escapeHtml(qAfter)}</span>`;
    } else {
      paragraphHtml = `<span class="doc-paragraph-highlight" id="${targetId}">${escapeHtml(paragraph)}</span>`;
    }

    return `<div class="markdown-fallback">${escapeHtml(before)}${paragraphHtml}${escapeHtml(after)}</div>`;
  }

  async function openDocumentViewer(docId, chunkId = null, quote = null, chunkText = null) {
    if (!docId) return;
    const reqId = ++state.docViewerRequestSeq;

    el.docViewerTitle.textContent = "Loading document...";
    el.docViewerSubtitle.textContent = chunkId ? `${docId} • ${chunkId}` : docId;
    el.docViewerBody.textContent = "Loading document content...";
    el.docViewerModal.classList.remove("hidden");

    try {
      const record = await fetchDocumentRecord(docId);

      if (reqId !== state.docViewerRequestSeq) return;

      const meta = record.metadata || {};
      el.docViewerTitle.textContent = meta.title || record.slug || record.doc_id || docId;
      const subtitleParts = [
        meta.doc_type || "document",
        meta.local_restriction || "none",
        record.doc_id || docId,
      ];
      if (chunkId) subtitleParts.push(chunkId);
      el.docViewerSubtitle.textContent = subtitleParts.join(" • ");

      el.docViewerBody.innerHTML = renderDocumentContentWithHighlight(
        record.content || "",
        quote,
        chunkText,
        "doc-viewer-target"
      );

      scrollPaneToTarget(el.docViewerBody, "#doc-viewer-target");
    } catch (err) {
      if (reqId !== state.docViewerRequestSeq) return;
      el.docViewerTitle.textContent = "Document Viewer Error";
      el.docViewerBody.innerHTML = `<div class="empty-state">Failed to load document: ${escapeHtml(err.message)}</div>`;
    }
  }

  function closeDocumentViewer() {
    el.docViewerModal.classList.add("hidden");
  }

  async function handleQuerySubmit(event) {
    event.preventDefault();
    if (state.isQuerying) return;

    const queryText = el.chatInput.value.trim();
    if (!queryText) return;

    state.isQuerying = true;
    el.sendQueryBtn.disabled = true;
    el.chatInput.value = "";

    appendUserMessage(queryText);
    resetPipelineVisualizer();

    const payload = {
      query: queryText,
      session_id: state.sessionId,
      messages: state.chatHistory.slice(-10),
      retrieval_mode: state.retrievalMode || "auto",
      allow_web_fallback: Boolean(el.webSearchToggle && el.webSearchToggle.checked),
      include_trace: true,
    };

    try {
      const res = await apiFetch("/v1/query/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        throw new Error(await parseErrorDetail(res));
      }
      if (!res.body) {
        throw new Error("Streaming response body is unavailable.");
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      let donePayload = null;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // Normalize CRLF and split SSE frames on double newline
        const normalized = buffer.replace(/\r\n/g, "\n");
        const frames = normalized.split("\n\n");
        buffer = frames.pop() || "";

        for (const frame of frames) {
          if (!frame.trim()) continue;
          let eventName = "message";
          const dataLines = [];

          for (const line of frame.split("\n")) {
            if (line.startsWith("event:")) {
              eventName = line.slice(6).trim();
            } else if (line.startsWith("data:")) {
              dataLines.push(line.slice(5).trim());
            }
          }

          if (dataLines.length === 0) continue;
          let parsed;
          try {
            parsed = JSON.parse(dataLines.join("\n"));
          } catch {
            continue;
          }

          const resolvedEvent = parsed.event || parsed.type || eventName;
          if (resolvedEvent === "error") {
            throw new Error(parsed.detail || parsed.error || "Query stream error");
          }
          handlePipelineEvent(resolvedEvent, parsed);
          if (resolvedEvent === "done") {
            donePayload = parsed;
          }
        }
      }

      if (donePayload) {
        appendAssistantMessage(donePayload);
        state.chatHistory.push(
          { role: "user", content: queryText },
          { role: "assistant", content: donePayload.answer || "" }
        );
      }
    } catch (err) {
      el.pipelineStatusText.textContent = "Error";
      el.stageSynthesis.classList.remove("active");
      el.synthStatusPill.classList.remove("synthesizing", "done");
      el.synthStatusPill.innerHTML = '<span class="synth-idle-text">Error</span>';
      appendAssistantMessage({ answer: `Error: ${err.message}`, citations: [] });
    } finally {
      state.isQuerying = false;
      el.sendQueryBtn.disabled = false;
      el.chatInput.focus();
    }
  }

  function zoomGraphBy(factor) {
    const cy = state.cy;
    if (!cy) return;
    const nextZoom = Math.min(cy.maxZoom(), Math.max(cy.minZoom(), cy.zoom() * factor));
    cy.zoom({
      level: nextZoom,
      renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 },
    });
  }

  // --- API Implementation & Integration Guide Modal ---
  function getApiSnippets(lang) {
    const baseUrl = window.location.origin && window.location.origin !== "null"
      ? window.location.origin
      : "http://localhost:8000";

    const specs = [
      {
        method: "POST",
        path: "/v1/query",
        note: "Header: X-OpenAI-API-Key: sk-... (optional if server env set)",
        python: `import httpx

resp = httpx.post(
    "${baseUrl}/v1/query",
    headers={"X-OpenAI-API-Key": "sk-..."},  # optional if server env set
    json={
        "query": "What are the contraindications for metformin?",
        "retrieval_mode": "hybrid",
        "top_k": 5,
    },
    timeout=60.0,
)
resp.raise_for_status()
data = resp.json()
print(data["answer"])`,
        curl: `curl -X POST "${baseUrl}/v1/query" \\
  -H "Content-Type: application/json" \\
  -H "X-OpenAI-API-Key: sk-..." \\
  -d '{
    "query": "What are the contraindications for metformin?",
    "retrieval_mode": "hybrid",
    "top_k": 5
  }'`,
      },
      {
        method: "POST",
        path: "/v1/query/stream",
        note: "SSE streaming pipeline events + grounded answer",
        python: `import httpx

with httpx.stream(
    "POST",
    "${baseUrl}/v1/query/stream",
    headers={"X-OpenAI-API-Key": "sk-..."},  # optional if server env set
    json={"query": "Summarize SGLT2 inhibitor recommendations", "retrieval_mode": "hybrid"},
    timeout=60.0,
) as resp:
    resp.raise_for_status()
    for line in resp.iter_lines():
        if line.startswith("data: "):
            print(line[6:])`,
        curl: `curl -N -X POST "${baseUrl}/v1/query/stream" \\
  -H "Content-Type: application/json" \\
  -H "X-OpenAI-API-Key: sk-..." \\
  -d '{
    "query": "Summarize SGLT2 inhibitor recommendations",
    "retrieval_mode": "hybrid"
  }'`,
      },
      {
        method: "POST",
        path: "/v1/documents/upload",
        note: "Multipart PDF / MD / TXT ingestion",
        python: `import httpx

with open("guideline.pdf", "rb") as f:
    resp = httpx.post(
        "${baseUrl}/v1/documents/upload",
        headers={"X-OpenAI-API-Key": "sk-..."},  # optional if server env set
        files={"file": ("guideline.pdf", f, "application/pdf")},
        data={"title": "Diabetes Guideline", "local_restriction": "DE"},
        timeout=60.0,
    )
resp.raise_for_status()
print(resp.json())  # {"job_id": "...", "status": "queued"}`,
        curl: `curl -X POST "${baseUrl}/v1/documents/upload" \\
  -H "X-OpenAI-API-Key: sk-..." \\
  -F "file=@guideline.pdf" \\
  -F "title=Diabetes Guideline" \\
  -F "local_restriction=DE"`,
      },
      {
        method: "GET",
        path: "/v1/graph",
        note: "No API key required — full knowledge graph JSON",
        python: `import httpx

resp = httpx.get("${baseUrl}/v1/graph", timeout=30.0)
resp.raise_for_status()
graph = resp.json()
print(len(graph["nodes"]), len(graph["edges"]))`,
        curl: `curl -s "${baseUrl}/v1/graph" | jq '{nodes: (.nodes | length), edges: (.edges | length)}'`,
      },
    ];

    return specs.map((spec) => ({
      method: spec.method,
      path: spec.path,
      note: spec.note,
      code: lang === "curl" ? spec.curl : spec.python,
    }));
  }

  function renderApiSnippets(lang) {
    const activeLang = lang === "curl" ? "curl" : "python";
    state.apiGuideLang = activeLang;

    if (el.snippetLangCurl && el.snippetLangPython) {
      const isCurl = activeLang === "curl";
      el.snippetLangCurl.classList.toggle("active", isCurl);
      el.snippetLangCurl.setAttribute("aria-selected", String(isCurl));
      el.snippetLangPython.classList.toggle("active", !isCurl);
      el.snippetLangPython.setAttribute("aria-selected", String(!isCurl));
    }

    if (!el.apiSnippetsContainer) return;
    el.apiSnippetsContainer.innerHTML = "";

    const snippets = getApiSnippets(activeLang);
    snippets.forEach((snippet) => {
      const card = document.createElement("div");
      card.className = "snippet-card";

      const header = document.createElement("div");
      header.className = "snippet-card-header";

      const left = document.createElement("div");
      left.className = "snippet-endpoint-left";

      const badge = document.createElement("span");
      badge.className = `method-badge ${snippet.method.toLowerCase()}`;
      badge.textContent = snippet.method;

      const pathEl = document.createElement("span");
      pathEl.className = "snippet-path";
      pathEl.textContent = snippet.path;

      left.appendChild(badge);
      left.appendChild(pathEl);

      const noteEl = document.createElement("span");
      noteEl.className = "snippet-note";
      noteEl.textContent = snippet.note;

      header.appendChild(left);
      header.appendChild(noteEl);

      const codeWrap = document.createElement("div");
      codeWrap.className = "snippet-code-wrap";

      const pre = document.createElement("pre");
      pre.className = "snippet-pre";
      const codeEl = document.createElement("code");
      codeEl.textContent = snippet.code;
      pre.appendChild(codeEl);

      const copyBtn = document.createElement("button");
      copyBtn.type = "button";
      copyBtn.className = "snippet-copy-btn";
      copyBtn.textContent = "Copy";
      let resetTimer = null;

      copyBtn.addEventListener("click", async () => {
        try {
          if (navigator.clipboard && navigator.clipboard.writeText) {
            await navigator.clipboard.writeText(snippet.code);
          } else {
            const ta = document.createElement("textarea");
            ta.value = snippet.code;
            ta.style.position = "fixed";
            ta.style.opacity = "0";
            document.body.appendChild(ta);
            ta.select();
            document.execCommand("copy");
            document.body.removeChild(ta);
          }
          copyBtn.textContent = "Copied!";
          copyBtn.classList.add("copied");
          if (resetTimer) clearTimeout(resetTimer);
          resetTimer = setTimeout(() => {
            copyBtn.textContent = "Copy";
            copyBtn.classList.remove("copied");
          }, 1600);
        } catch (err) {
          console.error("Failed to copy snippet:", err);
        }
      });

      codeWrap.appendChild(pre);
      codeWrap.appendChild(copyBtn);

      card.appendChild(header);
      card.appendChild(codeWrap);
      el.apiSnippetsContainer.appendChild(card);
    });
  }

  function openApiGuideModal() {
    if (!el.apiGuideModal) return;
    renderApiSnippets(state.apiGuideLang || "python");
    el.apiGuideModal.classList.remove("hidden");
  }

  function closeApiGuideModal() {
    if (!el.apiGuideModal) return;
    el.apiGuideModal.classList.add("hidden");
  }

  // --- View Switcher & Benchmarks ---
  function switchView(viewName) {
    const isWorkspace = viewName === "workspace";
    state.activeView = isWorkspace ? "workspace" : "overview";

    if (el.overviewView) {
      el.overviewView.classList.toggle("hidden", isWorkspace);
    }
    if (el.workspaceView) {
      el.workspaceView.classList.toggle("hidden", !isWorkspace);
    }
    if (el.navOverviewBtn) {
      el.navOverviewBtn.classList.toggle("active", !isWorkspace);
      el.navOverviewBtn.setAttribute("aria-pressed", String(!isWorkspace));
    }
    if (el.navWorkspaceBtn) {
      el.navWorkspaceBtn.classList.toggle("active", isWorkspace);
      el.navWorkspaceBtn.setAttribute("aria-pressed", String(isWorkspace));
    }

    if (isWorkspace && state.cy) {
      requestAnimationFrame(() => {
        state.cy.resize();
        if (!state.workspaceInitializedFit) {
          fitGraphToVisibleArea(state.cy);
          state.baseFitZoom = state.cy.zoom();
          state.workspaceInitializedFit = true;
        }
      });
    }
  }

  function formatCostUsd(val) {
    if (typeof val !== "number" || Number.isNaN(val)) return "—";
    return `$${val.toFixed(4)}`;
  }

  function formatLatencyMs(val) {
    if (typeof val !== "number" || Number.isNaN(val)) return "—";
    return `${Math.round(val).toLocaleString()} ms`;
  }

  function formatTokens(val) {
    if (typeof val !== "number" || Number.isNaN(val)) return "—";
    return val.toLocaleString();
  }

  function renderBenchmarksEmpty(message) {
    if (el.benchmarkSummaryGrid) {
      el.benchmarkSummaryGrid.innerHTML = `<div class="benchmark-empty">${escapeHtml(message)}</div>`;
    }
    if (el.benchmarkQueriesList) {
      el.benchmarkQueriesList.innerHTML = "";
    }
  }

  function renderUnitEvals(unitEvals) {
    if (!el.unitEvalsGrid) return;
    if (!Array.isArray(unitEvals) || unitEvals.length === 0) {
      el.unitEvalsGrid.classList.add("hidden");
      el.unitEvalsGrid.innerHTML = "";
      return;
    }
    el.unitEvalsGrid.innerHTML = unitEvals
      .map((u) => {
        const hasScore = typeof u.passed === "number" && typeof u.total === "number";
        const value = hasScore ? `${u.passed} / ${u.total}` : `${u.cases ?? 0} cases`;
        const subParts = [];
        if (typeof u.avg_latency_ms === "number") {
          subParts.push(`Avg: <strong>${escapeHtml(formatLatencyMs(u.avg_latency_ms))}</strong>`);
        }
        if (typeof u.avg_sources === "number") {
          subParts.push(`Sources: <strong>${escapeHtml(String(u.avg_sources))}</strong>`);
        }
        return `
          <div class="benchmark-kpi-card">
            <div class="benchmark-kpi-label"><span>${escapeHtml(u.label || "Unit Eval")}</span></div>
            <div class="benchmark-kpi-value">${escapeHtml(value)}</div>
            <div class="benchmark-kpi-sub">${subParts.join(" ") || "<span>Unit stage check</span>"}</div>
          </div>
        `;
      })
      .join("");
    el.unitEvalsGrid.classList.remove("hidden");
  }

  function renderBenchmarks(data) {
    renderUnitEvals(data && data.unit_evals);
    if (!data || data.available === false) {
      renderBenchmarksEmpty("No benchmark telemetry available yet. Run the evaluation suite to populate GET /v1/benchmarks.");
      return;
    }

    const summary = data.summary || {};
    const byMode = summary.by_mode || {};
    const totalQueries = summary.total_queries ?? 0;
    const autoMatches = summary.auto_matches ?? 0;
    const accuracyPct =
      typeof summary.auto_accuracy_pct === "number"
        ? `${summary.auto_accuracy_pct.toFixed(1)}%`
        : "—";

    if (el.benchmarkSummaryGrid) {
      const modeKeys = ["rag", "hybrid", "auto"].filter(
        (mode) => byMode[mode] && typeof byMode[mode] === "object"
      );
      const accuracyCardHtml = `
        <div class="benchmark-kpi-card">
          <div class="benchmark-kpi-label"><span>Auto Router Accuracy</span></div>
          <div class="benchmark-kpi-value">${escapeHtml(accuracyPct)}</div>
          <div class="benchmark-kpi-sub"><span>${escapeHtml(String(autoMatches))} / ${escapeHtml(String(totalQueries))} queries matched</span></div>
        </div>
      `;

      const modeCardsHtml = modeKeys
        .map((mode) => {
          const stats = byMode[mode] || {};
          const isAuto = mode === "auto";
          return `
            <div class="benchmark-kpi-card${isAuto ? " benchmark-kpi-card--auto" : ""}">
              <div class="benchmark-kpi-label">
                <span>${escapeHtml(mode)}</span>
                <span class="benchmark-mode-pill benchmark-mode-pill--${escapeHtml(mode)}">${escapeHtml(mode)}</span>
              </div>
              <div class="benchmark-kpi-value">${escapeHtml(formatLatencyMs(stats.avg_latency_ms))}</div>
              <div class="benchmark-kpi-sub">
                <span>Tokens: <strong>${escapeHtml(formatTokens(stats.total_tokens))}</strong></span>
                <span>Cost: <strong>${escapeHtml(formatCostUsd(stats.total_cost_usd))}</strong></span>
              </div>
            </div>
          `;
        })
        .join("");

      el.benchmarkSummaryGrid.innerHTML = accuracyCardHtml + modeCardsHtml;
    }

    const queries = data.results || data.queries || [];
    if (!el.benchmarkQueriesList) return;

    if (!Array.isArray(queries) || queries.length === 0) {
      el.benchmarkQueriesList.innerHTML = `<div class="benchmark-empty">No benchmark query records found.</div>`;
      return;
    }

    el.benchmarkQueriesList.innerHTML = queries
      .map((item) => {
        const id = item.id || "query";
        const queryText = item.query || "";
        const expectedMode = item.expected_mode || "auto";
        const autoMatch = Boolean(item.auto_matches_expected);
        const rationale = item.rationale || "";
        const runs = item.runs || item.modes || {};

        const runsHtml = ["rag", "hybrid", "auto"]
          .map((mode) => {
            const run = runs[mode];
            if (!run) return "";
            const chosenMode = run.chosen_mode || mode;
            const citationsCount =
              typeof run.citation_count === "number"
                ? run.citation_count
                : Array.isArray(run.citations)
                  ? run.citations.length
                  : 0;
            const nodesCount = Array.isArray(run.seed_nodes)
              ? run.seed_nodes.length
              : (run.seed_nodes_count ?? run.nodes_count ?? 0);
            const edgesCount = Array.isArray(run.traversed_edges)
              ? run.traversed_edges.length
              : (run.traversed_edges_count ?? run.edges_count ?? 0);
            const preview = run.answer_preview || run.answer || "";

            return `
              <div class="benchmark-run-card${mode === "auto" ? " benchmark-run-card--auto" : ""}">
                <div class="benchmark-run-head">
                  <span class="benchmark-mode-pill benchmark-mode-pill--${escapeHtml(mode)}">${escapeHtml(mode)}</span>
                  <span class="benchmark-run-chosen">routed: <strong>${escapeHtml(chosenMode)}</strong></span>
                </div>
                <div class="benchmark-run-metrics">
                  <div class="benchmark-run-metric"><span>Latency</span><strong>${escapeHtml(formatLatencyMs(run.latency_ms))}</strong></div>
                  <div class="benchmark-run-metric"><span>Tokens</span><strong>${escapeHtml(formatTokens(run.total_tokens))}</strong></div>
                  <div class="benchmark-run-metric"><span>Cost</span><strong>${escapeHtml(formatCostUsd(run.cost_usd))}</strong></div>
                  <div class="benchmark-run-metric"><span>Citations</span><strong>${escapeHtml(String(citationsCount))}</strong></div>
                  <div class="benchmark-run-metric"><span>Nodes</span><strong>${escapeHtml(String(nodesCount))}</strong></div>
                  <div class="benchmark-run-metric"><span>Edges</span><strong>${escapeHtml(String(edgesCount))}</strong></div>
                </div>
                <p class="benchmark-run-preview">${escapeHtml(preview)}</p>
              </div>
            `;
          })
          .join("");

        return `
          <article class="benchmark-query-card">
            <button type="button" class="benchmark-query-toggle" aria-expanded="false">
              <div class="benchmark-query-main">
                <span class="benchmark-query-id">${escapeHtml(id)}</span>
                <span class="benchmark-query-text">${escapeHtml(queryText)}</span>
              </div>
              <div class="benchmark-query-badges">
                <span class="benchmark-mode-pill benchmark-mode-pill--${escapeHtml(expectedMode)}">expected: ${escapeHtml(expectedMode)}</span>
                <span class="benchmark-match-badge ${autoMatch ? "match-ok" : "match-miss"}">${autoMatch ? "Auto Match" : "Mismatch"}</span>
                <span class="benchmark-chevron" aria-hidden="true">&#9662;</span>
              </div>
            </button>
            <div class="benchmark-query-details">
              ${rationale ? `<p class="benchmark-rationale"><strong>Rationale:</strong> ${escapeHtml(rationale)}</p>` : ""}
              <div class="benchmark-runs-grid">${runsHtml}</div>
            </div>
          </article>
        `;
      })
      .join("");
  }

  async function loadBenchmarks() {
    try {
      const resp = await fetch("/v1/benchmarks");
      if (!resp.ok) {
        renderBenchmarksEmpty(`Benchmark telemetry unavailable (HTTP ${resp.status}).`);
        return;
      }
      const data = await resp.json();
      renderBenchmarks(data);
    } catch {
      renderBenchmarksEmpty("Unable to load benchmark telemetry from /v1/benchmarks.");
    }
  }

  // --- Event Listeners ---
  function initEventListeners() {
    if (el.navOverviewBtn) {
      el.navOverviewBtn.addEventListener("click", () => switchView("overview"));
    }
    if (el.navWorkspaceBtn) {
      el.navWorkspaceBtn.addEventListener("click", () => switchView("workspace"));
    }
    if (el.openWorkspaceCta) {
      el.openWorkspaceCta.addEventListener("click", () => switchView("workspace"));
    }
    if (el.overviewApiGuideBtn) {
      el.overviewApiGuideBtn.addEventListener("click", openApiGuideModal);
    }
    if (el.benchmarkQueriesList) {
      el.benchmarkQueriesList.addEventListener("click", (e) => {
        const toggle = e.target.closest(".benchmark-query-toggle");
        if (!toggle) return;
        const card = toggle.closest(".benchmark-query-card");
        if (!card) return;
        const expanded = card.classList.toggle("expanded");
        toggle.setAttribute("aria-expanded", String(expanded));
      });
    }

    if (el.nodeLegend) {
      el.nodeLegend.addEventListener("click", (e) => {
        const pill = e.target.closest(".legend-item[data-node-type]");
        if (!pill) return;
        toggleNodeTypeFilter(pill.dataset.nodeType);
      });
    }

    if (el.zoomInBtn) {
      el.zoomInBtn.addEventListener("click", () => zoomGraphBy(1.25));
    }

    if (el.zoomOutBtn) {
      el.zoomOutBtn.addEventListener("click", () => zoomGraphBy(1 / 1.25));
    }

    el.resetFocusBtn.addEventListener("click", () => {
      clearSelectedDocument();
      resetGraphFocus({ fit: true });
    });

    el.apiKeyBtn.addEventListener("click", () => openApiKeyModal(false));
    if (el.apiGuideBtn) {
      el.apiGuideBtn.addEventListener("click", openApiGuideModal);
    }
    if (el.closeApiGuideModalBtn) {
      el.closeApiGuideModalBtn.addEventListener("click", closeApiGuideModal);
    }
    if (el.apiGuideModal) {
      el.apiGuideModal.addEventListener("click", (e) => {
        if (e.target === el.apiGuideModal) closeApiGuideModal();
      });
    }
    if (el.snippetLangCurl) {
      el.snippetLangCurl.addEventListener("click", () => renderApiSnippets("curl"));
    }
    if (el.snippetLangPython) {
      el.snippetLangPython.addEventListener("click", () => renderApiSnippets("python"));
    }
    el.closeApiKeyModalBtn.addEventListener("click", closeApiKeyModal);
    el.clearApiKeyBtn.addEventListener("click", () => {
      sessionStorage.removeItem(API_KEY_STORAGE_KEY);
      updateApiKeyStatusUI();
      if (!state.hasServerApiKey) {
        openApiKeyModal(true);
      } else {
        closeApiKeyModal();
      }
    });

    el.apiKeyForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const key = el.apiKeyInput.value.trim();
      if (!key) return;
      sessionStorage.setItem(API_KEY_STORAGE_KEY, key);
      el.apiKeyModal.dataset.blocking = "false";
      el.apiKeyModal.classList.add("hidden");
      updateApiKeyStatusUI();
    });

    if (el.webSearchToggle) {
      el.webSearchToggle.addEventListener("change", () => {
        if (el.webSearchToggle.checked) {
          // Require explicit confirmation before enabling external web search
          el.webSearchToggle.checked = false;
          openWebSearchModal();
        } else {
          syncWebSearchStageState();
        }
      });
    }
    if (el.confirmWebSearchBtn) {
      el.confirmWebSearchBtn.addEventListener("click", () => closeWebSearchModal(true));
    }
    if (el.cancelWebSearchBtn) {
      el.cancelWebSearchBtn.addEventListener("click", () => closeWebSearchModal(false));
    }
    if (el.closeWebSearchModalBtn) {
      el.closeWebSearchModalBtn.addEventListener("click", () => closeWebSearchModal(false));
    }
    if (el.webSearchModal) {
      el.webSearchModal.addEventListener("click", (e) => {
        if (e.target === el.webSearchModal) closeWebSearchModal(false);
      });
    }

    if (el.toggleDocsBtn && el.sidebarLeft) {
      el.toggleDocsBtn.addEventListener("click", () => {
        setDocsCollapsed(!el.sidebarLeft.classList.contains("collapsed"));
      });
    }

    el.openUploadBtn.addEventListener("click", openUploadModal);
    el.closeUploadModalBtn.addEventListener("click", closeUploadModal);
    el.cancelUploadBtn.addEventListener("click", closeUploadModal);
    el.uploadForm.addEventListener("submit", handleUploadSubmit);

    if (el.closeGuardUploadModalBtn) {
      el.closeGuardUploadModalBtn.addEventListener("click", closeGuardUploadConfirmModal);
    }
    if (el.cancelGuardUploadBtn) {
      el.cancelGuardUploadBtn.addEventListener("click", closeGuardUploadConfirmModal);
    }
    if (el.guardUploadConfirmModal) {
      el.guardUploadConfirmModal.addEventListener("click", (e) => {
        if (e.target === el.guardUploadConfirmModal) closeGuardUploadConfirmModal();
      });
    }
    if (el.confirmGuardUploadBtn) {
      el.confirmGuardUploadBtn.addEventListener("click", async () => {
        closeGuardUploadConfirmModal();
        await performUpload();
      });
    }

    el.closeInspectorBtn.addEventListener("click", resetGraphFocus);
    if (el.closePipelineBtn && el.pipelineBar) {
      el.closePipelineBtn.addEventListener("click", () => {
        el.pipelineBar.classList.add("hidden");
      });
    }
    el.closeCitationModalBtn.addEventListener("click", closeCitationModal);
    el.citationModal.addEventListener("click", (e) => {
      if (e.target === el.citationModal) closeCitationModal();
    });
    el.closeDocViewerBtn.addEventListener("click", closeDocumentViewer);
    document.addEventListener("keydown", (e) => {
      if (e.key !== "Escape") return;
      if (el.guardUploadConfirmModal && !el.guardUploadConfirmModal.classList.contains("hidden")) {
        closeGuardUploadConfirmModal();
      } else if (el.apiGuideModal && !el.apiGuideModal.classList.contains("hidden")) {
        closeApiGuideModal();
      } else if (!el.docViewerModal.classList.contains("hidden")) {
        closeDocumentViewer();
      } else if (!el.citationModal.classList.contains("hidden")) {
        closeCitationModal();
      }
    });

    el.closeNoticeModalBtn.addEventListener("click", hideNoticeModal);
    el.okNoticeModalBtn.addEventListener("click", hideNoticeModal);

    if (el.retrievalModeGroup) {
      el.retrievalModeGroup.querySelectorAll(".mode-pill").forEach((pill) => {
        pill.addEventListener("click", () => {
          const mode = pill.dataset.mode || "auto";
          state.retrievalMode = mode;
          el.retrievalModeGroup.querySelectorAll(".mode-pill").forEach((btn) => {
            const isActive = btn === pill;
            btn.classList.toggle("active", isActive);
            btn.setAttribute("aria-pressed", String(isActive));
          });
        });
      });
    }

    el.chatForm.addEventListener("submit", handleQuerySubmit);
    el.chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        el.chatForm.requestSubmit();
      }
    });
  }

  // --- Bootstrap ---
  async function init() {
    initEventListeners();
    await checkSystemStatus();
    await Promise.all([loadDocuments(), loadGraph(), loadBenchmarks()]);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
