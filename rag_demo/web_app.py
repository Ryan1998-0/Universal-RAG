import argparse
import json
import html
import mimetypes
import os
from dataclasses import replace
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from time import perf_counter
from urllib.parse import parse_qs, quote, unquote, urlparse

from rag_demo.config import RagConfig
from rag_demo.access_control import AccessControlError, AccessPolicyStore
from rag_demo.conversation_store import ConversationStore
from rag_demo.document_pipeline import (
    MAX_DOCUMENT_UPLOAD_BYTES,
    SUPPORTED_FORMATS,
    DocumentPipelineError,
    DocumentStore,
)
from rag_demo.hybrid_retrieval import (
    evaluate_retrieval_evidence,
    get_hybrid_retriever,
    list_profile_retrieval_configs,
    load_profile_retrieval_config,
    register_uploaded_document,
)
from rag_demo.fine_evidence import retrieve_fine_evidence
from rag_demo.knowledge_base import active_knowledge_base
from rag_demo.query_rewriter import QueryRewriteDecision, decide_and_rewrite_query_for_retrieval
from rag_demo.model_providers import ask_model, parse_model_spec
from rag_demo.rag_pipeline import (
    RagPipeline,
    RagPipelineInputError,
    RagPipelineRequest,
    allowed_models_from_env,
    answer_from_contexts as _answer_from_browser_contexts,
    conversation_context as _conversation_context,
    normalize_contexts,
    rag_conversation_context as _rag_conversation_context,
)
DEFAULT_MODEL = os.getenv("RAG_MODEL", "ollama:qwen2.5:7b")
DEFAULT_PROFILE = os.getenv("RAG_PROFILE", "default")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATIC_ROOT = PROJECT_ROOT / "docs" / "rag-demo"
if not DEFAULT_STATIC_ROOT.is_dir():
    DEFAULT_STATIC_ROOT = PROJECT_ROOT / "docs"
STATIC_ROOT = Path(
    os.getenv("RAG_STATIC_ROOT", str(DEFAULT_STATIC_ROOT))
).expanduser().resolve(strict=False)
CONVERSATION_STORE = None
DOCUMENT_STORE = None
RAG_PIPELINE = None
ACCESS_POLICY_STORE = None


def _conversation_store():
    global CONVERSATION_STORE
    if CONVERSATION_STORE is None:
        CONVERSATION_STORE = ConversationStore()
    return CONVERSATION_STORE


def _document_store():
    global DOCUMENT_STORE
    if DOCUMENT_STORE is None:
        DOCUMENT_STORE = DocumentStore()
    return DOCUMENT_STORE


def _rag_pipeline():
    global RAG_PIPELINE
    if RAG_PIPELINE is None:
        RAG_PIPELINE = RagPipeline(conversation_store=_conversation_store())
    return RAG_PIPELINE


def _access_policy_store():
    global ACCESS_POLICY_STORE
    if ACCESS_POLICY_STORE is None:
        ACCESS_POLICY_STORE = AccessPolicyStore()
    return ACCESS_POLICY_STORE


def render_home(
    answer_html: str = "",
    question: str = "",
    model: str = DEFAULT_MODEL,
    top_k: int = None,
) -> str:
    config = RagConfig.from_env()
    knowledge_base = active_knowledge_base()
    top_k = top_k or config.top_k
    escaped_question = html.escape(question)
    escaped_model = html.escape(model)
    escaped_profile = html.escape(knowledge_base.name)
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>泛用 RAG 工作台</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #202124;
      --muted: #5f6368;
      --line: #d8dee4;
      --panel: #ffffff;
      --page: #f6f8fa;
      --accent: #0f766e;
      --accent-2: #b42318;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: var(--page);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.55;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }}
    .wrap {{
      width: min(1080px, calc(100vw - 32px));
      margin: 0 auto;
    }}
    .top {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 0;
    }}
    h1 {{
      margin: 0;
      font-size: 22px;
      letter-spacing: 0;
    }}
    .badge {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 6px 10px;
      color: var(--muted);
      background: #fafafa;
      font-size: 13px;
      white-space: nowrap;
    }}
    main {{
      padding: 28px 0 40px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(280px, 380px) 1fr;
      gap: 20px;
      align-items: start;
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }}
    h2 {{
      margin: 0 0 14px;
      font-size: 16px;
      letter-spacing: 0;
    }}
    label {{
      display: block;
      margin-bottom: 8px;
      font-size: 14px;
      color: var(--muted);
    }}
    textarea {{
      width: 100%;
      min-height: 132px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      font: inherit;
      color: var(--ink);
      background: #ffffff;
    }}
    input, select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 12px;
      font: inherit;
      color: var(--ink);
      background: #ffffff;
    }}
    input:focus, select:focus {{
      outline: 2px solid rgba(15, 118, 110, 0.22);
      border-color: var(--accent);
    }}
    .field {{
      margin-top: 12px;
    }}
    textarea:focus {{
      outline: 2px solid rgba(15, 118, 110, 0.22);
      border-color: var(--accent);
    }}
    button {{
      width: 100%;
      margin-top: 12px;
      border: 0;
      border-radius: 6px;
      padding: 11px 14px;
      background: var(--accent);
      color: #ffffff;
      font: inherit;
      font-weight: 650;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      min-height: 46px;
    }}
    button[disabled] {{
      cursor: wait;
      opacity: 0.82;
    }}
    .submit-spinner {{
      width: 16px;
      height: 16px;
      border: 2px solid rgba(255, 255, 255, 0.45);
      border-top-color: #ffffff;
      border-radius: 50%;
      display: none;
      flex: 0 0 auto;
    }}
    .is-loading .submit-spinner {{
      display: inline-block;
      animation: spin 0.8s linear infinite;
    }}
    @keyframes spin {{
      to {{ transform: rotate(360deg); }}
    }}
    .samples {{
      display: grid;
      gap: 8px;
      margin-top: 14px;
    }}
    .sample {{
      display: block;
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      background: #fafafa;
      color: var(--ink);
      text-align: left;
      font: inherit;
      cursor: pointer;
    }}
    pre {{
      min-height: 320px;
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 14px;
      line-height: 1.7;
    }}
    .empty {{
      color: var(--muted);
    }}
    .warn {{
      margin-top: 12px;
      color: var(--accent-2);
      font-size: 13px;
    }}
    @media (max-width: 760px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .top {{ align-items: flex-start; flex-direction: column; }}
      .badge {{ white-space: normal; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="wrap top">
      <h1>泛用 RAG 工作台</h1>
      <div class="badge">知識庫：{escaped_profile}／模型：{escaped_model}</div>
    </div>
  </header>
  <main class="wrap">
    <div class="grid">
      <section>
        <h2>提問</h2>
        <form method="post" action="/ask" id="ask-form">
          <label for="question">問題</label>
          <textarea id="question" name="question" autofocus>{escaped_question}</textarea>
          <div class="field">
            <label for="model">模型</label>
            <input id="model" name="model" value="{escaped_model}">
          </div>
          <div class="field">
            <label for="top_k">取回片段數量（top_k）</label>
            <input id="top_k" name="top_k" type="number" min="1" max="10" value="{top_k}">
          </div>
          <button type="submit" id="submit-button" data-loading-text="處理中">
            <span class="submit-spinner" aria-hidden="true"></span>
            <span class="submit-label">送出問題</span>
          </button>
        </form>
        <div class="warn">本機模型可能需要幾秒鐘；雲端模型需要設定對應的 API 金鑰。</div>
      </section>
      <section>
        <h2>回答</h2>
        <pre>{answer_html or '<span class="empty">尚未送出問題。</span>'}</pre>
      </section>
    </div>
  </main>
  <script>
    const askForm = document.querySelector("#ask-form");
    const submitButton = document.querySelector("#submit-button");
    if (askForm && submitButton) {{
      askForm.addEventListener("submit", () => {{
        const label = submitButton.querySelector(".submit-label");
        submitButton.classList.add("is-loading");
        submitButton.disabled = true;
        submitButton.setAttribute("aria-busy", "true");
        if (label) {{
          label.textContent = submitButton.dataset.loadingText || "處理中";
        }}
      }});
    }}
  </script>
</body>
</html>"""


class RagRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._send_json({
                "ok": True,
                "model": DEFAULT_MODEL,
                "profile": DEFAULT_PROFILE,
                "static_root": str(STATIC_ROOT),
                "retrieval": "parent-child(1024/256)+bm25+dense+rrf+complexity-gated-cross-encoder",
                "document_upload": sorted(SUPPORTED_FORMATS),
                "document_folders": True,
            })
            return
        if parsed.path == "/api/config":
            self._send_json(_runtime_config_payload())
            return
        if parsed.path == "/api/access/session":
            self._send_json(self._access_session_payload())
            return
        if parsed.path.startswith("/api/access/policies/"):
            source_id = unquote(parsed.path.removeprefix("/api/access/policies/").strip("/"))
            self._handle_access_policy_get(source_id)
            return
        if parsed.path.startswith("/api/profiles/"):
            profile = unquote(parsed.path.removeprefix("/api/profiles/").strip("/"))
            if not profile:
                self._send_json({"error": "profile is required"}, status=400)
                return
            try:
                self._send_json(_profile_data_payload(profile, principal=self._current_principal()))
            except (ValueError, FileNotFoundError) as exc:
                self._send_json({"error": str(exc)}, status=400)
            return
        if parsed.path == "/api/documents":
            self._send_json({"documents": self._visible_uploaded_documents()})
            return
        if parsed.path == "/api/folders":
            self._send_json({"folders": self._visible_folders()})
            return
        if parsed.path.startswith("/api/documents/") and parsed.path.endswith("/download"):
            self._handle_document_download(parsed.path)
            return
        if parsed.path == "/api/conversations":
            self._send_json({"conversations": _conversation_store().list_conversations()})
            return
        if parsed.path.startswith("/api/conversations/"):
            conversation_id = parsed.path.removeprefix("/api/conversations/").strip("/")
            conversation = _conversation_store().get_conversation_with_messages(conversation_id)
            if conversation is None:
                self._send_json({"error": "conversation not found"}, status=404)
            else:
                self._send_json(conversation)
            return
        if parsed.path == "/api/memories":
            self._send_json({"memories": _conversation_store().list_memories()})
            return
        if self._send_static_file(parsed.path):
            return
        if not is_home_path(parsed.path):
            self._send_not_found()
            return
        self._send_html(render_home())

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/retrieve":
            self._handle_api_retrieve()
            return
        if parsed.path == "/api/access/session":
            self._handle_access_session_set()
            return
        if parsed.path == "/api/route":
            self._handle_api_route()
            return
        if parsed.path == "/api/ask":
            self._handle_api_ask()
            return
        if parsed.path == "/api/conversations":
            self._handle_create_conversation()
            return
        if parsed.path == "/api/documents/upload":
            self._handle_document_upload()
            return
        if parsed.path == "/api/folders":
            self._handle_create_folder()
            return

        if parsed.path != "/ask":
            self._send_not_found()
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        question = parse_qs(body).get("question", [""])[0].strip()
        model = parse_qs(body).get("model", [DEFAULT_MODEL])[0].strip() or DEFAULT_MODEL
        top_k = _parse_int(parse_qs(body).get("top_k", [str(RagConfig.from_env().top_k)])[0], RagConfig.from_env().top_k)

        if question:
            source_ids, _ = self._authorized_source_ids(DEFAULT_PROFILE, None)
            answer = _answer_standalone_question(
                question,
                model=model,
                profile=DEFAULT_PROFILE,
                top_k=top_k,
                source_ids=source_ids,
            )
            answer_html = html.escape(answer)
        else:
            answer_html = '<span class="empty">請先輸入問題。</span>'

        self._send_html(render_home(answer_html=answer_html, question=question, model=model, top_k=top_k))

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/folders/"):
            folder_id = unquote(parsed.path.removeprefix("/api/folders/").strip("/"))
            self._handle_delete_folder(folder_id)
            return
        if not parsed.path.startswith("/api/conversations/"):
            self._send_not_found()
            return

        conversation_id = unquote(
            parsed.path.removeprefix("/api/conversations/").strip("/")
        )
        if not conversation_id:
            self._send_json({"error": "conversation id is required"}, status=400)
            return
        if not _conversation_store().delete_conversation(conversation_id):
            self._send_json({"error": "conversation not found"}, status=404)
            return
        self._send_json({"deleted": True, "id": conversation_id})

    def do_PUT(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/access/policies/"):
            source_id = unquote(parsed.path.removeprefix("/api/access/policies/").strip("/"))
            self._handle_access_policy_put(source_id)
            return
        if parsed.path.startswith("/api/folders/"):
            folder_id = unquote(parsed.path.removeprefix("/api/folders/").strip("/"))
            self._handle_rename_folder(folder_id)
            return
        if parsed.path.startswith("/api/documents/") and parsed.path.endswith("/folder"):
            relative = parsed.path.removeprefix("/api/documents/")
            source_id = unquote(relative.removesuffix("/folder").strip("/"))
            self._handle_move_document(source_id)
            return
        self._send_not_found()

    def _handle_api_route(self) -> None:
        started_at = perf_counter()
        try:
            payload = self._read_json_body()
            question = str(payload.get("question", "")).strip()
            if not question:
                self._send_json({"error": "question is required"}, status=400)
                return

            model = _model_string(payload.get("model") or {})
            conversation_id = str(payload.get("conversation_id") or "").strip()
            history = _conversation_store().get_recent_messages(conversation_id, limit=10) if conversation_id else []
            try:
                decision = decide_and_rewrite_query_for_retrieval(
                    question,
                    model=model,
                    conversation_context=_conversation_context(history),
                )
            except Exception as exc:
                decision = QueryRewriteDecision(
                    needs_retrieval=True,
                    reason=f"檢索判斷失敗，為避免漏掉文件證據，保守改走檢索：{exc}",
                    retrieval_query=question,
                )

            self._send_json({
                "needs_retrieval": decision.needs_retrieval,
                "reason": decision.reason,
                "retrieval_query": decision.retrieval_query,
                "sub_questions": list(getattr(decision, "sub_questions", ()) or ()),
                "query_variants": list(getattr(decision, "query_variants", ()) or ()),
                "rewrite_semantic_validation": {
                    "status": str(getattr(decision, "semantic_validation_status", "skipped") or "skipped"),
                    "similarity": getattr(decision, "semantic_similarity", None),
                    "accepted": bool(getattr(decision, "semantic_accepted", True)),
                    "threshold": getattr(decision, "semantic_validation_threshold", None),
                    "reason": str(getattr(decision, "semantic_validation_reason", "") or ""),
                },
                "timing_ms": round((perf_counter() - started_at) * 1000, 2),
            })
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def _handle_api_retrieve(self) -> None:
        try:
            payload = self._read_json_body()
            question = str(payload.get("question", "")).strip()
            if not question:
                self._send_json({"error": "question is required"}, status=400)
                return

            profile = str(payload.get("profile") or DEFAULT_PROFILE).strip()

            source_ids, denied_source_ids = self._authorized_source_ids(
                profile,
                _source_ids_from_payload(payload),
            )
            settings = RagConfig.from_env()
            query_variants = [
                str(query).strip()[:500]
                for query in (payload.get("query_variants") or [])[:8]
                if str(query).strip()
            ] if isinstance(payload.get("query_variants"), list) else []
            result = get_hybrid_retriever(profile).retrieve(
                question=question,
                retrieval_query=str(payload.get("retrieval_query") or "").strip(),
                query_variants=query_variants,
                evidence_query=str(payload.get("evidence_query") or "").strip()[:2000],
                source_ids=source_ids,
                top_k=_parse_int(
                    str(payload.get("top_k", settings.hybrid_top_k)),
                    settings.hybrid_top_k,
                ),
                candidate_k=_parse_int(
                    str(payload.get("candidate_k", settings.hybrid_candidate_k)),
                    settings.hybrid_candidate_k,
                ),
            )
            contexts = normalize_contexts(
                result.get("contexts"),
                max_contexts=settings.hybrid_max_top_k,
            )
            if contexts and settings.fine_evidence_enabled:
                raw_contexts = [dict(context) for context in contexts]
                contexts, evidence_focus = retrieve_fine_evidence(
                    question=question,
                    contexts=raw_contexts,
                    settings=settings,
                    chunk_fraction=settings.fine_evidence_chunk_fraction,
                )
                result["contexts"] = contexts
                result["raw_contexts"] = raw_contexts
                result["evidence_focus"] = evidence_focus
                result["evidenceEvaluation"] = evaluate_retrieval_evidence(
                    contexts,
                    settings=settings,
                    question=question,
                )
            result["access"] = self._access_result_payload(denied_source_ids)
            self._send_json(result)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def _handle_create_conversation(self) -> None:
        try:
            payload = self._read_json_body()
            conversation = _conversation_store().create_conversation(
                profile=str(payload.get("profile") or DEFAULT_PROFILE),
            )
            conversation["messages"] = []
            self._send_json(conversation, status=201)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def _handle_create_folder(self) -> None:
        try:
            if not self._require_access_admin():
                return
            payload = self._read_json_body()
            folder = _document_store().create_folder(payload.get("name"))
            self._send_json({"folder": folder}, status=201)
        except DocumentPipelineError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def _handle_rename_folder(self, folder_id: str) -> None:
        try:
            if not self._require_access_admin():
                return
            payload = self._read_json_body()
            folder = _document_store().rename_folder(folder_id, payload.get("name"))
            self._send_json({"folder": folder})
        except DocumentPipelineError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def _handle_delete_folder(self, folder_id: str) -> None:
        try:
            if not self._require_access_admin():
                return
            self._send_json(_document_store().delete_folder(folder_id))
        except DocumentPipelineError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def _handle_move_document(self, source_id: str) -> None:
        try:
            if not self._require_access_admin():
                return
            payload = self._read_json_body()
            document = _document_store().move_document(source_id, payload.get("folder_id"))
            self._send_json({"document": document})
        except DocumentPipelineError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def _handle_document_upload(self) -> None:
        if not self._require_access_admin():
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except (TypeError, ValueError):
            length = 0
        if length <= 0:
            self._send_json({"error": "檔案內容是空的。"}, status=400)
            return
        if length > MAX_DOCUMENT_UPLOAD_BYTES:
            self._send_json({"error": "檔案超過 50 MB 上限。"}, status=413)
            return

        filename = unquote(str(self.headers.get("X-File-Name") or "")).strip()
        content_type = str(self.headers.get("Content-Type") or "")
        folder_header = unquote(str(self.headers.get("X-Folder-Id") or "")).strip()
        payload = self.rfile.read(length)
        try:
            document = _document_store().import_document(
                filename,
                payload,
                content_type=content_type,
                folder_id=folder_header or None,
            )
            register_uploaded_document(
                source=document.metadata,
                chunks=document.chunks,
                embeddings=document.embeddings,
            )
            self._send_json(
                document.response_payload(),
                status=200 if document.duplicate else 201,
            )
        except DocumentPipelineError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def _handle_document_download(self, path: str) -> None:
        relative = path.removeprefix("/api/documents/")
        source_id = relative.removesuffix("/download").strip("/")
        original_path = _document_store().original_path(source_id)
        if original_path is None:
            self._send_not_found()
            return
        allowed_ids, _ = self._authorized_source_ids(DEFAULT_PROFILE, [source_id])
        if source_id not in allowed_ids:
            self._send_json({"error": "沒有下載此文件的權限。"}, status=403)
            return
        metadata = next(
            (
                document
                for document in _document_store().list_documents()
                if document.get("source_id") == source_id
            ),
            {},
        )
        filename = str(metadata.get("name") or "document")
        data = original_path.read_bytes()
        self.send_response(200)
        content_type = str(metadata.get("mime_type") or "").strip()
        self.send_header(
            "Content-Type",
            content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream",
        )
        self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quote(filename)}")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_api_ask(self) -> None:
        try:
            payload = self._read_json_body()
            request = RagPipelineRequest.from_payload(
                payload,
                default_model=DEFAULT_MODEL,
                default_profile=DEFAULT_PROFILE,
                allowed_models=allowed_models_from_env(DEFAULT_MODEL),
            )
            source_ids, denied_source_ids = self._authorized_source_ids(
                request.profile,
                request.source_ids,
            )
            result = _rag_pipeline().run(replace(request, source_ids=source_ids))
            result["access"] = self._access_result_payload(denied_source_ids)
            self._send_json(result)
        except RagPipelineInputError as exc:
            self._send_json(
                {"error": str(exc), "code": "INVALID_RAG_REQUEST"},
                status=400,
            )
        except Exception:
            self._send_json(
                {
                    "error": "RAG pipeline request failed",
                    "code": "RAG_PIPELINE_ERROR",
                },
                status=500,
            )

    def log_message(self, format: str, *args) -> None:
        return

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _current_principal(self) -> dict:
        # This local workbench uses a selectable demo profile.  Production must
        # resolve the equivalent principal from a verified OIDC/JWT token.
        cookie_header = str(self.headers.get("Cookie") or "")
        principal_id = ""
        for item in cookie_header.split(";"):
            key, _, value = item.strip().partition("=")
            if key == "rag_demo_principal":
                principal_id = unquote(value)
                break
        return _access_policy_store().resolve_principal(principal_id or "admin-demo")

    def _access_session_payload(self) -> dict:
        store = _access_policy_store()
        principal = self._current_principal()
        return {
            "mode": "local-demo-profile",
            "notice": "此本機頁面的身分切換用於驗證權限規則；正式環境必須改由 OIDC/JWT 驗證。",
            "principal": principal,
            "principals": store.list_principals(),
            "roles": store.list_roles(),
            "can_manage": store.is_admin(principal),
        }

    def _handle_access_session_set(self) -> None:
        try:
            payload = self._read_json_body()
            principal = _access_policy_store().resolve_principal(payload.get("principal_id"))
            encoded_id = quote(principal["id"], safe="")
            self._send_json(
                self._access_session_payload_for(principal),
                headers={"Set-Cookie": f"rag_demo_principal={encoded_id}; Path=/; SameSite=Lax"},
            )
        except AccessControlError as exc:
            self._send_json({"error": str(exc)}, status=400)

    def _access_session_payload_for(self, principal: dict) -> dict:
        store = _access_policy_store()
        return {
            "mode": "local-demo-profile",
            "notice": "此本機頁面的身分切換用於驗證權限規則；正式環境必須改由 OIDC/JWT 驗證。",
            "principal": principal,
            "principals": store.list_principals(),
            "roles": store.list_roles(),
            "can_manage": store.is_admin(principal),
        }

    def _require_access_admin(self) -> bool:
        if _access_policy_store().is_admin(self._current_principal()):
            return True
        self._send_json({"error": "只有系統管理員可以修改文件與權限設定。"}, status=403)
        return False

    def _available_source_ids(self, profile: str) -> list[str]:
        description = get_hybrid_retriever(profile).describe()
        return list(dict.fromkeys(
            str(source.get("source_id") or "").strip()
            for source in (description.get("sources") or [])
            if str(source.get("source_id") or "").strip()
        ))

    def _authorized_source_ids(self, profile: str, requested_source_ids):
        principal = self._current_principal()
        return _access_policy_store().allowed_source_ids(
            principal,
            self._available_source_ids(profile),
            requested_source_ids,
        )

    def _access_result_payload(self, denied_source_ids: list[str]) -> dict:
        principal = self._current_principal()
        return {
            "principal": {"id": principal["id"], "label": principal["label"]},
            "denied_source_count": len(denied_source_ids),
        }

    def _visible_uploaded_documents(self) -> list[dict]:
        documents = _document_store().list_documents()
        allowed_ids, _ = _access_policy_store().allowed_source_ids(
            self._current_principal(),
            [document.get("source_id") for document in documents],
        )
        allowed_set = set(allowed_ids)
        return [document for document in documents if document.get("source_id") in allowed_set]

    def _visible_folders(self) -> list[dict]:
        documents = self._visible_uploaded_documents()
        counts = {}
        for document in documents:
            folder_id = str(document.get("folder_id") or "uncategorized")
            counts[folder_id] = counts.get(folder_id, 0) + 1
        folders = []
        for folder in _document_store().list_folders():
            folder_id = str(folder.get("id") or "")
            if folder.get("system") or folder_id in counts:
                folders.append({**folder, "document_count": counts.get(folder_id, 0)})
        return folders

    def _handle_access_policy_get(self, source_id: str) -> None:
        try:
            if not self._require_access_admin():
                return
            self._ensure_known_source(source_id)
            self._send_json({"policy": _access_policy_store().policy_for_source(source_id)})
        except AccessControlError as exc:
            self._send_json({"error": str(exc)}, status=400)

    def _handle_access_policy_put(self, source_id: str) -> None:
        try:
            if not self._require_access_admin():
                return
            self._ensure_known_source(source_id)
            payload = self._read_json_body()
            policy = _access_policy_store().set_policy(source_id, payload.get("allowed_roles"))
            self._send_json({"policy": policy})
        except AccessControlError as exc:
            self._send_json({"error": str(exc)}, status=400)

    def _ensure_known_source(self, source_id: str) -> None:
        configured_profiles = list_profile_retrieval_configs(default_profile=DEFAULT_PROFILE)
        known_ids = set()
        for config in configured_profiles:
            known_ids.update(self._available_source_ids(config["profile"]))
        if source_id not in known_ids:
            raise AccessControlError("找不到要設定權限的文件。")

    def _send_static_file(self, raw_path: str) -> bool:
        relative_path = "index.html" if raw_path == "/" else unquote(raw_path).lstrip("/")
        candidate = (STATIC_ROOT / relative_path).resolve()
        if not _is_relative_to(candidate, STATIC_ROOT.resolve()) or not candidate.is_file():
            return False
        data = candidate.read_bytes()
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        if candidate.suffix == ".js":
            content_type = "text/javascript"
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8" if _is_text_file(candidate) else content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        return True

    def _send_html(self, content: str) -> None:
        data = content.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_not_found(self) -> None:
        self.send_response(404)
        self.end_headers()

    def _send_json(self, payload: dict, status: int = 200, headers: dict | None = None) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        for key, value in (headers or {}).items():
            self.send_header(str(key), str(value))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def run_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), RagRequestHandler)
    print(f"RAG demo running at http://{host}:{port}")
    server.serve_forever()


def is_home_path(path: str) -> bool:
    return path in {"/", "/ask"}


def _parse_int(value: str, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _source_ids_from_payload(payload: dict):
    if "source_ids" not in payload:
        return None
    source_ids = payload.get("source_ids")
    if not isinstance(source_ids, list):
        return []
    return list(
        dict.fromkeys(
            str(source_id).strip()
            for source_id in source_ids
            if str(source_id).strip()
        )
    )


def _model_string(model_payload: object) -> str:
    if isinstance(model_payload, dict):
        provider = str(model_payload.get("provider") or "ollama")
        name = str(model_payload.get("name") or "qwen2.5:7b")
        return f"{provider}:{name}"
    return str(model_payload or DEFAULT_MODEL)


def _runtime_config_payload() -> dict:
    model = parse_model_spec(DEFAULT_MODEL)
    profiles = list_profile_retrieval_configs(default_profile=DEFAULT_PROFILE)
    settings = RagConfig.from_env().normalized()
    return {
        "default_profile": DEFAULT_PROFILE,
        "default_model": DEFAULT_MODEL,
        "models": [
            {
                "id": DEFAULT_MODEL,
                "provider": model.provider,
                "name": model.model,
                "label": f"{model.provider} {model.model}",
            }
        ],
        "retrieval": {
            "chunk_strategy": settings.chunk_strategy,
            "chunk_size": settings.chunk_size,
            "chunk_overlap_tokens": settings.chunk_overlap_tokens,
            "query_rewrite_semantic_enabled": settings.query_rewrite_semantic_enabled,
            "query_rewrite_min_similarity": settings.query_rewrite_min_similarity,
            "top_k": settings.hybrid_top_k,
            "candidate_k": settings.hybrid_candidate_k,
            "rerank_top_k": settings.rerank_top_k,
            "max_top_k": settings.hybrid_max_top_k,
            "fusion_method": settings.hybrid_fusion_method,
            "complexity_routing_enabled": settings.complexity_routing_enabled,
            "query_complexity_threshold": settings.query_complexity_threshold,
            "simple_query_top_k": settings.simple_query_top_k,
            "keyword_weight": settings.keyword_weight,
            "embedding_weight": settings.embedding_weight,
        },
        "profiles": [
            {
                "id": config["profile"],
                "label": config["label"],
                "sample_queries": config["sample_queries"],
            }
            for config in profiles
        ],
    }


def _profile_data_payload(profile: str, principal: dict | None = None) -> dict:
    config = load_profile_retrieval_config(profile)
    payload = get_hybrid_retriever(config["profile"]).describe()
    uploaded_documents = {
        document["source_id"]: document
        for document in _document_store().list_documents()
    }
    payload["sources"] = [
        {**source, **uploaded_documents.get(source.get("source_id"), {})}
        for source in payload.get("sources") or []
    ]
    if principal is not None:
        allowed_ids, _ = _access_policy_store().allowed_source_ids(
            principal,
            [source.get("source_id") for source in payload["sources"]],
        )
        allowed_set = set(allowed_ids)
        payload["sources"] = [
            source for source in payload["sources"]
            if source.get("source_id") in allowed_set
        ]
    folders = _document_store().list_folders()
    if principal is not None:
        visible_documents = [
            document
            for document in _document_store().list_documents()
            if document.get("source_id") in allowed_set
        ]
        visible_counts = {}
        for document in visible_documents:
            folder_id = str(document.get("folder_id") or "uncategorized")
            visible_counts[folder_id] = visible_counts.get(folder_id, 0) + 1
        folders = [
            {**folder, "document_count": visible_counts.get(str(folder.get("id") or ""), 0)}
            for folder in folders
            if folder.get("system") or str(folder.get("id") or "") in visible_counts
        ]
    payload["folders"] = folders
    payload.update(
        {
            "profile": config["profile"],
            "label": config["label"],
            "sample_queries": config["sample_queries"],
        }
    )
    return payload


def _answer_standalone_question(
    question: str,
    model: str,
    profile: str = DEFAULT_PROFILE,
    top_k: int = None,
    source_ids=None,
) -> str:
    request = RagPipelineRequest(
        question=question,
        model=model,
        profile=profile,
        top_k=top_k,
        source_ids=source_ids,
        persist_conversation=False,
    )
    return _rag_pipeline().run(request)["answer"]


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _is_text_file(path: Path) -> bool:
    return path.suffix.lower() in {".html", ".css", ".js", ".json", ".md", ".txt", ".svg"}


def _normalize_retrieval_decision(raw_decision: object):
    if not isinstance(raw_decision, dict):
        return None
    normalized = {
        "needs_retrieval": bool(raw_decision.get("needs_retrieval")),
        "reason": str(raw_decision.get("reason") or ""),
        "retrieval_query": str(raw_decision.get("retrieval_query") or ""),
    }
    if "rewrite_semantic_validation" in raw_decision:
        normalized["rewrite_semantic_validation"] = dict(
            raw_decision.get("rewrite_semantic_validation") or {}
        )
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local RAG web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
