"""Chainlit interface for the existing local RAG pipeline (loopback only)."""
import asyncio
import json
import logging
import os
from pathlib import Path
import threading

import chainlit as cl
from chainlit.input_widget import MultiSelect, Select, Slider, Tags

from rag_demo import web_app as backend
from rag_demo.rag_pipeline import RagPipelineRequest, allowed_models_from_env
from rag_demo.hybrid_retrieval import register_uploaded_document
from rag_demo.document_pipeline import MAX_DOCUMENT_UPLOAD_BYTES

# The local RAG app must not activate unrelated chat-platform integrations
# merely because tokens exist in the inherited shell environment.
for key in ("DISCORD_BOT_TOKEN", "SLACK_BOT_TOKEN", "SLACK_WEBSOCKET_TOKEN", "TEAMS_APP_ID", "TEAMS_APP_PASSWORD"):
    os.environ.pop(key, None)

from chainlit.server import app
from starlette.responses import FileResponse
from starlette.routing import Route

async def architecture_page(request):
    return FileResponse(backend.PROJECT_ROOT / "docs/rag-demo/architecture.html")

app.router.routes.insert(0, Route("/architecture.html", architecture_page))
for filename in ("architecture-workbench.css", "architecture-workbench.js"):
    async def asset(request, filename=filename):
        return FileResponse(backend.PROJECT_ROOT / "docs/rag-demo/assets" / filename)
    app.router.routes.insert(0, Route("/assets/" + filename, asset))

logger = logging.getLogger(__name__)
UPLOAD_LOCK = threading.Lock()


def principal():
    return backend._access_policy_store().resolve_principal(
        cl.user_session.get("settings", {}).get("principal", "admin-demo")
    )


def profile_data(profile):
    return backend._profile_data_payload(profile, principal=principal())


async def settings_panel(current=None):
    current = current or cl.user_session.get("settings", {})
    runtime = backend._runtime_config_payload()
    profiles = [p["id"] for p in runtime["profiles"]]
    profile = current.get("profile", backend.DEFAULT_PROFILE)
    if profile not in profiles:
        profile = profiles[0]
    data = await asyncio.to_thread(profile_data, profile)
    sources = {str(s["source_id"]): str(s.get("name") or s.get("title") or s["source_id"])
               for s in data.get("sources", [])}
    cl.user_session.set("sources", sources)
    models = allowed_models_from_env(backend.DEFAULT_MODEL)
    identities = [p["id"] for p in backend._access_policy_store().list_principals()]
    folders = backend._document_store().list_folders()
    widgets = [
        Select(id="profile", label="知識庫", values=profiles, initial_value=profile),
        Select(id="model", label="問答模型", values=models,
               initial_value=current.get("model", backend.DEFAULT_MODEL)),
        Select(id="principal", label="本機測試身分", values=identities,
               initial_value=current.get("principal", "admin-demo")),
        Slider(id="top_k", label="檢索片段數", initial=current.get("top_k", runtime["retrieval"]["top_k"]),
               min=1, max=runtime["retrieval"]["max_top_k"], step=1),
        Select(id="folder", label="上傳資料夾", items={f["name"]: f["id"] for f in folders},
               initial_value=current.get("folder", "uncategorized")),
        (MultiSelect(id="source_ids", label="查詢文件（留空搜尋全部授權文件）",
                     items={f"{name} · {sid}": sid for sid, name in sources.items()},
                     initial=[sid for sid in current.get("source_ids", []) if sid in sources])
         if sources else Tags(id="source_ids", label="尚無文件，請先上傳", disabled=True)),
    ]
    settings = await cl.ChatSettings(widgets).send()
    cl.user_session.set("settings", settings)


@cl.on_chat_start
async def start():
    cl.user_session.set("conversation_id", "")
    await settings_panel()
    await cl.Message(content="歡迎使用泛用 RAG 工作台。可直接提問，或附加 PDF、圖片、DOCX、TXT、Markdown、JSON。\n\n"
                     "使用設定選擇知識庫、模型與文件。輸入 `/文件` 查看文件，`/紀錄` 查看舊對話，"
                     "`/載入 對話ID` 接續對話；`/新增資料夾 名稱` 建立上傳資料夾。"
                     "本機版本使用測試身分，請僅在自己的電腦使用。",
                     actions=[cl.Action(name="documents", label="查看文件", payload={}),
                              cl.Action(name="history", label="歷史對話", payload={})]).send()


@cl.on_settings_update
async def update(settings):
    old = cl.user_session.get("settings", {})
    cl.user_session.set("settings", settings)
    if any(old.get(key) != settings.get(key) for key in ("profile", "principal")):
        cl.user_session.set("conversation_id", "")
        settings["source_ids"] = []
        await settings_panel(settings)


@cl.action_callback("documents")
async def documents(action=None):
    data = await asyncio.to_thread(profile_data, cl.user_session.get("settings")["profile"])
    lines = [f"- **{s.get('name') or s.get('title') or s['source_id']}**：`{s['source_id']}`"
             for s in data.get("sources", [])]
    await cl.Message(content="### 授權文件\n" + ("\n".join(lines) or "尚無文件，可附加檔案上傳。")).send()


@cl.action_callback("history")
async def history(action=None):
    rows = await asyncio.to_thread(backend._conversation_store().list_conversations)
    await cl.Message(content="### 歷史對話\n" + ("\n".join(
        f"- {row['title']}（{row['profile']}）：`/載入 {row['id']}`" for row in rows
    ) or "尚無對話。")).send()


def import_file(element, folder):
    with UPLOAD_LOCK:
        return _import_file(element, folder)


def _import_file(element, folder):
    if not element.path:
        raise ValueError("找不到上傳檔案，請重新附加。")
    path = Path(element.path)
    if path.stat().st_size > MAX_DOCUMENT_UPLOAD_BYTES:
        raise ValueError("檔案超過 50 MB 上限。")
    document = backend._document_store().import_document(
        element.name, path.read_bytes(), content_type=element.mime or "", folder_id=folder,
    )
    register_uploaded_document(source=document.metadata, chunks=document.chunks, embeddings=document.embeddings)
    return document.metadata


def run_question(content, settings, conversation_id):
    data = profile_data(settings["profile"])
    available = [s["source_id"] for s in data.get("sources", [])]
    requested = settings.get("source_ids") or None
    allowed, _ = backend._access_policy_store().allowed_source_ids(principal(), available, requested)
    request = RagPipelineRequest.from_payload(
        {"question": content, "model": settings["model"], "profile": settings["profile"],
         "source_ids": allowed, "top_k": int(settings["top_k"]), "conversation_id": conversation_id},
        default_model=backend.DEFAULT_MODEL, default_profile=backend.DEFAULT_PROFILE,
        allowed_models=allowed_models_from_env(backend.DEFAULT_MODEL),
    )
    return backend._rag_pipeline().run(request)


@cl.on_message
async def message(message: cl.Message):
    settings = dict(cl.user_session.get("settings") or {})
    try:
        text = message.content.strip()
        if text == "/文件":
            await documents(); return
        if text == "/紀錄":
            await history(); return
        if text.startswith("/載入 "):
            row = await asyncio.to_thread(backend._conversation_store().get_conversation_with_messages, text.split(maxsplit=1)[1])
            if not row:
                raise ValueError("找不到對話。")
            settings["profile"] = row["profile"]
            settings["source_ids"] = []
            await settings_panel(settings)
            cl.user_session.set("conversation_id", row["id"])
            for item in row["messages"]:
                await cl.Message(content=item["content"], type="user_message" if item["role"] == "user" else "assistant_message").send()
            return
        if text.startswith("/新增資料夾 "):
            if not backend._access_policy_store().is_admin(principal()):
                raise ValueError("只有管理員可以建立資料夾。")
            folder = await asyncio.to_thread(backend._document_store().create_folder, text.split(maxsplit=1)[1])
            settings["folder"] = folder["id"]
            await settings_panel(settings)
            await cl.Message(content=f"已建立資料夾：{folder['name']}").send(); return
        if message.elements:
            if not backend._access_policy_store().is_admin(principal()):
                raise ValueError("只有管理員可以上傳文件。")
            async with cl.Step(name="匯入文件與建立索引", type="tool") as step:
                imported = []
                for element in message.elements:
                    doc = await asyncio.to_thread(import_file, element, settings.get("folder", "uncategorized"))
                    imported.append(doc["source_id"])
                step.output = f"已匯入 {len(imported)} 份文件。"
            # An explicit selection must include newly attached documents.
            if settings.get("source_ids"):
                settings["source_ids"] = list(dict.fromkeys(settings["source_ids"] + imported))
            await settings_panel(settings)
        if not text:
            await cl.Message(content="文件已可檢索，請輸入問題。").send(); return
        async with cl.Step(name="RAG 檢索與回答", type="tool") as step:
            result = await asyncio.to_thread(run_question, text, settings, cl.user_session.get("conversation_id", ""))
            step.output = "已完成。" if result["retrieval"]["needed"] else "本次直接回答。"
        cl.user_session.set("conversation_id", result["conversation_id"])
        elements = [cl.Text(name=f"證據 {c.get('rank', i)}", content=f"來源：{c.get('title') or c.get('source_id') or '文件'}\n頁碼：{c.get('page', '')}\n\n{c.get('content', '')}", display="side")
                    for i, c in enumerate(result["retrieval"].get("contexts", []), 1)]
        elements.append(cl.Text(name="檢索詳情", content=json.dumps(
            {"retrieval": result["retrieval"], "timings": result.get("timings"),
             "grounding_warnings": result.get("grounding_warnings")}, ensure_ascii=False, indent=2), display="side"))
        await cl.Message(content=result["answer"] + "\n\n" + " · ".join(element.name for element in elements), elements=elements).send()
    except Exception as exc:
        logger.exception("Chainlit request failed")
        await cl.Message(content=f"處理失敗：{exc}").send()
