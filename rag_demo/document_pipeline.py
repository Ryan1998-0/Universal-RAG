from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import platform
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from rag_demo.config import RagConfig
from rag_demo.embeddings import DEFAULT_EMBEDDING_MODEL, embed_chunks
from rag_demo.ollama_client import ask_ollama_vision
from rag_demo.word_documents import extract_docx_blocks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPLOAD_ROOT = Path(
    os.getenv("RAG_UPLOAD_ROOT", str(PROJECT_ROOT / ".local" / "uploaded_documents"))
).expanduser()
VISION_OCR_SCRIPT = PROJECT_ROOT / "scripts" / "macos_vision_ocr.swift"
MAX_DOCUMENT_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_TEXT_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 200
MAX_EXTRACTED_CHARACTERS = 5_000_000
MIN_PDF_TEXT_CHARACTERS = 24
DOCX_RASTER_IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DOCX_VISION_PROMPT = """你正在辨識公司內部操作手冊的畫面截圖。
請使用繁體中文輸出，並完成以下工作：
1. 逐字保留可辨識的介面文字、欄位名稱、按鈕、日期、代碼、檔名與路徑。
2. 說明畫面顯示的操作動作與選取條件。
3. 表格或樞紐分析請列出欄位配置、重要數值及篩選狀態。
4. 看不清楚的內容標記為「無法辨識」，不要猜測。
只輸出可供知識庫檢索的內容，不要加入開場白。"""
_SOURCE_ID_PATTERN = re.compile(r"^upload-[0-9a-f]{20}$")
_FOLDER_ID_PATTERN = re.compile(r"^folder-[0-9a-f]{12}$")
FOLDER_REGISTRY_FILENAME = "_folders.json"
UNCATEGORIZED_FOLDER_ID = "uncategorized"
UNCATEGORIZED_FOLDER_NAME = "未分類"
MAX_FOLDER_NAME_LENGTH = 80

DOCX_MIME_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
SUPPORTED_FORMATS: Dict[str, dict] = {
    ".docx": {"source_type": "word", "mime_type": DOCX_MIME_TYPE},
    ".pdf": {"source_type": "pdf", "mime_type": "application/pdf"},
    ".txt": {"source_type": "text", "mime_type": "text/plain"},
    ".md": {"source_type": "markdown", "mime_type": "text/markdown"},
    ".markdown": {"source_type": "markdown", "mime_type": "text/markdown"},
    ".json": {"source_type": "json", "mime_type": "application/json"},
    ".png": {"source_type": "image", "mime_type": "image/png"},
    ".jpg": {"source_type": "image", "mime_type": "image/jpeg"},
    ".jpeg": {"source_type": "image", "mime_type": "image/jpeg"},
    ".webp": {"source_type": "image", "mime_type": "image/webp"},
    ".tif": {"source_type": "image", "mime_type": "image/tiff"},
    ".tiff": {"source_type": "image", "mime_type": "image/tiff"},
    ".bmp": {"source_type": "image", "mime_type": "image/bmp"},
    ".heic": {"source_type": "image", "mime_type": "image/heic"},
}
SUPPORTED_EXTENSION_LABEL = ", ".join(sorted(SUPPORTED_FORMATS))


class DocumentPipelineError(ValueError):
    pass


@dataclass(frozen=True)
class ExtractionResult:
    units: List[dict]
    method: str
    source_type: str
    mime_type: str
    ocr_used: bool = False
    warnings: Sequence[str] = ()
    details: Optional[dict] = None


@dataclass(frozen=True)
class IndexedDocument:
    metadata: dict
    chunks: List[dict]
    embeddings: List[List[float]]
    duplicate: bool = False

    def response_payload(self) -> dict:
        return {"document": dict(self.metadata), "duplicate": self.duplicate}


class DocumentStore:
    def __init__(
        self,
        root: Path = DEFAULT_UPLOAD_ROOT,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embed_chunks_fn: Optional[Callable[[Sequence[dict]], Sequence[Sequence[float]]]] = None,
        now_fn: Optional[Callable[[], datetime]] = None,
        ocr_fn: Optional[Callable[[Path], dict]] = None,
        image_understanding_fn: Optional[Callable[[Path], dict]] = None,
        pdf_reader_factory: Optional[Callable[[BytesIO], object]] = None,
        pdf_page_renderer: Optional[Callable[[Path, int, Path], Path]] = None,
    ):
        self.root = Path(root)
        self.embedding_model = embedding_model
        self._embed_chunks_fn = embed_chunks_fn or (
            lambda chunks: embed_chunks(chunks, model_name=self.embedding_model)
        )
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self._ocr_fn = ocr_fn or perform_ocr
        self._image_understanding_fn = image_understanding_fn or perform_image_understanding
        self._pdf_reader_factory = pdf_reader_factory
        self._pdf_page_renderer = pdf_page_renderer or render_pdf_page
        self._lock = threading.RLock()

    def import_document(
        self,
        filename: str,
        payload: bytes,
        content_type: str = "",
        folder_id: Optional[str] = None,
    ) -> IndexedDocument:
        started_at = perf_counter()
        clean_filename, extension, format_spec = normalize_document_filename(filename)
        _validate_payload(payload, extension)

        digest = hashlib.sha256(payload).hexdigest()
        identity_digest = (
            digest
            if extension == ".docx"
            else hashlib.sha256(extension.encode("ascii") + b"\0" + payload).hexdigest()
        )
        source_id = f"upload-{identity_digest[:20]}"
        with self._lock:
            clean_folder_id = (
                self._require_folder_id(folder_id)
                if folder_id is not None
                else UNCATEGORIZED_FOLDER_ID
            )
            existing = self._load_indexed_document(source_id)
            if existing is not None:
                if folder_id is not None and existing.metadata.get("folder_id") != clean_folder_id:
                    metadata = self.move_document(source_id, clean_folder_id)
                else:
                    metadata = self._decorate_folder(existing.metadata)
                return IndexedDocument(
                    metadata=metadata,
                    chunks=existing.chunks,
                    embeddings=existing.embeddings,
                    duplicate=True,
                )

        extraction = extract_document(
            filename=clean_filename,
            payload=payload,
            extension=extension,
            ocr_fn=self._ocr_fn,
            image_understanding_fn=self._image_understanding_fn,
            pdf_reader_factory=self._pdf_reader_factory,
            pdf_page_renderer=self._pdf_page_renderer,
        )
        chunks = build_document_chunks(
            extraction.units,
            source_id=source_id,
            filename=clean_filename,
            source_type=extraction.source_type,
            extraction_method=extraction.method,
        )
        if not chunks:
            raise DocumentPipelineError("找不到可建立索引的文字內容。")

        embeddings = np.asarray(self._embed_chunks_fn(chunks), dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks) or embeddings.shape[1] <= 0:
            raise DocumentPipelineError("Embedding 建立失敗，回傳維度與文件 chunks 不一致。")

        stored_filename = f"original{extension}"
        extraction_details = dict(extraction.details or {})
        metadata = {
            "source_id": source_id,
            "name": clean_filename,
            "source_type": extraction.source_type,
            "file_extension": extension,
            "mime_type": extraction.mime_type or format_spec["mime_type"],
            "sha256": digest,
            "chunk_count": len(chunks),
            "uploaded_at": self._now_fn().astimezone(timezone.utc).isoformat(),
            "embedding_model": self.embedding_model,
            "stored_filename": stored_filename,
            "selected_by_default": False,
            "status": "ready",
            "folder_id": clean_folder_id,
            "extraction": {
                "method": extraction.method,
                "ocr_used": extraction.ocr_used,
                "warnings": list(extraction.warnings),
                **extraction_details,
            },
            "pipeline": [
                {"stage": "detect_format", "status": "completed"},
                {"stage": "extract_text", "status": "completed"},
                {"stage": "normalize", "status": "completed"},
                {"stage": "chunk", "status": "completed", "count": len(chunks)},
                {"stage": "embed", "status": "completed", "model": self.embedding_model},
                {"stage": "persist", "status": "completed"},
            ],
            "processing_ms": round((perf_counter() - started_at) * 1000.0, 2),
            "url": f"/api/documents/{source_id}/download",
        }

        self.root.mkdir(parents=True, exist_ok=True)
        temporary_dir = self.root / f".tmp-{uuid.uuid4().hex}"
        destination = self.root / source_id
        try:
            temporary_dir.mkdir(parents=False, exist_ok=False)
            (temporary_dir / stored_filename).write_bytes(payload)
            (temporary_dir / "chunks.json").write_text(
                json.dumps(chunks, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            np.save(temporary_dir / "embeddings.npy", embeddings)
            (temporary_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            with self._lock:
                existing = self._load_indexed_document(source_id)
                if existing is not None:
                    return IndexedDocument(
                        metadata=self._decorate_folder(existing.metadata),
                        chunks=existing.chunks,
                        embeddings=existing.embeddings,
                        duplicate=True,
                    )
                temporary_dir.replace(destination)
        finally:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir, ignore_errors=True)

        return IndexedDocument(
            metadata=self._decorate_folder(metadata),
            chunks=chunks,
            embeddings=embeddings.astype(float).tolist(),
            duplicate=False,
        )

    def list_documents(self) -> List[dict]:
        documents = [
            self._decorate_folder(document.metadata)
            for document in self.load_indexed_documents()
        ]
        return sorted(documents, key=lambda item: str(item.get("uploaded_at") or ""), reverse=True)

    def list_folders(self) -> List[dict]:
        with self._lock:
            folders = self._load_folder_registry()
            counts = {UNCATEGORIZED_FOLDER_ID: 0}
            for folder in folders:
                counts[folder["id"]] = 0
            for metadata in self._iter_document_metadata():
                folder_id = str(metadata.get("folder_id") or UNCATEGORIZED_FOLDER_ID)
                if folder_id not in counts:
                    folder_id = UNCATEGORIZED_FOLDER_ID
                counts[folder_id] += 1
            uncategorized = {
                "id": UNCATEGORIZED_FOLDER_ID,
                "name": UNCATEGORIZED_FOLDER_NAME,
                "system": True,
                "document_count": counts[UNCATEGORIZED_FOLDER_ID],
            }
            return [
                uncategorized,
                *[
                    {**folder, "system": False, "document_count": counts.get(folder["id"], 0)}
                    for folder in sorted(folders, key=lambda item: str(item["name"]).casefold())
                ],
            ]

    def create_folder(self, name: str) -> dict:
        clean_name = normalize_folder_name(name)
        with self._lock:
            folders = self._load_folder_registry()
            self._ensure_unique_folder_name(clean_name, folders)
            timestamp = self._now_fn().astimezone(timezone.utc).isoformat()
            folder = {
                "id": f"folder-{uuid.uuid4().hex[:12]}",
                "name": clean_name,
                "created_at": timestamp,
                "updated_at": timestamp,
            }
            folders.append(folder)
            self._save_folder_registry(folders)
            return {**folder, "system": False, "document_count": 0}

    def rename_folder(self, folder_id: str, name: str) -> dict:
        clean_name = normalize_folder_name(name)
        with self._lock:
            clean_folder_id = self._require_mutable_folder_id(folder_id)
            folders = self._load_folder_registry()
            self._ensure_unique_folder_name(clean_name, folders, exclude_id=clean_folder_id)
            folder = next(item for item in folders if item["id"] == clean_folder_id)
            folder["name"] = clean_name
            folder["updated_at"] = self._now_fn().astimezone(timezone.utc).isoformat()
            self._save_folder_registry(folders)
            document_count = sum(
                1
                for metadata in self._iter_document_metadata()
                if metadata.get("folder_id") == clean_folder_id
            )
            return {**folder, "system": False, "document_count": document_count}

    def delete_folder(self, folder_id: str) -> dict:
        with self._lock:
            clean_folder_id = self._require_mutable_folder_id(folder_id)
            folders = self._load_folder_registry()
            folder = next(item for item in folders if item["id"] == clean_folder_id)
            moved_document_count = 0
            timestamp = self._now_fn().astimezone(timezone.utc).isoformat()
            for metadata in self._iter_document_metadata():
                if metadata.get("folder_id") != clean_folder_id:
                    continue
                metadata["folder_id"] = UNCATEGORIZED_FOLDER_ID
                metadata["folder_updated_at"] = timestamp
                self._write_document_metadata(metadata["source_id"], metadata)
                moved_document_count += 1
            self._save_folder_registry(
                [item for item in folders if item["id"] != clean_folder_id]
            )
            return {
                "deleted": True,
                "folder": {**folder, "system": False},
                "moved_document_count": moved_document_count,
                "destination_folder_id": UNCATEGORIZED_FOLDER_ID,
            }

    def move_document(self, source_id: str, folder_id: str) -> dict:
        with self._lock:
            clean_folder_id = self._require_folder_id(folder_id)
            document = self._load_indexed_document(source_id)
            if document is None:
                raise DocumentPipelineError("找不到要移動的文件。")
            metadata = dict(document.metadata)
            metadata["folder_id"] = clean_folder_id
            metadata["folder_updated_at"] = self._now_fn().astimezone(timezone.utc).isoformat()
            self._write_document_metadata(source_id, metadata)
            return self._decorate_folder(metadata)

    def load_indexed_documents(self) -> List[IndexedDocument]:
        if not self.root.exists():
            return []
        documents = []
        for path in sorted(self.root.iterdir()):
            if not path.is_dir() or not _SOURCE_ID_PATTERN.fullmatch(path.name):
                continue
            try:
                document = self._load_indexed_document(path.name)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                document = None
            if document is not None:
                documents.append(document)
        return documents

    def original_path(self, source_id: str) -> Optional[Path]:
        if not _SOURCE_ID_PATTERN.fullmatch(str(source_id or "")):
            return None
        directory = (self.root / source_id).resolve()
        try:
            directory.relative_to(self.root.resolve())
        except ValueError:
            return None
        metadata_path = directory / "metadata.json"
        if not metadata_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        candidates = [str(metadata.get("stored_filename") or "")]
        suffix = Path(str(metadata.get("name") or "")).suffix.lower()
        if suffix:
            candidates.append(f"original{suffix}")
        candidates.append("original.docx")
        for candidate in candidates:
            if not candidate:
                continue
            path = (directory / Path(candidate).name).resolve()
            try:
                path.relative_to(directory)
            except ValueError:
                continue
            if path.is_file():
                return path
        return None

    def _load_indexed_document(self, source_id: str) -> Optional[IndexedDocument]:
        if not _SOURCE_ID_PATTERN.fullmatch(str(source_id or "")):
            return None
        directory = self.root / source_id
        metadata_path = directory / "metadata.json"
        chunks_path = directory / "chunks.json"
        embeddings_path = directory / "embeddings.npy"
        if not metadata_path.is_file() or not chunks_path.is_file() or not embeddings_path.is_file():
            return None

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        embeddings_array = np.load(embeddings_path, allow_pickle=False).astype(np.float32)
        if metadata.get("source_id") != source_id:
            raise ValueError("Stored document source ID is inconsistent.")
        if not isinstance(chunks, list) or embeddings_array.ndim != 2:
            raise ValueError("Stored document index is invalid.")
        if len(chunks) != embeddings_array.shape[0]:
            raise ValueError("Stored document chunk and embedding counts differ.")
        if metadata.get("embedding_model") != self.embedding_model:
            raise ValueError("Stored document uses a different embedding model.")
        metadata.setdefault("selected_by_default", False)
        metadata.setdefault("status", "ready")
        metadata.setdefault("folder_id", UNCATEGORIZED_FOLDER_ID)
        metadata.setdefault("source_type", _source_type_for_name(metadata.get("name")))
        metadata.setdefault("file_extension", Path(str(metadata.get("name") or "")).suffix.lower())
        metadata.setdefault("mime_type", mimetypes.guess_type(str(metadata.get("name") or ""))[0] or "application/octet-stream")
        return IndexedDocument(
            metadata=metadata,
            chunks=chunks,
            embeddings=embeddings_array.astype(float).tolist(),
            duplicate=False,
        )

    def _folder_registry_path(self) -> Path:
        return self.root / FOLDER_REGISTRY_FILENAME

    def _load_folder_registry(self) -> List[dict]:
        path = self._folder_registry_path()
        if not path.is_file():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DocumentPipelineError("資料夾設定檔損壞，無法讀取。") from exc
        folders = payload.get("folders") if isinstance(payload, dict) else None
        if not isinstance(folders, list):
            raise DocumentPipelineError("資料夾設定檔格式錯誤。")
        clean_folders = []
        seen_ids = set()
        seen_names = set()
        for folder in folders:
            if not isinstance(folder, dict):
                raise DocumentPipelineError("資料夾設定檔包含無效項目。")
            folder_id = str(folder.get("id") or "")
            name = normalize_folder_name(folder.get("name"))
            if not _FOLDER_ID_PATTERN.fullmatch(folder_id):
                raise DocumentPipelineError("資料夾設定檔包含無效 ID。")
            if folder_id in seen_ids or name.casefold() in seen_names:
                raise DocumentPipelineError("資料夾設定檔包含重複項目。")
            seen_ids.add(folder_id)
            seen_names.add(name.casefold())
            clean_folders.append(
                {
                    "id": folder_id,
                    "name": name,
                    "created_at": str(folder.get("created_at") or ""),
                    "updated_at": str(folder.get("updated_at") or ""),
                }
            )
        return clean_folders

    def _save_folder_registry(self, folders: Sequence[dict]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._folder_registry_path()
        temporary_path = self.root / f".{FOLDER_REGISTRY_FILENAME}.{uuid.uuid4().hex}.tmp"
        temporary_path.write_text(
            json.dumps({"schema_version": 1, "folders": list(folders)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(path)

    def _require_folder_id(self, folder_id: object) -> str:
        clean_folder_id = str(folder_id or "").strip()
        if clean_folder_id == UNCATEGORIZED_FOLDER_ID:
            return clean_folder_id
        if not _FOLDER_ID_PATTERN.fullmatch(clean_folder_id):
            raise DocumentPipelineError("資料夾 ID 無效。")
        if not any(folder["id"] == clean_folder_id for folder in self._load_folder_registry()):
            raise DocumentPipelineError("找不到指定的資料夾。")
        return clean_folder_id

    def _require_mutable_folder_id(self, folder_id: object) -> str:
        clean_folder_id = str(folder_id or "").strip()
        if clean_folder_id == UNCATEGORIZED_FOLDER_ID:
            raise DocumentPipelineError("系統的未分類資料夾不能重新命名或刪除。")
        return self._require_folder_id(clean_folder_id)

    @staticmethod
    def _ensure_unique_folder_name(
        name: str,
        folders: Sequence[dict],
        exclude_id: str = "",
    ) -> None:
        if name.casefold() == UNCATEGORIZED_FOLDER_NAME.casefold():
            raise DocumentPipelineError("這個資料夾名稱由系統保留。")
        if any(
            folder["id"] != exclude_id and str(folder["name"]).casefold() == name.casefold()
            for folder in folders
        ):
            raise DocumentPipelineError("已有相同名稱的資料夾。")

    def _iter_document_metadata(self):
        if not self.root.exists():
            return
        for directory in sorted(self.root.iterdir()):
            if not directory.is_dir() or not _SOURCE_ID_PATTERN.fullmatch(directory.name):
                continue
            metadata_path = directory / "metadata.json"
            if not metadata_path.is_file():
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if metadata.get("source_id") == directory.name:
                metadata.setdefault("folder_id", UNCATEGORIZED_FOLDER_ID)
                yield metadata

    def _write_document_metadata(self, source_id: str, metadata: dict) -> None:
        if not _SOURCE_ID_PATTERN.fullmatch(str(source_id or "")):
            raise DocumentPipelineError("文件 ID 無效。")
        directory = self.root / source_id
        if not directory.is_dir():
            raise DocumentPipelineError("找不到文件索引目錄。")
        temporary_path = directory / f".metadata.{uuid.uuid4().hex}.tmp"
        temporary_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(directory / "metadata.json")

    def _decorate_folder(self, metadata: dict) -> dict:
        item = dict(metadata)
        folder_id = str(item.get("folder_id") or UNCATEGORIZED_FOLDER_ID)
        folders = {folder["id"]: folder for folder in self._load_folder_registry()}
        folder = folders.get(folder_id)
        if folder is None:
            folder_id = UNCATEGORIZED_FOLDER_ID
            folder_name = UNCATEGORIZED_FOLDER_NAME
        else:
            folder_name = folder["name"]
        item["folder_id"] = folder_id
        item["folder_name"] = folder_name
        return item


def normalize_document_filename(filename: str):
    clean_filename = Path(str(filename or "").replace("\\", "/")).name.strip()
    clean_filename = re.sub(r"[\x00-\x1f\x7f]", "", clean_filename).strip()
    if not clean_filename:
        raise DocumentPipelineError("缺少檔名。")
    extension = Path(clean_filename).suffix.lower()
    if extension == ".doc":
        raise DocumentPipelineError("舊式 .doc 請先在 Word 另存為 .docx。")
    if extension not in SUPPORTED_FORMATS:
        raise DocumentPipelineError(f"不支援 {extension or '無副檔名'}；可用格式：{SUPPORTED_EXTENSION_LABEL}。")
    if len(clean_filename) > 180:
        stem = Path(clean_filename).stem[: max(1, 179 - len(extension))].rstrip()
        clean_filename = f"{stem or 'document'}{extension}"
    return clean_filename, extension, SUPPORTED_FORMATS[extension]


def normalize_folder_name(name: object) -> str:
    clean_name = re.sub(r"[\x00-\x1f\x7f]", "", str(name or "")).strip()
    clean_name = re.sub(r"\s+", " ", clean_name)
    if not clean_name:
        raise DocumentPipelineError("資料夾名稱不能是空白。")
    if len(clean_name) > MAX_FOLDER_NAME_LENGTH:
        raise DocumentPipelineError(f"資料夾名稱不能超過 {MAX_FOLDER_NAME_LENGTH} 個字元。")
    return clean_name


def extract_document(
    filename: str,
    payload: bytes,
    extension: str,
    ocr_fn: Callable[[Path], dict] = None,
    image_understanding_fn: Callable[[Path], dict] = None,
    pdf_reader_factory: Optional[Callable[[BytesIO], object]] = None,
    pdf_page_renderer: Optional[Callable[[Path, int, Path], Path]] = None,
) -> ExtractionResult:
    ocr_fn = ocr_fn or perform_ocr
    image_understanding_fn = image_understanding_fn or perform_image_understanding
    pdf_page_renderer = pdf_page_renderer or render_pdf_page
    if extension == ".docx":
        return _extract_docx(
            filename,
            payload,
            ocr_fn=ocr_fn,
            image_understanding_fn=image_understanding_fn,
        )
    if extension == ".pdf":
        return _extract_pdf(
            filename,
            payload,
            ocr_fn=ocr_fn,
            pdf_reader_factory=pdf_reader_factory,
            pdf_page_renderer=pdf_page_renderer,
        )
    if extension in {".txt", ".md", ".markdown", ".json"}:
        text = decode_text_payload(payload)
        if extension == ".json":
            return _extract_json(filename, text)
        if extension in {".md", ".markdown"}:
            return _extract_markdown(filename, text)
        return _extract_plain_text(filename, text)
    if extension in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".heic"}:
        return _extract_image(filename, payload, extension, ocr_fn)
    raise DocumentPipelineError(f"尚未實作 {extension} 的解析器。")


def build_document_chunks(
    units: Sequence[dict],
    source_id: str,
    filename: str,
    source_type: str,
    extraction_method: str,
    chunk_size: Optional[int] = None,
    chunk_stride: Optional[int] = None,
) -> List[dict]:
    config = RagConfig.from_env()
    chunk_size = max(200, int(chunk_size or config.chunk_size))
    chunk_stride = min(chunk_size, max(1, int(chunk_stride or config.chunk_stride)))
    parent_title = Path(filename).stem or filename
    chunks = []
    extracted_character_count = 0
    for unit_index, unit in enumerate(units, start=1):
        content = normalize_extracted_text(unit.get("content"))
        if not content:
            continue
        remaining_characters = MAX_EXTRACTED_CHARACTERS - extracted_character_count
        if remaining_characters <= 0:
            raise DocumentPipelineError(
                f"文件可索引文字超過 {MAX_EXTRACTED_CHARACTERS:,} 字上限。"
            )
        if len(content) > remaining_characters:
            raise DocumentPipelineError(
                f"文件可索引文字超過 {MAX_EXTRACTED_CHARACTERS:,} 字上限。"
            )
        extracted_character_count += len(content)
        unit_title = str(unit.get("title") or f"Section {unit_index}").strip()
        page = str(unit.get("page") or f"區段 {unit_index}")
        pieces = list(split_sized_text(content, chunk_size, chunk_stride))
        for part_index, piece in enumerate(pieces, start=1):
            chunk_index = len(chunks)
            title = f"{filename} | {unit_title}"
            if len(pieces) > 1:
                title = f"{title} / part {part_index}"
            chunk = {
                "id": f"{source_id}::{chunk_index}",
                "source_id": source_id,
                "source": source_id,
                "source_type": source_type,
                "chunk_index": chunk_index,
                "parent_title": parent_title,
                "title": title,
                "page": page,
                "content": piece,
                "extraction_method": str(unit.get("extraction_method") or extraction_method),
            }
            for key in ("ocr_confidence", "ocr_engine", "json_path"):
                if key in unit:
                    chunk[key] = unit[key]
            chunks.append(chunk)
    return chunks


def decode_text_payload(payload: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "cp950", "big5", "gb18030"):
        try:
            text = payload.decode(encoding)
            if "\x00" not in text:
                return text
        except (UnicodeDecodeError, LookupError):
            continue
    raise DocumentPipelineError("文字檔編碼無法辨識；建議另存為 UTF-8。")


def perform_ocr(image_path: Path) -> dict:
    errors = []
    if platform.system() == "Darwin" and VISION_OCR_SCRIPT.is_file():
        swift = shutil.which("swift") or "/usr/bin/swift"
        try:
            result = subprocess.run(
                [swift, str(VISION_OCR_SCRIPT), str(image_path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=int(os.getenv("RAG_OCR_TIMEOUT_SECONDS", "240")),
            )
            if result.returncode == 0:
                payload = json.loads(result.stdout)
                if normalize_extracted_text(payload.get("text")):
                    return payload
            errors.append(result.stderr.strip() or "macOS Vision 沒有辨識到文字")
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            errors.append(str(exc))

    tesseract = shutil.which("tesseract")
    if tesseract:
        languages = os.getenv("RAG_OCR_LANGUAGES", "chi_tra+eng").strip()
        if not re.fullmatch(r"[A-Za-z0-9_+-]+", languages):
            raise DocumentPipelineError("RAG_OCR_LANGUAGES 設定包含不合法字元。")
        try:
            result = subprocess.run(
                [
                    tesseract,
                    str(image_path),
                    "stdout",
                    "-l",
                    languages,
                    "--psm",
                    "6",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=int(os.getenv("RAG_OCR_TIMEOUT_SECONDS", "240")),
            )
            text = normalize_extracted_text(result.stdout)
            if result.returncode == 0 and text:
                return {
                    "text": text,
                    "confidence": None,
                    "engine": "tesseract",
                    "languages": languages.split("+"),
                }
            errors.append(result.stderr.strip() or "Tesseract 沒有辨識到文字")
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(str(exc))

    detail = "；".join(error for error in errors if error)
    raise DocumentPipelineError(f"圖片 OCR 失敗。{detail}".rstrip("。"))


def perform_image_understanding(image_path: Path) -> dict:
    model = os.getenv("RAG_VLM_MODEL", "").strip()
    if not model:
        return {"text": "", "engine": "disabled", "model": ""}
    timeout_seconds = float(os.getenv("RAG_VLM_TIMEOUT_SECONDS", "300"))
    text = ask_ollama_vision(
        image_path=image_path,
        prompt=DOCX_VISION_PROMPT,
        model=model,
        timeout_seconds=timeout_seconds,
    )
    return {
        "text": text,
        "engine": "ollama-vlm",
        "model": model,
    }


def render_pdf_page(pdf_path: Path, page_number: int, output_dir: Path) -> Path:
    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        raise DocumentPipelineError("缺少 pdftoppm，無法對掃描型 PDF 執行 OCR。")
    output_prefix = output_dir / f"page-{page_number}"
    result = subprocess.run(
        [
            pdftoppm,
            "-f", str(page_number),
            "-l", str(page_number),
            "-r", os.getenv("RAG_PDF_OCR_DPI", "200"),
            "-png",
            "-singlefile",
            str(pdf_path),
            str(output_prefix),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=int(os.getenv("RAG_OCR_TIMEOUT_SECONDS", "240")),
    )
    output_path = output_prefix.with_suffix(".png")
    if result.returncode != 0 or not output_path.is_file():
        raise DocumentPipelineError(f"PDF 第 {page_number} 頁轉圖片失敗：{result.stderr.strip()}")
    return output_path


def normalize_extracted_text(text: object) -> str:
    lines = []
    for line in str(text or "").replace("\u00a0", " ").splitlines():
        clean_line = re.sub(r"[ \t]+", " ", line).strip()
        if clean_line:
            lines.append(clean_line)
    return "\n".join(lines)[:MAX_EXTRACTED_CHARACTERS]


def split_sized_text(text: str, chunk_size: int, chunk_stride: int):
    if len(text) <= chunk_size:
        yield text
        return
    start = 0
    while start < len(text):
        target_end = min(len(text), start + chunk_size)
        end = target_end
        if target_end < len(text):
            minimum_boundary = start + int(chunk_size * 0.65)
            boundary = max(
                text.rfind("\n", minimum_boundary, target_end),
                text.rfind("。", minimum_boundary, target_end),
                text.rfind(". ", minimum_boundary, target_end),
            )
            if boundary >= minimum_boundary:
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            yield piece
        if end >= len(text):
            break
        start = max(start + 1, min(start + chunk_stride, end))


def _extract_docx(
    filename: str,
    payload: bytes,
    ocr_fn,
    image_understanding_fn,
) -> ExtractionResult:
    blocks = extract_docx_blocks(payload, include_images=True)
    units = []
    warnings = []
    current_title = Path(filename).stem
    current_lines: List[str] = []
    current_section = 1
    image_count = sum(1 for block in blocks if block.get("kind") == "image")
    recognized_image_count = 0
    ocr_image_count = 0
    vlm_image_count = 0
    vision_models = set()
    recognition_cache = {}

    def flush() -> None:
        content = normalize_extracted_text("\n\n".join(current_lines))
        if content:
            units.append(
                {
                    "title": current_title,
                    "page": f"章節 {current_section}",
                    "content": content,
                    "extraction_method": "docx-xml",
                }
            )
        current_lines.clear()

    with tempfile.TemporaryDirectory(prefix="rag-docx-images-") as directory:
        image_dir = Path(directory)
        for block in blocks:
            kind = block.get("kind")
            text = normalize_extracted_text(block.get("text"))
            if kind == "heading" and text:
                had_content = bool(current_lines or units)
                flush()
                if had_content:
                    current_section += 1
                current_title = text
                continue
            if kind == "table" and text:
                current_lines.append(f"[表格 {int(block.get('table_index') or 1)}]\n{text}")
                continue
            if kind != "image":
                if text:
                    current_lines.append(text)
                continue

            flush()
            image_index = int(block.get("image_index") or 1)
            extension = str(block.get("extension") or "").lower()
            image_payload = bytes(block.get("payload") or b"")
            if extension not in DOCX_RASTER_IMAGE_EXTENSIONS or not image_payload:
                warnings.append(
                    f"圖片 {image_index} 格式 {extension or 'unknown'} 不支援 OCR/VLM，已略過。"
                )
                continue

            digest = hashlib.sha256(image_payload).hexdigest()
            cached = recognition_cache.get(digest)
            if cached is None:
                image_path = image_dir / f"image-{image_index}{extension}"
                image_path.write_bytes(image_payload)
                cached = {
                    "ocr_text": "",
                    "ocr_engine": "",
                    "ocr_confidence": None,
                    "vision_text": "",
                    "vision_engine": "",
                    "vision_model": "",
                }
                try:
                    ocr_result = ocr_fn(image_path)
                    cached["ocr_text"] = normalize_extracted_text(ocr_result.get("text"))
                    cached["ocr_engine"] = str(ocr_result.get("engine") or "ocr")
                    if ocr_result.get("confidence") is not None:
                        cached["ocr_confidence"] = round(float(ocr_result["confidence"]), 6)
                except Exception as exc:
                    warnings.append(f"圖片 {image_index} OCR 失敗：{exc}")
                try:
                    vision_result = image_understanding_fn(image_path)
                    cached["vision_text"] = normalize_extracted_text(vision_result.get("text"))
                    cached["vision_engine"] = str(vision_result.get("engine") or "")
                    cached["vision_model"] = str(vision_result.get("model") or "")
                except Exception as exc:
                    warnings.append(f"圖片 {image_index} VLM 辨識失敗：{exc}")
                recognition_cache[digest] = cached

            content_parts = []
            alt_text = normalize_extracted_text(block.get("alt_text"))
            if alt_text:
                content_parts.append(f"[圖片替代文字]\n{alt_text}")
            if cached["ocr_text"]:
                content_parts.append(f"[圖片 OCR 文字]\n{cached['ocr_text']}")
                ocr_image_count += 1
            if cached["vision_text"]:
                content_parts.append(f"[VLM 畫面說明]\n{cached['vision_text']}")
                vlm_image_count += 1
                if cached["vision_model"]:
                    vision_models.add(cached["vision_model"])
            content = normalize_extracted_text("\n\n".join(content_parts))
            if not content:
                warnings.append(f"圖片 {image_index} 沒有取得可索引內容。")
                continue
            recognized_image_count += 1
            unit = {
                "title": f"{current_title} / 圖片 {image_index}",
                "page": f"章節 {current_section} / 圖片 {image_index}",
                "content": content,
                "extraction_method": (
                    "docx-image-ocr+vlm"
                    if cached["ocr_text"] and cached["vision_text"]
                    else ("docx-image-vlm" if cached["vision_text"] else "docx-image-ocr")
                ),
                "ocr_engine": cached["ocr_engine"],
            }
            if cached["ocr_confidence"] is not None:
                unit["ocr_confidence"] = cached["ocr_confidence"]
            units.append(unit)
    flush()
    method = "docx-xml"
    if vlm_image_count and ocr_image_count:
        method = "docx-xml+ocr+vlm"
    elif vlm_image_count:
        method = "docx-xml+vlm"
    elif ocr_image_count:
        method = "docx-xml+ocr"
    return ExtractionResult(
        units=units,
        method=method,
        source_type="word",
        mime_type=DOCX_MIME_TYPE,
        ocr_used=ocr_image_count > 0,
        warnings=warnings,
        details={
            "section_count": len(units),
            "embedded_image_count": image_count,
            "recognized_image_count": recognized_image_count,
            "ocr_image_count": ocr_image_count,
            "vlm_image_count": vlm_image_count,
            "vision_models": sorted(vision_models),
        },
    )


def _extract_plain_text(filename: str, text: str) -> ExtractionResult:
    content = normalize_extracted_text(text)
    return ExtractionResult(
        units=[{"title": Path(filename).stem, "page": "全文", "content": content}],
        method="text-decode",
        source_type="text",
        mime_type="text/plain",
        details={"character_count": len(content)},
    )


def _extract_markdown(filename: str, text: str) -> ExtractionResult:
    units = []
    current_title = Path(filename).stem
    current_lines: List[str] = []

    def flush() -> None:
        content = normalize_extracted_text("\n".join(current_lines))
        if content:
            units.append({"title": current_title, "page": f"區段 {len(units) + 1}", "content": content})

    for line in str(text or "").splitlines():
        heading = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if heading:
            flush()
            current_title = heading.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)
    flush()
    return ExtractionResult(
        units=units,
        method="markdown-headings",
        source_type="markdown",
        mime_type="text/markdown",
        details={"section_count": len(units)},
    )


def _extract_json(filename: str, text: str) -> ExtractionResult:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DocumentPipelineError(f"JSON 格式錯誤：第 {exc.lineno} 行，第 {exc.colno} 欄。") from exc

    units = []
    if isinstance(data, dict):
        items = list(data.items())
    else:
        items = [("root", data)]
    entry_budget = [100_000]
    for key, value in items:
        lines: List[str] = []
        _flatten_json(value, f"$.{key}" if key != "root" else "$", lines, entry_budget, depth=0)
        content = normalize_extracted_text("\n".join(lines))
        if content:
            units.append(
                {
                    "title": str(key),
                    "page": f"JSON {key}",
                    "json_path": f"$.{key}" if key != "root" else "$",
                    "content": content,
                }
            )
    return ExtractionResult(
        units=units,
        method="json-path-flatten",
        source_type="json",
        mime_type="application/json",
        details={"top_level_sections": len(units)},
    )


def _extract_image(filename: str, payload: bytes, extension: str, ocr_fn) -> ExtractionResult:
    with tempfile.TemporaryDirectory(prefix="rag-image-") as directory:
        image_path = Path(directory) / f"input{extension}"
        image_path.write_bytes(payload)
        result = ocr_fn(image_path)
    text = normalize_extracted_text(result.get("text"))
    if not text:
        raise DocumentPipelineError("圖片 OCR 完成，但沒有辨識到可索引文字。")
    confidence = result.get("confidence")
    unit = {
        "title": "OCR",
        "page": "圖片",
        "content": text,
        "ocr_engine": str(result.get("engine") or "ocr"),
    }
    if confidence is not None:
        unit["ocr_confidence"] = round(float(confidence), 6)
    mime_type = SUPPORTED_FORMATS[extension]["mime_type"]
    return ExtractionResult(
        units=[unit],
        method=str(result.get("engine") or "ocr"),
        source_type="image",
        mime_type=mime_type,
        ocr_used=True,
        details={
            "ocr_engine": unit["ocr_engine"],
            "ocr_confidence": unit.get("ocr_confidence"),
            "languages": list(result.get("languages") or []),
        },
    )


def _extract_pdf(
    filename: str,
    payload: bytes,
    ocr_fn,
    pdf_reader_factory=None,
    pdf_page_renderer=None,
) -> ExtractionResult:
    if pdf_reader_factory is None:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise DocumentPipelineError("缺少 pypdf，請先安裝 requirements.txt。") from exc
        pdf_reader_factory = PdfReader

    try:
        reader = pdf_reader_factory(BytesIO(payload))
    except Exception as exc:
        raise DocumentPipelineError("PDF 結構損壞或無法讀取。") from exc
    if getattr(reader, "is_encrypted", False):
        try:
            if reader.decrypt("") == 0:
                raise DocumentPipelineError("PDF 已加密，請先解除密碼後再上傳。")
        except DocumentPipelineError:
            raise
        except Exception as exc:
            raise DocumentPipelineError("PDF 已加密，請先解除密碼後再上傳。") from exc

    pages = list(reader.pages)
    if not pages:
        raise DocumentPipelineError("PDF 沒有頁面。")
    if len(pages) > MAX_PDF_PAGES:
        raise DocumentPipelineError(f"PDF 超過 {MAX_PDF_PAGES} 頁上限。")

    units = []
    warnings = []
    text_pages = 0
    ocr_pages = 0
    with tempfile.TemporaryDirectory(prefix="rag-pdf-") as directory:
        temp_dir = Path(directory)
        pdf_path = temp_dir / "input.pdf"
        pdf_path.write_bytes(payload)
        for page_number, page in enumerate(pages, start=1):
            try:
                extracted_text = page.extract_text() or ""
            except Exception:
                extracted_text = ""
            text = normalize_extracted_text(extracted_text)
            method = "pdf-text"
            unit_extra = {}
            if len(re.sub(r"\s+", "", text)) < MIN_PDF_TEXT_CHARACTERS:
                try:
                    image_path = pdf_page_renderer(pdf_path, page_number, temp_dir)
                    ocr_result = ocr_fn(image_path)
                    ocr_text = normalize_extracted_text(ocr_result.get("text"))
                    if ocr_text:
                        text = ocr_text
                        method = str(ocr_result.get("engine") or "ocr")
                        unit_extra["ocr_engine"] = method
                        if ocr_result.get("confidence") is not None:
                            unit_extra["ocr_confidence"] = round(float(ocr_result["confidence"]), 6)
                        ocr_pages += 1
                except Exception as exc:
                    warnings.append(f"第 {page_number} 頁 OCR 失敗：{exc}")
            if method == "pdf-text" and text:
                text_pages += 1
            if not text:
                warnings.append(f"第 {page_number} 頁沒有可索引文字。")
                continue
            units.append(
                {
                    "title": f"第 {page_number} 頁",
                    "page": f"page {page_number}",
                    "content": text,
                    "extraction_method": method,
                    **unit_extra,
                }
            )
    if not units:
        raise DocumentPipelineError("PDF 沒有可索引文字；文字層與 OCR 都未取得內容。")
    method = "pdf-text+ocr" if ocr_pages and text_pages else ("pdf-ocr" if ocr_pages else "pdf-text")
    return ExtractionResult(
        units=units,
        method=method,
        source_type="pdf",
        mime_type="application/pdf",
        ocr_used=ocr_pages > 0,
        warnings=warnings,
        details={"page_count": len(pages), "text_pages": text_pages, "ocr_pages": ocr_pages},
    )


def _flatten_json(value, path: str, lines: List[str], budget: List[int], depth: int) -> None:
    if budget[0] <= 0:
        raise DocumentPipelineError("JSON 項目超過 100,000 筆上限。")
    if depth > 24:
        raise DocumentPipelineError("JSON 巢狀深度超過 24 層上限。")
    if isinstance(value, dict):
        if not value:
            lines.append(f"{path}: {{}}")
            budget[0] -= 1
        for key, child in value.items():
            _flatten_json(child, f"{path}.{key}", lines, budget, depth + 1)
        return
    if isinstance(value, list):
        if not value:
            lines.append(f"{path}: []")
            budget[0] -= 1
        for index, child in enumerate(value):
            _flatten_json(child, f"{path}[{index}]", lines, budget, depth + 1)
        return
    rendered = json.dumps(value, ensure_ascii=False)
    lines.append(f"{path}: {rendered}")
    budget[0] -= 1


def _validate_payload(payload: bytes, extension: str) -> None:
    if not payload:
        raise DocumentPipelineError("檔案內容是空的。")
    limit = MAX_TEXT_UPLOAD_BYTES if extension in {".txt", ".md", ".markdown", ".json"} else MAX_DOCUMENT_UPLOAD_BYTES
    if len(payload) > limit:
        raise DocumentPipelineError(f"檔案超過 {limit // (1024 * 1024)} MB 上限。")


def _source_type_for_name(name: object) -> str:
    extension = Path(str(name or "")).suffix.lower()
    return str(SUPPORTED_FORMATS.get(extension, {}).get("source_type") or "document")
