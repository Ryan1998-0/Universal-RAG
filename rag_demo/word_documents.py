import hashlib
import json
import os
import posixpath
import re
import shutil
import threading
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence
from xml.etree import ElementTree

import numpy as np

from rag_demo.config import RagConfig
from rag_demo.chunk_strategies import split_text
from rag_demo.embeddings import DEFAULT_EMBEDDING_MODEL, embed_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPLOAD_ROOT = Path(
    os.getenv(
        "RAG_UPLOAD_ROOT",
        str(PROJECT_ROOT / ".local" / "uploaded_documents"),
    )
).expanduser()
MAX_WORD_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_DOCX_UNCOMPRESSED_BYTES = 100 * 1024 * 1024
MAX_DOCX_MEMBERS = 2000
DOCX_MIME_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

_WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_WORD = f"{{{_WORD_NAMESPACE}}}"
_DRAWING_NAMESPACE = "http://schemas.openxmlformats.org/drawingml/2006/main"
_DRAWING = f"{{{_DRAWING_NAMESPACE}}}"
_RELATIONSHIP_NAMESPACE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_RELATIONSHIP = f"{{{_RELATIONSHIP_NAMESPACE}}}"
_PACKAGE_RELATIONSHIP_NAMESPACE = "http://schemas.openxmlformats.org/package/2006/relationships"
_PACKAGE_RELATIONSHIP = f"{{{_PACKAGE_RELATIONSHIP_NAMESPACE}}}"
_WORDPROCESSING_DRAWING_NAMESPACE = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
_WORDPROCESSING_DRAWING = f"{{{_WORDPROCESSING_DRAWING_NAMESPACE}}}"
_VML_NAMESPACE = "urn:schemas-microsoft-com:vml"
_VML = f"{{{_VML_NAMESPACE}}}"
_DOCUMENT_XML = "word/document.xml"
_DOCUMENT_RELS_XML = "word/_rels/document.xml.rels"
_STYLES_XML = "word/styles.xml"
_SOURCE_ID_PATTERN = re.compile(r"^upload-[0-9a-f]{20}$")
MAX_DOCX_IMAGES = 500


class WordDocumentError(ValueError):
    pass


@dataclass(frozen=True)
class IndexedWordDocument:
    metadata: dict
    chunks: List[dict]
    embeddings: List[List[float]]
    duplicate: bool = False

    def response_payload(self) -> dict:
        return {
            "document": dict(self.metadata),
            "duplicate": self.duplicate,
        }


class WordDocumentStore:
    def __init__(
        self,
        root: Path = DEFAULT_UPLOAD_ROOT,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embed_chunks_fn: Optional[Callable[[Sequence[dict]], Sequence[Sequence[float]]]] = None,
        now_fn: Optional[Callable[[], datetime]] = None,
    ):
        self.root = Path(root)
        self.embedding_model = embedding_model
        self._embed_chunks_fn = embed_chunks_fn or (
            lambda chunks: embed_chunks(chunks, model_name=self.embedding_model)
        )
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self._lock = threading.RLock()

    def import_docx(self, filename: str, payload: bytes) -> IndexedWordDocument:
        clean_filename = normalize_word_filename(filename)
        if not payload:
            raise WordDocumentError("Word 檔案內容是空的。")
        if len(payload) > MAX_WORD_UPLOAD_BYTES:
            raise WordDocumentError("Word 檔案超過 20 MB 上限。")

        digest = hashlib.sha256(payload).hexdigest()
        source_id = f"upload-{digest[:20]}"
        with self._lock:
            existing = self._load_indexed_document(source_id)
            if existing is not None:
                return IndexedWordDocument(
                    metadata=existing.metadata,
                    chunks=existing.chunks,
                    embeddings=existing.embeddings,
                    duplicate=True,
                )

        blocks = extract_docx_blocks(payload)
        chunks = build_word_chunks(
            blocks,
            source_id=source_id,
            filename=clean_filename,
        )
        if not chunks:
            raise WordDocumentError("找不到可建立索引的文字；圖片型 Word 需要先做 OCR。")

        embeddings = np.asarray(self._embed_chunks_fn(chunks), dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks) or embeddings.shape[1] <= 0:
            raise WordDocumentError("Embedding 建立失敗，回傳維度與文件 chunks 不一致。")

        metadata = {
            "source_id": source_id,
            "name": clean_filename,
            "source_type": "word",
            "sha256": digest,
            "chunk_count": len(chunks),
            "uploaded_at": self._now_fn().astimezone(timezone.utc).isoformat(),
            "embedding_model": self.embedding_model,
            "url": f"/api/documents/{source_id}/download",
        }

        self.root.mkdir(parents=True, exist_ok=True)
        temporary_dir = self.root / f".tmp-{uuid.uuid4().hex}"
        destination = self.root / source_id
        try:
            temporary_dir.mkdir(parents=False, exist_ok=False)
            (temporary_dir / "original.docx").write_bytes(payload)
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
                    return IndexedWordDocument(
                        metadata=existing.metadata,
                        chunks=existing.chunks,
                        embeddings=existing.embeddings,
                        duplicate=True,
                    )
                temporary_dir.replace(destination)
        finally:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir, ignore_errors=True)

        return IndexedWordDocument(
            metadata=metadata,
            chunks=chunks,
            embeddings=embeddings.astype(float).tolist(),
            duplicate=False,
        )

    def list_documents(self) -> List[dict]:
        documents = [document.metadata for document in self.load_indexed_documents()]
        return sorted(
            documents,
            key=lambda document: str(document.get("uploaded_at") or ""),
            reverse=True,
        )

    def load_indexed_documents(self) -> List[IndexedWordDocument]:
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
        path = (self.root / source_id / "original.docx").resolve()
        try:
            path.relative_to(self.root.resolve())
        except ValueError:
            return None
        return path if path.is_file() else None

    def _load_indexed_document(self, source_id: str) -> Optional[IndexedWordDocument]:
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
            raise ValueError("Stored Word document source ID is inconsistent.")
        if not isinstance(chunks, list) or embeddings_array.ndim != 2:
            raise ValueError("Stored Word document index is invalid.")
        if len(chunks) != embeddings_array.shape[0]:
            raise ValueError("Stored Word document chunk and embedding counts differ.")
        if metadata.get("embedding_model") != self.embedding_model:
            raise ValueError("Stored Word document uses a different embedding model.")
        return IndexedWordDocument(
            metadata=metadata,
            chunks=chunks,
            embeddings=embeddings_array.astype(float).tolist(),
            duplicate=False,
        )


def normalize_word_filename(filename: str) -> str:
    clean_filename = Path(str(filename or "").replace("\\", "/")).name.strip()
    clean_filename = re.sub(r"[\x00-\x1f\x7f]", "", clean_filename).strip()
    if not clean_filename:
        raise WordDocumentError("缺少 Word 檔名。")
    if Path(clean_filename).suffix.lower() != ".docx":
        raise WordDocumentError("目前支援 .docx；舊式 .doc 請先在 Word 另存為 .docx。")
    if len(clean_filename) > 180:
        stem = Path(clean_filename).stem[:170].rstrip()
        clean_filename = f"{stem or 'document'}.docx"
    return clean_filename


def extract_docx_blocks(payload: bytes, include_images: bool = False) -> List[dict]:
    document_xml, styles_xml, image_parts = _read_docx_parts(
        payload,
        include_images=include_images,
    )
    try:
        document = ElementTree.fromstring(document_xml)
        styles = _style_names(styles_xml)
    except ElementTree.ParseError as exc:
        raise WordDocumentError("Word XML 結構損壞，無法讀取。") from exc

    body = document.find(f"{_WORD}body")
    if body is None:
        raise WordDocumentError("Word 文件缺少主要內容。")

    blocks = []
    table_index = 0
    image_index = 0
    for child in body:
        if child.tag == f"{_WORD}p":
            text = _paragraph_text(child)
            if text:
                blocks.append(
                    {
                        "kind": "heading" if _is_heading_paragraph(child, styles) else "paragraph",
                        "text": text,
                    }
                )
        elif child.tag == f"{_WORD}tbl":
            table_text = _table_text(child)
            if table_text:
                table_index += 1
                blocks.append(
                    {
                        "kind": "table",
                        "text": table_text,
                        "table_index": table_index,
                    }
                )
        if include_images:
            alt_text = _container_alt_text(child)
            for relationship_id in _container_image_relationship_ids(child):
                image = image_parts.get(relationship_id)
                if image is None:
                    continue
                image_index += 1
                blocks.append(
                    {
                        "kind": "image",
                        "image_index": image_index,
                        "relationship_id": relationship_id,
                        "media_name": image["media_name"],
                        "extension": image["extension"],
                        "payload": image["payload"],
                        "alt_text": alt_text,
                    }
                )
    return blocks


def build_word_chunks(
    blocks: Sequence[dict],
    source_id: str,
    filename: str,
    chunk_size: Optional[int] = None,
    chunk_stride: Optional[int] = None,
) -> List[dict]:
    config = RagConfig.from_env()
    chunk_size = max(200, int(chunk_size or config.chunk_size))
    chunk_stride = min(chunk_size, max(1, int(chunk_stride or config.chunk_stride)))
    parent_title = Path(filename).stem or filename
    current_title = parent_title
    current_lines: List[str] = []
    sections = []

    def flush_section() -> None:
        content = "\n\n".join(line for line in current_lines if line).strip()
        if content:
            sections.append((current_title, content))

    for block in blocks:
        text = _normalize_block_text(block.get("text"))
        if not text:
            continue
        if block.get("kind") == "heading":
            flush_section()
            current_title = text
            current_lines = []
            continue
        if block.get("kind") == "table":
            table_index = int(block.get("table_index") or 1)
            current_lines.append(f"[表格 {table_index}]\n{text}")
        else:
            current_lines.append(text)
    flush_section()

    chunks = []
    for section_index, (section_title, content) in enumerate(sections, start=1):
        pieces = list(
            _split_sized_text(
                content,
                chunk_size,
                chunk_stride,
                strategy=config.chunk_strategy,
                overlap_tokens=config.chunk_overlap_tokens,
            )
        )
        for part_index, piece in enumerate(pieces, start=1):
            chunk_index = len(chunks)
            title = f"{filename} | {section_title}"
            if len(pieces) > 1:
                title = f"{title} / part {part_index}"
            chunks.append(
                {
                    "id": f"{source_id}::{chunk_index}",
                    "source_id": source_id,
                    "source": source_id,
                    "chunk_index": chunk_index,
                    "parent_title": parent_title,
                    "title": title,
                    "page": f"章節 {section_index}",
                    "content": piece,
                }
            )
    return chunks


def _read_docx_xml(payload: bytes):
    document_xml, styles_xml, _ = _read_docx_parts(payload, include_images=False)
    return document_xml, styles_xml


def _read_docx_parts(payload: bytes, include_images: bool):
    try:
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            members = archive.infolist()
            if len(members) > MAX_DOCX_MEMBERS:
                raise WordDocumentError("Word 壓縮內容異常，檔案項目過多。")
            uncompressed_bytes = sum(member.file_size for member in members)
            if uncompressed_bytes > MAX_DOCX_UNCOMPRESSED_BYTES:
                raise WordDocumentError("Word 解壓後內容超過 100 MB 上限。")
            names = {member.filename for member in members}
            if _DOCUMENT_XML not in names:
                raise WordDocumentError("檔案不是有效的 .docx Word 文件。")
            document_xml = archive.read(_DOCUMENT_XML)
            styles_xml = archive.read(_STYLES_XML) if _STYLES_XML in names else b""
            image_parts = (
                _read_docx_image_parts(archive, names)
                if include_images and _DOCUMENT_RELS_XML in names
                else {}
            )
            return document_xml, styles_xml, image_parts
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError) as exc:
        raise WordDocumentError("檔案不是有效或可讀取的 .docx Word 文件。") from exc


def _read_docx_image_parts(archive: zipfile.ZipFile, names: set) -> Dict[str, dict]:
    try:
        relationships = ElementTree.fromstring(archive.read(_DOCUMENT_RELS_XML))
    except ElementTree.ParseError as exc:
        raise WordDocumentError("Word 圖片關聯結構損壞，無法讀取。") from exc

    image_parts: Dict[str, dict] = {}
    for relationship in relationships.findall(f"{_PACKAGE_RELATIONSHIP}Relationship"):
        relationship_id = relationship.get("Id") or ""
        relationship_type = relationship.get("Type") or ""
        target = relationship.get("Target") or ""
        if (
            not relationship_id
            or not relationship_type.endswith("/image")
            or relationship.get("TargetMode") == "External"
        ):
            continue
        part_name = (
            target.lstrip("/")
            if target.startswith("/")
            else posixpath.normpath(posixpath.join("word", target))
        )
        if not part_name.startswith("word/") or part_name not in names:
            continue
        extension = Path(part_name).suffix.lower()
        image_parts[relationship_id] = {
            "media_name": Path(part_name).name,
            "extension": extension,
            "payload": archive.read(part_name),
        }
        if len(image_parts) > MAX_DOCX_IMAGES:
            raise WordDocumentError(f"Word 內嵌圖片超過 {MAX_DOCX_IMAGES} 張上限。")
    return image_parts


def _container_image_relationship_ids(container) -> List[str]:
    relationship_ids = []
    for blip in container.findall(f".//{_DRAWING}blip"):
        relationship_id = blip.get(f"{_RELATIONSHIP}embed")
        if relationship_id:
            relationship_ids.append(relationship_id)
    for image_data in container.findall(f".//{_VML}imagedata"):
        relationship_id = image_data.get(f"{_RELATIONSHIP}id")
        if relationship_id:
            relationship_ids.append(relationship_id)
    return relationship_ids


def _container_alt_text(container) -> str:
    candidates = []
    for properties in container.findall(f".//{_WORDPROCESSING_DRAWING}docPr"):
        for attribute in ("descr", "title", "name"):
            value = str(properties.get(attribute) or "").strip()
            if value and not re.fullmatch(r"Picture\s+\d+", value, flags=re.IGNORECASE):
                candidates.append(value)
    return " / ".join(dict.fromkeys(candidates))


def _style_names(styles_xml: bytes) -> Dict[str, str]:
    if not styles_xml:
        return {}
    root = ElementTree.fromstring(styles_xml)
    names = {}
    for style in root.findall(f"{_WORD}style"):
        style_id = style.get(f"{_WORD}styleId") or ""
        name = style.find(f"{_WORD}name")
        if style_id and name is not None:
            names[style_id] = name.get(f"{_WORD}val") or ""
    return names


def _is_heading_paragraph(paragraph, style_names: Dict[str, str]) -> bool:
    properties = paragraph.find(f"{_WORD}pPr")
    if properties is None:
        return False
    if properties.find(f"{_WORD}outlineLvl") is not None:
        return True
    style = properties.find(f"{_WORD}pStyle")
    style_id = style.get(f"{_WORD}val") if style is not None else ""
    style_label = f"{style_id} {style_names.get(style_id, '')}".casefold()
    return bool(re.search(r"heading|title|標題|章節", style_label))


def _paragraph_text(paragraph) -> str:
    parts = []
    for node in paragraph.iter():
        if node.tag == f"{_WORD}t":
            parts.append(node.text or "")
        elif node.tag == f"{_WORD}tab":
            parts.append("\t")
        elif node.tag in {f"{_WORD}br", f"{_WORD}cr"}:
            parts.append("\n")
    return _normalize_block_text("".join(parts))


def _table_text(table) -> str:
    rows = []
    for row in table.findall(f".//{_WORD}tr"):
        cells = []
        for cell in row.findall(f"{_WORD}tc"):
            paragraphs = [
                _paragraph_text(paragraph)
                for paragraph in cell.findall(f".//{_WORD}p")
            ]
            cells.append(" / ".join(text for text in paragraphs if text))
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _normalize_block_text(text: object) -> str:
    lines = []
    for line in str(text or "").replace("\u00a0", " ").splitlines():
        clean_line = re.sub(r"[ \t]+", " ", line).strip()
        if clean_line:
            lines.append(clean_line)
    return "\n".join(lines)


def _split_sized_text(
    text: str,
    chunk_size: int,
    chunk_stride: int,
    strategy: str = "boundary",
    overlap_tokens: int = 200,
):
    """Compatibility wrapper around the shared chunking strategies."""

    yield from split_text(
        text,
        chunk_size=chunk_size,
        chunk_stride=chunk_stride,
        strategy=strategy,
        overlap_tokens=overlap_tokens,
    )
