from pathlib import Path
import re
from typing import Dict, List

from rag_demo.config import RagConfig
from rag_demo.chunk_strategies import split_text
from rag_demo.parent_child import build_parent_child_index


Chunk = Dict[str, object]


def chunk_markdown(markdown_text: str, source: str) -> List[Chunk]:
    config = RagConfig.from_env()
    chunks: List[Chunk] = []
    current_parent_title = ""
    current_title = None
    current_lines: List[str] = []
    chunk_index = 0

    for line in markdown_text.splitlines():
        if line.startswith("## ") and not line.startswith("### "):
            if current_title and current_lines:
                chunk_index = _append_sized_chunks(
                    chunks,
                    source,
                    chunk_index,
                    current_parent_title,
                    current_title,
                    current_lines,
                    config.chunk_size,
                    config.chunk_stride,
                    config.chunk_strategy,
                    config.chunk_overlap_tokens,
                )
                current_title = None
                current_lines = []
            current_parent_title = line.removeprefix("## ").strip()
            continue

        if line.startswith("### "):
            if current_title and current_lines:
                chunk_index = _append_sized_chunks(
                    chunks,
                    source,
                    chunk_index,
                    current_parent_title,
                    current_title,
                    current_lines,
                    config.chunk_size,
                    config.chunk_stride,
                    config.chunk_strategy,
                    config.chunk_overlap_tokens,
                )
            current_title = line.removeprefix("### ").strip()
            current_lines = []
            continue

        if current_title:
            current_lines.append(line)

    if current_title and current_lines:
        _append_sized_chunks(
            chunks,
            source,
            chunk_index,
            current_parent_title,
            current_title,
            current_lines,
            config.chunk_size,
            config.chunk_stride,
            config.chunk_strategy,
            config.chunk_overlap_tokens,
        )

    return chunks


def load_markdown_chunks(raw_dir: Path) -> List[Chunk]:
    chunks: List[Chunk] = []
    for path in sorted(raw_dir.glob("*.md")):
        chunks.extend(chunk_markdown(path.read_text(encoding="utf-8"), source=str(path)))
    return chunks


def chunk_sectioned_record_text(
    record_text: str,
    source: str,
    parent_title: str = "Sectioned Record",
    chunk_size: int = None,
    chunk_stride: int = None,
) -> List[Chunk]:
    return chunk_sectioned_text(
        record_text,
        source=source,
        parent_title=parent_title,
        chunk_size=chunk_size,
        chunk_stride=chunk_stride,
    )


def chunk_avalon_record_text(
    record_text: str,
    source: str,
    chunk_size: int = None,
    chunk_stride: int = None,
) -> List[Chunk]:
    return chunk_sectioned_record_text(
        record_text,
        source=source,
        parent_title="阿瓦隆對局紀錄",
        chunk_size=chunk_size,
        chunk_stride=chunk_stride,
    )


def chunk_sectioned_text(
    text: str,
    source: str,
    parent_title: str = "Sectioned Text",
    chunk_size: int = None,
    chunk_stride: int = None,
) -> List[Chunk]:
    config = RagConfig.from_env()
    chunk_size = chunk_size or config.chunk_size
    chunk_stride = chunk_stride or config.chunk_stride
    chunks: List[Chunk] = []
    current_title = None
    current_lines: List[str] = []
    chunk_index = 0

    for line in text.splitlines():
        stripped = line.strip()
        if current_title and _is_delimited_section_footer(stripped):
            continue
        if _is_delimited_section_heading(stripped):
            if current_title and current_lines:
                chunk_index = _append_sized_chunks(
                    chunks,
                    source,
                    chunk_index,
                    parent_title,
                    current_title,
                    current_lines,
                    chunk_size,
                    chunk_stride,
                    config.chunk_strategy,
                    config.chunk_overlap_tokens,
                )
            current_title = _clean_delimited_section_heading(stripped)
            current_lines = []
            continue

        if current_title:
            current_lines.append(line)

    if current_title and current_lines:
        _append_sized_chunks(
            chunks,
            source,
            chunk_index,
            parent_title,
            current_title,
            current_lines,
            chunk_size,
            chunk_stride,
            config.chunk_strategy,
            config.chunk_overlap_tokens,
        )

    return chunks


def chunk_plain_text(
    text: str,
    source: str,
    chunk_size: int = None,
    chunk_stride: int = None,
) -> List[Chunk]:
    config = RagConfig.from_env()
    chunk_size = chunk_size or config.chunk_size
    chunk_stride = chunk_stride or config.chunk_stride
    chunks: List[Chunk] = []
    _append_sized_chunks(
        chunks,
        source,
        0,
        "Plain Text",
        Path(source).name,
        text.splitlines(),
        chunk_size,
        chunk_stride,
        config.chunk_strategy,
        config.chunk_overlap_tokens,
    )
    return chunks


def chunk_narrative_text(
    text: str,
    source: str,
    chunk_size: int = None,
    chunk_stride: int = None,
) -> List[Chunk]:
    config = RagConfig.from_env()
    chunk_size = chunk_size or config.chunk_size
    chunk_stride = chunk_stride or config.chunk_stride
    chunks: List[Chunk] = []
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    front_lines = nonempty_lines[:12]
    chunk_index = 0

    if front_lines:
        chunks.append(
            _make_chunk_from_content(
                source,
                chunk_index,
                "Narrative Text",
                "文件開頭 / metadata",
                "\n".join(front_lines),
            )
        )
        chunk_index += 1

    current_title = Path(source).name
    current_lines: List[str] = []
    seen_heading = False
    for line in text.splitlines():
        stripped = line.strip()
        if _is_narrative_heading(stripped):
            if seen_heading and current_lines:
                chunk_index = _append_sized_chunks(
                    chunks,
                    source,
                    chunk_index,
                    "Narrative Text",
                    current_title,
                    current_lines,
                    chunk_size,
                    chunk_stride,
                    config.chunk_strategy,
                    config.chunk_overlap_tokens,
                )
            current_title = _clean_narrative_heading(stripped)
            current_lines = []
            seen_heading = True
            continue
        if seen_heading and stripped:
            current_lines.append(line)

    if seen_heading and current_lines:
        _append_sized_chunks(
            chunks,
            source,
            chunk_index,
            "Narrative Text",
            current_title,
            current_lines,
            chunk_size,
            chunk_stride,
            config.chunk_strategy,
            config.chunk_overlap_tokens,
        )

    return chunks


def load_knowledge_base_chunks(kb_dir: Path) -> List[Chunk]:
    config = RagConfig.from_env()
    chunks: List[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        chunks.extend(chunk_markdown(path.read_text(encoding="utf-8"), source=str(path)))
    for path in sorted(kb_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        sectioned_chunks = chunk_sectioned_text(
            text,
            source=str(path),
            parent_title=path.stem,
            chunk_size=config.chunk_size,
            chunk_stride=config.chunk_stride,
        )
        if sectioned_chunks:
            chunks.extend(sectioned_chunks)
            continue
        if _has_narrative_headings(text):
            chunks.extend(
                chunk_narrative_text(
                    text,
                    source=str(path),
                    chunk_size=config.chunk_size,
                    chunk_stride=config.chunk_stride,
                )
            )
            continue
        chunks.extend(
            chunk_plain_text(
                text,
                source=str(path),
                chunk_size=config.chunk_size,
                chunk_stride=config.chunk_stride,
            )
        )
    if not config.parent_child_enabled or not chunks:
        return chunks

    # Legacy local knowledge-base files already carry section titles. Convert
    # those sections into the same child-index schema used by uploaded files;
    # the canonical document pipeline can still build larger parents directly
    # from extraction units.
    units = [
        {
            "title": chunk.get("title") or chunk.get("parent_title") or "Section",
            "page": chunk.get("page") or "",
            "content": chunk.get("content") or "",
        }
        for chunk in chunks
        if str(chunk.get("content") or "").strip()
    ]
    if not units:
        return chunks
    parent_child = build_parent_child_index(
        units,
        source_id=str(kb_dir),
        filename=kb_dir.name or "knowledge-base",
        source_type="local",
        extraction_method="legacy-local-chunking",
        parent_size_tokens=config.parent_chunk_size_tokens,
        child_size_tokens=config.child_chunk_size_tokens,
        parent_overlap_tokens=config.parent_chunk_overlap_tokens,
        child_overlap_tokens=config.child_chunk_overlap_tokens,
    )
    return parent_child.children


def _is_delimited_section_heading(line: str) -> bool:
    return bool(re.match(r"^={3,}\s*\S.*?\s*={3,}$", line))


def _is_delimited_section_footer(line: str) -> bool:
    return bool(re.match(r"^={8,}.*={8,}$", line))


def _clean_delimited_section_heading(line: str) -> str:
    return line.strip("= ").strip()


def _has_narrative_headings(text: str) -> bool:
    return sum(1 for line in text.splitlines() if _is_narrative_heading(line.strip())) >= 2


def _is_narrative_heading(line: str) -> bool:
    if not line:
        return False
    if re.match(r"^\d{1,3}[.．]\s*\S.{0,80}$", line):
        return True
    if re.match(r"^【[^】]{1,40}】$", line):
        return True
    if re.match(r"^第[一二三四五六七八九十百千0-9]+[章節部卷回]\b", line):
        return True
    if line in {"序", "序章", "楔子", "前言", "後記", "尾聲", "上部", "中部", "下部", "上卷", "中卷", "下卷"}:
        return True
    return False


def _clean_narrative_heading(line: str) -> str:
    return line.strip()


def _make_chunk(
    source: str,
    chunk_index: int,
    parent_title: str,
    title: str,
    lines: List[str],
) -> Chunk:
    content = "\n".join(line for line in lines).strip()
    return _make_chunk_from_content(source, chunk_index, parent_title, title, content)


def _make_chunk_from_content(
    source: str,
    chunk_index: int,
    parent_title: str,
    title: str,
    content: str,
) -> Chunk:
    return {
        "id": f"{Path(source).name}::{chunk_index}",
        "source": source,
        "chunk_index": chunk_index,
        "parent_title": parent_title,
        "title": title,
        "content": content,
    }


def _append_sized_chunks(
    chunks: List[Chunk],
    source: str,
    start_index: int,
    parent_title: str,
    title: str,
    lines: List[str],
    chunk_size: int,
    chunk_stride: int,
    chunk_strategy: str = "boundary",
    overlap_tokens: int = 200,
) -> int:
    content = "\n".join(line for line in lines).strip()
    index = start_index
    pieces = list(
        split_text(
            content,
            chunk_size=chunk_size,
            chunk_stride=chunk_stride,
            strategy=chunk_strategy,
            overlap_tokens=overlap_tokens,
        )
    )
    for part, piece in enumerate(pieces, start=1):
        if piece:
            chunks.append(
                _make_chunk_from_content(
                    source,
                    index,
                    parent_title,
                    f"{title} / part {part}" if len(pieces) > 1 else title,
                    piece,
                )
            )
            index += 1
    return index
