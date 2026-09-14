"""Parent-child chunking for high-recall retrieval and coherent evidence.

The index stores 128-token child chunks because they are precise retrieval
units.  Each child keeps a small amount of parent metadata and the complete
512-token parent text so a retriever can expand a hit before it is sent to a
generation model.  Embeddings should be built from the child ``content``;
``parent_content`` is deliberately only an evidence expansion field.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from rag_demo.chunk_strategies import (
    CHUNK_STRATEGY_DYNAMIC,
    estimate_token_count,
    split_text,
)


@dataclass(frozen=True)
class ParentChildIndex:
    """The parent documents and the child records used by the index."""

    parents: list[dict[str, Any]]
    children: list[dict[str, Any]]

    @property
    def parent_by_id(self) -> dict[str, dict[str, Any]]:
        return {str(parent["id"]): parent for parent in self.parents}


def build_parent_child_index(
    units: Sequence[dict[str, Any]],
    *,
    source_id: str,
    filename: str,
    source_type: str,
    extraction_method: str,
    parent_size_tokens: int = 512,
    child_size_tokens: int = 128,
    parent_overlap_tokens: int = 0,
    child_overlap_tokens: int = 0,
    strategy: str = CHUNK_STRATEGY_DYNAMIC,
) -> ParentChildIndex:
    """Build sentence-aware 512-token parents and 128-token children.

    Parent boundaries are created first.  Children are then split inside each
    parent, which makes every child point to exactly one parent and prevents a
    child from crossing a document section boundary.
    """

    parent_size = max(1, int(parent_size_tokens))
    child_size = max(1, int(child_size_tokens))
    parent_overlap = max(0, min(int(parent_overlap_tokens), parent_size - 1))
    child_overlap = max(0, min(int(child_overlap_tokens), child_size - 1))
    parents: list[dict[str, Any]] = []
    children: list[dict[str, Any]] = []
    parent_title = _clean_text(filename.rsplit(".", 1)[0] if "." in filename else filename)

    for unit_index, unit in enumerate(units, start=1):
        content = _clean_text(unit.get("content"))
        if not content:
            continue
        unit_title = _clean_text(unit.get("title") or f"Section {unit_index}")
        page = _clean_text(unit.get("page") or f"區段 {unit_index}")
        parent_pieces = list(
            split_text(
                content,
                parent_size,
                parent_size,
                strategy=strategy,
                overlap_tokens=parent_overlap,
            )
        )
        for parent_index, parent_content in enumerate(parent_pieces, start=1):
            parent_id = f"{source_id}::parent::{unit_index}-{parent_index}"
            parent_label = f"{filename} | {unit_title}"
            if len(parent_pieces) > 1:
                parent_label = f"{parent_label} / parent {parent_index}"
            parent = {
                "id": parent_id,
                "source_id": source_id,
                "source": source_id,
                "source_type": source_type,
                "chunk_level": "parent",
                "parent_chunk_id": parent_id,
                "parent_chunk_index": parent_index,
                "chunk_index": len(parents),
                "parent_title": parent_title,
                "title": parent_label,
                "page": page,
                "content": parent_content,
                "token_count": estimate_token_count(parent_content),
                "extraction_method": str(unit.get("extraction_method") or extraction_method),
            }
            _copy_unit_metadata(parent, unit)
            parents.append(parent)

            child_pieces = list(
                split_text(
                    parent_content,
                    child_size,
                    child_size,
                    strategy=strategy,
                    overlap_tokens=child_overlap,
                )
            )
            for child_index, child_content in enumerate(child_pieces, start=1):
                child_id = f"{source_id}::child::{unit_index}-{parent_index}-{child_index}"
                child = {
                    "id": child_id,
                    "source_id": source_id,
                    "source": source_id,
                    "source_type": source_type,
                    "chunk_level": "child",
                    "parent_id": parent_id,
                    "parent_chunk_id": parent_id,
                    "parent_chunk_index": parent_index,
                    "child_chunk_index": child_index,
                    "chunk_index": len(children),
                    "parent_title": parent_title,
                    "title": f"{parent_label} / child {child_index}",
                    "parent_title_context": parent_label,
                    "page": page,
                    "content": child_content,
                    "parent_content": parent_content,
                    "token_count": estimate_token_count(child_content),
                    "parent_token_count": estimate_token_count(parent_content),
                    "extraction_method": str(unit.get("extraction_method") or extraction_method),
                }
                _copy_unit_metadata(child, unit)
                children.append(child)

    return ParentChildIndex(parents=parents, children=children)


def build_parent_child_chunks(*args, **kwargs) -> list[dict[str, Any]]:
    """Compatibility helper returning only child records for indexing."""

    return build_parent_child_index(*args, **kwargs).children


def expand_child_contexts(
    selected_children: Iterable[dict[str, Any]],
    *,
    parent_by_id: dict[str, dict[str, Any]] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Replace selected child records with unique parent evidence records."""

    parents = parent_by_id or {}
    expanded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for child in selected_children:
        parent_id = str(child.get("parent_id") or child.get("parent_chunk_id") or "")
        parent = parents.get(parent_id)
        if parent is None and child.get("parent_content"):
            parent = {
                **child,
                "id": parent_id or str(child.get("id") or ""),
                "title": child.get("parent_title_context") or child.get("title") or "Untitled",
                "content": child.get("parent_content"),
                "chunk_level": "parent",
            }
        if parent is None:
            parent = child
        identity = str(parent.get("id") or parent_id or child.get("id") or "")
        if identity in seen:
            continue
        seen.add(identity)
        expanded_item = dict(parent)
        expanded_item["retrieval_child_id"] = str(child.get("id") or "")
        expanded_item["retrieval_child_ids"] = [str(child.get("id") or "")]
        expanded_item["retrieval_child_score"] = child.get("rerank_score", child.get("fusion_score", child.get("rrf_score", 0.0)))
        expanded_item["retrieval_child_bm25_score"] = child.get("bm25_score", 0.0)
        expanded_item["retrieval_child_embedding_score"] = child.get("embedding_score", 0.0)
        expanded_item["retrieval_child_fusion_score"] = child.get("fusion_score", child.get("rrf_score", 0.0))
        expanded_item["retrieval_child_rrf_score"] = child.get("rrf_score", child.get("fusion_score", 0.0))
        expanded_item["retrieval_child_rerank_score"] = child.get("rerank_score", expanded_item["retrieval_child_score"])
        expanded.append(expanded_item)
        if limit is not None and len(expanded) >= max(1, int(limit)):
            break
    return expanded


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _copy_unit_metadata(target: dict[str, Any], unit: dict[str, Any]) -> None:
    for key in ("ocr_confidence", "ocr_engine", "json_path"):
        if key in unit:
            target[key] = unit[key]
