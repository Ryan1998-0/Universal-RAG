from __future__ import annotations

import hashlib
import json
from typing import Iterable


INDEX_ENTRY_FIELDS = (
    "qdrant_point_id",
    "chunk_id",
    "chunk_record_id",
    "document_id",
    "document_version_id",
    "content_sha256",
)


def index_entries_sha256(entries: Iterable[dict]) -> str:
    normalized = []
    point_ids = set()
    chunk_ids = set()
    record_ids = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("index manifest entries must be objects")
        item = {
            field: str(entry.get(field) or "").strip()
            for field in INDEX_ENTRY_FIELDS
        }
        if not all(item.values()):
            raise ValueError("index manifest entry is incomplete")
        digest = item["content_sha256"].lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("index manifest content digest is invalid")
        item["content_sha256"] = digest
        if "content" in entry:
            computed = hashlib.sha256(
                str(entry.get("content") or "").encode("utf-8")
            ).hexdigest()
            if computed != digest:
                raise ValueError("index content does not match its stored digest")
        for value, seen, label in (
            (item["qdrant_point_id"], point_ids, "Qdrant point ID"),
            (item["chunk_id"], chunk_ids, "chunk ID"),
            (item["chunk_record_id"], record_ids, "chunk record ID"),
        ):
            if value in seen:
                raise ValueError(f"duplicate {label} in index manifest")
            seen.add(value)
        normalized.append(item)

    normalized.sort(
        key=lambda item: (
            item["qdrant_point_id"],
            item["chunk_record_id"],
        )
    )
    body = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()
