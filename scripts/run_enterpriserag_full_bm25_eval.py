#!/usr/bin/env python3
"""Run a full EnterpriseRAG-Bench retrieval comparison.

The benchmark contains a large zip archive, so the evaluator builds a local
SQLite FTS5 index instead of keeping the whole corpus in Python memory.  Both
branches use the same direct document-level BM25 Top 30 parameters and the
same original user question, with query rewriting and reranking disabled.  The
optimized branch additionally expands each selected document through 40-token
children to 200-token parent context.

The JSON, summary, and Markdown report can be copied into ``evals/`` as
versioned benchmark artifacts after the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sqlite3
import statistics
import time
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = (
    ROOT
    / "RAG測試題庫"
    / "03_EnterpriseRAG-Bench"
    / "source"
    / "questions.jsonl"
)
DEFAULT_TEMP_ROOT = Path(os.environ.get("TEMP", ".")) / "Universal-RAG-EnterpriseRAG-100"
DEFAULT_ARCHIVE = DEFAULT_TEMP_ROOT / "all_documents.zip"
DEFAULT_DB = DEFAULT_TEMP_ROOT / "corpus-bm25.sqlite"
DEFAULT_OUTPUT = DEFAULT_TEMP_ROOT / "retrieval-results.json"
DEFAULT_REPORT = DEFAULT_TEMP_ROOT / "retrieval-report.md"
DEFAULT_SUMMARY = DEFAULT_TEMP_ROOT / "retrieval-summary.json"
# Keep this one-off run in Qdrant's in-memory local engine.  Persisting each
# of half a million points to disk dominates the runtime and the user asked
# for a temporary report only.
DEFAULT_QDRANT = Path(":memory:")

SAMPLE_COUNT = 500
SAMPLE_SEED = 42
BM25_K1 = 1.4
BM25_B = 0.72
FINAL_TOP_K = 30
OPTIMIZED_FINAL_TOP_K = 30
# Enterprise experiment: smaller parent/child evidence windows.  The child
# overlap is kept at 25% of the child size so neighbouring evidence retains a
# small amount of context without producing duplicate one-token windows.
PARENT_TOKENS = 200
CHILD_TOKENS = 40
CHILD_OVERLAP = 10
DOC_ID_RE = re.compile(r"(dsid_[0-9a-f]{32})", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:-]*")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_questions(path: Path, *, seed: int = SAMPLE_SEED, count: int = SAMPLE_COUNT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) < count:
        raise ValueError(f"Expected at least {count} EnterpriseRAG questions, got {len(rows)}")
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), count))
    selected = [rows[index] for index in indices]
    categories = Counter(str(row.get("question_type") or "unknown") for row in selected)
    return selected, {
        "available_questions": len(rows),
        "sample_count": len(selected),
        "sample_seed": seed,
        "selection": "uniform_without_replacement_over_question_file_order",
        "question_type_counts": dict(sorted(categories.items())),
    }


def _doc_id(path: str) -> str:
    match = DOC_ID_RE.search(Path(path).name)
    if not match:
        raise ValueError(f"Document path has no dsid identifier: {path}")
    return match.group(1).lower()


def _ensure_bm25_database(archive_path: Path, db_path: Path) -> dict[str, Any]:
    """Create or reuse an FTS5 document index from the release zip."""

    archive_hash = _sha256(archive_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.is_file():
        try:
            with sqlite3.connect(db_path) as connection:
                meta = dict(connection.execute("SELECT key, value FROM metadata").fetchall())
                count = int(connection.execute("SELECT count(*) FROM documents").fetchone()[0])
            if meta.get("archive_sha256") == archive_hash and count > 0:
                return {
                    "archive_sha256": archive_hash,
                    "corpus_documents": count,
                    "index_reused": True,
                    "index_path": str(db_path),
                }
        except (sqlite3.Error, OSError, ValueError):
            pass
        try:
            db_path.unlink()
        except OSError:
            pass

    started = time.perf_counter()
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute(
            "CREATE VIRTUAL TABLE documents USING fts5("
            "doc_id UNINDEXED, path UNINDEXED, content, "
            "tokenize='unicode61 remove_diacritics 2'"
            ")"
        )
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        insert_sql = "INSERT INTO documents(doc_id, path, content) VALUES (?, ?, ?)"
        count = 0
        with zipfile.ZipFile(archive_path) as archive:
            names = sorted(
                name
                for name in archive.namelist()
                if name.lower().endswith(".txt") and not name.endswith("/")
            )
            for name in names:
                body = archive.read(name).decode("utf-8", errors="replace")
                if not body.strip():
                    continue
                connection.execute(insert_sql, (_doc_id(name), name, body))
                count += 1
                if count % 5000 == 0:
                    connection.commit()
                    print(f"[BM25 index] {count:,}/{len(names):,} documents", flush=True)
        connection.commit()
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [
                ("archive_sha256", archive_hash),
                ("corpus_documents", str(count)),
                ("created_at", datetime.now().astimezone().isoformat(timespec="seconds")),
            ],
        )
        connection.commit()
    finally:
        connection.close()
    elapsed = (time.perf_counter() - started) * 1000.0
    return {
        "archive_sha256": archive_hash,
        "corpus_documents": count,
        "index_reused": False,
        "index_path": str(db_path),
        "index_build_ms": round(elapsed, 3),
    }


def _query_terms(text: str) -> list[str]:
    terms = []
    for token in WORD_RE.findall(str(text or "").lower()):
        token = token.strip("._:-/")
        if len(token) >= 2 and token not in terms:
            terms.append(token)
    return terms


def _fts_query(text: str) -> str:
    terms = _query_terms(text)
    if not terms:
        return '""'
    # Quoting each token prevents punctuation in model names or paths from
    # becoming FTS5 operators.  OR mirrors a permissive keyword baseline.
    return " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)


def _bm25_search(
    connection: sqlite3.Connection,
    question: str,
    top_k: int,
) -> list[dict[str, Any]]:
    match = _fts_query(question)
    if match == '""':
        return []
    rows = connection.execute(
        "SELECT rowid, doc_id, path, content, bm25(documents) AS bm25_score "
        "FROM documents WHERE documents MATCH ? ORDER BY bm25_score LIMIT ?",
        (match, max(1, int(top_k))),
    ).fetchall()
    return [
        {
            "rowid": int(row[0]),
            "doc_id": str(row[1]),
            "path": str(row[2]),
            "content": str(row[3]),
            # SQLite FTS5 returns lower-is-better negative BM25 values.
            "score": round(-float(row[4]), 8),
        }
        for row in rows
    ]


def _parent_child_context(
    text: str,
    question: str,
) -> dict[str, Any]:
    """Return parent/child sizing metadata for an already selected document.

    Document ranking remains the direct BM25 Top-30 result.  This helper only
    chooses the most query-bearing parent segment inside each selected
    document for evidence expansion; it does not reorder the fifteen documents.
    """

    tokens = WORD_RE.findall(str(text or ""))
    if not tokens:
        return {
            "parent_chunk_count": 0,
            "child_chunk_count": 0,
            "parent_chunk_index": None,
            "parent_token_count": 0,
        }
    parent_stride = PARENT_TOKENS
    parents = [
        tokens[start : start + PARENT_TOKENS]
        for start in range(0, len(tokens), parent_stride)
        if tokens[start : start + PARENT_TOKENS]
    ]
    child_stride = max(1, CHILD_TOKENS - CHILD_OVERLAP)
    child_count = sum(
        max(1, (len(parent) - 1) // child_stride + 1)
        for parent in parents
    )
    query_terms = set(_query_terms(question))
    if query_terms:
        parent_scores = [
            sum(1 for token in parent if token.lower() in query_terms)
            for parent in parents
        ]
        parent_index = max(range(len(parents)), key=lambda index: (parent_scores[index], -index))
    else:
        parent_index = 0
    selected_parent = parents[parent_index]
    return {
        "parent_chunk_count": len(parents),
        "child_chunk_count": child_count,
        "parent_chunk_index": parent_index,
        "parent_token_count": len(selected_parent),
    }


def _gold_metrics(retrieved: Sequence[dict[str, Any]], question: dict[str, Any]) -> dict[str, Any]:
    gold = {str(value).lower() for value in (question.get("expected_doc_ids") or []) if value}
    returned = [str(item.get("doc_id") or "").lower() for item in retrieved]
    returned_set = set(returned)
    overlap = gold & returned_set
    return {
        "gold_document_count": len(gold),
        "retrieved_document_ids": returned,
        "document_recall": (len(overlap) / len(gold)) if gold else None,
        "any_gold_document_hit": bool(overlap) if gold else None,
    }


def _summarize(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(case["timings_ms"]["question_stage"]) for case in cases]
    scored = [case["metrics"] for case in cases if case["metrics"]["gold_document_count"] > 0]
    if not cases:
        return {"question_count": 0, "gold_document_questions": 0}
    return {
        "question_count": len(cases),
        "gold_document_questions": len(scored),
        "document_recall_at_30": round(
            statistics.mean(metric["document_recall"] for metric in scored), 8
        ) if scored else None,
        "any_evidence_hit_rate": round(
            sum(bool(metric["any_gold_document_hit"]) for metric in scored) / len(scored), 8
        ) if scored else None,
        "average_question_seconds": round(statistics.mean(latencies) / 1000.0, 6),
    }


def _run_bm25_version(
    connection: sqlite3.Connection,
    questions: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    cases = []
    label = "無優化版 BM25 Top 30"
    for position, question in enumerate(questions, start=1):
        started = time.perf_counter()
        retrieved = _bm25_search(
            connection,
            str(question.get("question") or ""),
            FINAL_TOP_K,
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        metrics = _gold_metrics(retrieved, question)
        cases.append(
            {
                "question_id": question.get("question_id"),
                "question_type": question.get("question_type"),
                "question": question.get("question"),
                "retrieval": {
                    "method": "BM25",
                    "candidate_k": FINAL_TOP_K,
                    "query_rewrite": False,
                    "rerank": False,
                    "embedding": False,
                    "parent_chunk_tokens": PARENT_TOKENS,
                    "child_chunk_tokens": CHILD_TOKENS,
                    "child_overlap_tokens": CHILD_OVERLAP,
                    "parent_expansion": False,
                    "final_top_k": FINAL_TOP_K,
                },
                "metrics": metrics,
                "timings_ms": {"question_stage": round(elapsed, 3)},
                "contexts": [
                    {
                        "rank": rank,
                        "doc_id": item["doc_id"],
                        "path": item["path"],
                        "score": item["score"],
                    }
                    for rank, item in enumerate(retrieved, start=1)
                ],
            }
        )
        if position == 1 or position % 10 == 0 or position == len(questions):
            hit = metrics.get("any_gold_document_hit")
            print(f"[{label}] {position}/{len(questions)} | {elapsed:.1f} ms | any={hit}", flush=True)
    return {
        "name": label,
        "version": "unoptimized_bm25_top30",
        "configuration": (
            f"Document-level BM25 direct Top 30; no query rewrite, Embedding, "
            f"reranking, or parent-child expansion (shared chunk settings {PARENT_TOKENS}/{CHILD_TOKENS})"
        ),
        "summary": _summarize(cases),
        "cases": cases,
    }


def _run_optimized_bm25_top30(
    connection: sqlite3.Connection,
    questions: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Run the requested no-rerank, BM25-only Top-30 optimized branch."""

    cases = []
    label = "優化版 BM25 Top 30"
    for position, question in enumerate(questions, start=1):
        started = time.perf_counter()
        question_text = str(question.get("question") or "")
        retrieved = _bm25_search(
            connection,
            question_text,
            OPTIMIZED_FINAL_TOP_K,
        )
        chunk_metadata = [
            _parent_child_context(item.get("content", ""), question_text)
            for item in retrieved
        ]
        elapsed = (time.perf_counter() - started) * 1000.0
        metrics = _gold_metrics(retrieved, question)
        cases.append(
            {
                "question_id": question.get("question_id"),
                "question_type": question.get("question_type"),
                "question": question.get("question"),
                "retrieval": {
                    "method": "BM25",
                    "candidate_k": OPTIMIZED_FINAL_TOP_K,
                    "query_rewrite": False,
                    "rerank": False,
                    "embedding": False,
                    "parent_chunk_tokens": PARENT_TOKENS,
                    "child_chunk_tokens": CHILD_TOKENS,
                    "child_overlap_tokens": CHILD_OVERLAP,
                    "parent_expansion": True,
                    "final_top_k": OPTIMIZED_FINAL_TOP_K,
                },
                "metrics": metrics,
                "timings_ms": {"question_stage": round(elapsed, 3)},
                "contexts": [
                    {
                        "rank": rank,
                        "doc_id": item["doc_id"],
                        "path": item["path"],
                        "score": item["score"],
                        "parent_chunk_tokens": PARENT_TOKENS,
                        "child_chunk_tokens": CHILD_TOKENS,
                        "child_overlap_tokens": CHILD_OVERLAP,
                        "parent_chunk_index": chunk_info["parent_chunk_index"],
                        "parent_chunk_count": chunk_info["parent_chunk_count"],
                        "child_chunk_count": chunk_info["child_chunk_count"],
                    }
                    for rank, (item, chunk_info) in enumerate(
                        zip(retrieved, chunk_metadata, strict=True),
                        start=1,
                    )
                ],
            }
        )
        if position == 1 or position % 10 == 0 or position == len(questions):
            hit = metrics.get("any_gold_document_hit")
            print(f"[{label}] {position}/{len(questions)} | {elapsed:.1f} ms | any={hit}", flush=True)
    return {
        "name": label,
        "version": "optimized_bm25_top30",
        "configuration": (
            f"Document-level BM25 direct Top 30 → {CHILD_TOKENS}-token child / "
            f"{PARENT_TOKENS}-token parent expansion (overlap {CHILD_OVERLAP}); "
            "no query rewrite, Embedding, RRF, or reranking"
        ),
        "summary": _summarize(cases),
        "cases": cases,
    }


def _build_vector_baseline(
    connection: sqlite3.Connection,
    qdrant_path: Path,
    *,
    model_name: str,
    batch_size: int,
) -> tuple[Any, Any, dict[str, Any]]:
    from qdrant_client import QdrantClient, models
    from sentence_transformers import SentenceTransformer

    if str(qdrant_path) == ":memory:":
        client = QdrantClient(location=":memory:")
    else:
        qdrant_path.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(qdrant_path))
    collection = "enterprise_rag_vector_baseline"
    if client.collection_exists(collection):
        client.delete_collection(collection)
    device = "cuda" if _cuda_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    dimension = int(model.get_sentence_embedding_dimension())
    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        # Defer HNSW construction until all points are loaded.  Local Qdrant
        # otherwise re-optimizes repeatedly at each indexing threshold.
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1_000_000),
    )

    started = time.perf_counter()
    cursor = connection.execute("SELECT rowid, doc_id, path, content FROM documents ORDER BY rowid")
    total = int(connection.execute("SELECT count(*) FROM documents").fetchone()[0])
    point_id = 0
    while True:
        rows = cursor.fetchmany(max(1, int(batch_size)))
        if not rows:
            break
        texts = [str(row[3]) for row in rows]
        vectors = model.encode(
            texts,
            batch_size=max(1, min(int(batch_size), 512)),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        points = [
            models.PointStruct(
                id=int(row[0]),
                vector=vector.tolist(),
                payload={"doc_id": str(row[1]), "path": str(row[2])},
            )
            for row, vector in zip(rows, vectors, strict=True)
        ]
        client.upsert(collection_name=collection, points=points, wait=False)
        point_id += len(points)
        if point_id % 5000 < len(points) or point_id == total:
            print(f"[Vector index] {point_id:,}/{total:,} documents ({device})", flush=True)
    # Build the HNSW index once after bulk loading rather than once per
    # threshold.  Local mode applies this update synchronously.
    client.update_collection(
        collection_name=collection,
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=20_000),
    )
    elapsed = (time.perf_counter() - started) * 1000.0
    return client, model, {
        "collection": collection,
        "vector_dimension": dimension,
        "vector_device": device,
        "vector_documents": point_id,
        "vector_index_ms": round(elapsed, 3),
        "vector_model": model_name,
    }


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _vector_search(
    client: Any,
    model: Any,
    vector_meta: dict[str, Any],
    question: str,
    top_k: int,
) -> list[dict[str, Any]]:
    vector = model.encode(
        [str(question or "")],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0].tolist()
    response = client.query_points(
        collection_name=vector_meta["collection"],
        query=vector,
        limit=max(1, int(top_k)),
        with_payload=True,
    )
    return [
        {
            "doc_id": str((point.payload or {}).get("doc_id") or ""),
            "path": str((point.payload or {}).get("path") or ""),
            "score": round(float(point.score), 8),
        }
        for point in response.points
    ]


def _run_vector_version(
    client: Any,
    model: Any,
    vector_meta: dict[str, Any],
    questions: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    cases = []
    for position, question in enumerate(questions, start=1):
        started = time.perf_counter()
        contexts = [
            {
                "rank": rank,
                "doc_id": item["doc_id"],
                "path": item["path"],
                "score": item["score"],
            }
            for rank, item in enumerate(
                _vector_search(
                    client,
                    model,
                    vector_meta,
                    str(question.get("question") or ""),
                    FINAL_TOP_K,
                ),
                start=1,
            )
        ]
        elapsed = (time.perf_counter() - started) * 1000.0
        metrics = _gold_metrics(contexts, question)
        cases.append(
            {
                "question_id": question.get("question_id"),
                "question_type": question.get("question_type"),
                "question": question.get("question"),
                "retrieval": {
                    "method": "Vector baseline",
                    "embedding_model": vector_meta["vector_model"],
                    "vector_database": "Qdrant local",
                    "final_top_k": FINAL_TOP_K,
                },
                "metrics": metrics,
                "timings_ms": {"question_stage": round(elapsed, 3)},
                "contexts": contexts,
            }
        )
        if position == 1 or position % 10 == 0 or position == len(questions):
            hit = metrics.get("any_gold_document_hit")
            print(f"[Vector baseline] {position}/{len(questions)} | {elapsed:.1f} ms | any={hit}", flush=True)
    return {
        "name": "Vector baseline",
        "version": "vector_baseline",
        "configuration": "Dense Embedding retrieval with Qdrant, final Top 5",
        "summary": _summarize(cases),
        "cases": cases,
    }


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2%}"


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# EnterpriseRAG-Bench 全量題目檢索比較報告",
        "",
        f"本次使用 EnterpriseRAG-Bench 題庫全部 {payload['sample_count']:,} 題，只執行證據檢索，不呼叫回答模型。",
        f"Corpus：{payload['corpus_documents']:,} 份企業文件。",
        "",
        "## 結果",
        "",
        "| 版本 | Document recall@30 | 任一證據命中 | 平均耗時（秒／題） |",
        "| --- | ---: | ---: | ---: |",
    ]
    for version in payload["versions"]:
        summary = version["summary"]
        lines.append(
            f"| {version['name']} | {_pct(summary.get('document_recall_at_30'))} | "
            f"{_pct(summary.get('any_evidence_hit_rate'))} | "
            f"{float(summary.get('average_question_seconds', 0.0)):.3f} |"
        )
    lines.extend(
        [
            "",
            "指標只對有 `expected_doc_ids` 的題目計算；高階與找不到資訊題沒有指定 Gold 文件，因此列為 n/a，不列入召回率分母。",
            "",
            f"測試設定：兩個版本均使用原始問題與文件級 BM25 Top 30，不使用 Query Rewrite、Embedding、RRF、重排或回答模型；優化版額外以 {CHILD_TOKENS}／{PARENT_TOKENS} 父子 Chunk（子 Chunk overlap {CHILD_OVERLAP}）展開證據。",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    questions, sample_meta = _load_questions(args.questions.resolve())
    archive = args.archive.resolve()
    db_meta = _ensure_bm25_database(archive, args.db.resolve())
    with sqlite3.connect(args.db.resolve()) as connection:
        unoptimized = _run_bm25_version(connection, questions)
        optimized = _run_optimized_bm25_top30(connection, questions)
    payload = {
        "schema_version": "enterpriserag-bench-full-500-retrieval-only-v14-200-40-top30-no-rewrite-public3",
        "dataset": "onyx-dot-app/EnterpriseRAG-Bench",
        "question_pool_count": sample_meta["available_questions"],
        "sample_count": sample_meta["sample_count"],
        "sample_seed": sample_meta["sample_seed"],
        "sample_selection": sample_meta["selection"],
        "question_type_counts": sample_meta["question_type_counts"],
        "archive_sha256": db_meta["archive_sha256"],
        "corpus_documents": db_meta["corpus_documents"],
        "bm25_index_reused": db_meta["index_reused"],
        "chunking": {
            "parent_chunk_tokens": PARENT_TOKENS,
            "child_chunk_tokens": CHILD_TOKENS,
            "child_overlap_tokens": CHILD_OVERLAP,
        },
        "query_rewrite_enabled": False,
        "reranking_enabled": False,
        "embedding_enabled": False,
        "evaluation_scope": "all_questions_two_architectures",
        "published_metrics": [
            "Document recall@30",
            "任一證據命中",
            "平均耗時（秒／題）",
        ],
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "retrieval_only": True,
        "model_calls": 0,
        "versions": [unoptimized, optimized],
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    payload = run(args)
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.resolve().write_text(_render_report(payload), encoding="utf-8")
    args.summary.resolve().parent.mkdir(parents=True, exist_ok=True)
    public_versions = []
    for version in payload["versions"]:
        summary = version["summary"]
        public_versions.append(
            {
                "name": version["name"],
                "version": version["version"],
                "configuration": version["configuration"],
                "summary": {
                    "question_count": summary.get("question_count"),
                    "gold_document_questions": summary.get("gold_document_questions"),
                    "document_recall_at_30": summary.get("document_recall_at_30"),
                    "any_evidence_hit_rate": summary.get("any_evidence_hit_rate"),
                    "average_question_seconds": summary.get("average_question_seconds"),
                },
            }
        )
    summary_payload = {
        key: payload[key]
        for key in (
            "schema_version",
            "dataset",
            "question_pool_count",
            "sample_count",
            "sample_seed",
            "sample_selection",
            "question_type_counts",
            "archive_sha256",
            "corpus_documents",
            "chunking",
            "query_rewrite_enabled",
            "reranking_enabled",
            "embedding_enabled",
            "evaluation_scope",
            "published_metrics",
            "run_at",
            "retrieval_only",
            "model_calls",
        )
        if key in payload
    }
    summary_payload["versions"] = public_versions
    args.summary.resolve().write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output.resolve()), "report": str(args.report.resolve()), "summary": str(args.summary.resolve())}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
