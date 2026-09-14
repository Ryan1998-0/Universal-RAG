"""Run the full Open RAG Benchmark retrieval comparison.

The benchmark is the Vectara ``open_ragbench`` PDF corpus with 3,045 queries,
1,000 papers, and section-level relevance labels.  This evaluator stops after
retrieval and runs the current BM25-only configurations:

* 無優化版: flat 200-token parent chunks, BM25 Top 30.
* 優化版: 40-token children (overlap 10), BM25 Top 30, then parent expansion
  to 200-token evidence windows.

No query rewriting, embeddings, reranking, or answer-model calls are made.
The dataset itself is downloaded separately from Hugging Face so the large PDF
corpus is not committed to this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import statistics
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = (
    Path(os.environ.get("TEMP", ".")) / "open_ragbench_data" / "pdf" / "arxiv"
)
DEFAULT_RUN_ROOT = Path(os.environ.get("TEMP", ".")) / "Universal-RAG-OpenRAG-full"
DEFAULT_DB = DEFAULT_RUN_ROOT / "open-rag-bench.sqlite"
DEFAULT_OUTPUT = DEFAULT_RUN_ROOT / "retrieval-results.json"
DEFAULT_REPORT = DEFAULT_RUN_ROOT / "retrieval-report.md"
DEFAULT_SUMMARY = DEFAULT_RUN_ROOT / "retrieval-summary.json"
DEFAULT_CHECKPOINT_ROOT = DEFAULT_RUN_ROOT / "checkpoints"

DATASET_NAME = "vectara/open_ragbench"
DATASET_URL = "https://huggingface.co/datasets/vectara/open_ragbench"
DATASET_REPOSITORY_URL = "https://github.com/vectara/open-rag-bench"
PUBLIC_REFERENCE_URL = (
    "https://github.com/Linkence-AI/Linkence-Benchmarks/blob/main/"
    "results/open_ragbench_full/metrics.json"
)
PUBLIC_REFERENCE_README_URL = "https://github.com/Linkence-AI/Linkence-Benchmarks/blob/main/README.md"

PARENT_TOKENS = 200
CHILD_TOKENS = 40
CHILD_OVERLAP = 10
FINAL_TOP_K = 30

PUBLISHED_METRICS = (
    "Document recall@30",
    "任一證據命中",
    "平均耗時（秒／題）",
)

# The dataset is English scientific text.  The final alternative keeps
# non-ASCII symbols searchable without putting punctuation into FTS operators.
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:+-]*|[^\W_]", re.UNICODE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _corpus_manifest(corpus_dir: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    files = sorted(corpus_dir.glob("*.json"))
    for path in files:
        stat = path.stat()
        digest.update(path.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
    return digest.hexdigest(), len(files)


def _load_questions(data_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    queries_path = data_root / "queries.json"
    qrels_path = data_root / "qrels.json"
    answers_path = data_root / "answers.json"
    corpus_dir = data_root / "corpus"
    required = [queries_path, qrels_path, corpus_dir]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Open RAG Benchmark data is missing: "
            + ", ".join(missing)
            + f". Download {DATASET_URL} into the expected data root."
        )

    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    qrels = json.loads(qrels_path.read_text(encoding="utf-8"))
    answers = (
        json.loads(answers_path.read_text(encoding="utf-8"))
        if answers_path.exists()
        else {}
    )
    if not isinstance(queries, dict) or not isinstance(qrels, dict):
        raise ValueError("queries.json and qrels.json must contain JSON objects")

    questions: list[dict[str, Any]] = []
    for question_id, query_data in queries.items():
        relevance = qrels.get(question_id)
        if not isinstance(query_data, dict) or not isinstance(relevance, dict):
            continue
        gold_doc_id = relevance.get("doc_id")
        gold_section_id = relevance.get("section_id")
        if not gold_doc_id or gold_section_id is None:
            continue
        questions.append(
            {
                "question_id": str(question_id),
                "question": str(query_data.get("query") or ""),
                "query_type": str(query_data.get("type") or ""),
                "query_source": str(query_data.get("source") or ""),
                "gold_doc_id": str(gold_doc_id),
                "gold_section_id": str(gold_section_id),
                "answer_available": question_id in answers,
            }
        )

    if not questions:
        raise ValueError("No valid query/qrels pairs were found")

    manifest_hash, corpus_count = _corpus_manifest(corpus_dir)
    metadata = {
        "question_pool_count": len(queries),
        "sample_count": len(questions),
        "corpus_document_count": corpus_count,
        "queries_sha256": _sha256(queries_path),
        "qrels_sha256": _sha256(qrels_path),
        "corpus_manifest_sha256": manifest_hash,
        "question_type_counts": _count_values(questions, "query_type"),
        "query_source_counts": _count_values(questions, "query_source"),
    }
    return questions, metadata


def _count_values(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def _section_text(record: dict[str, Any], section: dict[str, Any]) -> str:
    parts = [
        str(record.get("title") or ""),
        str(section.get("text") or ""),
    ]
    tables = section.get("tables") or {}
    if isinstance(tables, dict):
        parts.extend(str(value) for value in tables.values() if value)
    # Images are base64 blobs in the public dataset and have no OCR/caption.
    # A marker preserves the fact that the section contains an image without
    # polluting BM25 with millions of base64 tokens.
    if section.get("images"):
        parts.append("[IMAGE_CONTENT_UNINDEXED]")
    return "\n".join(part for part in parts if part.strip())


def _windows(
    tokens: Sequence[str],
    size: int,
    overlap: int = 0,
) -> Iterable[tuple[int, list[str]]]:
    if not tokens:
        yield 0, []
        return
    stride = max(1, int(size) - int(overlap))
    for start in range(0, len(tokens), stride):
        yield start, list(tokens[start : start + size])


def _fts_query(text: str) -> str:
    terms: list[str] = []
    seen: set[str] = set()
    for term in _tokenize(text):
        normalized = term.casefold()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        terms.append(term)
        if len(terms) >= 128:
            break
    if not terms:
        return '""'
    return " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)


def _remove_sqlite_files(path: Path) -> None:
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        if candidate.exists():
            candidate.unlink()


def _ready_metadata(connection: sqlite3.Connection) -> dict[str, str] | None:
    try:
        rows = connection.execute(
            "SELECT key, value FROM metadata WHERE key IN ('status', 'dataset_hash', 'parent_tokens', 'child_tokens', 'child_overlap', 'document_count', 'section_count')"
        ).fetchall()
    except sqlite3.DatabaseError:
        return None
    values = {str(key): str(value) for key, value in rows}
    return values if values.get("status") == "ready" else None


def _ensure_index(data_root: Path, db_path: Path, dataset_meta: dict[str, Any]) -> dict[str, Any]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    expected_hash = dataset_meta["corpus_manifest_sha256"]
    if db_path.exists():
        try:
            with sqlite3.connect(db_path) as connection:
                marker = _ready_metadata(connection)
                if marker and marker.get("dataset_hash") == expected_hash and all(
                    marker.get(key) == str(value)
                    for key, value in (
                        ("parent_tokens", PARENT_TOKENS),
                        ("child_tokens", CHILD_TOKENS),
                        ("child_overlap", CHILD_OVERLAP),
                    )
                ):
                    parent_count = int(
                        connection.execute("SELECT count(*) FROM parent_chunks").fetchone()[0]
                    )
                    child_count = int(
                        connection.execute("SELECT count(*) FROM child_chunks").fetchone()[0]
                    )
                    return {
                        "index_reused": True,
                        "document_count": int(marker.get("document_count", 0)),
                        "section_count": int(marker.get("section_count", 0)),
                        "parent_chunk_count": parent_count,
                        "child_chunk_count": child_count,
                    }
        except (sqlite3.DatabaseError, OSError, ValueError):
            pass
        _remove_sqlite_files(db_path)

    corpus_dir = data_root / "corpus"
    print(f"[index] building BM25 index from {len(list(corpus_dir.glob('*.json'))):,} documents", flush=True)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.executescript(
            """
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE VIRTUAL TABLE parent_chunks USING fts5(
                doc_id UNINDEXED,
                section_id UNINDEXED,
                chunk_index UNINDEXED,
                start_token UNINDEXED,
                text,
                tokenize='unicode61'
            );
            CREATE VIRTUAL TABLE child_chunks USING fts5(
                doc_id UNINDEXED,
                section_id UNINDEXED,
                parent_chunk_index UNINDEXED,
                parent_start_token UNINDEXED,
                child_index UNINDEXED,
                child_start_token UNINDEXED,
                text,
                tokenize='unicode61'
            );
            """
        )
        parent_rows: list[tuple[Any, ...]] = []
        child_rows: list[tuple[Any, ...]] = []
        parent_count = 0
        child_count = 0
        section_count = 0
        document_count = 0
        corpus_files = sorted(corpus_dir.glob("*.json"))
        for path in corpus_files:
            record = json.loads(path.read_text(encoding="utf-8"))
            doc_id = str(record.get("id") or path.stem)
            document_count += 1
            for section in record.get("sections") or []:
                if not isinstance(section, dict):
                    continue
                section_count += 1
                section_id = str(section.get("section_id"))
                tokens = _tokenize(_section_text(record, section))
                parents = list(_windows(tokens, PARENT_TOKENS))
                if not parents:
                    parents = [(0, [])]
                for parent_index, (parent_start, parent_tokens) in enumerate(parents):
                    parent_text = " ".join(parent_tokens)
                    parent_rows.append(
                        (doc_id, section_id, parent_index, parent_start, parent_text)
                    )
                    parent_count += 1
                    children = list(_windows(parent_tokens, CHILD_TOKENS, CHILD_OVERLAP))
                    for child_index, (child_start, child_tokens) in enumerate(children):
                        child_rows.append(
                            (
                                doc_id,
                                section_id,
                                parent_index,
                                parent_start,
                                child_index,
                                child_start,
                                " ".join(child_tokens),
                            )
                        )
                        child_count += 1
                if len(parent_rows) >= 4000:
                    connection.executemany(
                        "INSERT INTO parent_chunks(doc_id, section_id, chunk_index, start_token, text) VALUES (?, ?, ?, ?, ?)",
                        parent_rows,
                    )
                    parent_rows.clear()
                if len(child_rows) >= 8000:
                    connection.executemany(
                        "INSERT INTO child_chunks(doc_id, section_id, parent_chunk_index, parent_start_token, child_index, child_start_token, text) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        child_rows,
                    )
                    child_rows.clear()
            if document_count % 50 == 0 or document_count == len(corpus_files):
                print(
                    f"[index] {document_count:,}/{len(corpus_files):,} documents; "
                    f"{section_count:,} sections, {parent_count:,} parents, {child_count:,} children",
                    flush=True,
                )
        if parent_rows:
            connection.executemany(
                "INSERT INTO parent_chunks(doc_id, section_id, chunk_index, start_token, text) VALUES (?, ?, ?, ?, ?)",
                parent_rows,
            )
        if child_rows:
            connection.executemany(
                "INSERT INTO child_chunks(doc_id, section_id, parent_chunk_index, parent_start_token, child_index, child_start_token, text) VALUES (?, ?, ?, ?, ?, ?, ?)",
                child_rows,
            )
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [
                ("status", "ready"),
                ("dataset_hash", expected_hash),
                ("parent_tokens", str(PARENT_TOKENS)),
                ("child_tokens", str(CHILD_TOKENS)),
                ("child_overlap", str(CHILD_OVERLAP)),
                ("document_count", str(document_count)),
                ("section_count", str(section_count)),
            ],
        )
        connection.commit()
        return {
            "index_reused": False,
            "document_count": document_count,
            "section_count": section_count,
            "parent_chunk_count": parent_count,
            "child_chunk_count": child_count,
        }
    except Exception:
        connection.rollback()
        connection.close()
        _remove_sqlite_files(db_path)
        raise
    finally:
        connection.close()


def _search(
    connection: sqlite3.Connection,
    table: str,
    question: str,
    top_k: int,
) -> list[dict[str, Any]]:
    match = _fts_query(question)
    if match == '""':
        return []
    if table == "parent_chunks":
        select = "rowid, doc_id, section_id, chunk_index, start_token, text"
    elif table == "child_chunks":
        select = (
            "rowid, doc_id, section_id, parent_chunk_index, parent_start_token, "
            "child_index, child_start_token, text"
        )
    else:
        raise ValueError(f"unexpected FTS table: {table}")
    rows = connection.execute(
        f"SELECT {select}, bm25({table}) AS bm25_score "
        f"FROM {table} WHERE {table} MATCH ? ORDER BY bm25_score, rowid LIMIT ?",
        (match, max(1, int(top_k))),
    ).fetchall()
    results: list[dict[str, Any]] = []
    for row in rows:
        if table == "parent_chunks":
            rowid, doc_id, section_id, chunk_index, start_token, text = row[:6]
            result = {
                "rowid": int(rowid),
                "doc_id": str(doc_id),
                "section_id": str(section_id),
                "chunk_index": int(chunk_index),
                "start_token": int(start_token),
                "text": str(text),
            }
        else:
            (
                rowid,
                doc_id,
                section_id,
                parent_chunk_index,
                parent_start_token,
                child_index,
                child_start_token,
                text,
            ) = row[:8]
            result = {
                "rowid": int(rowid),
                "doc_id": str(doc_id),
                "section_id": str(section_id),
                "parent_chunk_index": int(parent_chunk_index),
                "parent_start_token": int(parent_start_token),
                "child_index": int(child_index),
                "child_start_token": int(child_start_token),
                "text": str(text),
            }
        # FTS5 bm25 is lower-is-better and normally negative.
        result["score"] = round(-float(row[-1]), 8)
        results.append(result)
    return results


def _retrieval_metrics(
    contexts: Sequence[dict[str, Any]],
    question: dict[str, Any],
) -> dict[str, Any]:
    gold_doc = str(question["gold_doc_id"]).casefold()
    gold_section = str(question["gold_section_id"])
    document_hit = any(str(context["doc_id"]).casefold() == gold_doc for context in contexts)
    evidence_hit = any(
        str(context["doc_id"]).casefold() == gold_doc
        and str(context["section_id"]) == gold_section
        for context in contexts
    )
    return {
        "document_recall": 1.0 if document_hit else 0.0,
        "any_evidence_hit": bool(evidence_hit),
    }


def _public_context(item: dict[str, Any], rank: int) -> dict[str, Any]:
    fields = {
        "rank": rank,
        "doc_id": item["doc_id"],
        "section_id": item["section_id"],
        "score": item["score"],
    }
    for key in (
        "chunk_index",
        "start_token",
        "parent_chunk_index",
        "parent_start_token",
        "child_index",
        "child_start_token",
        "child_rank",
    ):
        if key in item:
            fields[key] = item[key]
    return fields


def _run_unoptimized(
    connection: sqlite3.Connection,
    questions: Sequence[dict[str, Any]],
    checkpoint_path: Path,
) -> dict[str, Any]:
    return _run_version(
        connection,
        questions,
        checkpoint_path=checkpoint_path,
        version_name="無優化版 BM25 Top 30",
        version_id="unoptimized_bm25_parent200_top30",
        table="parent_chunks",
        optimized=False,
    )


def _run_optimized(
    connection: sqlite3.Connection,
    questions: Sequence[dict[str, Any]],
    checkpoint_path: Path,
) -> dict[str, Any]:
    return _run_version(
        connection,
        questions,
        checkpoint_path=checkpoint_path,
        version_name="優化版 BM25 Top 30",
        version_id="optimized_bm25_child40_parent200_top30",
        table="child_chunks",
        optimized=True,
    )


def _load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cases: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            case = json.loads(line)
            question_id = str(case.get("question_id") or "")
            if question_id:
                cases[question_id] = case
    return cases


def _run_version(
    connection: sqlite3.Connection,
    questions: Sequence[dict[str, Any]],
    *,
    checkpoint_path: Path,
    version_name: str,
    version_id: str,
    table: str,
    optimized: bool,
) -> dict[str, Any]:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _load_checkpoint(checkpoint_path)
    ordered_cases: list[dict[str, Any]] = []
    pending = [question for question in questions if question["question_id"] not in completed]
    print(
        f"[{version_name}] {len(completed):,} checkpointed, "
        f"{len(pending):,} remaining",
        flush=True,
    )
    with checkpoint_path.open("a", encoding="utf-8") as checkpoint:
        for position, question in enumerate(questions, start=1):
            question_id = question["question_id"]
            if question_id in completed:
                ordered_cases.append(completed[question_id])
                continue
            started = time.perf_counter()
            raw_hits = _search(connection, table, question["question"], FINAL_TOP_K)
            if optimized:
                # A child hit is promoted to its parent evidence window.  If
                # several children map to the same parent, retain the best
                # child score and its original rank.
                best_by_parent: dict[tuple[str, str, int], dict[str, Any]] = {}
                for child_rank, child in enumerate(raw_hits, start=1):
                    key = (
                        child["doc_id"],
                        child["section_id"],
                        child["parent_chunk_index"],
                    )
                    candidate = dict(child)
                    candidate["child_rank"] = child_rank
                    previous = best_by_parent.get(key)
                    if previous is None or candidate["score"] > previous["score"]:
                        best_by_parent[key] = candidate
                raw_hits = sorted(
                    best_by_parent.values(),
                    key=lambda item: (-item["score"], item["child_rank"]),
                )[:FINAL_TOP_K]
            contexts = [
                _public_context(item, rank)
                for rank, item in enumerate(raw_hits, start=1)
            ]
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            case = {
                "question_id": question_id,
                "question": question["question"],
                "query_type": question["query_type"],
                "query_source": question["query_source"],
                "gold_doc_id": question["gold_doc_id"],
                "gold_section_id": question["gold_section_id"],
                "retrieval": {
                    "method": "BM25",
                    "candidate_table": table,
                    "candidate_k": FINAL_TOP_K,
                    "final_top_k": FINAL_TOP_K,
                    "query_rewrite": False,
                    "embedding": False,
                    "reranking": False,
                    "parent_expansion": optimized,
                    "parent_chunk_tokens": PARENT_TOKENS,
                    "child_chunk_tokens": CHILD_TOKENS if optimized else None,
                    "child_overlap_tokens": CHILD_OVERLAP if optimized else None,
                },
                "metrics": _retrieval_metrics(contexts, question),
                "timings_ms": {"question_stage": round(elapsed_ms, 3)},
                "contexts": contexts,
            }
            checkpoint.write(json.dumps(case, ensure_ascii=False) + "\n")
            checkpoint.flush()
            ordered_cases.append(case)
            if position % 100 == 0 or position == len(questions):
                print(
                    f"[{version_name}] {position:,}/{len(questions):,} questions",
                    flush=True,
                )
    return {
        "name": version_name,
        "version": version_id,
        "configuration": (
            "Flat 200-token parent chunks with BM25 Top 30; no query rewrite, "
            "Embedding, reranking, or parent expansion"
            if not optimized
            else "40-token child chunks (overlap 10), BM25 Top 30, expanded to "
            "200-token parent evidence; no query rewrite, Embedding, or reranking"
        ),
        "summary": _summarize(ordered_cases),
        "cases": ordered_cases,
    }


def _summarize(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        return {
            "question_count": 0,
            "document_recall_at_30": None,
            "any_evidence_hit_rate": None,
            "average_question_seconds": None,
        }
    return {
        "question_count": len(cases),
        "document_recall_at_30": round(
            statistics.mean(float(case["metrics"]["document_recall"]) for case in cases),
            8,
        ),
        "any_evidence_hit_rate": round(
            statistics.mean(bool(case["metrics"]["any_evidence_hit"]) for case in cases),
            8,
        ),
        "average_question_seconds": round(
            statistics.mean(float(case["timings_ms"]["question_stage"]) for case in cases)
            / 1000.0,
            6,
        ),
    }


def _public_reference() -> dict[str, Any]:
    return {
        "name": "Linkence-Benchmarks 公開 full 結果",
        "repository": "https://github.com/Linkence-AI/Linkence-Benchmarks",
        "results": PUBLIC_REFERENCE_URL,
        "queries": 3045,
        "method": "hybrid hashed-TF-IDF sparse + text-embedding-3-small dense; section units; no reranker; top_k=20",
        "relaxed_document_hit_at_20": 0.9977,
        "strict_section_hit_at_20": 0.9691,
        "latency_p50_seconds": 0.4185,
        "comparison_note": "公開結果提供 @20 與 p50 latency，沒有本次 @30 平均耗時，因此只作外部參考，不併入本次主表。",
    }


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Open RAG Benchmark 全量檢索比較報告",
        "",
        f"本次使用 {payload['sample_count']:,} 題、{payload['corpus_documents']:,} 份 PDF 文件與 {payload.get('corpus_sections') or 0:,} 個 section，只執行證據檢索，不呼叫回答模型。",
        f"資料集：[vectara/open_ragbench]({DATASET_URL})；原始專案：[vectara/open-rag-bench]({DATASET_REPOSITORY_URL})。",
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
            f"{float(summary.get('average_question_seconds') or 0.0):.3f} |"
        )
    lines.extend(
        [
            "",
            "指標定義：Document recall@30 代表 Gold 文件出現在 Top 30 證據；任一證據命中代表 Gold 文件與 Gold section 同時出現在 Top 30 證據。",
            "",
            "測試設定：無優化版直接以 200-token parent chunk 做 BM25 Top 30；優化版以 40-token child chunk（overlap 10）檢索，再展開至 200-token parent evidence。兩者均不使用 Query Rewrite、Embedding、RRF、重排或回答模型。",
            "",
            "## 公開參考結果",
            "",
            f"[Linkence-Benchmarks 公開 full 結果]({PUBLIC_REFERENCE_README_URL})；[metrics.json]({PUBLIC_REFERENCE_URL}) 同樣評測 3,045 題，使用 hybrid hashed-TF-IDF + `text-embedding-3-small`、section 單位、Top 20、無重排；其公開 relaxed document hit@20 為 99.77%、strict section hit@20 為 96.91%、p50 latency 為 0.419 秒。這些指標與本次 Top 30 平均耗時不同，因此只作外部參考。",
            "",
            "資料集含文字、表格與圖片標記；圖片內容以 base64 儲存且沒有 OCR／caption，本次 BM25 只索引文字與表格內容。",
            "",
        ]
    )
    return "\n".join(lines)


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2%}"


def _summary_payload(payload: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version",
        "dataset",
        "dataset_url",
        "source_repository",
        "question_pool_count",
        "sample_count",
        "corpus_documents",
        "corpus_sections",
        "question_type_counts",
        "query_source_counts",
        "dataset_hashes",
        "chunking",
        "query_rewrite_enabled",
        "embedding_enabled",
        "reranking_enabled",
        "published_metrics",
        "public_reference",
        "retrieval_only",
        "model_calls",
        "run_at",
    )
    summary = {key: payload[key] for key in keys if key in payload}
    summary["versions"] = [
        {
            "name": version["name"],
            "version": version["version"],
            "configuration": version["configuration"],
            "summary": version["summary"],
        }
        for version in payload["versions"]
    ]
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_root = args.data_root.resolve()
    questions, question_meta = _load_questions(data_root)
    index_meta = _ensure_index(data_root, args.db.resolve(), question_meta)
    with sqlite3.connect(args.db.resolve()) as connection:
        unoptimized = _run_unoptimized(
            connection,
            questions,
            args.checkpoint_root.resolve() / "unoptimized.jsonl",
        )
        optimized = _run_optimized(
            connection,
            questions,
            args.checkpoint_root.resolve() / "optimized.jsonl",
        )

    payload = {
        "schema_version": "open-rag-bench-full-3045-retrieval-only-v1-parent200-child40-top30",
        "dataset": DATASET_NAME,
        "dataset_url": DATASET_URL,
        "source_repository": DATASET_REPOSITORY_URL,
        "question_pool_count": question_meta["question_pool_count"],
        "sample_count": len(questions),
        "corpus_documents": index_meta.get(
            "document_count", question_meta["corpus_document_count"]
        ),
        "corpus_sections": index_meta.get("section_count"),
        "parent_chunk_count": index_meta.get("parent_chunk_count"),
        "child_chunk_count": index_meta.get("child_chunk_count"),
        "question_type_counts": question_meta["question_type_counts"],
        "query_source_counts": question_meta["query_source_counts"],
        "dataset_hashes": {
            "queries_sha256": question_meta["queries_sha256"],
            "qrels_sha256": question_meta["qrels_sha256"],
            "corpus_manifest_sha256": question_meta["corpus_manifest_sha256"],
        },
        "chunking": {
            "parent_chunk_tokens": PARENT_TOKENS,
            "child_chunk_tokens": CHILD_TOKENS,
            "child_overlap_tokens": CHILD_OVERLAP,
        },
        "query_rewrite_enabled": False,
        "embedding_enabled": False,
        "reranking_enabled": False,
        "published_metrics": list(PUBLISHED_METRICS),
        "public_reference": _public_reference(),
        "retrieval_only": True,
        "model_calls": 0,
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "index": index_meta,
        "versions": [unoptimized, optimized],
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    args = parser.parse_args()
    payload = run(args)
    for path in (args.output, args.report, args.summary):
        path.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.report.resolve().write_text(_render_report(payload), encoding="utf-8")
    args.summary.resolve().write_text(
        json.dumps(_summary_payload(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "report": str(args.report.resolve()),
                "summary": str(args.summary.resolve()),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
