"""Run the Fujitsu RAG Hard Benchmark retrieval-only comparison.

The public benchmark contains 100 annotated questions and 34 referenced PDF
files.  PDFs are supplied separately under the benchmark terms of use; this
runner reads them from a local directory and never copies PDF content into the
published result files.

The two configurations follow the current Universal-RAG evaluation settings:

* 無優化版: flat 512-token page-local parent chunks, BM25 Top 30.
* 優化版: 128-token child chunks (overlap 10), BM25 Top 30, then expand each
  child hit to its 512-token parent evidence window.

Both versions are retrieval-only: no query rewrite, embedding, RRF,
cross-encoder reranking, or answer-model calls.  "Token" here means the
project's deterministic lexical token used for chunking, rather than a model
BPE token.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import sqlite3
import statistics
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "RAG測試題庫" / "05_Fujitsu-RAG-Hard-Benchmark" / "source"
DEFAULT_PDF_DIR = Path(
    os.environ.get("FUJITSU_RAG_HARD_PDF_DIR", "")
    or (Path(os.environ.get("TEMP", ".")) / "Fujitsu-RAG-Hard-Benchmark" / "dataset" / "PDFs")
)
DEFAULT_RUN_ROOT = Path(os.environ.get("TEMP", ".")) / "Universal-RAG-Fujitsu-RAG-Hard-full-512-128"
DEFAULT_DB = DEFAULT_RUN_ROOT / "fujitsu-rag-hard.sqlite"
DEFAULT_OUTPUT = DEFAULT_RUN_ROOT / "retrieval-results.json"
DEFAULT_REPORT = DEFAULT_RUN_ROOT / "retrieval-report.md"
DEFAULT_SUMMARY = DEFAULT_RUN_ROOT / "retrieval-summary.json"
DEFAULT_CHECKPOINT_ROOT = DEFAULT_RUN_ROOT / "checkpoints"

DATASET_NAME = "Fujitsu RAG Hard Benchmark"
DATASET_URL = "https://github.com/FujitsuResearch/Fujitsu-RAG-Hard-Benchmark"
BLOG_URL = "https://blog-en.fltech.dev/entry/2026/03/11/RAG-Hard-Benchmark-en"

PARENT_TOKENS = 512
CHILD_TOKENS = 128
CHILD_OVERLAP = 10
FINAL_TOP_K = 30

PUBLISHED_METRICS = (
    "Document recall@30",
    "任一證據命中",
    "平均耗時（秒／題）",
)

# Keep this in sync with the existing BM25-only benchmark runners.  Latin
# words/numbers remain whole terms; each non-ASCII word character is a lexical
# unit so Japanese questions and PDFs can be indexed without a language model.
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:+-]*|[^\W_]", re.UNICODE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


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


def _load_tasks(data_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    yaml_path = data_root / "dataset" / "FJ_KGQA_Hard.yaml"
    url_path = data_root / "dataset" / "DL_URL.csv"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Fujitsu annotation file is missing: {yaml_path}")
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks") if isinstance(payload, dict) else None
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("FJ_KGQA_Hard.yaml must contain a non-empty tasks array")

    questions: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        gold_pages: dict[str, set[int]] = {}
        for rationale in task.get("rationales") or []:
            if not isinstance(rationale, dict):
                continue
            filename = str(rationale.get("file_name") or "")
            if not filename:
                continue
            pages = {
                int(page.get("number"))
                for page in rationale.get("pages") or []
                if isinstance(page, dict) and page.get("number") is not None
            }
            if pages:
                gold_pages.setdefault(filename, set()).update(pages)
        if not gold_pages:
            continue
        questions.append(
            {
                "question_id": str(task.get("no.") or len(questions) + 1),
                "question": str(task.get("question") or ""),
                "retrieval_level": str(task.get("retrieval_level") or ""),
                "answer_level": str(task.get("answer_level") or ""),
                "gold_pages": {name: sorted(pages) for name, pages in gold_pages.items()},
            }
        )

    pdf_names = sorted({name for item in questions for name in item["gold_pages"]})
    metadata = {
        "question_pool_count": len(tasks),
        "sample_count": len(questions),
        "referenced_pdf_count": len(pdf_names),
        "referenced_pdf_names": pdf_names,
        "yaml_sha256": _sha256(yaml_path),
        "download_manifest_sha256": _sha256(url_path) if url_path.exists() else None,
        "retrieval_level_counts": _count_values(questions, "retrieval_level"),
        "answer_level_counts": _count_values(questions, "answer_level"),
    }
    return questions, metadata


def _count_values(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _pdf_pages(pdf_dir: Path, filename: str) -> list[str]:
    path = pdf_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Referenced PDF is missing: {path}. Download it according to the benchmark repository's DL_URL.csv."
        )
    # Some PDFs emit parser diagnostics to stderr.  They are not useful in a
    # checkpointed benchmark run, so keep the runner output focused on progress.
    with contextlib.redirect_stderr(io.StringIO()):
        reader = PdfReader(str(path), strict=False)
        pages: list[str] = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
    return pages


def _pdf_manifest(pdf_dir: Path, names: Sequence[str]) -> tuple[str, dict[str, Any]]:
    digest = hashlib.sha256()
    files: dict[str, Any] = {}
    for name in sorted(names):
        path = pdf_dir / name
        if not path.exists():
            raise FileNotFoundError(
                f"Referenced PDF is missing: {path}. Download it according to DL_URL.csv."
            )
        stat = path.stat()
        sha = _sha256(path)
        digest.update(name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(sha.encode("ascii"))
        files[name] = {"bytes": stat.st_size, "sha256": sha}
    return digest.hexdigest(), files


def _ready_metadata(connection: sqlite3.Connection) -> dict[str, str] | None:
    try:
        rows = connection.execute(
            "SELECT key, value FROM metadata WHERE key IN ("
            "'status', 'dataset_hash', 'parent_tokens', 'child_tokens', 'child_overlap',"
            "'document_count', 'page_count', 'indexed_page_count', 'parent_chunk_count', 'child_chunk_count'"
            ")"
        ).fetchall()
    except sqlite3.DatabaseError:
        return None
    values = {str(key): str(value) for key, value in rows}
    return values if values.get("status") == "ready" else None


def _ensure_index(
    pdf_dir: Path,
    db_path: Path,
    pdf_names: Sequence[str],
) -> dict[str, Any]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_hash, file_manifest = _pdf_manifest(pdf_dir, pdf_names)
    if db_path.exists():
        try:
            with sqlite3.connect(db_path) as connection:
                marker = _ready_metadata(connection)
                if marker and marker.get("dataset_hash") == dataset_hash and all(
                    marker.get(key) == str(value)
                    for key, value in (
                        ("parent_tokens", PARENT_TOKENS),
                        ("child_tokens", CHILD_TOKENS),
                        ("child_overlap", CHILD_OVERLAP),
                    )
                ):
                    return {
                        "index_reused": True,
                        "document_count": int(marker.get("document_count", 0)),
                        "page_count": int(marker.get("page_count", 0)),
                        "indexed_page_count": int(marker.get("indexed_page_count", 0)),
                        "parent_chunk_count": int(marker.get("parent_chunk_count", 0)),
                        "child_chunk_count": int(marker.get("child_chunk_count", 0)),
                        "pdf_manifest": file_manifest,
                    }
        except (sqlite3.DatabaseError, OSError, ValueError):
            pass
        _remove_sqlite_files(db_path)

    print(f"[index] building BM25 index from {len(pdf_names):,} PDF files", flush=True)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.executescript(
            """
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE VIRTUAL TABLE parent_chunks USING fts5(
                file_name UNINDEXED,
                page_number UNINDEXED,
                chunk_index UNINDEXED,
                start_token UNINDEXED,
                text,
                tokenize='unicode61'
            );
            CREATE VIRTUAL TABLE child_chunks USING fts5(
                file_name UNINDEXED,
                page_number UNINDEXED,
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
        document_count = 0
        page_count = 0
        indexed_page_count = 0
        parent_count = 0
        child_count = 0
        for document_count, filename in enumerate(sorted(pdf_names), start=1):
            pages = _pdf_pages(pdf_dir, filename)
            page_count += len(pages)
            for page_index, page_text in enumerate(pages, start=1):
                tokens = _tokenize(page_text)
                if tokens:
                    indexed_page_count += 1
                parents = list(_windows(tokens, PARENT_TOKENS)) or [(0, [])]
                for parent_index, (parent_start, parent_tokens) in enumerate(parents):
                    parent_rows.append(
                        (
                            filename,
                            page_index,
                            parent_index,
                            parent_start,
                            " ".join(parent_tokens),
                        )
                    )
                    parent_count += 1
                    children = list(_windows(parent_tokens, CHILD_TOKENS, CHILD_OVERLAP))
                    for child_index, (child_start, child_tokens) in enumerate(children):
                        child_rows.append(
                            (
                                filename,
                                page_index,
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
                        "INSERT INTO parent_chunks(file_name, page_number, chunk_index, start_token, text) VALUES (?, ?, ?, ?, ?)",
                        parent_rows,
                    )
                    parent_rows.clear()
                if len(child_rows) >= 8000:
                    connection.executemany(
                        "INSERT INTO child_chunks(file_name, page_number, parent_chunk_index, parent_start_token, child_index, child_start_token, text) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        child_rows,
                    )
                    child_rows.clear()
            print(
                f"[index] {document_count:,}/{len(pdf_names):,} files; "
                f"{page_count:,} pages, {parent_count:,} parents, {child_count:,} children",
                flush=True,
            )
        if parent_rows:
            connection.executemany(
                "INSERT INTO parent_chunks(file_name, page_number, chunk_index, start_token, text) VALUES (?, ?, ?, ?, ?)",
                parent_rows,
            )
        if child_rows:
            connection.executemany(
                "INSERT INTO child_chunks(file_name, page_number, parent_chunk_index, parent_start_token, child_index, child_start_token, text) VALUES (?, ?, ?, ?, ?, ?, ?)",
                child_rows,
            )
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [
                ("status", "ready"),
                ("dataset_hash", dataset_hash),
                ("parent_tokens", str(PARENT_TOKENS)),
                ("child_tokens", str(CHILD_TOKENS)),
                ("child_overlap", str(CHILD_OVERLAP)),
                ("document_count", str(document_count)),
                ("page_count", str(page_count)),
                ("indexed_page_count", str(indexed_page_count)),
                ("parent_chunk_count", str(parent_count)),
                ("child_chunk_count", str(child_count)),
            ],
        )
        connection.commit()
        return {
            "index_reused": False,
            "document_count": document_count,
            "page_count": page_count,
            "indexed_page_count": indexed_page_count,
            "parent_chunk_count": parent_count,
            "child_chunk_count": child_count,
            "pdf_manifest": file_manifest,
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
        select = "rowid, file_name, page_number, chunk_index, start_token, text"
    elif table == "child_chunks":
        select = (
            "rowid, file_name, page_number, parent_chunk_index, parent_start_token, "
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
            rowid, filename, page_number, chunk_index, start_token, text = row[:6]
            result = {
                "rowid": int(rowid),
                "file_name": str(filename),
                "page_number": int(page_number),
                "chunk_index": int(chunk_index),
                "start_token": int(start_token),
                "text": str(text),
            }
        else:
            (
                rowid,
                filename,
                page_number,
                parent_chunk_index,
                parent_start_token,
                child_index,
                child_start_token,
                text,
            ) = row[:8]
            result = {
                "rowid": int(rowid),
                "file_name": str(filename),
                "page_number": int(page_number),
                "parent_chunk_index": int(parent_chunk_index),
                "parent_start_token": int(parent_start_token),
                "child_index": int(child_index),
                "child_start_token": int(child_start_token),
                "text": str(text),
            }
        # SQLite FTS5 bm25 is lower-is-better (normally negative); expose a
        # positive score and retain the deterministic rowid tie-breaker.
        result["score"] = round(-float(row[-1]), 8)
        results.append(result)
    return results


def _public_context(item: dict[str, Any], rank: int) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "rank": rank,
        "file_name": item["file_name"],
        "page_number": item["page_number"],
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


def _retrieval_metrics(
    contexts: Sequence[dict[str, Any]],
    question: dict[str, Any],
) -> dict[str, Any]:
    gold_pages: dict[str, set[int]] = {
        str(name).casefold(): {int(page) for page in pages}
        for name, pages in question["gold_pages"].items()
    }
    retrieved_docs = {str(context["file_name"]).casefold() for context in contexts}
    gold_docs = set(gold_pages)
    document_recall = (
        len(retrieved_docs & gold_docs) / len(gold_docs) if gold_docs else 0.0
    )
    evidence_hit = any(
        str(context["file_name"]).casefold() in gold_pages
        and int(context["page_number"]) in gold_pages[str(context["file_name"]).casefold()]
        for context in contexts
    )
    return {
        "document_recall": round(document_recall, 8),
        "any_evidence_hit": bool(evidence_hit),
    }


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
    print(
        f"[{version_name}] {len(completed):,} checkpointed, "
        f"{len(questions) - len(completed):,} remaining",
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
                # Several child hits can point to one 512-token parent.  Keep
                # the best child score and its first child rank per parent.
                best_by_parent: dict[tuple[str, int, int], dict[str, Any]] = {}
                for child_rank, child in enumerate(raw_hits, start=1):
                    key = (
                        child["file_name"],
                        child["page_number"],
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
                "retrieval": {
                    "method": "BM25",
                    "candidate_table": table,
                    "candidate_k": FINAL_TOP_K,
                    "final_top_k": FINAL_TOP_K,
                    "query_rewrite": False,
                    "embedding": False,
                    "rrf": False,
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
            if position % 10 == 0 or position == len(questions):
                print(
                    f"[{version_name}] {position:,}/{len(questions):,} questions",
                    flush=True,
                )
    return {
        "name": version_name,
        "version": version_id,
        "configuration": (
            "Flat 512-token page-local parent chunks with BM25 Top 30; no query rewrite, "
            "Embedding, RRF, reranking, or parent expansion"
            if not optimized
            else "128-token page-local child chunks (overlap 10), BM25 Top 30, expanded to "
            "512-token parent evidence; no query rewrite, Embedding, RRF, or reranking"
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


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Fujitsu RAG Hard Benchmark 全量檢索比較報告",
        "",
        f"本次使用公開 benchmark 的 {payload['sample_count']:,} 題、{payload['corpus_documents']:,} 份參考 PDF（共 {payload['corpus_pages']:,} 頁），只執行證據檢索，不呼叫回答模型。",
        f"資料集：[Fujitsu RAG Hard Benchmark]({DATASET_URL})；背景介紹：[Fujitsu Research 技術文章]({BLOG_URL})。",
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
            "指標定義：Document recall@30 是每題 Gold 文件出現在 Top 30 證據的比例；若一題有多份 Gold 文件，先計算該題命中的 Gold 文件比例，再對 100 題取平均。任一證據命中代表至少一個 Top 30 證據同時符合 Gold 文件與 Gold 頁碼。",
            "",
            "測試設定：無優化版直接以 page-local 512-token parent chunk 做 BM25 Top 30；優化版以 page-local 128-token child chunk（overlap 10）檢索，再將命中的 child 展開至 512-token parent evidence。兩者均不使用問題改寫、Embedding、RRF、重排或回答模型。",
            "",
            f"文件處理：本次本機索引 {payload['indexed_pages']:,}/{payload['corpus_pages']:,} 頁有可抽取文字；圖片型頁面未加入 OCR，因此若 Gold 只存在於圖片，BM25 文字檢索可能無法命中。PDF 依 benchmark 與各原始發布者條款留在本機暫存，未放入本專案結果。",
            "",
            "截至本次檢查，官方公開 repository 與技術文章提供資料集、標註與評測腳本，但沒有可直接對齊本報告 Document recall@30／任一證據命中／平均耗時三欄的聚合基準數據，因此本報告不填入外部數值。",
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
        "source_blog_url",
        "question_pool_count",
        "sample_count",
        "corpus_documents",
        "corpus_pages",
        "indexed_pages",
        "referenced_pdf_names",
        "question_level_counts",
        "dataset_hashes",
        "chunking",
        "query_rewrite_enabled",
        "embedding_enabled",
        "rrf_enabled",
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
    pdf_dir = args.pdf_dir.resolve()
    questions, question_meta = _load_tasks(data_root)
    index_meta = _ensure_index(pdf_dir, args.db.resolve(), question_meta["referenced_pdf_names"])
    with sqlite3.connect(args.db.resolve()) as connection:
        unoptimized = _run_version(
            connection,
            questions,
            checkpoint_path=args.checkpoint_root.resolve() / "unoptimized.jsonl",
            version_name="無優化版 BM25 Top 30",
            version_id="unoptimized_bm25_parent512_top30",
            table="parent_chunks",
            optimized=False,
        )
        optimized = _run_version(
            connection,
            questions,
            checkpoint_path=args.checkpoint_root.resolve() / "optimized.jsonl",
            version_name="優化版 BM25 Top 30",
            version_id="optimized_bm25_child128_parent512_top30",
            table="child_chunks",
            optimized=True,
        )

    payload = {
        "schema_version": "fujitsu-rag-hard-100-retrieval-only-v2-parent512-child128-top30",
        "dataset": DATASET_NAME,
        "dataset_url": DATASET_URL,
        "source_blog_url": BLOG_URL,
        "question_pool_count": question_meta["question_pool_count"],
        "sample_count": len(questions),
        "corpus_documents": index_meta["document_count"],
        "corpus_pages": index_meta["page_count"],
        "indexed_pages": index_meta["indexed_page_count"],
        "referenced_pdf_names": question_meta["referenced_pdf_names"],
        "question_level_counts": {
            "retrieval_level": question_meta["retrieval_level_counts"],
            "answer_level": question_meta["answer_level_counts"],
        },
        "dataset_hashes": {
            "yaml_sha256": question_meta["yaml_sha256"],
            "download_manifest_sha256": question_meta["download_manifest_sha256"],
            "pdf_manifest_sha256": hashlib.sha256(
                "".join(
                    f"{name}:{item['sha256']}:{item['bytes']}\n"
                    for name, item in sorted(index_meta["pdf_manifest"].items())
                ).encode("utf-8")
            ).hexdigest(),
        },
        "chunking": {
            "parent_chunk_tokens": PARENT_TOKENS,
            "child_chunk_tokens": CHILD_TOKENS,
            "child_overlap_tokens": CHILD_OVERLAP,
        },
        "query_rewrite_enabled": False,
        "embedding_enabled": False,
        "rrf_enabled": False,
        "reranking_enabled": False,
        "published_metrics": list(PUBLISHED_METRICS),
        "public_reference": {
            "available": False,
            "note": "Official repository and technical article provide the dataset and evaluation scripts, but no directly comparable aggregate metrics for the three columns in this report.",
            "repository": DATASET_URL,
            "article": BLOG_URL,
        },
        "retrieval_only": True,
        "model_calls": 0,
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "index": {
            "index_reused": index_meta["index_reused"],
            "parent_chunk_count": index_meta["parent_chunk_count"],
            "child_chunk_count": index_meta["child_chunk_count"],
        },
        "versions": [unoptimized, optimized],
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
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
