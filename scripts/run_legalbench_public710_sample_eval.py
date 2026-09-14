#!/usr/bin/env python3
"""Compare three retrieval architectures on a reproducible sample of the
public 710-question LegalBench-RAG held-out split.

The sample is selected from the public model repository's
``test_predictions.jsonl`` IDs.  All three branches receive exactly the same
question records and stop after evidence retrieval; the answer model is not
called.

Use ``--document-scope specified`` to reproduce the public benchmark's
known-document setting.  The benchmark-provided ``document_path`` is used
only as a source-document filter; gold spans remain evaluation-only.

The three branches are:

* ``legacy_unoptimized``: the project's original hard chunks and weighted
  BM25+dense fusion with the existing lexical/semantic reranker;
* ``legacy_optimized``: the project's parent-child/RRF/complexity-routed
  architecture;
* ``public_ettin``: the published 384/96-token BM25 + LegalBench-RAG Ettin
  cross-encoder method.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import numpy as np
import os
import random
import statistics
import sys
import time
import urllib.request
import unicodedata
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_legalbench_rag_ettin_eval as public_eval
import run_legalbench_rag_retrieval_full_eval as legacy_eval
from rag_demo.config import RagConfig
from rag_demo.cross_encoder import (
    DEFAULT_CROSS_ENCODER_MODEL,
    CrossEncoderReranker,
)
from rag_demo.embeddings import DEFAULT_EMBEDDING_MODEL


PUBLIC_TEST_URL = (
    "https://huggingface.co/lxyuan/LegalBenchRAG-Ettin-150M-Reranker/"
    "resolve/main/test_predictions.jsonl"
)
PUBLIC_MODEL_CARD_URL = "https://huggingface.co/lxyuan/LegalBenchRAG-Ettin-150M-Reranker"
DEFAULT_ARCHIVE = legacy_eval.DEFAULT_ZIP
DEFAULT_OUTPUT_DIR = ROOT / "evals" / "legalbench_public710_sample100"
DEFAULT_SAMPLE_FILE = DEFAULT_OUTPUT_DIR / "sample.json"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "retrieval-results.json"
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "retrieval-report.md"
DEFAULT_SUMMARY = DEFAULT_OUTPUT_DIR / "retrieval-summary.json"
DEFAULT_CHECKPOINT_DIR = DEFAULT_OUTPUT_DIR / "checkpoints"
DEFAULT_QUERY_CACHE = DEFAULT_OUTPUT_DIR / "cache" / "query-embeddings.npy"

SAMPLE_SEED = 42
SAMPLE_COUNT = 100
DOMAINS = ("contractnli", "cuad", "maud", "privacy_qa")


def _load_all_archive(
    archive_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    """Load all 6,889 archive rows, including the 31 nda-10 rows.

    The old paper-aligned evaluator intentionally drops those rows to produce
    6,858 questions.  The public 710 held-out split was created from the full
    6,889-row release, so this evaluator must preserve them when resolving the
    public IDs.
    """

    if not archive_path.is_file():
        raise FileNotFoundError(f"LegalBench-RAG archive not found: {archive_path}")

    items: list[dict[str, Any]] = []
    downloaded_counts: dict[str, int] = {}
    corpus: dict[str, str] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for subset in legacy_eval.SUBSETS:
            benchmark_name = f"benchmarks/{subset}.json"
            payload = json.loads(archive.read(benchmark_name).decode("utf-8"))
            tests = payload.get("tests") if isinstance(payload, dict) else payload
            if not isinstance(tests, list):
                raise ValueError(f"Invalid benchmark payload: {benchmark_name}")
            downloaded_counts[subset] = len(tests)
            for subset_index, raw in enumerate(tests):
                record = dict(raw)
                query = str(record.get("query") or "").strip()
                record.update(
                    {
                        "id": f"legalbench-{subset}-{subset_index:04d}",
                        "subset": subset,
                        "subset_index": subset_index,
                        "query": query,
                        "snippets": [
                            {
                                **dict(snippet),
                                "file_path": legacy_eval._canonical_file_path(
                                    str(snippet.get("file_path") or "")
                                ),
                            }
                            for snippet in (record.get("snippets") or [])
                        ],
                    }
                )
                items.append(record)

        for name in archive.namelist():
            if not name.startswith("corpus/") or not name.endswith(".txt"):
                continue
            relative_path = legacy_eval._canonical_file_path(name[len("corpus/") :])
            corpus[relative_path] = archive.read(name).decode("utf-8", errors="replace")

    counts = Counter(str(item.get("subset") or "unknown") for item in items)
    expected_counts = {"contractnli": 977, "cuad": 4042, "maud": 1676, "privacy_qa": 194}
    if dict(counts) != expected_counts:
        raise ValueError(f"Unexpected full archive counts: {dict(counts)}")
    missing_files = sorted(
        {
            str(snippet.get("file_path") or "")
            for item in items
            for snippet in item.get("snippets") or []
            if str(snippet.get("file_path") or "") not in corpus
        }
    )
    if missing_files:
        raise ValueError(f"Gold snippets reference missing corpus files: {missing_files[:3]}")
    metadata = {
        "downloaded_counts": downloaded_counts,
        "evaluated_counts": dict(sorted(counts.items())),
        "downloaded_total": sum(downloaded_counts.values()),
        "evaluated_total": len(items),
        "corpus_documents": len(corpus),
        "corpus_characters": sum(len(text) for text in corpus.values()),
        "selection_rule": "Full 6,889-row LegalBench-RAG archive; no nda-10 rows removed.",
    }
    return items, corpus, metadata


def _fetch_public_ids() -> tuple[list[str], str]:
    request = urllib.request.Request(PUBLIC_TEST_URL, headers={"User-Agent": "Universal-RAG/1.0"})
    raw = urllib.request.urlopen(request, timeout=60).read()
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    ids = sorted({str(record.get("query_id") or "") for record in records if record.get("query_id")})
    if len(ids) != 710:
        raise ValueError(f"Expected 710 public test IDs, got {len(ids)}")
    return ids, hashlib.sha256(raw).hexdigest()


def _fetch_public_document_paths() -> dict[str, str]:
    """Load the benchmark-provided source document for each public query."""

    request = urllib.request.Request(
        PUBLIC_TEST_URL,
        headers={"User-Agent": "Universal-RAG/1.0"},
    )
    raw = urllib.request.urlopen(request, timeout=60).read()
    records = [
        json.loads(line)
        for line in raw.decode("utf-8").splitlines()
        if line.strip()
    ]
    paths = {
        str(record.get("query_id")): str(record.get("document_path") or "").strip()
        for record in records
        if record.get("query_id") and str(record.get("document_path") or "").strip()
    }
    if len(paths) != 710:
        raise ValueError(f"Expected 710 public document paths, got {len(paths)}")
    return paths


def _ensure_sample(sample_file: Path) -> dict[str, Any]:
    public_ids, public_file_sha256 = _fetch_public_ids()
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    if sample_file.is_file():
        try:
            payload = json.loads(sample_file.read_text(encoding="utf-8"))
            if (
                payload.get("public_count") == len(public_ids)
                and payload.get("sample_count") == SAMPLE_COUNT
                and payload.get("seed") == SAMPLE_SEED
                and payload.get("public_ids_sha256") == public_file_sha256
                and len(payload.get("query_ids") or []) == SAMPLE_COUNT
            ):
                selected = [str(value) for value in payload["query_ids"]]
                if len(set(selected)) == SAMPLE_COUNT and set(selected).issubset(public_ids):
                    return payload
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    rng = random.Random(SAMPLE_SEED)
    selected = sorted(rng.sample(public_ids, SAMPLE_COUNT))
    domain_counts = {
        domain: sum(value.startswith(f"{domain}:") for value in selected)
        for domain in DOMAINS
    }
    payload = {
        "source_url": PUBLIC_TEST_URL,
        "model_card_url": PUBLIC_MODEL_CARD_URL,
        "public_ids_sha256": public_file_sha256,
        "public_count": len(public_ids),
        "sample_count": len(selected),
        "seed": SAMPLE_SEED,
        "sampling": "uniform_without_replacement_over_sorted_query_ids",
        "domain_counts": domain_counts,
        "query_ids": selected,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    sample_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def _resolve_sample_items(
    all_items: Sequence[dict[str, Any]],
    sample_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    by_public_id = {
        f"{item['subset']}:{int(item['subset_index']):04d}": item
        for item in all_items
    }
    selected_ids = [str(value) for value in sample_payload.get("query_ids") or []]
    missing = [value for value in selected_ids if value not in by_public_id]
    if missing:
        raise ValueError(f"Public sample IDs missing from archive: {missing[:5]}")
    return [by_public_id[value] for value in selected_ids]


def _settings() -> RagConfig:
    base = RagConfig.from_env().normalized()
    return base.__class__(
        **{
            **base.__dict__,
            "hybrid_candidate_k": 100,
            "hybrid_max_candidate_k": 200,
            "hybrid_top_k": 5,
            "rerank_top_k": 5,
            "simple_query_top_k": 5,
            "hybrid_rrf_k": 60,
            "query_complexity_threshold": 2.0,
            "rerank_fusion_weight": 0.30,
            "rerank_bm25_weight": 0.32,
            "rerank_embedding_weight": 0.26,
            "rerank_coverage_weight": 0.10,
            "rerank_phrase_weight": 0.02,
        }
    ).normalized()


def _source_document_for_item(item: dict[str, Any]) -> str:
    """Return the benchmark-provided source document for one question.

    LegalBench-RAG questions in the public test split have one source
    document.  This metadata is an explicit benchmark input, not inferred from
    the answer span during retrieval.
    """

    benchmark_path = str(item.get("benchmark_document_path") or "").strip()
    if benchmark_path:
        return benchmark_path
    paths = sorted(
        {
            str(snippet.get("file_path") or "").strip()
            for snippet in item.get("snippets") or []
            if str(snippet.get("file_path") or "").strip()
        }
    )
    if len(paths) != 1:
        raise ValueError(
            f"Expected exactly one source document for {item.get('id')}; got {paths}"
        )
    return paths[0]


def _allowed_indices_by_item(
    items: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
) -> list[list[int]]:
    by_file: dict[str, list[int]] = {}
    for index, chunk in enumerate(chunks):
        file_path = str(chunk.get("file_path") or "").strip()
        if file_path:
            by_file.setdefault(file_path, []).append(index)
    return [
        list(by_file.get(_source_document_for_item(item), []))
        for item in items
    ]


def _scoped_dense_hits(
    matrix: np.ndarray,
    query_matrix: np.ndarray,
    allowed_indices_by_item: Sequence[Sequence[int]],
    *,
    top_k: int = 100,
) -> tuple[list[list[dict[str, Any]]], float]:
    """Search dense vectors only within each question's source document."""

    started = time.perf_counter()
    results: list[list[dict[str, Any]]] = []
    for query_vector, allowed_indices in zip(
        query_matrix,
        allowed_indices_by_item,
        strict=True,
    ):
        indices = np.asarray(list(allowed_indices), dtype=np.int64)
        if len(indices) == 0:
            results.append([])
            continue
        scores = np.asarray(matrix[indices] @ query_vector, dtype=np.float32)
        limit = min(max(1, int(top_k)), len(indices))
        selected = np.argpartition(-scores, kth=limit - 1)[:limit]
        order = np.argsort(-scores[selected], kind="stable")
        results.append(
            [
                {
                    "index": int(indices[selected[position]]),
                    "score": float(scores[selected[position]]),
                }
                for position in order
            ]
        )
    return results, (time.perf_counter() - started) * 1000.0


def _format_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# LegalBench-RAG 公開 710 題抽樣 100 題：三架構檢索比較",
        "",
        "本次從公開 held-out 710 題清單固定 seed 42 抽出 100 題，三個版本共用相同題目，只執行證據檢索，不呼叫回答模型。",
        "",
        f"公開來源：[LegalBenchRAG-Ettin-150M-Reranker 模型卡]({PUBLIC_MODEL_CARD_URL})",
        f"- 文件範圍：`{payload.get('document_scope', 'global')}`。`specified` 代表使用測試資料提供的 `document_path`，只在該文件內容內檢索。",
        "- `specified` 是已知文件的 benchmark 評估；若要測試全庫自動找文件，請使用 `global` 並另外報告文件召回率。",
        f"\n- 公開測試題數：{payload['public_test_count']:,} 題；本次抽樣：{payload['sample_count']:,} 題。",
        f"- 題目抽樣：固定 seed {payload['sample_seed']}、無放回、依排序後 ID 均勻抽樣。",
        f"- 子題型分布：{json.dumps(payload['sample_domain_counts'], ensure_ascii=False)}。",
        f"- 完整 archive：{payload['archive_question_count']:,} 題、{payload['corpus_documents']:,} 份文件。",
        f"- Archive SHA-256：`{payload['archive_sha256']}`。",
        "",
        "## 三個版本",
        "",
        "| 版本 | 架構 |",
        "| --- | --- |",
        "| 原本無優化版 | 硬切 600 字元、BM25/向量 50/50 直接加權、既有重排 Top 5 |",
        "| 原本優化版 | 1024 token Parent / 256 token Child、RRF、複雜度路由、複雜題 Cross-Encoder、父段落展開 Top 5 |",
        "| 最新公開方法 | Ettin 384 token / 96 overlap、BM25 Top 32、LegalBenchRAG Ettin Cross-Encoder Top 5 |",
        "",
        "## 結果",
        "",
        "| 版本 | Macro char recall | Micro char recall | 任一證據命中 | 完整證據命中 | Rank 1 命中 | 平均 ms/題 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for version in payload["versions"]:
        summary = version["summary"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(version["name"]),
                    _format_pct(summary.get("macro_char_recall")),
                    _format_pct(summary.get("micro_char_recall")),
                    _format_pct(summary.get("any_gold_span_hit_rate")),
                    _format_pct(summary.get("complete_gold_span_hit_rate")),
                    _format_pct(summary.get("rank1_gold_span_hit_rate")),
                    f"{float(summary.get('average_question_stage_ms', 0.0)):.1f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 指標定義",
            "",
            "- 任一證據命中：Top 5 中至少涵蓋一個標註證據區段。",
            "- 完整證據命中：Top 5 涵蓋該題全部標註證據區段。",
            "- Rank 1 命中：第一個回傳段落涵蓋至少一個標註證據區段。",
            "- 字元召回率：Top 5 回傳段落覆蓋標註證據字元的比例；重疊段落只計一次。",
            "",
            "## 題目與逐題結果",
            "",
            "- 抽樣清單：[sample.json](sample.json)",
            "- 完整逐題結果：[retrieval-results.json](retrieval-results.json)",
            "- 摘要：[retrieval-summary.json](retrieval-summary.json)",
            "",
        ]
    )
    return "\n".join(lines)


def _summary_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: payload[key]
        for key in (
            "schema_version",
            "dataset",
            "public_test_count",
            "sample_count",
            "sample_seed",
            "sample_domain_counts",
            "archive_question_count",
            "archive_sha256",
            "corpus_documents",
            "corpus_characters",
            "chunk_counts",
            "passage_count",
            "mapping_failures",
            "embedding_model",
            "cross_encoder_model",
            "run_at",
            "document_scope",
        )
    } | {
        "versions": [
            {
                "name": version["name"],
                "version": version["version"],
                "configuration": version["configuration"],
                "summary": version["summary"],
                "breakdown": version["breakdown"],
            }
            for version in payload["versions"]
        ]
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    archive_path = args.archive.resolve()
    archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    all_items, corpus, archive_metadata = _load_all_archive(archive_path)
    sample_payload = _ensure_sample(args.sample_file.resolve())
    items = _resolve_sample_items(all_items, sample_payload)
    if len(items) != SAMPLE_COUNT:
        raise ValueError(f"Expected {SAMPLE_COUNT} resolved sample items, got {len(items)}")
    public_document_paths = _fetch_public_document_paths()
    for item in items:
        query_id = f"{item['subset']}:{int(item['subset_index']):04d}"
        benchmark_path = public_document_paths.get(query_id)
        if not benchmark_path:
            raise ValueError(f"Public document_path missing for {query_id}")
        item["benchmark_document_path"] = benchmark_path
    print(
        f"Loaded public sample {len(items)} / {sample_payload['public_count']} questions and "
        f"{len(corpus)} corpus documents",
        flush=True,
    )
    print(f"Sample domains: {sample_payload['domain_counts']}", flush=True)

    settings = _settings()
    started_build = time.perf_counter()
    hard_chunks = legacy_eval._build_hard_chunks(corpus)
    parent_child, mapping_failures = legacy_eval._build_parent_child_chunks(corpus)
    print(
        f"Built {len(hard_chunks)} hard chunks, {len(parent_child.parents)} parents, "
        f"{len(parent_child.children)} children; mapping_failures={mapping_failures}",
        flush=True,
    )
    build_ms = (time.perf_counter() - started_build) * 1000.0

    # The requested comparison uses the legal-domain embedding model.  The
    # corresponding cache was built by the earlier 100-question legal
    # embedding run; the older full-eval cache contains a different 384-dim
    # model and must not be mixed with the 768-dim query vectors.
    legacy_cache = ROOT / "evals" / "legalbench_rag_retrieval_100_legal_embedding" / "embedding-cache"
    embedding_model = str(args.embedding_model)
    hard_matrix, hard_embedding_ms, hard_cache_hit = legacy_eval._load_or_build_embeddings(
        hard_chunks,
        legacy_cache / "hard-embeddings.npy",
        model_name=embedding_model,
        batch_size=args.batch_size,
        label="hard",
    )
    child_matrix, child_embedding_ms, child_cache_hit = legacy_eval._load_or_build_embeddings(
        parent_child.children,
        legacy_cache / "child-embeddings.npy",
        model_name=embedding_model,
        batch_size=args.batch_size,
        label="child",
    )
    query_matrix, query_embedding_ms, query_cache_hit = legacy_eval._load_or_build_query_embeddings(
        [str(item.get("query") or "") for item in items],
        args.query_cache.resolve(),
        model_name=embedding_model,
        batch_size=args.batch_size,
    )

    document_scope = str(getattr(args, "document_scope", "global") or "global").strip().lower()
    if document_scope not in {"global", "specified"}:
        raise ValueError(f"Unsupported document scope: {document_scope}")
    hard_allowed_indices = None
    child_allowed_indices = None
    if document_scope == "specified":
        hard_allowed_indices = _allowed_indices_by_item(items, hard_chunks)
        child_allowed_indices = _allowed_indices_by_item(items, parent_child.children)
        print(
            "Document-scoped mode: each question is restricted to its benchmark "
            "document_path before BM25, dense search, and reranking",
            flush=True,
        )

    hard_bm25 = legacy_eval.FastBm25Index(
        hard_chunks,
        k1=settings.hybrid_bm25_k1,
        b=settings.hybrid_bm25_b,
    )
    child_bm25 = legacy_eval.FastBm25Index(
        parent_child.children,
        k1=settings.hybrid_bm25_k1,
        b=settings.hybrid_bm25_b,
    )
    hard_token_sets = [
        set(legacy_eval.tokenize_bm25(str(chunk.get("content") or "")))
        for chunk in hard_chunks
    ]
    print("Computing dense Top 100 candidates for the hard branch", flush=True)
    if document_scope == "specified":
        hard_dense_hits, hard_dense_ms = _scoped_dense_hits(
            hard_matrix,
            query_matrix,
            hard_allowed_indices or [],
            top_k=100,
        )
    else:
        hard_dense_hits, hard_dense_ms = legacy_eval._dense_hits_all(
            hard_matrix,
            query_matrix,
            top_k=100,
            batch_size=args.dense_batch_size,
        )
    print("Computing dense Top 100 candidates for the child branch", flush=True)
    if document_scope == "specified":
        child_dense_hits, child_dense_ms = _scoped_dense_hits(
            child_matrix,
            query_matrix,
            child_allowed_indices or [],
            top_k=100,
        )
    else:
        child_dense_hits, child_dense_ms = legacy_eval._dense_hits_all(
            child_matrix,
            query_matrix,
            top_k=100,
            batch_size=args.dense_batch_size,
        )

    os.environ["RAG_CROSS_ENCODER_BATCH_SIZE"] = str(max(1, int(args.cross_encoder_batch_size)))
    cross_encoder = CrossEncoderReranker(args.cross_encoder_model)
    checkpoint_dir = args.checkpoint_dir.resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    legacy_unoptimized = legacy_eval._run_version(
        version="legacy_unoptimized",
        label="原本無優化版",
        items=items,
        chunks=hard_chunks,
        dense_hits=hard_dense_hits,
        bm25=hard_bm25,
        settings=settings,
        optimized=False,
        chunk_token_sets=hard_token_sets,
        dense_search_ms=hard_dense_ms,
        checkpoint_path=checkpoint_dir / "legacy-unoptimized.json",
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        allowed_indices_by_item=hard_allowed_indices,
    )
    legacy_optimized = legacy_eval._run_version(
        version="legacy_optimized",
        label="原本優化版",
        items=items,
        chunks=parent_child.children,
        dense_hits=child_dense_hits,
        bm25=child_bm25,
        settings=settings,
        optimized=True,
        parents=parent_child.parents,
        cross_encoder=cross_encoder,
        dense_search_ms=child_dense_ms,
        checkpoint_path=checkpoint_dir / "legacy-optimized.json",
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        allowed_indices_by_item=child_allowed_indices,
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.cross_encoder_model, use_fast=True)
    passage_started = time.perf_counter()
    public_passages, public_mapping_failures = public_eval._build_passages(
        corpus,
        tokenizer=tokenizer,
    )
    public_build_ms = (time.perf_counter() - passage_started) * 1000.0
    print(
        f"Built {len(public_passages)} public passages (384/96), "
        f"mapping_failures={public_mapping_failures}",
        flush=True,
    )
    public_bm25 = legacy_eval.FastBm25Index(public_passages)
    public_allowed_indices = (
        _allowed_indices_by_item(items, public_passages)
        if document_scope == "specified"
        else None
    )
    public_optimized = public_eval._run_version(
        version="public_ettin",
        label="最新公開方法（Ettin）",
        items=items,
        passages=public_passages,
        bm25=public_bm25,
        reranker=cross_encoder,
        checkpoint_path=checkpoint_dir / "public-ettin.json",
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        allowed_indices_by_item=public_allowed_indices,
    )

    model = getattr(cross_encoder, "_model", None)
    reranker_device = str(getattr(model, "device", None) or "unknown") if model is not None else "unknown"
    return {
        "schema_version": "legalbench-rag-public710-sample100-three-way-v1",
        "dataset": "ZeroEntropy-AI/legalbenchrag",
        "public_test_source_url": PUBLIC_TEST_URL,
        "public_test_model_card_url": PUBLIC_MODEL_CARD_URL,
        "public_test_count": int(sample_payload["public_count"]),
        "sample_count": len(items),
        "sample_seed": int(sample_payload["seed"]),
        "sample_sampling": sample_payload["sampling"],
        "sample_domain_counts": sample_payload["domain_counts"],
        "sample_query_ids": [str(item["subset"]) + ":" + f"{int(item['subset_index']):04d}" for item in items],
        "archive_question_count": len(all_items),
        "archive_sha256": archive_sha256,
        "archive_evaluated_question_count_paper_aligned": sum(legacy_eval.PUBLISHED_COUNTS.values()),
        "corpus_documents": len(corpus),
        "corpus_characters": sum(len(text) for text in corpus.values()),
        "chunk_counts": {
            "hard": len(hard_chunks),
            "parent": len(parent_child.parents),
            "child": len(parent_child.children),
        },
        "passage_count": len(public_passages),
        "mapping_failures": mapping_failures,
        "public_mapping_failures": public_mapping_failures,
        "build_ms": round(build_ms, 3),
        "public_passage_build_ms": round(public_build_ms, 3),
        "embedding_ms": {
            "hard": round(hard_embedding_ms, 3),
            "child": round(child_embedding_ms, 3),
            "queries": round(query_embedding_ms, 3),
        },
        "embedding_cache_hit": {
            "hard": hard_cache_hit,
            "child": child_cache_hit,
            "queries": query_cache_hit,
        },
        "embedding_model": embedding_model,
        "cross_encoder_model": args.cross_encoder_model,
        "reranker_device": reranker_device,
        "batch_size": int(args.batch_size),
        "dense_batch_size": int(args.dense_batch_size),
        "cross_encoder_batch_size": int(args.cross_encoder_batch_size),
        "retrieval_only": True,
        "model_calls": 0,
        "document_scope": document_scope,
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "versions": [legacy_unoptimized, legacy_optimized, public_optimized],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--sample-file", type=Path, default=DEFAULT_SAMPLE_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--query-cache", type=Path, default=DEFAULT_QUERY_CACHE)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dense-batch-size", type=int, default=64)
    parser.add_argument("--cross-encoder-batch-size", type=int, default=64)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--cross-encoder-model", default=DEFAULT_CROSS_ENCODER_MODEL)
    parser.add_argument(
        "--document-scope",
        choices=("global", "specified"),
        default="global",
        help=(
            "Search all corpus documents (global) or restrict every question "
            "to its benchmark-provided document_path (specified)."
        ),
    )
    args = parser.parse_args()

    payload = run(args)
    output_path = args.output.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    summary_path = args.summary.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(_render_report(payload), encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "report": str(report_path),
                "summary": str(summary_path),
                "sample_count": payload["sample_count"],
                "versions": {version["version"]: version["summary"] for version in payload["versions"]},
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
