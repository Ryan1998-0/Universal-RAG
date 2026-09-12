import argparse
import json
import os
import re
from pathlib import Path
from time import perf_counter

from rag_demo.action_result_resolution import answer_action_result
from rag_demo.chunking import load_knowledge_base_chunks
from rag_demo.config import RagConfig
from rag_demo.cross_encoder import CrossEncoderReranker
from rag_demo.embeddings import embed_query, load_embedding_matrix
from rag_demo.entity_aliases import expand_query_with_aliases
from rag_demo.entity_resolution import answer_entity_existence
from rag_demo.event_list_retrieval import find_event_list_chunks, find_result_constrained_chunks, merge_event_list_chunks
from rag_demo.evidence_policy import build_evidence_policy
from rag_demo.evidence_extraction_agent import extract_evidence
from rag_demo.general_answer import build_no_retrieval_answer_prompt, build_qwen_rag_system_prompt
from rag_demo.graph_retrieval import retrieve_graph_context
from rag_demo.index_store import load_index
from rag_demo.knowledge_base import active_knowledge_base
from rag_demo.keyword_extraction import extract_keywords
from rag_demo.lambdamart_fusion import fuse_candidates_with_lambdamart
from rag_demo.model_providers import ask_model
from rag_demo.prompting import build_answer_prompt
from rag_demo.qa_agent import answer_with_qa_agent
from rag_demo.question_extraction import extract_real_question
from rag_demo.rag_pipeline import RagPipeline, RagPipelineRequest
from rag_demo.query_rewriter import QueryRewriteDecision, decide_and_rewrite_query_for_retrieval
from rag_demo.query_complexity import classify_query_complexity
from rag_demo.parent_child import expand_child_contexts
from rag_demo.retrieval import embedding_search, hybrid_search, keyword_search
from rag_demo.self_rag_reflection import (
    AnswerSupportCritique,
    critique_answer_support,
    critique_passages,
    score_answer_utility,
    utility_from_answer_support_critique,
)
from rag_demo.vector_store import load_or_build_qdrant_vector_store


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def answer_question(question: str, model: str, top_k: int = 5) -> str:
    total_started_at = perf_counter()
    timing = {}
    config = RagConfig.from_env()
    top_k = top_k or config.top_k
    knowledge_base = active_knowledge_base(project_root=PROJECT_ROOT)
    kb_dir = knowledge_base.raw_dir
    index_dir = knowledge_base.index_dir
    index_path = index_dir / "chunks.json"
    started_at = perf_counter()
    if index_path.exists():
        chunks = load_index(index_path)
    else:
        chunks = load_knowledge_base_chunks(kb_dir)
    timing["load_index"] = perf_counter() - started_at

    started_at = perf_counter()
    keywords = extract_keywords(
        question,
        model=model,
    )
    timing["keyword_extraction_agent"] = perf_counter() - started_at

    started_at = perf_counter()
    refined_question = extract_real_question(
        question,
        model=model,
    )
    timing["question_extraction_agent"] = perf_counter() - started_at

    query_variants = _build_three_agent_query_variants(question, refined_question, keywords)
    embedding_path = index_dir / "embeddings.npy"
    embeddings = None
    vector_store = None
    if embedding_path.exists():
        started_at = perf_counter()
        try:
            embeddings = load_embedding_matrix(embedding_path)
            timing["load_embeddings"] = perf_counter() - started_at
            started_at = perf_counter()
            try:
                vector_store = load_or_build_qdrant_vector_store(index_dir=index_dir, chunks=chunks, embeddings=embeddings)
            except Exception:
                vector_store = None
            timing["load_vector_db"] = perf_counter() - started_at
        except Exception:
            timing["load_embeddings"] = perf_counter() - started_at

    started_at = perf_counter()
    results = _hybrid_rerank_results(
        query_variants=query_variants,
        question=question,
        refined_question=refined_question,
        keywords=keywords,
        chunks=chunks,
        embeddings=embeddings,
        vector_store=vector_store,
        top_k=top_k,
        candidate_k=config.retrieval_candidate_k,
        settings=config,
    )
    timing["retrieval"] = perf_counter() - started_at

    started_at = perf_counter()
    extracted_evidence = extract_evidence(
        original_question=question,
        refined_question=refined_question,
        keywords=keywords,
        chunks=results,
        model=model,
    )
    timing["evidence_extraction_agent"] = perf_counter() - started_at

    started_at = perf_counter()
    answer = answer_with_qa_agent(
        original_question=question,
        refined_question=refined_question,
        keywords=keywords,
        chunks=results,
        extracted_evidence=extracted_evidence,
        model=model,
    )
    timing["qa_agent"] = perf_counter() - started_at
    timing["total"] = perf_counter() - total_started_at
    return _format_three_agent_output(
        question,
        query_variants,
        keywords,
        refined_question,
        results,
        extracted_evidence,
        answer,
        timing=timing,
    )


def answer_question_sparse_dense_refined_keywords(question: str, model: str, top_k: int = 5) -> str:
    total_started_at = perf_counter()
    timing = {}
    config = RagConfig.from_env()
    top_k = top_k or config.top_k
    knowledge_base = active_knowledge_base(project_root=PROJECT_ROOT)
    kb_dir = knowledge_base.raw_dir
    index_dir = knowledge_base.index_dir
    index_path = index_dir / "chunks.json"
    started_at = perf_counter()
    if index_path.exists():
        chunks = load_index(index_path)
    else:
        chunks = load_knowledge_base_chunks(kb_dir)
    timing["load_index"] = perf_counter() - started_at

    started_at = perf_counter()
    refined_question = extract_real_question(
        question,
        model=model,
    )
    timing["question_extraction_agent"] = perf_counter() - started_at

    started_at = perf_counter()
    keywords = extract_keywords(
        refined_question,
        model=model,
    )
    timing["keyword_extraction_agent"] = perf_counter() - started_at

    query_variants = _build_sparse_dense_refined_query_variants(question, refined_question, keywords)
    embedding_path = index_dir / "embeddings.npy"
    embeddings = None
    vector_store = None
    if embedding_path.exists():
        started_at = perf_counter()
        try:
            embeddings = load_embedding_matrix(embedding_path)
            timing["load_embeddings"] = perf_counter() - started_at
            started_at = perf_counter()
            try:
                vector_store = load_or_build_qdrant_vector_store(index_dir=index_dir, chunks=chunks, embeddings=embeddings)
            except Exception:
                vector_store = None
            timing["load_vector_db"] = perf_counter() - started_at
        except Exception:
            timing["load_embeddings"] = perf_counter() - started_at

    started_at = perf_counter()
    results = _sparse_dense_refined_results(
        query_variants=query_variants,
        question=question,
        refined_question=refined_question,
        keywords=keywords,
        chunks=chunks,
        embeddings=embeddings,
        vector_store=vector_store,
        top_k=top_k,
        candidate_k=config.retrieval_candidate_k,
        settings=config,
    )
    timing["sparse_dense_retrieval"] = perf_counter() - started_at

    started_at = perf_counter()
    extracted_evidence = extract_evidence(
        original_question=question,
        refined_question=refined_question,
        keywords=keywords,
        chunks=results,
        model=model,
    )
    timing["evidence_extraction_agent"] = perf_counter() - started_at

    started_at = perf_counter()
    answer = answer_with_qa_agent(
        original_question=question,
        refined_question=refined_question,
        keywords=keywords,
        chunks=results,
        extracted_evidence=extracted_evidence,
        model=model,
    )
    timing["qa_agent"] = perf_counter() - started_at
    timing["total"] = perf_counter() - total_started_at
    return _format_three_agent_output(
        question,
        query_variants,
        keywords,
        refined_question,
        results,
        extracted_evidence,
        answer,
        timing=timing,
    )


def answer_question_sparse_dense_original_refined_keywords(question: str, model: str, top_k: int = 5) -> str:
    total_started_at = perf_counter()
    timing = {}
    config = RagConfig.from_env()
    top_k = top_k or config.top_k
    knowledge_base = active_knowledge_base(project_root=PROJECT_ROOT)
    kb_dir = knowledge_base.raw_dir
    index_dir = knowledge_base.index_dir
    index_path = index_dir / "chunks.json"
    started_at = perf_counter()
    if index_path.exists():
        chunks = load_index(index_path)
    else:
        chunks = load_knowledge_base_chunks(kb_dir)
    timing["load_index"] = perf_counter() - started_at

    started_at = perf_counter()
    refined_question = extract_real_question(
        question,
        model=model,
    )
    timing["question_extraction_agent"] = perf_counter() - started_at

    started_at = perf_counter()
    original_keywords = extract_keywords(
        question,
        model=model,
    )
    timing["original_keyword_extraction_agent"] = perf_counter() - started_at

    started_at = perf_counter()
    refined_keywords = extract_keywords(
        refined_question,
        model=model,
    )
    timing["refined_keyword_extraction_agent"] = perf_counter() - started_at
    keywords = _merge_keywords(original_keywords, refined_keywords)

    query_variants = _build_sparse_dense_original_refined_query_variants(
        original_question=question,
        refined_question=refined_question,
        original_keywords=original_keywords,
        refined_keywords=refined_keywords,
    )
    embedding_path = index_dir / "embeddings.npy"
    embeddings = None
    vector_store = None
    if embedding_path.exists():
        started_at = perf_counter()
        try:
            embeddings = load_embedding_matrix(embedding_path)
            timing["load_embeddings"] = perf_counter() - started_at
            started_at = perf_counter()
            try:
                vector_store = load_or_build_qdrant_vector_store(index_dir=index_dir, chunks=chunks, embeddings=embeddings)
            except Exception:
                vector_store = None
            timing["load_vector_db"] = perf_counter() - started_at
        except Exception:
            timing["load_embeddings"] = perf_counter() - started_at

    started_at = perf_counter()
    results = _sparse_dense_refined_results(
        query_variants=query_variants,
        question=question,
        refined_question=refined_question,
        keywords=keywords,
        chunks=chunks,
        embeddings=embeddings,
        vector_store=vector_store,
        top_k=top_k,
        candidate_k=config.retrieval_candidate_k,
        settings=config,
    )
    timing["sparse_dense_retrieval"] = perf_counter() - started_at

    started_at = perf_counter()
    extracted_evidence = extract_evidence(
        original_question=question,
        refined_question=refined_question,
        keywords=keywords,
        chunks=results,
        model=model,
    )
    timing["evidence_extraction_agent"] = perf_counter() - started_at

    started_at = perf_counter()
    answer = answer_with_qa_agent(
        original_question=question,
        refined_question=refined_question,
        keywords=keywords,
        chunks=results,
        extracted_evidence=extracted_evidence,
        model=model,
    )
    timing["qa_agent"] = perf_counter() - started_at
    timing["total"] = perf_counter() - total_started_at
    return _format_three_agent_output(
        question,
        query_variants,
        keywords,
        refined_question,
        results,
        extracted_evidence,
        answer,
        timing=timing,
    )


def answer_question_v2(question: str, model: str, top_k: int = 8) -> str:
    total_started_at = perf_counter()
    timing = {}
    config = RagConfig.from_env()
    final_context_k = max(5, min(8, int(top_k or 8)))
    rerank_top_k = final_context_k
    knowledge_base = active_knowledge_base(project_root=PROJECT_ROOT)
    kb_dir = knowledge_base.raw_dir
    index_dir = knowledge_base.index_dir
    index_path = index_dir / "chunks.json"

    started_at = perf_counter()
    if index_path.exists():
        chunks = load_index(index_path)
    else:
        chunks = load_knowledge_base_chunks(kb_dir)
    timing["load_index"] = perf_counter() - started_at

    started_at = perf_counter()
    try:
        rewrite_decision = decide_and_rewrite_query_for_retrieval(
            question,
            model=model,
        )
    except Exception:
        rewrite_decision = QueryRewriteDecision(
            needs_retrieval=True,
            reason="Query Rewrite / Retrieval Decision 失敗，保守改走 RAG retrieval。",
            retrieval_query=question,
        )
    rewritten_query = rewrite_decision.retrieval_query
    timing["query_rewrite"] = perf_counter() - started_at

    if not rewrite_decision.needs_retrieval:
        started_at = perf_counter()
        try:
            answer = ask_model(
                build_no_retrieval_answer_prompt(question, rewrite_decision.reason),
                model=model,
                system=build_qwen_rag_system_prompt(),
            )
        except Exception:
            answer = (
                "這個問題被判斷不需要查詢 knowledge base，因此已跳過 RAG retrieval。"
                "目前無法呼叫一般回答模型，請稍後再試。"
            )
        timing["general_fallback"] = perf_counter() - started_at
        timing["total"] = perf_counter() - total_started_at
        return _format_v2_no_retrieval_output(
            question=question,
            rewrite_decision=rewrite_decision,
            answer=answer,
            timing=timing,
        )

    embedding_path = index_dir / "embeddings.npy"
    embeddings = None
    vector_store = None
    if embedding_path.exists():
        started_at = perf_counter()
        try:
            embeddings = load_embedding_matrix(embedding_path)
            timing["load_embeddings"] = perf_counter() - started_at
            started_at = perf_counter()
            try:
                vector_store = load_or_build_qdrant_vector_store(index_dir=index_dir, chunks=chunks, embeddings=embeddings)
            except Exception:
                vector_store = None
            timing["load_vector_db"] = perf_counter() - started_at
        except Exception:
            timing["load_embeddings"] = perf_counter() - started_at

    started_at = perf_counter()
    results = _rrf_parent_context_results(
        question=question,
        rewritten_query=rewritten_query,
        chunks=chunks,
        embeddings=embeddings,
        top_k=rerank_top_k,
        candidate_k=config.retrieval_candidate_k,
        final_context_k=final_context_k,
        vector_store=vector_store,
    )
    timing["retrieval"] = perf_counter() - started_at

    started_at = perf_counter()
    answer = answer_with_qa_agent(
        original_question=question,
        refined_question=question,
        keywords=[],
        chunks=results,
        model=model,
    )
    timing["qa_agent"] = perf_counter() - started_at
    timing["total"] = perf_counter() - total_started_at
    return _format_v2_output(
        question=question,
        rewritten_query=rewritten_query,
        results=results,
        answer=answer,
        timing=timing,
        vector_db_enabled=vector_store is not None,
        rewrite_decision=rewrite_decision,
    )


def answer_question_self_rag(question: str, model: str, top_k: int = 8, max_attempts: int = 2) -> str:
    total_started_at = perf_counter()
    timing = {}
    config = RagConfig.from_env()
    final_context_k = max(5, min(8, int(top_k or 8)))
    rerank_top_k = final_context_k
    max_attempts = max(1, min(3, int(max_attempts or 2)))
    knowledge_base = active_knowledge_base(project_root=PROJECT_ROOT)
    kb_dir = knowledge_base.raw_dir
    index_dir = knowledge_base.index_dir
    index_path = index_dir / "chunks.json"

    started_at = perf_counter()
    if index_path.exists():
        chunks = load_index(index_path)
    else:
        chunks = load_knowledge_base_chunks(kb_dir)
    timing["load_index"] = perf_counter() - started_at

    started_at = perf_counter()
    try:
        rewrite_decision = decide_and_rewrite_query_for_retrieval(
            question,
            model=model,
        )
    except Exception:
        rewrite_decision = QueryRewriteDecision(
            needs_retrieval=True,
            reason="Query Rewrite / Retrieval Decision 失敗，保守改走 RAG retrieval。",
            retrieval_query=question,
        )
    timing["query_rewrite"] = perf_counter() - started_at

    if not rewrite_decision.needs_retrieval:
        started_at = perf_counter()
        try:
            answer = ask_model(
                build_no_retrieval_answer_prompt(question, rewrite_decision.reason),
                model=model,
                system=build_qwen_rag_system_prompt(),
            )
        except Exception:
            answer = (
                "這個問題被判斷不需要查詢 knowledge base，因此已跳過 RAG retrieval。"
                "目前無法呼叫一般回答模型，請稍後再試。"
            )
        timing["general_fallback"] = perf_counter() - started_at
        timing["total"] = perf_counter() - total_started_at
        return _format_self_rag_no_retrieval_output(
            question=question,
            rewrite_decision=rewrite_decision,
            answer=answer,
            timing=timing,
        )

    embedding_path = index_dir / "embeddings.npy"
    embeddings = None
    vector_store = None
    if embedding_path.exists():
        started_at = perf_counter()
        try:
            embeddings = load_embedding_matrix(embedding_path)
            timing["load_embeddings"] = perf_counter() - started_at
            started_at = perf_counter()
            try:
                vector_store = load_or_build_qdrant_vector_store(index_dir=index_dir, chunks=chunks, embeddings=embeddings)
            except Exception:
                vector_store = None
            timing["load_vector_db"] = perf_counter() - started_at
        except Exception:
            timing["load_embeddings"] = perf_counter() - started_at

    attempts = []
    current_query = rewrite_decision.retrieval_query
    final_answer = ""
    for attempt_index in range(1, max_attempts + 1):
        started_at = perf_counter()
        results = _rrf_parent_context_results(
            question=question,
            rewritten_query=current_query,
            chunks=chunks,
            embeddings=embeddings,
            top_k=rerank_top_k,
            candidate_k=config.retrieval_candidate_k,
            final_context_k=final_context_k,
            vector_store=vector_store,
        )
        timing[f"retrieval_attempt_{attempt_index}"] = perf_counter() - started_at

        started_at = perf_counter()
        passage_reflections = critique_passages(
            question=question,
            chunks=results,
            model=model,
        )
        timing[f"passage_critic_attempt_{attempt_index}"] = perf_counter() - started_at

        started_at = perf_counter()
        answer = answer_with_qa_agent(
            original_question=question,
            refined_question=question,
            keywords=[],
            chunks=results,
            model=model,
        )
        timing[f"qa_agent_attempt_{attempt_index}"] = perf_counter() - started_at
        final_answer = answer

        started_at = perf_counter()
        answer_critique = critique_answer_support(
            question=question,
            answer=answer,
            chunks=results,
            passage_reflections=passage_reflections,
            model=model,
        )
        answer_critique = _self_rag_enforce_passage_support(
            question=question,
            answer_critique=answer_critique,
            passage_reflections=passage_reflections,
        )
        timing[f"answer_critic_attempt_{attempt_index}"] = perf_counter() - started_at

        started_at = perf_counter()
        utility_critique = utility_from_answer_support_critique(answer_critique)
        timing[f"utility_from_combined_critic_attempt_{attempt_index}"] = perf_counter() - started_at

        should_retry = _self_rag_should_retry(
            attempt_index=attempt_index,
            max_attempts=max_attempts,
            passage_reflections=passage_reflections,
            answer_critique=answer_critique,
            utility_critique=utility_critique,
        )
        retry_query = _self_rag_retry_query(
            question=question,
            current_query=current_query,
            answer_critique=answer_critique,
            utility_critique=utility_critique,
        )
        attempts.append(
            {
                "attempt": attempt_index,
                "query": current_query,
                "results": results,
                "passage_reflections": passage_reflections,
                "answer": answer,
                "answer_critique": answer_critique,
                "utility_critique": utility_critique,
                "retry_needed": should_retry,
                "retry_query": retry_query if should_retry else "",
            }
        )
        if not should_retry:
            break
        current_query = retry_query

    timing["total"] = perf_counter() - total_started_at
    return _format_self_rag_output(
        question=question,
        rewrite_decision=rewrite_decision,
        attempts=attempts,
        answer=final_answer,
        timing=timing,
        vector_db_enabled=vector_store is not None,
        fine_tuned_self_rag=False,
    )


def _build_three_agent_query_variants(question: str, refined_question: str, keywords):
    keyword_query = " ".join(str(keyword).strip() for keyword in keywords if str(keyword).strip())
    original_query = " ".join(str(question).split())
    question_agent_query = " ".join(str(refined_question).split())
    variants = [
        {"name": "original", "query": original_query, "weight": 1.00},
        {"name": "question_agent", "query": question_agent_query, "weight": 1.10},
        {"name": "keywords", "query": keyword_query, "weight": 1.00},
        {"name": "keyword_only", "query": keyword_query, "weight": 0.85},
    ]

    unique = []
    seen = set()
    for variant in variants:
        query = str(variant["query"]).strip()
        key = (variant["name"], query)
        if not query or key in seen:
            continue
        seen.add(key)
        unique.append({**variant, "query": query})
    return unique


def _merge_keywords(*keyword_groups):
    merged = []
    seen = set()
    for keywords in keyword_groups:
        for keyword in keywords:
            cleaned = str(keyword).strip()
            key = _compact_rerank_text(cleaned)
            if cleaned and key not in seen:
                seen.add(key)
                merged.append(cleaned)
    return merged


def _build_sparse_dense_refined_query_variants(question: str, refined_question: str, keywords):
    keyword_query = " ".join(str(keyword).strip() for keyword in keywords if str(keyword).strip())
    refined_query = " ".join(str(refined_question).split())
    variants = [
        {"name": "sparse_refined", "query": refined_query, "retrieval": "sparse", "weight": 1.00},
        {"name": "sparse_keywords", "query": keyword_query, "retrieval": "sparse", "weight": 1.10},
        {"name": "dense_refined", "query": refined_query, "retrieval": "dense", "weight": 1.00},
        {"name": "dense_keywords", "query": keyword_query, "retrieval": "dense", "weight": 0.90},
    ]

    unique = []
    seen = set()
    for variant in variants:
        query = str(variant["query"]).strip()
        key = (variant["name"], query, variant["retrieval"])
        if not query or key in seen:
            continue
        seen.add(key)
        unique.append({**variant, "query": query})
    return unique


def _build_sparse_dense_original_refined_query_variants(
    original_question: str,
    refined_question: str,
    original_keywords,
    refined_keywords,
):
    original_keyword_query = " ".join(str(keyword).strip() for keyword in original_keywords if str(keyword).strip())
    refined_keyword_query = " ".join(str(keyword).strip() for keyword in refined_keywords if str(keyword).strip())
    all_keyword_query = " ".join(_merge_keywords(original_keywords, refined_keywords))
    refined_query = " ".join(str(refined_question).split())
    variants = [
        {"name": "sparse_refined", "query": refined_query, "retrieval": "sparse", "weight": 1.00},
        {"name": "sparse_original_keywords", "query": original_keyword_query, "retrieval": "sparse", "weight": 1.00},
        {"name": "sparse_refined_keywords", "query": refined_keyword_query, "retrieval": "sparse", "weight": 1.05},
        {"name": "sparse_all_keywords", "query": all_keyword_query, "retrieval": "sparse", "weight": 1.15},
        {"name": "dense_refined", "query": refined_query, "retrieval": "dense", "weight": 1.00},
        {"name": "dense_all_keywords", "query": all_keyword_query, "retrieval": "dense", "weight": 0.90},
    ]

    unique = []
    seen = set()
    for variant in variants:
        query = str(variant["query"]).strip()
        key = (variant["name"], query, variant["retrieval"])
        if not query or key in seen:
            continue
        seen.add(key)
        unique.append({**variant, "query": query})
    return unique


def _sparse_dense_refined_results(
    query_variants,
    question: str,
    refined_question: str,
    keywords,
    chunks,
    embeddings,
    top_k: int,
    candidate_k: int,
    vector_store=None,
    settings: RagConfig | None = None,
):
    settings = (settings or RagConfig.from_env()).normalized()
    candidates = {}
    candidate_k = max(top_k, min(100, int(candidate_k or 100)))

    for variant in query_variants:
        query_text = variant["query"]
        query_weight = float(variant.get("weight", 1.0))
        retrieval = str(variant.get("retrieval", "sparse"))

        if retrieval == "sparse":
            keyword_results = keyword_search(query_text, chunks, top_k=candidate_k)
            max_keyword_score = max((float(item.get("score", 0.0)) for item in keyword_results), default=1.0)
            for rank, chunk in enumerate(keyword_results, start=1):
                normalized_score = float(chunk.get("score", 0.0)) / max_keyword_score if max_keyword_score else 0.0
                _add_hybrid_candidate_score(
                    candidates,
                    chunk,
                    score=query_weight * (0.62 * normalized_score + _reciprocal_rank_score(rank, 0.12)),
                    method=f"sparse:{variant['name']}:{rank}",
                    keyword_score=float(chunk.get("score", 0.0)),
                )
            continue

        if retrieval == "dense" and (vector_store is not None or embeddings is not None):
            embedding_results = _dense_search_results(
                query_text,
                chunks=chunks,
                embeddings=embeddings,
                vector_store=vector_store,
                top_k=candidate_k,
            )
            embedding_scores = [float(item.get("embedding_score", 0.0)) for item in embedding_results]
            min_embedding = min(embedding_scores, default=0.0)
            max_embedding = max(embedding_scores, default=1.0)
            span = max(max_embedding - min_embedding, 1e-9)
            for rank, chunk in enumerate(embedding_results, start=1):
                raw_embedding = float(chunk.get("embedding_score", 0.0))
                normalized_score = (raw_embedding - min_embedding) / span
                _add_hybrid_candidate_score(
                    candidates,
                    chunk,
                    score=query_weight * (0.38 * normalized_score + _reciprocal_rank_score(rank, 0.08)),
                    method=f"dense:{variant['name']}:{rank}",
                    embedding_score=raw_embedding,
                )

    fused = fuse_candidates_with_lambdamart(
        list(candidates.values()),
        keyword_weight=settings.keyword_weight,
        embedding_weight=settings.embedding_weight,
    )
    candidate_map = {str(item["id"]): item for item in fused}
    for item in fused:
        item["score"] = float(item.get("lambda_mart_fusion_score", 0.0))

    complexity = classify_query_complexity(
        question,
        query_variants=[variant.get("query", "") for variant in query_variants],
        threshold=settings.query_complexity_threshold,
    )
    if settings.complexity_routing_enabled and not complexity.is_complex:
        ranked = sorted(
            fused,
            key=lambda item: float(item.get("lambda_mart_fusion_score", 0.0)),
            reverse=True,
        )
        selected = ranked[: min(top_k, settings.simple_query_top_k)]
    else:
        _apply_rerank_lexical_coverage_boost(
            candidate_map,
            question,
            refined_question,
            keywords,
        )
        ranked = sorted(candidate_map.values(), key=lambda item: item["score"], reverse=True)
        selected = ranked[:top_k]
    return [_strip_rerank_candidate_metadata(item) for item in selected]


def _rrf_parent_context_results(
    question: str,
    rewritten_query: str,
    chunks,
    embeddings,
    top_k: int,
    candidate_k: int,
    final_context_k: int = 8,
    graph=None,
    vector_store=None,
):
    final_context_k = max(1, min(8, int(final_context_k or 8)))
    query = _bm25_dense_retrieval_query(question, rewritten_query)
    metadata_query = _effective_rrf_retrieval_query(question, rewritten_query)
    filtered_chunks, filtered_embeddings = _metadata_filter_chunk_embedding_pairs(metadata_query, chunks, embeddings)
    candidate_k = max(top_k, int(candidate_k or top_k))

    bm25_results = keyword_search(query, filtered_chunks, top_k=candidate_k)
    ranked_lists = [("bm25", bm25_results)]
    definition_query = _definition_route_query(question)

    if vector_store is not None or filtered_embeddings is not None:
        allowed_chunk_ids = None
        if len(filtered_chunks) < len(chunks):
            allowed_chunk_ids = {chunk["id"] for chunk in filtered_chunks}
        dense_results = _dense_search_results(
            query,
            chunks=filtered_chunks,
            embeddings=filtered_embeddings,
            vector_store=vector_store,
            top_k=candidate_k,
            allowed_chunk_ids=allowed_chunk_ids,
        )
        ranked_lists.append(("qdrant_dense" if vector_store is not None else "dense", dense_results))

    if definition_query:
        definition_bm25_results = _definition_search_results(definition_query, filtered_chunks, top_k=candidate_k)
        ranked_lists.append(("definition_bm25", definition_bm25_results))
        if vector_store is not None or filtered_embeddings is not None:
            allowed_chunk_ids = None
            if len(filtered_chunks) < len(chunks):
                allowed_chunk_ids = {chunk["id"] for chunk in filtered_chunks}
            definition_dense_results = _dense_search_results(
                definition_query,
                chunks=filtered_chunks,
                embeddings=filtered_embeddings,
                vector_store=vector_store,
                top_k=candidate_k,
                allowed_chunk_ids=allowed_chunk_ids,
            )
            ranked_lists.append(
                (
                    "definition_qdrant_dense" if vector_store is not None else "definition_dense",
                    definition_dense_results,
                )
            )

    merged = _rrf_merge_ranked_results(ranked_lists, top_k=min(100, candidate_k))
    graph_results = retrieve_graph_context(
        question,
        chunks,
        graph=graph,
        project_root=PROJECT_ROOT,
        max_results=final_context_k,
    )
    if graph_results:
        merged = _merge_graph_and_vector_candidates(graph_results, merged, top_k=candidate_k)
    _apply_definition_route_boost(merged, definition_query)
    complexity = classify_query_complexity(question, threshold=RagConfig.from_env().query_complexity_threshold)
    if complexity.is_complex:
        cross_encoder = CrossEncoderReranker()
        documents = [str(item.get("content") or "") for item in merged]
        scores = cross_encoder.score(question, documents)
        for item, score in zip(merged, scores):
            item["score"] = float(score)
            item["rerank_score"] = float(score)
        reranked = sorted(
            merged,
            key=lambda item: (float(item.get("rerank_score", 0.0)), float(item.get("rrf_score", 0.0))),
            reverse=True,
        )[:5]
    else:
        _rerank_rrf_candidates(merged, question=question, rewritten_query=" ".join([query, definition_query]))
        reranked = sorted(merged, key=lambda item: item["score"], reverse=True)[:5]
    if any(str(chunk.get("chunk_level") or "").lower() == "child" for chunk in chunks):
        expanded = expand_child_contexts(reranked, limit=min(5, final_context_k))
    else:
        expanded = _expand_parent_chunks(reranked, chunks, max_contexts=min(5, final_context_k))
    structured_results = find_result_constrained_chunks(question, chunks, max_results=final_context_k)
    if structured_results:
        expanded = merge_event_list_chunks(structured_results, expanded)[:final_context_k]

    results = []
    for chunk in expanded:
        result = dict(chunk)
        result["retrieval_method"] = _merge_retrieval_method(result.get("retrieval_method", ""), "rrf_parent_context")
        result.setdefault("rerank_trace", "")
        results.append(result)
    return results


def _merge_graph_and_vector_candidates(graph_results, vector_results, top_k: int):
    merged = {}
    order = []
    for chunk in list(graph_results) + list(vector_results):
        chunk_id = chunk.get("id")
        if chunk_id not in merged:
            candidate = dict(chunk)
            candidate["score"] = float(candidate.get("score", 0.0))
            merged[chunk_id] = candidate
            order.append(chunk_id)
            continue
        existing = merged[chunk_id]
        existing["score"] = max(float(existing.get("score", 0.0)), float(chunk.get("score", 0.0)))
        existing["retrieval_method"] = _merge_retrieval_method(
            existing.get("retrieval_method", ""),
            chunk.get("retrieval_method", ""),
        )
        if chunk.get("graph_trace"):
            existing["graph_trace"] = chunk["graph_trace"]
        if chunk.get("rerank_trace"):
            existing["rerank_trace"] = _merge_trace(existing.get("rerank_trace", ""), chunk["rerank_trace"])

    for index, chunk_id in enumerate(order):
        method = str(merged[chunk_id].get("retrieval_method", ""))
        if "graph" in method:
            merged[chunk_id]["score"] += 3.0 - (0.05 * index)

    return sorted(merged.values(), key=lambda item: item.get("score", 0.0), reverse=True)[:top_k]


def _merge_retrieval_method(left: str, right: str) -> str:
    methods = [method for method in (str(left), str(right)) if method]
    return "+".join(dict.fromkeys(methods))


def _merge_trace(left: str, right: str) -> str:
    traces = [trace for trace in (str(left), str(right)) if trace]
    return ", ".join(dict.fromkeys(traces))


def _metadata_filter_chunk_embedding_pairs(query: str, chunks, embeddings):
    filtered_chunks = list(chunks)
    filtered_embeddings = list(embeddings) if embeddings is not None else None
    metadata_filters = _explicit_metadata_filters(query, filtered_chunks)
    if not metadata_filters:
        return filtered_chunks, filtered_embeddings

    selected_chunks = []
    selected_embeddings = [] if filtered_embeddings is not None else None
    for index, chunk in enumerate(filtered_chunks):
        if _chunk_matches_metadata_filters(chunk, metadata_filters):
            selected_chunks.append(chunk)
            if selected_embeddings is not None:
                selected_embeddings.append(filtered_embeddings[index])

    if not selected_chunks:
        return filtered_chunks, filtered_embeddings
    return selected_chunks, selected_embeddings


def _bm25_dense_retrieval_query(question: str, rewritten_query: str) -> str:
    original_query = " ".join(str(question).split())
    return expand_query_with_aliases(original_query, project_root=PROJECT_ROOT)


def _effective_rrf_retrieval_query(question: str, rewritten_query: str) -> str:
    if _is_low_information_rewrite(question, rewritten_query):
        return " ".join(str(question).split())
    return _combine_original_and_rewritten_query(question, rewritten_query)


def _is_low_information_rewrite(question: str, rewritten_query: str) -> bool:
    cleaned = " ".join(str(rewritten_query or "").split())
    if not cleaned:
        return True
    compact_question = _compact_rerank_text(question)
    compact_rewrite = _compact_rerank_text(cleaned)
    if compact_rewrite == compact_question:
        return False

    generic_terms = {
        "文明",
        "世界",
        "問題",
        "问题",
        "遊戲",
        "游戏",
    }
    terms = [term.strip() for term in cleaned.split() if term.strip()]
    if not terms:
        terms = [cleaned]
    if len(terms) <= 2:
        low_information_terms = 0
        for term in terms:
            compact_term = _compact_rerank_text(term)
            if compact_term in generic_terms or compact_term in compact_question:
                low_information_terms += 1
        return low_information_terms == len(terms)
    return False


def _definition_route_query(question: str) -> str:
    subject = _definition_route_subject(question)
    if not subject:
        return ""
    definition_terms = [
        subject,
        "定義",
        "定义",
        "正式稱呼",
        "正式称呼",
        "稱呼",
        "称呼",
        "又稱",
        "又称",
        "稱為",
        "称为",
        "被稱為",
        "被称为",
        "叫它",
        "叫做",
        "叫作",
        "指的是",
        "代表",
        "意思",
        "身份",
        "角色",
        "功能",
        "用途",
        "作用",
        "來源",
        "来源",
        "材料",
        "構成",
        "构成",
        "形狀",
        "形状",
    ]
    return " ".join(dict.fromkeys(term for term in definition_terms if term))


def _definition_route_subject(question: str) -> str:
    text = " ".join(str(question or "").split())
    if not text:
        return ""
    if re.search(r"(為什麼|为什么|爲什麼|如何|怎麼|怎么|何時|何时|哪裡|哪里)", text):
        return ""
    text = re.sub(r"^[請请]\s*問\s*", "", text)
    text = text.strip(" 　\t\r\n？?。.!！")
    patterns = (
        r"^(.{1,32}?)(?:是|為|为)(?:什麼|什么|誰|谁)(?:東西|东西|人|角色|概念|意思|內容|内容)?$",
        r"^(.{1,32}?)(?:指的是|代表的是|代表|意味著|意味着)(?:什麼|什么)?$",
        r"^(?:何謂|何谓|什麼是|什么是)(.{1,32}?)$",
    )
    for pattern in patterns:
        match = re.match(pattern, text)
        if not match:
            continue
        subject = match.group(1).strip(" 　\t\r\n，,。.!！？?「」『』“”\"'")
        if subject:
            return subject
    return ""


def _definition_search_results(definition_query: str, chunks, top_k: int):
    terms = str(definition_query or "").split()
    if not terms:
        return []
    subject = terms[0]
    subject_compact = _compact_rerank_text(subject)
    marker_terms = _definition_marker_terms()
    scored = []
    for chunk in chunks:
        text = _compact_rerank_text(
            f"{chunk.get('parent_title', '')} {chunk.get('title', '')} {chunk.get('content', '')}"
        )
        if not subject_compact or subject_compact not in text:
            continue
        covered_markers = _definition_marker_hits_near_subject(subject_compact, text, marker_terms)
        if not covered_markers:
            continue
        result = dict(chunk)
        marker_score = sum(_definition_marker_weight(term) for term in covered_markers)
        result["score"] = 8 + marker_score + _definition_subject_density(subject_compact, text)
        scored.append(result)
    return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]


def _definition_subject_density(subject_compact: str, text: str) -> int:
    if not subject_compact:
        return 0
    return min(6, text.count(subject_compact))


def _definition_marker_hits_near_subject(subject_compact: str, text: str, marker_terms, max_distance: int = 90):
    subject_positions = [match.start() for match in re.finditer(re.escape(subject_compact), text)]
    if not subject_positions:
        return []
    hits = []
    for term in marker_terms:
        marker = _compact_rerank_text(term)
        if not marker:
            continue
        marker_positions = [match.start() for match in re.finditer(re.escape(marker), text)]
        if not marker_positions:
            continue
        if any(abs(subject_position - marker_position) <= max_distance for subject_position in subject_positions for marker_position in marker_positions):
            hits.append(term)
    return hits


def _apply_definition_route_boost(candidates, definition_query: str) -> None:
    subject = _definition_route_subject(definition_query.split()[0] if definition_query else "")
    if not subject and definition_query:
        subject = definition_query.split()[0]
    if not subject:
        return
    subject_compact = _compact_rerank_text(subject)
    marker_terms = _definition_marker_terms()
    for candidate in candidates:
        definition_rank_boost = _definition_route_rank_boost(candidate)
        if definition_rank_boost:
            candidate["score"] += definition_rank_boost
        text = _compact_rerank_text(
            f"{candidate.get('parent_title', '')} {candidate.get('title', '')} {candidate.get('content', '')}"
        )
        if subject_compact not in text:
            continue
        marker_score = sum(
            _definition_marker_weight(term)
            for term in _definition_marker_hits_near_subject(subject_compact, text, marker_terms)
        )
        if marker_score:
            candidate["score"] += min(0.32, 0.025 * marker_score)


def _definition_route_rank_boost(candidate) -> float:
    trace = str(candidate.get("rerank_trace", ""))
    match = re.search(r"\bdefinition_bm25:(\d+)\b", trace)
    if not match:
        return 0.0
    rank = int(match.group(1))
    if rank <= 3:
        return 0.35 - (0.05 * (rank - 1))
    if rank <= 8:
        return 0.10
    return 0.0


def _definition_marker_terms():
    return (
        "定義",
        "定义",
        "正式稱呼",
        "正式称呼",
        "稱呼",
        "称呼",
        "又稱",
        "又称",
        "稱為",
        "称为",
        "被稱為",
        "被称为",
        "叫它",
        "叫做",
        "叫作",
        "指的是",
        "代表",
        "意思",
        "身份",
        "角色",
        "功能",
        "用途",
        "作用",
        "來源",
        "来源",
        "材料",
        "構成",
        "构成",
        "形狀",
        "形状",
    )


def _definition_marker_weight(term: str) -> int:
    compact = _compact_rerank_text(term)
    naming_markers = {
        "正式稱呼",
        "正式称呼",
        "稱呼",
        "称呼",
        "又稱",
        "又称",
        "稱為",
        "称为",
        "被稱為",
        "被称为",
        "叫它",
        "叫做",
        "叫作",
        "指的是",
    }
    identity_markers = {"定義", "定义", "身份", "角色"}
    concrete_markers = {"材料", "構成", "构成", "形狀", "形状", "功能", "用途"}
    if compact in {_compact_rerank_text(marker) for marker in naming_markers}:
        return 10
    if compact in {_compact_rerank_text(marker) for marker in identity_markers}:
        return 8
    if compact in {_compact_rerank_text(marker) for marker in concrete_markers}:
        return 4
    return 1


def _explicit_metadata_filters(query: str, chunks):
    compact_query = _compact_rerank_text(query)
    filters = {}
    sequence_filters = _explicit_sequence_filters(query)
    if sequence_filters:
        filters["sequence"] = set(sequence_filters)

    metadata_fields = ("book", "volume", "document_type", "author", "department", "chapter")
    for field in metadata_fields:
        values = {
            str(chunk.get(field, "")).strip()
            for chunk in chunks
            if str(chunk.get(field, "")).strip()
        }
        matched = [value for value in values if _compact_rerank_text(value) in compact_query]
        if matched:
            filters[field] = set(matched)
    return filters


def _chunk_matches_metadata_filters(chunk, metadata_filters) -> bool:
    for field, allowed_values in metadata_filters.items():
        if field == "sequence":
            if not _chunk_matches_sequence_filter(chunk, allowed_values):
                return False
            continue
        value = str(chunk.get(field, "")).strip()
        if value not in allowed_values:
            return False
    return True


def _explicit_sequence_filters(query: str):
    sequence_labels = []
    patterns = (
        (r"第\s*(\d+)\s*(輪|章|節|段|步|天|次|回合|階段)", lambda number, unit: f"第{number}{unit}"),
        (r"\b(round|chapter|section|step|phase|day)\s*(\d+)\b", lambda unit, number: f"{unit.lower()} {number}"),
        (r"\b(\d+)\s*(round|chapter|section|step|phase|day)\b", lambda number, unit: f"{unit.lower()} {number}"),
    )
    for pattern, formatter in patterns:
        for match in re.finditer(pattern, str(query), flags=re.IGNORECASE):
            sequence_labels.append(formatter(*match.groups()))
    return list(dict.fromkeys(sequence_labels))


def _chunk_matches_sequence_filter(chunk, allowed_values) -> bool:
    metadata_text = "\n".join(
        str(chunk.get(field, ""))
        for field in ("parent_title", "title", "chapter", "section", "phase", "step")
    )
    compact_metadata = _compact_rerank_text(metadata_text)
    for value in allowed_values:
        compact_value = _compact_rerank_text(value)
        if compact_value and compact_value in compact_metadata:
            return True
    return False


def _rrf_merge_ranked_results(ranked_lists, top_k: int, rrf_k: int = 60):
    candidates = {}
    for source_name, results in ranked_lists:
        for rank, chunk in enumerate(results, start=1):
            chunk_id = chunk["id"]
            if chunk_id not in candidates:
                candidate = dict(chunk)
                candidate["score"] = 0.0
                candidate["rrf_score"] = 0.0
                candidate["retrieval_methods"] = []
                candidates[chunk_id] = candidate
            score = 1.0 / (rrf_k + rank)
            candidates[chunk_id]["score"] += score
            candidates[chunk_id]["rrf_score"] += score
            candidates[chunk_id]["retrieval_methods"].append(f"{source_name}:{rank}")

    ranked = sorted(candidates.values(), key=lambda item: item["score"], reverse=True)
    results = []
    for item in ranked[:top_k]:
        result = dict(item)
        methods = result.get("retrieval_methods", [])
        result["rerank_trace"] = ", ".join(methods)
        result["retrieval_method"] = "rrf"
        results.append(result)
    return results


def _rerank_rrf_candidates(candidates, question: str, rewritten_query: str) -> None:
    terms = _relevant_rerank_terms(" ".join([question, rewritten_query]))
    terms.extend(_cjk_query_phrase_terms(question))
    terms = list(dict.fromkeys(terms))[:48]
    if not terms:
        return
    for candidate in candidates:
        text = _compact_rerank_text(
            f"{candidate.get('parent_title', '')} {candidate.get('title', '')} {candidate.get('content', '')}"
        )
        covered = sum(1 for term in terms if _compact_rerank_text(term) in text)
        candidate["score"] += 0.20 * (covered / len(terms))


def _cjk_query_phrase_terms(text: str):
    compact = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", "", str(text))
    stop_fragments = {
        "的是",
        "什麼",
        "哪一",
        "哪個",
        "哪本",
        "一本",
        "問題",
        "主要",
        "開始",
        "後來",
        "要求",
    }
    terms = []
    for length in (4, 3):
        for index in range(max(0, len(compact) - length + 1)):
            term = compact[index : index + length]
            if any(fragment in term for fragment in stop_fragments):
                continue
            terms.append(term)
    return terms


def _expand_parent_chunks(ranked_chunks, all_chunks, max_contexts: int = 8, neighbor_window: int = 1):
    selected = []
    selected_ids = set()
    chunks_by_parent = _chunks_by_parent_section(all_chunks)

    for chunk in ranked_chunks:
        for candidate in _parent_expansion_candidates(chunk, chunks_by_parent, neighbor_window):
            if candidate.get("id") == chunk.get("id"):
                candidate = chunk
            chunk_id = candidate.get("id")
            if chunk_id in selected_ids:
                continue
            selected.append(candidate)
            selected_ids.add(chunk_id)
            if len(selected) >= max_contexts:
                return selected
    return selected


def _chunks_by_parent_section(chunks):
    groups = {}
    for chunk in chunks:
        key = (
            str(chunk.get("source", "")),
            str(chunk.get("parent_title", "")),
            _base_chunk_title(str(chunk.get("title", ""))),
        )
        groups.setdefault(key, []).append(chunk)
    for grouped in groups.values():
        grouped.sort(key=lambda item: int(item.get("chunk_index", 0)))
    return groups


def _parent_expansion_candidates(chunk, chunks_by_parent, neighbor_window: int):
    key = (
        str(chunk.get("source", "")),
        str(chunk.get("parent_title", "")),
        _base_chunk_title(str(chunk.get("title", ""))),
    )
    siblings = chunks_by_parent.get(key, [chunk])
    chunk_index = int(chunk.get("chunk_index", 0))
    nearby = [
        sibling
        for sibling in siblings
        if abs(int(sibling.get("chunk_index", 0)) - chunk_index) <= neighbor_window
    ]
    nearby.sort(
        key=lambda sibling: (
            abs(int(sibling.get("chunk_index", 0)) - chunk_index),
            int(sibling.get("chunk_index", 0)),
        )
    )
    return nearby or [chunk]


def _base_chunk_title(title: str) -> str:
    return re.sub(r"\s*/\s*part\s+\d+\s*$", "", str(title), flags=re.IGNORECASE).strip()


def _hybrid_rerank_results(
    query_variants,
    question: str,
    refined_question: str,
    keywords,
    chunks,
    embeddings,
    top_k: int,
    candidate_k: int,
    vector_store=None,
    settings: RagConfig | None = None,
):
    settings = (settings or RagConfig.from_env()).normalized()
    candidates = {}
    candidate_k = max(top_k, int(candidate_k or top_k))

    for variant in query_variants:
        query_text = variant["query"]
        query_weight = float(variant.get("weight", 1.0))

        keyword_results = keyword_search(query_text, chunks, top_k=candidate_k)
        max_keyword_score = max((float(item.get("score", 0.0)) for item in keyword_results), default=1.0)
        for rank, chunk in enumerate(keyword_results, start=1):
            normalized_score = float(chunk.get("score", 0.0)) / max_keyword_score if max_keyword_score else 0.0
            _add_hybrid_candidate_score(
                candidates,
                chunk,
                score=query_weight * (0.58 * normalized_score + _reciprocal_rank_score(rank, 0.10)),
                method=f"kw:{variant['name']}:{rank}",
                keyword_score=float(chunk.get("score", 0.0)),
            )

        if vector_store is None and embeddings is None:
            continue

        embedding_results = _dense_search_results(
            query_text,
            chunks=chunks,
            embeddings=embeddings,
            top_k=candidate_k,
            vector_store=vector_store,
        )
        embedding_scores = [float(item.get("embedding_score", 0.0)) for item in embedding_results]
        min_embedding = min(embedding_scores, default=0.0)
        max_embedding = max(embedding_scores, default=1.0)
        span = max(max_embedding - min_embedding, 1e-9)
        for rank, chunk in enumerate(embedding_results, start=1):
            raw_embedding = float(chunk.get("embedding_score", 0.0))
            normalized_score = (raw_embedding - min_embedding) / span
            _add_hybrid_candidate_score(
                candidates,
                chunk,
                score=query_weight * (0.32 * normalized_score + _reciprocal_rank_score(rank, 0.07)),
                method=f"emb:{variant['name']}:{rank}",
                embedding_score=raw_embedding,
            )

    fused = fuse_candidates_with_lambdamart(
        list(candidates.values()),
        keyword_weight=settings.keyword_weight,
        embedding_weight=settings.embedding_weight,
    )
    for item in fused:
        item["score"] = float(item.get("lambda_mart_fusion_score", 0.0))
    complexity = classify_query_complexity(
        question,
        query_variants=[variant.get("query", "") for variant in query_variants],
        threshold=settings.query_complexity_threshold,
    )
    if settings.complexity_routing_enabled and not complexity.is_complex:
        ranked = sorted(
            fused,
            key=lambda item: float(item.get("lambda_mart_fusion_score", 0.0)),
            reverse=True,
        )
        selected = ranked[: min(top_k, settings.simple_query_top_k)]
    else:
        candidate_map = {str(item["id"]): item for item in fused}
        _apply_rerank_lexical_coverage_boost(
            candidate_map,
            question,
            refined_question,
            keywords,
        )
        ranked = sorted(candidate_map.values(), key=lambda item: item["score"], reverse=True)
        selected = ranked[:top_k]
    return [_strip_rerank_candidate_metadata(item) for item in selected]


def _dense_search_results(
    query: str,
    chunks,
    embeddings,
    top_k: int,
    vector_store=None,
    allowed_chunk_ids=None,
):
    if vector_store is not None:
        return vector_store.search(query, top_k=top_k, allowed_chunk_ids=allowed_chunk_ids)
    if embeddings is None:
        return []
    return embedding_search(
        query,
        chunks,
        embeddings=embeddings,
        embed_query_fn=embed_query,
        top_k=top_k,
    )


def _add_hybrid_candidate_score(
    candidates,
    chunk,
    score: float,
    method: str,
    keyword_score: float = 0.0,
    embedding_score: float = 0.0,
):
    chunk_id = chunk["id"]
    if chunk_id not in candidates:
        candidate = dict(chunk)
        candidate["score"] = 0.0
        candidate["keyword_score"] = 0.0
        candidate["embedding_score"] = 0.0
        candidate["retrieval_methods"] = []
        candidates[chunk_id] = candidate

    candidate = candidates[chunk_id]
    candidate["score"] += score
    candidate["keyword_score"] = max(float(candidate.get("keyword_score", 0.0)), keyword_score)
    candidate["embedding_score"] = max(float(candidate.get("embedding_score", 0.0)), embedding_score)
    candidate["retrieval_methods"].append(method)


def _reciprocal_rank_score(rank: int, weight: float) -> float:
    return weight / max(rank, 1)


def _apply_rerank_lexical_coverage_boost(candidates, question: str, refined_question: str, keywords) -> None:
    terms = _relevant_rerank_terms(" ".join([question, refined_question, " ".join(keywords)]))
    choice_terms = _choice_rerank_terms(question, keywords)
    choice_anchor = _choice_subject_anchor(keywords, choice_terms)
    if not terms:
        return
    for candidate in candidates.values():
        text = _compact_rerank_text(
            f"{candidate.get('parent_title', '')} {candidate.get('title', '')} {candidate.get('content', '')}"
        )
        covered = sum(1 for term in terms if _compact_rerank_text(term) in text)
        candidate["score"] += 0.16 * (covered / len(terms))
        if choice_anchor and choice_terms:
            anchor_hit = _compact_rerank_text(choice_anchor) in text
            choice_hits = sum(1 for term in choice_terms if _compact_rerank_text(term) in text)
            if anchor_hit and choice_hits:
                candidate["score"] += 2.2 + (0.18 * min(choice_hits, 2))


def _choice_rerank_terms(question_text: str, keywords) -> list:
    compact_question = _compact_rerank_text(question_text)
    if "還是" not in compact_question and "或" not in compact_question:
        return []
    choice_terms = []
    for keyword in keywords:
        cleaned = str(keyword).strip()
        compact_keyword = _compact_rerank_text(cleaned)
        if len(compact_keyword) < 2:
            continue
        for marker in ("還是", "或"):
            marker_index = compact_question.find(marker)
            if marker_index < 0:
                continue
            window = compact_question[max(0, marker_index - 10) : marker_index + 12]
            if compact_keyword in window:
                choice_terms.append(cleaned)
                break
    return list(dict.fromkeys(choice_terms))


def _choice_subject_anchor(keywords, choice_terms) -> str:
    compact_choices = {_compact_rerank_text(term) for term in choice_terms}
    for keyword in keywords:
        cleaned = str(keyword).strip()
        if cleaned and _compact_rerank_text(cleaned) not in compact_choices:
            return cleaned
    return ""


def _relevant_rerank_terms(text: str):
    raw_terms = re.findall(r"[A-Za-z0-9_.-]+|[\u4e00-\u9fff]{2,}", str(text))
    stop_terms = {
        "什麼",
        "為什麼",
        "如何",
        "哪裡",
        "哪種",
        "問題",
        "核心",
        "主要",
        "代表",
        "得到",
        "提到",
        "造成",
        "認為",
        "需要",
        "希望",
    }
    terms = []
    for term in raw_terms:
        cleaned = term.strip().lower()
        if len(cleaned) < 2 or cleaned in stop_terms:
            continue
        terms.append(cleaned)
    return list(dict.fromkeys(terms))[:24]


def _compact_rerank_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text)).lower()


def _strip_rerank_candidate_metadata(candidate):
    result = dict(candidate)
    result["retrieval_method"] = "hybrid_rerank"
    methods = result.get("retrieval_methods", [])
    result["rerank_trace"] = ", ".join(methods[:10])
    if len(methods) > 10:
        result["rerank_trace"] += f", +{len(methods) - 10} more"
    return result


def _build_source_insufficient_answer(verification) -> str:
    reason = str((verification or {}).get("reason", "")).strip()
    if reason:
        return f"無法從目前檢索來源確認。原因：{reason}"
    return "無法從目前檢索來源確認。"


def _combine_original_and_rewritten_query(question: str, rewritten_query: str) -> str:
    cleaned_question = " ".join(str(question).split())
    cleaned_rewrite = " ".join(str(rewritten_query or "").split())
    if not cleaned_rewrite or cleaned_rewrite == cleaned_question:
        return cleaned_question
    if cleaned_question in cleaned_rewrite:
        return cleaned_rewrite
    return f"{cleaned_question} {cleaned_rewrite}"


def _select_retrieval_query(question: str, rewritten_query: str, chunks) -> str:
    if _is_narrative_knowledge_base(chunks):
        return " ".join(str(question).split())
    return _combine_original_and_rewritten_query(question, rewritten_query)


def _is_narrative_knowledge_base(chunks) -> bool:
    return any(str(chunk.get("parent_title", "")) == "Narrative Text" for chunk in chunks)


def _planned_retrieval_results(retrieval_plan, chunks, embeddings, top_k: int, config: RagConfig):
    variants = list(retrieval_plan.query_variants)
    if len(variants) <= 1:
        return []

    selected = []
    selected_ids = set()
    max_results = max(top_k * 2, 8)
    per_query_k = max(2, min(top_k, 5))

    for variant in variants[1:]:
        expanded_variant = expand_query_with_aliases(variant, project_root=PROJECT_ROOT)
        candidates = _search_chunks_for_planned_query(
            expanded_variant,
            chunks,
            embeddings=embeddings,
            top_k=per_query_k,
            config=config,
        )
        for candidate in candidates:
            chunk_id = candidate.get("id") or f"{candidate.get('source', '')}:{candidate.get('title', '')}"
            if chunk_id in selected_ids:
                continue
            result = dict(candidate)
            result["retrieval_plan_query"] = expanded_variant
            selected.append(result)
            selected_ids.add(chunk_id)
            if len(selected) >= max_results:
                return selected

    return selected


def _search_chunks_for_planned_query(query: str, chunks, embeddings, top_k: int, config: RagConfig):
    if embeddings is not None:
        try:
            return hybrid_search(
                query,
                chunks,
                embeddings=embeddings,
                embed_query_fn=embed_query,
                top_k=top_k,
                keyword_weight=config.keyword_weight,
                embedding_weight=config.embedding_weight,
                metadata_boost_max=config.metadata_boost_max,
            )
        except Exception:
            pass
    return keyword_search(query, chunks, top_k=top_k)


def _diversify_retrieval_results(question: str, retrieval_query: str, chunks, results, extra_limit: int = 6):
    if not results or not _is_narrative_knowledge_base(chunks):
        return results

    anchors = _diversity_anchor_queries(question, retrieval_query)
    if not anchors:
        return results

    diversified = list(results)
    selected_ids = {chunk.get("id") for chunk in diversified}
    added = 0

    for anchor in anchors:
        fallback_anchor = _fallback_anchor(anchor)
        candidate_anchors = [fallback_anchor, anchor] if fallback_anchor and fallback_anchor != anchor else [anchor]
        for candidate_anchor in candidate_anchors:
            candidate = _first_diversity_candidate(candidate_anchor, chunks, selected_ids)
            if candidate is None:
                continue
            diversified.append(candidate)
            selected_ids.add(candidate.get("id"))
            added += 1
            if added >= extra_limit:
                break
        if added >= extra_limit:
            break

    return diversified


def _first_diversity_candidate(anchor: str, chunks, selected_ids):
    for candidate in keyword_search(anchor, chunks, top_k=5):
        chunk_id = candidate.get("id")
        if chunk_id in selected_ids:
            continue
        if _is_metadata_candidate(candidate):
            continue
        if not _anchor_matches_chunk(anchor, candidate):
            continue
        return candidate
    return None


def _fallback_anchor(anchor: str) -> str:
    terms = [term.strip() for term in str(anchor).split() if term.strip()]
    if len(terms) < 2:
        return ""
    return terms[-1]


def _is_metadata_candidate(chunk) -> bool:
    title = str(chunk.get("title", ""))
    return "metadata" in title.lower() or title == "文件開頭 / metadata"


def _diversity_anchor_queries(question: str, retrieval_query: str, max_anchors: int = 12):
    subject = _first_known_subject(question)
    original_terms = set(str(question).split())
    anchors = []
    for term in str(retrieval_query).split():
        normalized = term.strip()
        if not normalized or normalized in original_terms:
            continue
        if _is_low_value_diversity_term(normalized):
            continue
        anchor = f"{subject} {normalized}".strip() if subject and subject not in normalized else normalized
        anchors.append(anchor)
    return list(dict.fromkeys(anchors))[:max_anchors]


def _first_known_subject(question: str) -> str:
    for term in re.findall(r"[A-Za-z0-9_.+-]+|[\u4e00-\u9fff]{2,}", str(question)):
        if not _is_low_value_diversity_term(term):
            return term
    return ""


def _is_low_value_diversity_term(term: str) -> bool:
    compact = "".join(str(term).split())
    if len(compact) < 2:
        return True
    if any(marker in compact for marker in ("人生經歷", "後來的選擇", "影響選擇", "對人類失望")):
        return True
    stop_terms = {
        "什麼",
        "為什麼",
        "如何",
        "哪些",
        "小說",
        "遊戲",
        "文明",
        "問題",
        "背景",
        "主要",
        "時期",
        "選擇",
        "影響",
        "工作",
    }
    return compact in stop_terms or compact.endswith("？")


def _anchor_matches_chunk(anchor: str, chunk) -> bool:
    haystack = f"{chunk.get('parent_title', '')}\n{chunk.get('title', '')}\n{chunk.get('content', '')}"
    terms = [term for term in str(anchor).split() if len(term.strip()) >= 2]
    if not terms:
        return False
    return all(term in haystack for term in terms)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the canonical local RAG pipeline.")
    parser.add_argument("question", help="Question to ask against local raw documents.")
    parser.add_argument("--model", default=os.getenv("RAG_MODEL", "ollama:qwen2.5:7b"))
    parser.add_argument("--profile", default=os.getenv("RAG_PROFILE", "default"))
    parser.add_argument("--source-id", action="append", dest="source_ids")
    parser.add_argument("--top-k", type=int, default=RagConfig.from_env().hybrid_top_k)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Run the pre-canonical query path for historical comparison only.",
    )
    args = parser.parse_args()

    if args.legacy:
        print(answer_question(args.question, model=args.model, top_k=args.top_k))
        return

    result = RagPipeline().run(RagPipelineRequest(
        question=args.question,
        model=args.model,
        profile=args.profile,
        source_ids=args.source_ids,
        top_k=args.top_k,
        persist_conversation=False,
    ))
    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result["answer"])


def _format_output(
    question: str,
    retrieval_query: str,
    results,
    answer: str,
    verification=None,
    timing=None,
    context_summary: str = "",
) -> str:
    source_lines = []
    for index, chunk in enumerate(results, start=1):
        detail = f"score={chunk['score']}"
        if "keyword_score" in chunk and "embedding_score" in chunk:
            detail = (
                f"score={float(chunk['score']):.4f}, "
                f"keyword={float(chunk['keyword_score']):.1f}, "
                f"embedding={float(chunk['embedding_score']):.4f}"
            )
        source_lines.append(
            f"{index}. {Path(str(chunk['source'])).name} / {chunk['title']} / {detail}"
        )

    sources = "\n".join(source_lines) if source_lines else "沒有找到來源"
    verifier_text = ""
    if verification:
        verifier_text = f"""
Verifier Agent:
is_related={verification["is_related"]}, confidence={verification["confidence"]}, reason={verification["reason"]}
"""
    summary_text = ""
    if context_summary:
        summary_text = f"""
彙整資料：
{context_summary}
"""
    timing_text = _format_timing(timing or {})

    return f"""問題：{question}

向量檢索用查詢：
{retrieval_query}

檢索來源：
{sources}
{verifier_text}
{summary_text}
{timing_text}

模型回答：
{answer}
"""


def _format_three_agent_output(
    question: str,
    query_variants,
    keywords,
    refined_question: str,
    results,
    extracted_evidence: str,
    answer: str,
    timing=None,
) -> str:
    source_lines = []
    for index, chunk in enumerate(results, start=1):
        detail = f"score={chunk.get('score', 0)}"
        if "embedding_score" in chunk:
            detail = (
                f"score={float(chunk.get('score', 0.0)):.4f}, "
                f"keyword={float(chunk.get('keyword_score', 0.0)):.4f}, "
                f"embedding={float(chunk['embedding_score']):.4f}"
            )
        if chunk.get("rerank_trace"):
            detail = f"{detail}, trace={chunk['rerank_trace']}"
        source_lines.append(
            f"{index}. {Path(str(chunk['source'])).name} / {chunk['title']} / {detail}"
        )

    sources = "\n".join(source_lines) if source_lines else "沒有找到來源"
    keyword_text = ", ".join(str(keyword) for keyword in keywords) if keywords else "無"
    query_variant_text = "\n".join(
        f"- {variant['name']}: {variant['query']}" for variant in query_variants
    )
    timing_text = _format_timing(timing or {})

    return f"""問題：{question}

Keyword Extraction Agent:
{keyword_text}

Question Extraction Agent:
{refined_question}

Hybrid Retrieval Query Variants:
{query_variant_text}

檢索來源 Top {len(results)}：
{sources}

Evidence Extraction Agent:
{str(extracted_evidence or '').strip() or '無'}
{timing_text}

Final Answer:
{answer}
"""


def _self_rag_should_retry(
    attempt_index: int,
    max_attempts: int,
    passage_reflections,
    answer_critique,
    utility_critique,
) -> bool:
    if attempt_index >= max_attempts:
        return False
    has_supporting_passage = any(item.is_supported for item in passage_reflections)
    if not has_supporting_passage:
        return True
    if answer_critique.retry_needed or not answer_critique.is_supported:
        return True
    return bool(utility_critique.retry_needed or int(utility_critique.score) <= 2)


def _self_rag_enforce_passage_support(
    question: str,
    answer_critique,
    passage_reflections,
):
    if any(item.is_supported for item in passage_reflections):
        return answer_critique
    reason = (
        f"沒有 supporting passage，因此不可把回答標記為 [Supported]。"
        f"原始 critic reason：{answer_critique.reason}"
    )
    return AnswerSupportCritique(
        support_label="[Unsupported]",
        is_supported=False,
        retry_needed=True,
        retry_query=answer_critique.retry_query or question,
        reason=reason,
        raw_output=answer_critique.raw_output,
    )


def _self_rag_retry_query(
    question: str,
    current_query: str,
    answer_critique,
    utility_critique,
) -> str:
    retry_query = str(answer_critique.retry_query or utility_critique.retry_query or current_query or question).strip()
    if not retry_query:
        retry_query = question
    return _combine_original_and_rewritten_query(question, retry_query)


def _format_self_rag_output(
    question: str,
    rewrite_decision,
    attempts,
    answer: str,
    timing=None,
    vector_db_enabled: bool = False,
    fine_tuned_self_rag: bool = False,
) -> str:
    final_attempt = attempts[-1] if attempts else {}
    results = final_attempt.get("results", []) if final_attempt else []
    source_lines = []
    for index, chunk in enumerate(results, start=1):
        detail = f"score={float(chunk.get('score', 0.0)):.4f}"
        if chunk.get("rerank_trace"):
            detail = f"{detail}, trace={chunk['rerank_trace']}"
        source_lines.append(
            f"{index}. {Path(str(chunk['source'])).name} / {chunk['title']} / {detail}"
        )
    sources = "\n".join(source_lines) if source_lines else "沒有找到來源"
    timing_text = _format_timing(timing or {})
    vector_db_text = "Qdrant Vector DB" if vector_db_enabled else "local embedding fallback"
    attempt_text = _format_self_rag_attempts(attempts)
    fine_tuned_text = "true" if fine_tuned_self_rag else "false (engineering controller, not fine-tuned Self-RAG LM)"

    return f"""問題：{question}

Workflow:
Question
↓
Query Rewrite / Retrieval Decision
↓
[Retrieve]=true
↓
Metadata Filter
↓
BM25(original question) + Dense(original question via {vector_db_text}) + Graph
↓
Child Chunk Candidate Merge + RRF Fusion (Top-100)
↓
Query Complexity Gate
↓
Simple Top-5 / Complex Cross-Encoder Reranker
↓
Parent Chunk Expansion
↓
Passage Critic [Relevant]
↓
QA Agent
↓
Answer Support Critic [Supported]
↓
Utility Scoring [Utility]
↓
Iterative Retry when critique fails

Engineering Self-RAG Controller:
- fine_tuned_self_reflective_lm={fine_tuned_text}
- [Retrieve]=true, reason={rewrite_decision.reason}

Query Rewrite Output:
{rewrite_decision.retrieval_query}

Self-RAG Attempts:
{attempt_text}

檢索來源 Top {len(results)}：
{sources}
{timing_text}

Final Answer:
{answer}
"""


def _format_self_rag_no_retrieval_output(
    question: str,
    rewrite_decision,
    answer: str,
    timing=None,
) -> str:
    timing_text = _format_timing(timing or {})
    return f"""問題：{question}

Workflow:
Question
↓
Query Rewrite / Retrieval Decision
↓
[Retrieve]=false
↓
Skip Retrieval
↓
General Answer

Engineering Self-RAG Controller:
- fine_tuned_self_reflective_lm=false (engineering controller, not fine-tuned Self-RAG LM)
- [Retrieve]=false, reason={rewrite_decision.reason}

Query Rewrite Output:
{rewrite_decision.retrieval_query}

檢索來源 Top 0：
已判斷此問題不需要檢索 knowledge base，因此未執行 BM25 / Dense / Graph Retrieval。
{timing_text}

Final Answer:
{answer}
"""


def _format_self_rag_attempts(attempts) -> str:
    if not attempts:
        return "無"
    blocks = []
    for attempt in attempts:
        passage_lines = []
        for reflection in attempt.get("passage_reflections", []):
            passage_lines.append(
                f"  - Source {reflection.source_index}: {reflection.relevance_label}{reflection.support_label} {reflection.reason}"
            )
        passage_text = "\n".join(passage_lines) if passage_lines else "  - 無 passage reflection"
        answer_critique = attempt.get("answer_critique")
        utility_critique = attempt.get("utility_critique")
        retry_query = attempt.get("retry_query") or "無"
        blocks.append(
            "\n".join(
                [
                    f"Attempt {attempt.get('attempt')}:",
                    f"- Retrieval Query: {attempt.get('query')}",
                    "- Passage Critic:",
                    passage_text,
                    (
                        "- Answer Support Critic: "
                        f"{answer_critique.support_label}, retry_needed={str(bool(answer_critique.retry_needed)).lower()}, "
                        f"reason={answer_critique.reason}"
                    ),
                    (
                        "- Utility Critic: "
                        f"{utility_critique.utility_label}, retry_needed={str(bool(utility_critique.retry_needed)).lower()}, "
                        f"reason={utility_critique.reason}"
                    ),
                    f"- Retry Query: {retry_query}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _format_v2_output(
    question: str,
    rewritten_query: str,
    results,
    answer: str,
    timing=None,
    vector_db_enabled: bool = False,
    rewrite_decision=None,
) -> str:
    source_lines = []
    for index, chunk in enumerate(results, start=1):
        detail = f"score={float(chunk.get('score', 0.0)):.4f}"
        if chunk.get("rerank_trace"):
            detail = f"{detail}, trace={chunk['rerank_trace']}"
        source_lines.append(
            f"{index}. {Path(str(chunk['source'])).name} / {chunk['title']} / {detail}"
        )
    sources = "\n".join(source_lines) if source_lines else "沒有找到來源"
    timing_text = _format_timing(timing or {})
    vector_db_text = "Qdrant Vector DB" if vector_db_enabled else "local embedding fallback"
    decision_text = ""
    if rewrite_decision is not None:
        decision_text = f"""
Retrieval Decision:
needs_retrieval={str(bool(rewrite_decision.needs_retrieval)).lower()}, reason={rewrite_decision.reason}
"""

    return f"""問題：{question}

Workflow:
Question
↓
Query Rewrite / Retrieval Decision
↓
Metadata Filter
↓
BM25(original question) + Dense(original question via {vector_db_text})
↓
Child Chunk Candidate Merge + RRF Fusion (Top-100)
↓
Query Complexity Gate
↓
Simple Top-5 / Complex Cross-Encoder Reranker
↓
Parent Chunk Expansion
↓
Top {len(results)} Context
↓
QA Agent

Query Rewrite Output:
{rewritten_query}
{decision_text}

檢索來源 Top {len(results)}：
{sources}
{timing_text}

Final Answer:
{answer}
"""


def _format_v2_no_retrieval_output(
    question: str,
    rewrite_decision,
    answer: str,
    timing=None,
) -> str:
    timing_text = _format_timing(timing or {})
    return f"""問題：{question}

Workflow:
Question
↓
Query Rewrite / Retrieval Decision
↓
Skip Retrieval
↓
General Answer

Retrieval Decision:
needs_retrieval=false, reason={rewrite_decision.reason}

Query Rewrite Output:
{rewrite_decision.retrieval_query}

檢索來源 Top 0：
已判斷此問題不需要檢索 knowledge base，因此未執行 BM25 / Dense / Graph Retrieval。
{timing_text}

Final Answer:
{answer}
"""


def _format_timing(timing) -> str:
    if not timing:
        return ""
    preferred_order = [
        "load_index",
        "keyword_extraction_agent",
        "question_extraction_agent",
        "query_rewrite",
        "load_embeddings",
        "retrieval",
        "retrieval_attempt_1",
        "passage_critic_attempt_1",
        "qa_agent_attempt_1",
        "answer_critic_attempt_1",
        "utility_scoring_attempt_1",
        "retrieval_attempt_2",
        "passage_critic_attempt_2",
        "qa_agent_attempt_2",
        "answer_critic_attempt_2",
        "utility_scoring_attempt_2",
        "retrieval_attempt_3",
        "passage_critic_attempt_3",
        "qa_agent_attempt_3",
        "answer_critic_attempt_3",
        "utility_scoring_attempt_3",
        "evidence_extraction_agent",
        "planned_retrieval",
        "event_list_retrieval",
        "sparse_fact_retrieval",
        "evidence_diversification",
        "deterministic_evidence",
        "entity_resolution",
        "action_result_resolution",
        "verifier",
        "context_summary",
        "answer_generation",
        "qa_agent",
        "general_fallback",
        "total",
    ]
    lines = ["節點耗時："]
    for key in preferred_order:
        if key in timing:
            lines.append(f"- {key}: {float(timing[key]):.2f}s")
    for key, value in timing.items():
        if key not in preferred_order:
            lines.append(f"- {key}: {float(value):.2f}s")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
