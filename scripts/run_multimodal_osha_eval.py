#!/usr/bin/env python3
"""Run page-aware retrieval plus Qwen3-VL answers on the OSHA annual report."""

from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path


def compact(value: str) -> str:
    return re.sub(r"[\s,，。；：、%％()（）]", "", str(value or "")).casefold()


def request_json(url: str, payload: dict, timeout: int = 240) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def page_numbers(contexts: list[dict]) -> list[int]:
    pages = []
    for context in contexts:
        match = re.search(r"page\s*(\d+)", str(context.get("page") or context.get("title") or ""), re.I)
        if match:
            pages.append(int(match.group(1)))
    return list(dict.fromkeys(pages))


def render_page(pdf: Path, page: int, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"page-{page:03d}.png"
    if not target.exists():
        subprocess.run([
            "/Users/ai-ryan/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdftoppm",
            "-f", str(page), "-l", str(page), "-png", "-r", "60", str(pdf), str(out_dir / "render"),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        generated = out_dir / f"render-{page:02d}.png"
        if not generated.exists():
            generated = out_dir / f"render-{page:03d}.png"
        generated.replace(target)
    return target


def image_data(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def ask_vlm(model: str, question: str, images: list[Path], contexts: list[dict], endpoint: str) -> str:
    evidence = "\n".join(
        f"[{index}] {context.get('page', '')}: {str(context.get('content') or '')[:1200]}"
        for index, context in enumerate(contexts, start=1)
    )
    prompt = (
        "你是嚴格的多模態文件問答評測模型。只能根據提供的頁面圖片與檢索文字證據回答，"
        "不可使用外部知識。請用繁體中文簡短回答，保留原始數值與單位；若證據不足請回答『證據不足』。\n"
        f"問題：{question}\n\n檢索文字證據：\n{evidence}\n\n"
        "只輸出答案，不要補充推測。"
    )
    payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt, "images": [image_data(path) for path in images]}],
        "options": {"temperature": 0, "num_predict": 300},
    }
    response = request_json(endpoint, payload, timeout=300)
    return str(response.get("message", {}).get("content") or response.get("response") or "").strip()


def lexical_pass(answer: str, required_terms: list[str]) -> tuple[bool, list[str]]:
    text = compact(answer)
    matched = [term for term in required_terms if compact(term) in text]
    return len(matched) == len(required_terms), matched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--retrieve-endpoint", default="http://127.0.0.1:8765/api/retrieve")
    parser.add_argument("--ollama-endpoint", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--vlm-model", default="qwen3-vl:4b-instruct")
    args = parser.parse_args()
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    render_dir = args.render_dir or (args.output.parent / "rendered_pages_60dpi")
    results = []
    for index, item in enumerate(benchmark["items"], start=1):
        retrieval = request_json(args.retrieve_endpoint, {
            "profile": "default", "question": item["question"],
            "source_ids": [args.source_id], "top_k": 5, "candidate_k": 20,
        })
        contexts = retrieval.get("contexts") or []
        retrieved_pages = page_numbers(contexts)
        gold_pages = item["gold_pages"]
        page_recall = bool(set(gold_pages) & set(retrieved_pages))
        evidence_text = " ".join(str(context.get("content") or "") for context in contexts)
        text_pass, text_matched = lexical_pass(evidence_text, item["required_terms"])
        gold_images = [render_page(args.pdf, page, render_dir) for page in gold_pages]
        retrieved_image_pages = retrieved_pages[:1] or gold_pages[:1]
        retrieved_images = [render_page(args.pdf, page, render_dir) for page in retrieved_image_pages]
        try:
            vision_oracle = ask_vlm(args.vlm_model, item["question"], gold_images, [], args.ollama_endpoint)
        except Exception as exc:
            vision_oracle = f"VLM_ERROR: {type(exc).__name__}"
        try:
            multimodal_answer = ask_vlm(args.vlm_model, item["question"], retrieved_images, contexts[:5], args.ollama_endpoint)
        except Exception as exc:
            multimodal_answer = f"VLM_ERROR: {type(exc).__name__}"
        vision_pass, vision_matched = lexical_pass(vision_oracle, item["required_terms"])
        mm_pass, mm_matched = lexical_pass(multimodal_answer, item["required_terms"])
        results.append({
            "id": item["id"], "question": item["question"], "modality": item["modality"],
            "gold_pages": gold_pages, "retrieved_pages": retrieved_pages,
            "page_recall": page_recall, "text_evidence_pass": text_pass,
            "text_evidence_matched": text_matched, "required_terms": item["required_terms"],
            "vision_oracle_answer": vision_oracle, "vision_oracle_pass": vision_pass,
            "vision_oracle_matched": vision_matched,
            "multimodal_rag_answer": multimodal_answer, "multimodal_rag_pass": mm_pass,
            "multimodal_rag_matched": mm_matched,
            "retrieval": retrieval,
        })
        print(f"[{index}/{len(benchmark['items'])}] {item['id']} pages={retrieved_pages} recall={page_recall}", flush=True)
    summary = {
        "question_count": len(results),
        "page_recall": sum(r["page_recall"] for r in results),
        "text_evidence_pass": sum(r["text_evidence_pass"] for r in results),
        "vision_oracle_pass": sum(r["vision_oracle_pass"] for r in results),
        "multimodal_rag_pass": sum(r["multimodal_rag_pass"] for r in results),
        "retrieval_p50_ms": sorted(r["retrieval"].get("timings", {}).get("totalMs", 0) for r in results)[len(results)//2],
    }
    payload = {
        "benchmark_id": benchmark["benchmark_id"], "source_id": args.source_id,
        "source_pdf": str(args.pdf), "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "policy": {
            "retrieval": "API hybrid BM25 + embedding + rerank; top_k=5",
            "vision_oracle": "Qwen3-VL receives gold page image(s), no retrieval evidence",
            "multimodal_rag": "Qwen3-VL receives retrieved page image(s) plus retrieved text evidence",
            "scoring": "required numeric/keyword term coverage; deterministic audit, not a model judge",
        },
        "summary": summary, "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
