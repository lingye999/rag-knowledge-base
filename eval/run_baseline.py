"""Run retrieval baseline on the structured retrieval dataset.

Usage:
    python eval/run_baseline.py
    python eval/run_baseline.py --top 3
    python eval/run_baseline.py --queries eval/retrieval_queries.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.embedding import EmbeddingService
from src.ingestion import IngestionService
from src.retriever import Retriever
from src.reranker import Reranker
from src.vector_store.faiss_store import FaissVectorStore
from eval.evaluation import evaluate_retrieval, normalize


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_QUERIES = ROOT / "eval" / "retrieval_queries.jsonl"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_queries(path: Path) -> list[dict]:
    if path.suffix != ".jsonl":
        raise ValueError(f"Only JSONL retrieval datasets are supported: {path}")
    return _load_jsonl(path)


def _print_group_report(results: list[dict], field: str):
    groups = defaultdict(list)
    for result in results:
        groups[result[field]].append(result)

    print(f"\nBy {field}:")
    for key in sorted(groups, key=lambda x: str(x)):
        group = groups[key]
        hits = sum(1 for r in group if r["hit"])
        print(f"  {str(key):18s} {hits:2d}/{len(group):2d} ({hits / len(group):.1%})")


def _clip(text: str, limit: int = 240) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _score(hit: dict) -> str:
    score = hit.get("score")
    if isinstance(score, (int, float)):
        return f"{score:.4f}"
    return "-"


def _evidence_terms(query: dict) -> list[str]:
    if query.get("relevant_doc") is None:
        return query.get("forbidden_evidence", [])
    terms = []
    for group in query.get("evidence", []):
        terms.extend(group.get("must_contain", []))
    return terms


def _matched_terms(text: str, terms: list[str]) -> list[str]:
    normalized = normalize(text)
    return [term for term in terms if normalize(term) in normalized]


def _print_match_details(query: dict, hits: list[dict], score: dict,
                         top_k: int, show_top: int):
    terms = _evidence_terms(query)
    print(f"      expected_doc={query.get('relevant_doc') or '<negative>'}")
    print(f"      expected_terms={terms}")
    print(f"      policy={score.get('evidence_policy', 'same_chunk')}")

    if query.get("relevant_doc") is None:
        if score["unsafe_hits"]:
            print(f"      unsafe_hits={score['unsafe_hits']}")
        else:
            print("      safe=true forbidden evidence not found")
    elif score["evidence_rank"]:
        hit = hits[score["evidence_rank"] - 1]
        print(
            f"      evidence_hit rank={score['evidence_rank']} "
            f"score={_score(hit)} doc={hit.get('doc', '?')}"
        )
        print(f"      matched={score['matched']}")
        if score.get("supporting_hits"):
            print(f"      supporting_hits={score['supporting_hits']}")
        if score.get("evidence_policy", "same_chunk") == "same_chunk":
            print(f"      chunk={_clip(hit.get('text', ''))}")
    else:
        print(
            f"      evidence_hit=false document_hit={score['document_hit']} "
            f"matched={score['matched']}"
        )

    if show_top <= 0:
        return

    print(f"      top_results:")
    for rank, hit in enumerate(hits[:min(show_top, top_k)], start=1):
        matched = _matched_terms(hit.get("text", ""), terms)
        print(
            f"        #{rank} score={_score(hit)} doc={hit.get('doc', '?')} "
            f"matched={matched}"
        )
        print(f"           {_clip(hit.get('text', ''), 180)}")


def _load_all_documents(ingestion: IngestionService,
                        selected_docs: set[str] | None = None,
                        allow_ocr: bool = True):
    doc_paths = sorted(
        path for path in DATA_DIR.iterdir()
        if path.suffix.lower() in {".pdf", ".docx", ".txt"}
    )
    if selected_docs is not None:
        doc_paths = [path for path in doc_paths if path.name in selected_docs]

    for path in doc_paths:
        try:
            t0 = time.time()
            chunks, _ = ingestion.add(
                str(path), chunk_method="auto", allow_ocr=allow_ocr
            )
            elapsed = time.time() - t0
            print(f"  OK {path.name} ({chunks} chunks, {elapsed:.1f}s)", flush=True)
        except Exception as exc:
            print(f"  FAIL {path.name}: {exc}", flush=True)


def run_baseline(top_k: int = 5, query_path: Path = DEFAULT_QUERIES,
                 ids: set[str] | None = None, show_matches: bool = False,
                 show_top: int = 0, use_reranker: bool = True,
                 only_query_docs: bool = False, allow_ocr: bool = True):
    print("=" * 72)
    print("RAG retrieval baseline")
    print("=" * 72)

    print("\n[1/4] Loading models...")
    emb = EmbeddingService()
    db = FaissVectorStore(emb.dimension)

    try:
        if not use_reranker:
            raise RuntimeError("disabled by --no-reranker")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        reranker = Reranker("BAAI/bge-reranker-base", device=device)
    except Exception as exc:
        reranker = None
        print(f"  Reranker skipped: {exc}")

    retriever = Retriever(db, reranker=reranker)
    ingestion = IngestionService(emb, db, retriever)

    print("\n[2/4] Loading queries...")
    queries = _load_queries(query_path)
    if ids is not None:
        queries = [query for query in queries if query["id"] in ids]
    if not queries:
        print("No queries selected.")
        return
    docs = Counter(q.get("relevant_doc") for q in queries)
    print(f"  Queries: {len(queries)}")
    print(f"  Covered docs: {len(docs)}")

    print("\n[3/4] Ingesting documents...")
    selected_docs = None
    if only_query_docs:
        selected_docs = {
            query["relevant_doc"] for query in queries
            if query.get("relevant_doc") is not None
        }
    _load_all_documents(
        ingestion, selected_docs=selected_docs, allow_ocr=allow_ocr
    )
    if db.count == 0:
        print("No documents were ingested.")
        return
    print(f"  Total chunks: {db.count}")

    print(f"\n[4/4] Searching top_k={top_k}...\n")
    results = []
    for idx, query in enumerate(queries, start=1):
        query_text = query["query"]
        t0 = time.time()
        vec = emb.encode(query_text)
        hits = retriever.search(query_text, vec, top_k=top_k)
        elapsed_ms = (time.time() - t0) * 1000

        score = evaluate_retrieval(query, hits, top_k)
        result = {
            "id": query["id"],
            "query": query_text[:60],
            "doc": query.get("relevant_doc") or "<negative>",
            "difficulty": query.get("difficulty", "?"),
            "query_type": query.get("query_type", "?"),
            "parser_mode": query.get("parser_mode", "?"),
            "hit": score["hit"],
            "document_hit": score["document_hit"],
            "evidence_rank": score["evidence_rank"],
            "matched": score["matched"],
            "evidence_policy": score["evidence_policy"],
            "supporting_hits": score["supporting_hits"],
            "unsafe_hits": score["unsafe_hits"],
            "time_ms": elapsed_ms,
        }
        results.append(result)

        status = "OK" if score["hit"] else "MISS"
        print(f"  [{idx:02d}/{len(queries):02d}] {status:4s} {query['id']} | {result['query']}")
        if show_matches:
            _print_match_details(query, hits, score, top_k, show_top)

    total = len(results)
    hit_count = sum(1 for r in results if r["hit"])
    document_hit_count = sum(1 for r in results if r["document_hit"])
    positive_scores = [r for r in results if r["doc"] != "<negative>"]
    negative_scores = [r for r in results if r["doc"] == "<negative>"]
    avg_time = sum(r["time_ms"] for r in results) / max(total, 1)

    print("\n" + "=" * 72)
    print("Report")
    print("=" * 72)
    print(f"Total:     {total}")
    print(f"Hit:       {hit_count} ({hit_count / max(total, 1):.1%})")
    print(f"Doc hit:   {document_hit_count} ({document_hit_count / max(total, 1):.1%})")
    if positive_scores:
        mrr = sum(1 / r["evidence_rank"] for r in positive_scores
                  if r["evidence_rank"]) / len(positive_scores)
        print(f"Positive MRR: {mrr:.4f}")
    if negative_scores:
        safe = sum(1 for r in negative_scores if r["hit"])
        print(f"Negative safe: {safe}/{len(negative_scores)} ({safe / len(negative_scores):.1%})")
    print(f"Miss:      {total - hit_count}")
    print(f"Avg time:  {avg_time:.0f}ms")

    _print_group_report(results, "difficulty")
    _print_group_report(results, "query_type")
    _print_group_report(results, "parser_mode")
    _print_group_report(results, "doc")

    failures = [r for r in results if not r["hit"]]
    if failures:
        print("\nMissed queries:")
        for failure in failures:
            print(f"  [{failure['difficulty']}/{failure['query_type']}] {failure['id']}: {failure['query']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--ids", help="Comma-separated query IDs to run")
    parser.add_argument("--show-matches", action="store_true",
                        help="Print matched evidence chunks")
    parser.add_argument("--show-top", type=int, default=0,
                        help="Also print the first N retrieved chunks")
    parser.add_argument("--no-reranker", action="store_true",
                        help="Skip Cross-Encoder reranking")
    parser.add_argument("--only-query-docs", action="store_true",
                        help="Ingest only documents referenced by selected queries")
    parser.add_argument("--no-ocr-fallback", action="store_true",
                        help="Do not activate OCR fallback during ingestion")
    args = parser.parse_args()
    selected_ids = set(args.ids.split(",")) if args.ids else None
    run_baseline(
        top_k=args.top,
        query_path=args.queries,
        ids=selected_ids,
        show_matches=args.show_matches,
        show_top=args.show_top,
        use_reranker=not args.no_reranker,
        only_query_docs=args.only_query_docs,
        allow_ocr=not args.no_ocr_fallback,
    )


if __name__ == "__main__":
    main()
