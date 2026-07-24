"""Run retrieval baseline evaluation on the structured dataset."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from eval.evaluation import evaluate_retrieval, normalize
from config import config
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import Retriever
from src.services.embedding import EmbeddingService
from src.services.ingestion import IngestionService
from src.vector_store.faiss_store import FaissVectorStore

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATASET_DIR = ROOT / "eval" / "datasets"
DATASET_PATHS = {
    "smoke": DATASET_DIR / "retrieval_smoke.jsonl",
    "dev": DATASET_DIR / "retrieval_dev.jsonl",
    "test": DATASET_DIR / "retrieval_test.jsonl",
}
DEFAULT_QUERIES = DATASET_PATHS["smoke"]

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
        raise ValueError(f"仅支持 JSONL 格式的数据集：{path}")
    return _load_jsonl(path)


def _print_group_report(results: list[dict], field: str):
    groups = defaultdict(list)
    for result in results:
        groups[result[field]].append(result)

    print(f"\n按 {field} 分组：")
    for key in sorted(groups, key=lambda x: str(x)):
        group = groups[key]
        hits = sum(1 for r in group if r["hit"])
        candidate_hits = sum(1 for r in group if r["candidate_hit"])
        print(
            f"  {str(key):18s} {hits:2d}/{len(group):2d} ({hits / len(group):.1%})"
            f" | pool {candidate_hits:2d}/{len(group):2d}"
        )
        positives = [result for result in group if result["positive"]]
        if positives:
            print(
                f"{' ' * 21}Recall@K={_mean(positives, 'evidence_recall'):.3f} "
                f"PoolRecall={_mean(positives, 'candidate_evidence_recall'):.3f} "
                f"Precision@K={_mean(positives, 'context_precision'):.3f} "
                f"PoolPrecision={_mean(positives, 'candidate_context_precision'):.3f}"
            )


def _mean(results: list[dict], field: str) -> float:
    values = [result[field] for result in results if result[field] is not None]
    return sum(values) / len(values) if values else 0.0


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


def _print_match_details(
    query: dict,
    hits: list[dict],
    score: dict,
    top_k: int,
    show_top: int,
):
    terms = _evidence_terms(query)
    print(f"      expected_doc={query.get('relevant_doc') or '<negative>'}")
    print(f"      expected_terms={terms}")
    print(f"      policy={score.get('evidence_policy', 'same_chunk')}")
    if score["positive"]:
        print(
            f"      evidence_recall@{top_k}={score['evidence_recall']:.3f} "
            f"context_precision@{top_k}={score['context_precision']:.3f} "
            f"mrr={score['mrr']:.3f} ndcg={score['ndcg']:.3f}"
        )

    if query.get("relevant_doc") is None:
        if score["unsafe_hits"]:
            print(f"      unsafe_hits={score['unsafe_hits']}")
        else:
            print("      safe=true，未找到禁用证据")
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

    print("      top_results:")
    for rank, hit in enumerate(hits[:min(show_top, top_k)], start=1):
        matched = _matched_terms(hit.get("text", ""), terms)
        print(
            f"        #{rank} score={_score(hit)} doc={hit.get('doc', '?')} "
            f"page={hit.get('page', '?')} matched={matched}"
        )
        print(f"           {_clip(hit.get('text', ''), 180)}")


def _load_all_documents(
    ingestion: IngestionService,
    selected_docs: set[str] | None = None,
    allow_ocr: bool = True,
):
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


def run_baseline(
    top_k: int = 5,
    retrieve_top: int | None = None,
    query_path: Path = DEFAULT_QUERIES,
    ids: set[str] | None = None,
    show_matches: bool = False,
    show_top: int = 0,
    use_reranker: bool = True,
    only_query_docs: bool = False,
    allow_ocr: bool = True,
    report_path: Path | None = None,
):
    print("=" * 72)
    print("RAG 检索基线评测")
    print("=" * 72)

    print("\n[1/4] 正在加载模型...")
    emb = EmbeddingService()
    db = FaissVectorStore(emb.dimension)

    try:
        if not use_reranker:
            raise RuntimeError("已通过 --no-reranker 禁用")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        reranker_config = config["reranker"]
        reranker = Reranker(
            reranker_config["model"],
            device=device,
            local_files_only=reranker_config.get("local_files_only", False),
        )
        if reranker_config.get("preload", False):
            reranker.preload()
    except Exception as exc:
        reranker = None
        print(f"  已跳过精排器：{exc}")

    retriever = Retriever(db, reranker=reranker)
    ingestion = IngestionService(emb, db, retriever)

    print("\n[2/4] 正在加载查询集...")
    queries = _load_queries(query_path)
    if ids is not None:
        queries = [query for query in queries if query["id"] in ids]
    if not queries:
        print("未选中任何查询。")
        return
    docs = Counter(q.get("relevant_doc") for q in queries)
    print(f"  查询数：{len(queries)}")
    print(f"  覆盖文档数：{len(docs)}")

    print("\n[3/4] 正在导入文档...")
    selected_docs = None
    if only_query_docs:
        selected_docs = {
            query["relevant_doc"] for query in queries
            if query.get("relevant_doc") is not None
        }
    _load_all_documents(
        ingestion,
        selected_docs=selected_docs,
        allow_ocr=allow_ocr,
    )
    if db.count == 0:
        print("没有成功导入任何文档。")
        return
    print(f"  Chunk 总数：{db.count}")

    pool_label = retrieve_top if retrieve_top is not None else "config"
    print(f"\n[4/4] 正在检索 top_k={top_k}, retrieve_top={pool_label}...\n")
    results = []
    for idx, query in enumerate(queries, start=1):
        query_text = query["query"]
        t0 = time.time()
        vec = emb.encode(query_text)
        trace = retriever.search_with_trace(
            query_text,
            vec,
            top_k=top_k,
            retrieve_top=retrieve_top,
        )
        hits = trace["final"]
        # Neighbor expansion preserves insertion order. Candidate metrics such
        # as MRR and nDCG must inspect the same score-ranked order as retrieval.
        candidate_hits = sorted(
            trace["candidates"],
            key=lambda item: item.get("score", 0.0),
            reverse=True,
        )
        elapsed_ms = (time.time() - t0) * 1000

        score = evaluate_retrieval(query, hits, top_k)
        candidate_k = max(len(candidate_hits), 1)
        candidate_score = evaluate_retrieval(query, candidate_hits, candidate_k)
        result = {
            "id": query["id"],
            "query": query_text[:60],
            "doc": query.get("relevant_doc") or "<negative>",
            "difficulty": query.get("difficulty", "?"),
            "query_type": query.get("query_type", "?"),
            "parser_mode": query.get("parser_mode", "?"),
            "positive": score["positive"],
            "candidate_positive": candidate_score["positive"],
            "hit": score["hit"],
            "candidate_hit": candidate_score["hit"],
            "document_hit": score["document_hit"],
            "candidate_document_hit": candidate_score["document_hit"],
            "evidence_rank": score["evidence_rank"],
            "candidate_evidence_rank": candidate_score["evidence_rank"],
            "evidence_recall": score["evidence_recall"],
            "candidate_evidence_recall": candidate_score["evidence_recall"],
            "context_precision": score["context_precision"],
            "candidate_context_precision": candidate_score["context_precision"],
            "mrr": score["mrr"],
            "candidate_mrr": candidate_score["mrr"],
            "ndcg": score["ndcg"],
            "candidate_ndcg": candidate_score["ndcg"],
            "matched": score["matched"],
            "candidate_matched": candidate_score["matched"],
            "evidence_policy": score["evidence_policy"],
            "supporting_hits": score["supporting_hits"],
            "candidate_supporting_hits": candidate_score["supporting_hits"],
            "relevant_contexts": score["relevant_contexts"],
            "candidate_relevant_contexts": candidate_score["relevant_contexts"],
            "unsafe_hits": score["unsafe_hits"],
            "candidate_unsafe_hits": candidate_score["unsafe_hits"],
            "candidate_count": len(candidate_hits),
            "first_stage_count": len(trace["first_stage"]),
            "doc_internal_count": len(trace["doc_internal"]),
            "expanded_count": len(trace["expanded"]),
            "time_ms": elapsed_ms,
        }
        results.append(result)

        if score["hit"]:
            status = "OK"
        elif candidate_score["hit"]:
            status = "POOL"
        else:
            status = "MISS"
        print(
            f"  [{idx:02d}/{len(queries):02d}] {status:4s} "
            f"{query['id']} | {result['query']}"
        )
        if show_matches:
            _print_match_details(query, hits, score, top_k, show_top)
            if not score["hit"] and candidate_score["hit"]:
                print("      evidence found in candidate pool but not final context")
                _print_match_details(
                    query,
                    candidate_hits,
                    candidate_score,
                    len(candidate_hits),
                    show_top,
                )

    total = len(results)
    hit_count = sum(1 for r in results if r["hit"])
    candidate_hit_count = sum(1 for r in results if r["candidate_hit"])
    document_hit_count = sum(1 for r in results if r["document_hit"])
    candidate_document_hit_count = sum(
        1 for r in results if r["candidate_document_hit"]
    )
    positive_scores = [r for r in results if r["doc"] != "<negative>"]
    negative_scores = [r for r in results if r["doc"] == "<negative>"]
    avg_time = sum(r["time_ms"] for r in results) / max(total, 1)
    metric_summary = {
        "evidence_recall_at_k": _mean(positive_scores, "evidence_recall"),
        "candidate_evidence_recall": _mean(
            positive_scores, "candidate_evidence_recall"
        ),
        "context_precision_at_k": _mean(positive_scores, "context_precision"),
        "candidate_context_precision": _mean(
            positive_scores, "candidate_context_precision"
        ),
        "mrr": _mean(positive_scores, "mrr"),
        "candidate_mrr": _mean(positive_scores, "candidate_mrr"),
        "ndcg": _mean(positive_scores, "ndcg"),
        "candidate_ndcg": _mean(positive_scores, "candidate_ndcg"),
    }

    print("\n" + "=" * 72)
    print("评测报告")
    print("=" * 72)
    print(f"总数：       {total}")
    print(f"证据命中：   {hit_count} ({hit_count / max(total, 1):.1%})")
    print(
        f"候选池证据命中： {candidate_hit_count} "
        f"({candidate_hit_count / max(total, 1):.1%})"
    )
    print(f"文档命中：   {document_hit_count} ({document_hit_count / max(total, 1):.1%})")
    print(
        f"候选池文档命中： {candidate_document_hit_count} "
        f"({candidate_document_hit_count / max(total, 1):.1%})"
    )
    if positive_scores:
        print(f"Evidence Recall@{top_k}:  {metric_summary['evidence_recall_at_k']:.4f}")
        print(
            f"Candidate Evidence Recall: "
            f"{metric_summary['candidate_evidence_recall']:.4f}"
        )
        print(
            f"Context Precision@{top_k}: "
            f"{metric_summary['context_precision_at_k']:.4f}"
        )
        print(
            f"Candidate Precision:       "
            f"{metric_summary['candidate_context_precision']:.4f}"
        )
        print(f"正样本 MRR：              {metric_summary['mrr']:.4f}")
        print(f"候选池 MRR：              {metric_summary['candidate_mrr']:.4f}")
        print(f"正样本 nDCG@{top_k}:         {metric_summary['ndcg']:.4f}")
        print(f"候选池 nDCG：             {metric_summary['candidate_ndcg']:.4f}")
    if negative_scores:
        safe = sum(1 for r in negative_scores if r["hit"])
        print(f"负样本安全： {safe}/{len(negative_scores)} ({safe / len(negative_scores):.1%})")
    print(f"未命中：     {total - hit_count}")
    print(f"平均耗时：   {avg_time:.0f}ms")

    _print_group_report(results, "difficulty")
    _print_group_report(results, "query_type")
    _print_group_report(results, "parser_mode")
    _print_group_report(results, "doc")

    failures = [r for r in results if not r["hit"]]
    if failures:
        print("\n未命中的查询：")
        for failure in failures:
            print(
                f"  [{failure['difficulty']}/{failure['query_type']}] "
                f"{failure['id']}: {failure['query']}"
            )

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "query_path": str(query_path),
            "top_k": top_k,
            "retrieve_top": retrieve_top,
            "use_reranker": use_reranker,
            "only_query_docs": only_query_docs,
            "summary": {
                "total": total,
                "evidence_hit_rate": hit_count / max(total, 1),
                "candidate_evidence_hit_rate": (
                    candidate_hit_count / max(total, 1)
                ),
                "document_hit_rate": document_hit_count / max(total, 1),
                "candidate_document_hit_rate": (
                    candidate_document_hit_count / max(total, 1)
                ),
                "negative_safe_rate": (
                    sum(1 for r in negative_scores if r["hit"])
                    / len(negative_scores) if negative_scores else None
                ),
                "avg_time_ms": avg_time,
                **metric_summary,
            },
            "results": results,
        }
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"评测报告已写入：{report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--retrieve-top", type=int, default=None)
    parser.add_argument("--dataset", choices=sorted(DATASET_PATHS), default="smoke",
                        help="选择 smoke、dev 或 test 数据集")
    parser.add_argument("--queries", type=Path,
                        help="使用指定 JSONL 路径，覆盖 --dataset")
    parser.add_argument("--ids", help="要执行的查询 ID，多个 ID 用英文逗号分隔")
    parser.add_argument("--show-matches", action="store_true",
                        help="打印命中的证据 chunk")
    parser.add_argument("--show-top", type=int, default=0,
                        help="额外打印前 N 个召回的 chunk")
    parser.add_argument("--no-reranker", action="store_true",
                        help="跳过 Cross-Encoder 精排")
    parser.add_argument("--only-query-docs", action="store_true",
                        help="仅导入所选查询引用的文档。")
    parser.add_argument("--no-ocr-fallback", action="store_true",
                        help="导入时不启用 OCR 回退")
    parser.add_argument("--report", type=Path,
                        help="将逐题指标和汇总写入 JSON 文件")
    args = parser.parse_args()
    selected_ids = set(args.ids.split(",")) if args.ids else None
    query_path = args.queries or DATASET_PATHS[args.dataset]
    run_baseline(
        top_k=args.top,
        retrieve_top=args.retrieve_top,
        query_path=query_path,
        ids=selected_ids,
        show_matches=args.show_matches,
        show_top=args.show_top,
        use_reranker=not args.no_reranker,
        only_query_docs=args.only_query_docs,
        allow_ocr=not args.no_ocr_fallback,
        report_path=args.report,
    )


if __name__ == "__main__":
    main()
