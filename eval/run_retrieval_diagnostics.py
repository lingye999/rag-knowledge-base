"""生成检索链路诊断报告，定位证据在哪个阶段丢失。"""
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
from config import config as _cfg
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import Retriever
from src.services.embedding import EmbeddingService
from src.services.ingestion import IngestionService
from src.vector_store.faiss_store import FaissVectorStore

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATASET_DIR = ROOT / "eval" / "datasets"
REPORT_DIR = ROOT / "eval" / "reports"
DATASET_PATHS = {
    "smoke": DATASET_DIR / "retrieval_smoke.jsonl",
    "dev": DATASET_DIR / "retrieval_dev.jsonl",
    "test": DATASET_DIR / "retrieval_test.jsonl",
}
STAGES = ("first_stage", "doc_internal", "expanded", "final")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _selected_queries(path: Path, ids: set[str] | None) -> list[dict]:
    if path.suffix != ".jsonl":
        raise ValueError(f"仅支持 JSONL 格式的数据集：{path}")
    rows = _load_jsonl(path)
    if ids is not None:
        rows = [row for row in rows if row["id"] in ids]
    if not rows:
        raise ValueError("没有选中任何查询")
    return rows


def _load_documents(
    ingestion: IngestionService,
    queries: list[dict],
    only_query_docs: bool,
    allow_ocr: bool,
):
    selected_docs = None
    if only_query_docs:
        selected_docs = {
            query["relevant_doc"] for query in queries
            if query.get("relevant_doc") is not None
        }

    doc_paths = sorted(
        path for path in DATA_DIR.iterdir()
        if path.suffix.lower() in {".pdf", ".docx", ".txt"}
    )
    if selected_docs is not None:
        doc_paths = [path for path in doc_paths if path.name in selected_docs]

    for path in doc_paths:
        started = time.perf_counter()
        try:
            chunks, _ = ingestion.add(str(path), chunk_method="auto", allow_ocr=allow_ocr)
            elapsed = time.perf_counter() - started
            print(f"  OK {path.name} ({chunks} chunks, {elapsed:.1f}s)", flush=True)
        except Exception as exc:
            print(f"  FAIL {path.name}: {exc}", flush=True)


def _ranked(items: list[dict]) -> list[dict]:
    """按检索分数排序，保证阶段指标和最终排名口径一致。"""
    return sorted(items, key=lambda item: item.get("score", 0.0), reverse=True)


def _stage_top_k(stage: str, hits: list[dict], final_top_k: int) -> int:
    """阶段池指标看完整池，final 只看最终 top_k。"""
    if stage == "final":
        return final_top_k
    return max(len(hits), 1)


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


def _clip(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def _format_score(score) -> str:
    return f"{score:.4f}" if isinstance(score, (int, float)) else "-"


def _diagnose_positive(stage_scores: dict[str, dict]) -> str:
    """正样本失败/命中归因。"""
    if stage_scores["final"]["hit"]:
        return "final_hit"
    if not stage_scores["first_stage"]["document_hit"]:
        return "document_not_recalled"
    if stage_scores["first_stage"]["hit"]:
        return "final_selector_dropped"
    if stage_scores["doc_internal"]["hit"]:
        return "doc_internal_recovered_but_final_lost"
    if stage_scores["expanded"]["hit"]:
        return "neighbor_recovered_but_final_lost"
    if stage_scores["expanded"]["document_hit"]:
        return "evidence_missing_inside_recalled_doc"
    return "evidence_not_found"


def _diagnose_negative(stage_scores: dict[str, dict]) -> str:
    """负样本失败/安全归因。"""
    for stage in STAGES:
        if stage_scores[stage]["unsafe_hits"]:
            if stage == "final":
                return "unsafe_evidence_in_final"
            return f"unsafe_filtered_before_final_from_{stage}"
    if stage_scores["final"]["hit"]:
        return "negative_safe"
    return "negative_unsafe_unknown"


def diagnose_trace(query: dict, trace: dict, top_k: int) -> dict:
    """把 retriever trace 转成单题诊断记录。"""
    stage_hits = {
        "first_stage": _ranked(trace.get("first_stage", [])),
        "doc_internal": _ranked(trace.get("doc_internal", [])),
        "expanded": _ranked(trace.get("expanded", [])),
        "final": trace.get("final", []),
    }
    stage_scores = {}
    for stage, hits in stage_hits.items():
        stage_scores[stage] = evaluate_retrieval(
            query,
            hits,
            _stage_top_k(stage, hits, top_k),
        )

    positive = query.get("relevant_doc") is not None
    diagnosis = (
        _diagnose_positive(stage_scores)
        if positive else _diagnose_negative(stage_scores)
    )
    return {
        "stage_hits": stage_hits,
        "stage_scores": stage_scores,
        "diagnosis": diagnosis,
    }


def _stage_summary(records: list[dict]) -> dict[str, dict]:
    summary = {}
    positives = [record for record in records if record["positive"]]
    for stage in STAGES:
        scores = [record["stages"][stage] for record in records]
        positive_scores = [record["stages"][stage] for record in positives]
        hits = sum(1 for score in scores if score["hit"])
        docs = sum(1 for score in scores if score["document_hit"])
        evidence_recall = [
            score["evidence_recall"] for score in positive_scores
            if score["evidence_recall"] is not None
        ]
        precision = [
            score["context_precision"] for score in positive_scores
            if score["context_precision"] is not None
        ]
        summary[stage] = {
            "hit_rate": hits / max(len(scores), 1),
            "document_hit_rate": docs / max(len(scores), 1),
            "evidence_recall": (
                sum(evidence_recall) / len(evidence_recall)
                if evidence_recall else None
            ),
            "context_precision": (
                sum(precision) / len(precision)
                if precision else None
            ),
        }
    return summary


def _group_summary(records: list[dict], field: str) -> dict[str, dict]:
    groups = defaultdict(list)
    for record in records:
        groups[str(record.get(field, "?"))].append(record)
    return {
        group: {
            "total": len(items),
            "final_hits": sum(
                1 for item in items if item["stages"]["final"]["hit"]
            ),
            "expanded_hits": sum(
                1 for item in items if item["stages"]["expanded"]["hit"]
            ),
            "diagnoses": dict(Counter(item["diagnosis"] for item in items)),
        }
        for group, items in sorted(groups.items())
    }


def _build_summary(records: list[dict], started_at: float) -> dict:
    diagnoses = Counter(record["diagnosis"] for record in records)
    positives = [record for record in records if record["positive"]]
    negatives = [record for record in records if not record["positive"]]
    return {
        "total": len(records),
        "positive_total": len(positives),
        "negative_total": len(negatives),
        "elapsed_s": round(time.perf_counter() - started_at, 2),
        "diagnoses": dict(diagnoses),
        "stages": _stage_summary(records),
        "by_difficulty": _group_summary(records, "difficulty"),
        "by_query_type": _group_summary(records, "query_type"),
        "by_parser_mode": _group_summary(records, "parser_mode"),
        "by_doc": _group_summary(records, "doc"),
    }


def _top_examples(stage_hits: dict[str, list[dict]], query: dict, limit: int) -> dict:
    terms = _evidence_terms(query)
    examples = {}
    for stage, hits in stage_hits.items():
        examples[stage] = [
            {
                "rank": rank,
                "index": hit.get("index"),
                "score": hit.get("score"),
                "doc": hit.get("doc"),
                "page": hit.get("page"),
                "recall_stage": hit.get("recall_stage"),
                "fusion_mode": hit.get("fusion_mode"),
                "fusion_profile": hit.get("fusion_profile"),
                "rrf_score": hit.get("rrf_score"),
                "dense_rank": hit.get("dense_rank"),
                "dense_score": hit.get("dense_score"),
                "dense_rrf": hit.get("dense_rrf"),
                "bm25_rank": hit.get("bm25_rank"),
                "bm25_score": hit.get("bm25_score"),
                "bm25_rrf": hit.get("bm25_rrf"),
                "matched_terms": _matched_terms(hit.get("text", ""), terms),
                "text": _clip(hit.get("text", "")),
            }
            for rank, hit in enumerate(hits[:limit], start=1)
        ]
    return examples


def _print_console_report(summary: dict, records: list[dict]):
    print("\n" + "=" * 72)
    print("检索诊断报告")
    print("=" * 72)
    print(f"总查询数：{summary['total']}")
    print(f"正样本：{summary['positive_total']}，负样本：{summary['negative_total']}")
    print("\n阶段命中：")
    for stage in STAGES:
        item = summary["stages"][stage]
        recall = item["evidence_recall"]
        precision = item["context_precision"]
        recall_text = "-" if recall is None else f"{recall:.3f}"
        precision_text = "-" if precision is None else f"{precision:.3f}"
        print(
            f"  {stage:12s} hit={item['hit_rate']:.1%} "
            f"doc={item['document_hit_rate']:.1%} "
            f"recall={recall_text} precision={precision_text}"
        )

    print("\n诊断归因：")
    for name, count in sorted(
        summary["diagnoses"].items(),
        key=lambda item: (-item[1], item[0]),
    ):
        print(f"  {name:36s} {count}")

    misses = [record for record in records if record["diagnosis"] != "final_hit"]
    misses = [
        record for record in misses
        if record["positive"] or record["diagnosis"] != "negative_safe"
    ]
    if misses:
        print("\n需要关注的查询：")
        for record in misses:
            final = record["stages"]["final"]
            expanded = record["stages"]["expanded"]
            print(
                f"  {record['id']} | {record['diagnosis']} | "
                f"final_hit={final['hit']} expanded_hit={expanded['hit']} | "
                f"{record['query']}"
            )


def _write_report(path: Path, report: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n诊断报告已写入：{path}")


def _default_report_path(dataset_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPORT_DIR / f"retrieval_diagnostics_{dataset_name}_{timestamp}.json"


def run_diagnostics(
    query_path: Path,
    dataset_name: str,
    ids: set[str] | None = None,
    top_k: int = 5,
    retrieve_top: int | None = None,
    use_reranker: bool = True,
    only_query_docs: bool = False,
    allow_ocr: bool = True,
    show_top: int = 3,
    report_path: Path | None = None,
    fusion_mode: str | None = None,
    reranker_model: str | None = None,
) -> dict:
    """运行检索链路诊断并返回完整报告。"""
    started_at = time.perf_counter()
    print("=" * 72)
    print("RAG 检索链路诊断")
    print("=" * 72)

    print("\n[1/4] 正在加载模型...")
    embedding = EmbeddingService()
    db = FaissVectorStore(embedding.dimension)
    try:
        if not use_reranker:
            raise RuntimeError("已通过 --no-reranker 禁用")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = reranker_model or _cfg["reranker"]["model"]
        reranker = Reranker(
            model_name,
            device=device,
            local_files_only=_cfg["reranker"].get("local_files_only", False),
        )
        if _cfg["reranker"].get("preload", False):
            reranker.preload()
    except Exception as exc:
        reranker = None
        print(f"  已跳过精排器：{exc}")

    retriever = Retriever(db, reranker=reranker)
    ingestion = IngestionService(embedding, db, retriever)

    print("\n[2/4] 正在加载查询集...")
    queries = _selected_queries(query_path, ids)
    print(f"  查询数：{len(queries)}")

    print("\n[3/4] 正在导入文档...")
    _load_documents(
        ingestion,
        queries=queries,
        only_query_docs=only_query_docs,
        allow_ocr=allow_ocr,
    )
    if db.count == 0:
        raise RuntimeError("没有成功导入任何文档，无法生成诊断报告")
    print(f"  Chunk 总数：{db.count}")

    print("\n[4/4] 正在逐题诊断...")
    records = []
    for number, query in enumerate(queries, start=1):
        query_text = query["query"]
        vec = embedding.encode(query_text)
        trace = retriever.search_with_trace(
            query_text,
            vec,
            top_k=top_k,
            retrieve_top=retrieve_top,
            fusion_mode=fusion_mode,
        )
        diagnosis = diagnose_trace(query, trace, top_k)
        stage_scores = diagnosis["stage_scores"]
        record = {
            "id": query["id"],
            "query": query_text,
            "doc": query.get("relevant_doc") or "<negative>",
            "difficulty": query.get("difficulty", "?"),
            "query_type": query.get("query_type", "?"),
            "parser_mode": query.get("parser_mode", "?"),
            "positive": query.get("relevant_doc") is not None,
            "evidence_policy": query.get("evidence_policy", "same_chunk"),
            "expected_terms": _evidence_terms(query),
            "diagnosis": diagnosis["diagnosis"],
            "stage_counts": {
                stage: len(diagnosis["stage_hits"][stage])
                for stage in STAGES
            },
            "stages": stage_scores,
            "examples": _top_examples(diagnosis["stage_hits"], query, show_top),
            "final_selection": trace.get("final_selector", {}),
        }
        records.append(record)
        status = "OK" if stage_scores["final"]["hit"] else "CHECK"
        print(
            f"  [{number:02d}/{len(queries):02d}] {status:5s} "
            f"{query['id']} | {record['diagnosis']}"
        )

    summary = _build_summary(records, started_at)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_name,
        "query_path": str(query_path),
        "top_k": top_k,
        "retrieve_top": retrieve_top,
        "use_reranker": use_reranker,
        "reranker_model": reranker_model or _cfg["reranker"]["model"],
        "only_query_docs": only_query_docs,
        "allow_ocr": allow_ocr,
        "fusion_mode": fusion_mode or "config",
        "summary": summary,
        "records": records,
    }
    _print_console_report(summary, records)
    if report_path is not None:
        _write_report(report_path, report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成 RAG 检索链路诊断报告")
    parser.add_argument("--dataset", choices=sorted(DATASET_PATHS), default="smoke",
                        help="选择 smoke、dev 或 test 数据集")
    parser.add_argument("--queries", type=Path,
                        help="使用指定 JSONL 路径，覆盖 --dataset")
    parser.add_argument("--ids", help="要执行的查询 ID，多个 ID 用英文逗号分隔")
    parser.add_argument("--top", type=int, default=5,
                        help="最终上下文 top_k")
    parser.add_argument("--retrieve-top", type=int, default=None,
                        help="首轮内部候选池大小，默认使用配置")
    parser.add_argument(
        "--fusion-mode",
        choices=("dense_only", "bm25_only", "rrf_equal", "rrf_weighted"),
        help="覆盖配置中的首轮融合模式，用于消融对比",
    )
    parser.add_argument("--no-reranker", action="store_true",
                        help="跳过 Cross-Encoder 精排")
    parser.add_argument(
        "--reranker-model",
        help="本地 Cross-Encoder 模型路径；不传则使用配置中的 reranker.model",
    )
    parser.add_argument("--only-query-docs", action="store_true",
                        help="仅导入所选查询引用的文档")
    parser.add_argument("--no-ocr-fallback", action="store_true",
                        help="导入时不启用 OCR 回退")
    parser.add_argument("--show-top", type=int, default=3,
                        help="报告中每阶段保留的样例 chunk 数")
    parser.add_argument("--report", type=Path,
                        help="JSON 报告输出路径；不传则只打印控制台摘要")
    parser.add_argument("--default-report", action="store_true",
                        help="写入 eval/reports 下的默认时间戳报告")
    return parser


def main():
    args = _build_parser().parse_args()
    ids = set(args.ids.split(",")) if args.ids else None
    query_path = args.queries or DATASET_PATHS[args.dataset]
    dataset_name = args.dataset if args.queries is None else query_path.stem
    report_path = args.report
    if report_path is None and args.default_report:
        report_path = _default_report_path(dataset_name)
    run_diagnostics(
        query_path=query_path,
        dataset_name=dataset_name,
        ids=ids,
        top_k=args.top,
        retrieve_top=args.retrieve_top,
        use_reranker=not args.no_reranker,
        only_query_docs=args.only_query_docs,
        allow_ocr=not args.no_ocr_fallback,
        show_top=args.show_top,
        report_path=report_path,
        fusion_mode=args.fusion_mode,
        reranker_model=args.reranker_model,
    )


if __name__ == "__main__":
    main()
