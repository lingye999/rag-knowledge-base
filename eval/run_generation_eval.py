"""运行生成侧 RAG 评测，并保存每题的上下文、答案和可追溯判定结果。

用法：
    python eval/run_generation_eval.py --dataset smoke --ids qa_smoke_evac_001
    python eval/run_generation_eval.py --dataset dev --judge none
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import config
from eval.evaluation import (
    evaluate_answer,
    evaluate_citations,
    evaluate_context_coverage,
)
from eval.generation_judge import GenerationJudge


EVAL_DIR = ROOT / "eval"
DATASET_DIR = EVAL_DIR / "datasets"
QA_DATASETS = {
    "smoke": DATASET_DIR / "qa_smoke.jsonl",
    "dev": DATASET_DIR / "qa_dev.jsonl",
    "test": DATASET_DIR / "qa_test.jsonl",
}
RETRIEVAL_DATASETS = {
    "smoke": DATASET_DIR / "retrieval_smoke.jsonl",
    "dev": DATASET_DIR / "retrieval_dev.jsonl",
    "test": DATASET_DIR / "retrieval_test.jsonl",
}
DEFAULT_QA = QA_DATASETS["smoke"]
DEFAULT_RETRIEVAL = RETRIEVAL_DATASETS["smoke"]
ARTIFACT_DIR = EVAL_DIR / "artifacts"


@dataclass(slots=True)
class EvaluationRuntime:
    """一次评测运行所需的模型、检索器和可选裁判。"""

    embedding: Any
    retriever: Any
    ingestion: Any
    llm: Any
    judge: GenerationJudge | None


@dataclass(frozen=True, slots=True)
class EvaluationThresholds:
    """决定单题是否通过的生成侧指标门槛。"""

    faithfulness: float
    relevancy: float


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _select_qa_rows(path: Path, ids: set[str] | None) -> list[dict]:
    rows = _load_jsonl(path)
    if ids is not None:
        rows = [row for row in rows if row["id"] in ids]
    if not rows:
        raise ValueError("没有选中任何 QA 样本")
    return rows


def _load_retrieval_contracts(path: Path, qa_rows: list[dict]) -> dict[str, dict]:
    contracts = {row["id"]: row for row in _load_jsonl(path)}
    missing_links = [
        row["id"] for row in qa_rows
        if row.get("retrieval_id") not in contracts
    ]
    if missing_links:
        raise ValueError(f"QA 样本缺少有效 retrieval_id：{', '.join(missing_links)}")
    return contracts


def _load_documents(ingestion: Any, allow_ocr: bool,
                    retrieval_contracts: dict[str, dict]) -> None:
    data_dir = ROOT / "data"
    referenced_docs = {
        contract["relevant_doc"] for contract in retrieval_contracts.values()
        if contract.get("relevant_doc") is not None
    }
    # A negative test is only meaningful against the whole available corpus.
    if any(contract.get("relevant_doc") is None
           for contract in retrieval_contracts.values()):
        paths = sorted(
            path for path in data_dir.iterdir()
            if path.suffix.lower() in {".pdf", ".docx", ".txt"}
        )
    else:
        paths = [data_dir / doc for doc in sorted(referenced_docs)]
    for path in paths:
        try:
            started = time.perf_counter()
            chunks, _ = ingestion.add(
                str(path), chunk_method="auto", allow_ocr=allow_ocr
            )
            elapsed = time.perf_counter() - started
            print(f"已导入 {path.name}：{chunks} 个 chunk，耗时 {elapsed:.1f} 秒")
        except Exception as exc:
            print(f"导入失败 {path.name}：{exc}")


def _contexts_from_hits(hits: list[dict]) -> list[dict]:
    return [
        {
            "id": f"chunk-{hit.get('index')}",
            "index": hit.get("index"),
            "rank": rank,
            "doc": hit.get("doc", ""),
            "page": hit.get("page"),
            "score": hit.get("score"),
            "text": hit.get("text", ""),
        }
        for rank, hit in enumerate(hits, start=1)
    ]


def _build_runtime(use_reranker: bool, judge_mode: str) -> EvaluationRuntime:
    # 延迟导入大模型和 PDF 依赖，使 --help 和模块导入保持轻量。
    import torch

    from src.retrieval.reranker import Reranker
    from src.retrieval.retriever import Retriever
    from src.services.embedding import EmbeddingService
    from src.services.ingestion import IngestionService
    from src.services.llm_service import LLMService
    from src.vector_store.faiss_store import FaissVectorStore

    embedding_cfg = config["embedding"]
    embedding = EmbeddingService(
        model_name=embedding_cfg["model"],
        device=embedding_cfg["device"],
    )
    db = FaissVectorStore(embedding.dimension)
    reranker = None
    if use_reranker:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            reranker = Reranker(config["reranker"]["model"], device=device)
        except Exception as exc:
            print(f"精排器不可用，使用基础检索：{exc}")

    llm_cfg = config["llm"]
    llm = LLMService(
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        model=llm_cfg["model"],
        base_url=llm_cfg["base_url"],
    )
    retriever = Retriever(db, reranker=reranker)
    ingestion = IngestionService(embedding, db, retriever)
    return EvaluationRuntime(
        embedding=embedding,
        retriever=retriever,
        ingestion=ingestion,
        llm=llm,
        judge=GenerationJudge(llm) if judge_mode == "llm" else None,
    )


def _default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ARTIFACT_DIR / f"generation_eval_{timestamp}.jsonl"


def _evaluate_one(
    golden: dict,
    retrieval_contract: dict,
    runtime: EvaluationRuntime,
    top_k: int,
    thresholds: EvaluationThresholds,
) -> dict:
    """完成单题检索、生成、规则评测和可选 LLM 裁判。"""
    started = time.perf_counter()
    query = golden["query"]
    hits = runtime.retriever.search(
        query, runtime.embedding.encode(query), top_k=top_k
    )
    contexts = _contexts_from_hits(hits)
    record = {
        "id": golden["id"],
        "retrieval_id": golden["retrieval_id"],
        "query": query,
        "retrieval_contract": {
            "relevant_doc": retrieval_contract.get("relevant_doc"),
            "evidence": retrieval_contract.get("evidence", []),
        },
        "contexts": contexts,
    }

    try:
        llm_cfg = config["llm"]
        answer = runtime.llm.ask_with_sources(
            query,
            contexts,
            top_k=top_k,
            temperature=llm_cfg["temperature"],
            max_tokens=llm_cfg["max_tokens"],
        )
        answer_check = evaluate_answer(answer, golden)
        citation_check = evaluate_citations(
            answer, contexts, golden["citation_required"]
        )
        record.update({
            "answer": answer,
            "answer_correctness": answer_check,
            "context_coverage": evaluate_context_coverage(golden, contexts),
            "citation_validation": citation_check,
        })

        judge_passed = True
        if runtime.judge is not None:
            faithfulness = runtime.judge.judge_faithfulness(answer, contexts)
            relevancy = runtime.judge.judge_answer_relevancy(query, answer)
            record.update({
                "faithfulness": faithfulness,
                "answer_relevancy": relevancy,
                "context_utilization": runtime.judge.judge_context_utilization(
                    query, answer, contexts
                ),
            })
            judge_passed = (
                faithfulness["score"] >= thresholds.faithfulness
                and relevancy["score"] >= thresholds.relevancy
            )

        record["passed"] = (
            answer_check["passed"]
            and citation_check["passed"]
            and judge_passed
        )
    except Exception as exc:
        record.update({"error": str(exc), "passed": False})

    record["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return record


def _mean(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def _metric_values(results: list[dict], group: str, metric: str) -> list[float | None]:
    return [record.get(group, {}).get(metric) for record in results]


def _summarize(results: list[dict], output_path: Path) -> dict:
    """汇总规则指标和 LLM 裁判指标；未启用裁判时相关指标为 null。"""
    return {
        "output": str(output_path),
        "total": len(results),
        "passed": sum(record["passed"] for record in results),
        "answer_correctness": _mean(
            _metric_values(results, "answer_correctness", "required_fact_ratio")
        ),
        "context_coverage": _mean(
            _metric_values(results, "context_coverage", "required_fact_ratio")
        ),
        "citation_precision": _mean(
            _metric_values(results, "citation_validation", "citation_precision")
        ),
        "faithfulness": _mean(_metric_values(results, "faithfulness", "score")),
        "answer_relevancy": _mean(
            _metric_values(results, "answer_relevancy", "score")
        ),
        "context_utilization": _mean(
            _metric_values(results, "context_utilization", "score")
        ),
    }


def run_generation_eval(
    qa_path: Path = DEFAULT_QA,
    retrieval_path: Path = DEFAULT_RETRIEVAL,
    output_path: Path | None = None,
    ids: set[str] | None = None,
    top_k: int = 5,
    judge_mode: str = "llm",
    use_reranker: bool = True,
    allow_ocr: bool = True,
    faithfulness_threshold: float = 0.9,
    relevancy_threshold: float = 0.8,
) -> tuple[list[dict], dict]:
    """执行完整生成评测，并返回逐题结果和汇总指标。"""
    qa_rows = _select_qa_rows(qa_path, ids)
    retrieval_contracts = _load_retrieval_contracts(retrieval_path, qa_rows)
    runtime = _build_runtime(use_reranker, judge_mode)

    print("正在导入评测文档...")
    _load_documents(
        runtime.ingestion,
        allow_ocr=allow_ocr,
        retrieval_contracts=retrieval_contracts,
    )
    if runtime.retriever.db.count == 0:
        raise RuntimeError("没有成功导入任何文档，无法执行生成评测")

    output_path = output_path or _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds = EvaluationThresholds(
        faithfulness=faithfulness_threshold,
        relevancy=relevancy_threshold,
    )

    results = []
    with output_path.open("w", encoding="utf-8") as handle:
        for number, golden in enumerate(qa_rows, start=1):
            record = _evaluate_one(
                golden,
                retrieval_contracts[golden["retrieval_id"]],
                runtime,
                top_k,
                thresholds,
            )
            results.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            status = "通过" if record["passed"] else "失败"
            print(f"[{number:02d}/{len(qa_rows):02d}] {status} {golden['id']}")

    return results, _summarize(results, output_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行生成侧 RAG 质量评测")
    parser.add_argument("--dataset", choices=sorted(QA_DATASETS), default="smoke",
                        help="选择 smoke、dev 或 test 数据集")
    parser.add_argument("--qa", type=Path, help="QA 黄金集路径，覆盖 --dataset")
    parser.add_argument("--retrieval", type=Path,
                        help="检索证据集路径，覆盖 --dataset")
    parser.add_argument("--output", type=Path, help="结果 JSONL 输出路径")
    parser.add_argument("--ids", help="要执行的 QA ID，多个 ID 用英文逗号分隔")
    parser.add_argument("--top", type=int, default=5, help="送入模型的上下文数量")
    parser.add_argument("--judge", choices=["llm", "none"], default="llm",
                        help="是否运行 LLM 裁判，none 仅执行确定性规则")
    parser.add_argument("--no-reranker", action="store_true", help="跳过精排器")
    parser.add_argument("--no-ocr", action="store_true", help="不启用 OCR 回退")
    parser.add_argument("--faithfulness-threshold", type=float, default=0.9,
                        help="忠实度通过阈值")
    parser.add_argument("--relevancy-threshold", type=float, default=0.8,
                        help="答案相关性通过阈值")
    return parser


def _print_summary(summary: dict) -> None:
    print("\n生成侧评测汇总：")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def main() -> None:
    args = _build_parser().parse_args()
    ids = set(args.ids.split(",")) if args.ids else None
    qa_path = args.qa or QA_DATASETS[args.dataset]
    retrieval_path = args.retrieval or RETRIEVAL_DATASETS[args.dataset]
    _, summary = run_generation_eval(
        qa_path=qa_path,
        retrieval_path=retrieval_path,
        output_path=args.output,
        ids=ids,
        top_k=args.top,
        judge_mode=args.judge,
        use_reranker=not args.no_reranker,
        allow_ocr=not args.no_ocr,
        faithfulness_threshold=args.faithfulness_threshold,
        relevancy_threshold=args.relevancy_threshold,
    )
    _print_summary(summary)


if __name__ == "__main__":
    main()
