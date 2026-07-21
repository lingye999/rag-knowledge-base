"""消融实验：对比 chunk 参数 × 检索模式

用法:
    python eval/benchmark.py              # 跑全部实验
    python eval/benchmark.py --quick      # 快速模式（只跑一组）
"""
import sys
import os
import json
import time
import jieba
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedding import EmbeddingService
from src.vector_store.faiss_store import FaissVectorStore
from src.retriever import Retriever
from src.ingestion import IngestionService
from eval.metrics import evaluate, aggregate


# ── 实验参数 ──

CHUNK_CONFIGS = [
    # (method, label, extra_args for chunk_text)
    ("sentence", "sentence(5句,0ov)", {}),
    ("sentence", "sentence(8句,1ov)", {}),
    ("jieba",    "jieba(60词,0ov)",  {}),
    ("jieba",    "jieba(120词,20ov)", {}),
    ("paragraph","paragraph",         {}),
    ("size",     "size(200字,50ov)",  {}),
    ("size",     "size(400字,80ov)",  {}),
]

RETRIEVAL_MODES = ["dense_only", "bm25_only", "rrf_full"]

MODES = {
    "dense_only": "纯 Dense",
    "bm25_only": "纯 BM25",
    "rrf_full": "Dense + BM25 (RRF)",
}


# ── 主流程 ──

def run_single_experiment(emb, doc_paths, chunk_method, retrieval_mode,
                          top_k=5):
    """用指定参数跑一次完整实验：入库 → 对每个 query 搜 → 算指标"""
    db = FaissVectorStore(emb.dimension)
    retriever = Retriever(db)
    ingestion = IngestionService(emb, db, retriever)

    # 批量入库
    for path in doc_paths:
        if os.path.exists(path):
            ingestion.add(path, chunk_method=chunk_method)

    if db.count == 0:
        return None, "无数据"

    # 加载测试 queries
    with open(os.path.join(os.path.dirname(__file__), "queries.json"),
              "r", encoding="utf-8") as f:
        queries = json.load(f)

    all_scores = []
    for q in queries:
        vec = emb.encode(q["query"])
        top = top_k * 5 * (5 if retriever.reranker else 1)

        if retrieval_mode == "bm25_only":
            results = _bm25_search(retriever, q["query"], top_k)
        elif retrieval_mode == "dense_only":
            results = _dense_search(db, vec, top_k)
        else:
            results = retriever.search(q["query"], vec, top_k=top_k)

        score = evaluate(results, q, k=top_k)
        all_scores.append(score)

    aggr = aggregate(all_scores)
    return aggr, None


def _dense_search(db, query_vec, top_k):
    """纯 Dense 检索"""
    raw = db.search(query_vec, top_k=top_k)
    return raw


def _bm25_search(retriever, query_text, top_k):
    """纯 BM25 检索"""
    if retriever._bm25 is None:
        return []
    tokens = jieba.lcut(query_text)
    bm25_scores = retriever._bm25.get_scores(tokens)
    top_idx = np.argsort(bm25_scores)[::-1][:top_k]
    results = []
    for idx in top_idx:
        if idx < len(retriever.db.texts):
            results.append({
                "text": retriever.db.texts[idx],
                "score": float(bm25_scores[idx]),
                "index": int(idx),
            })
    return results


# ── 输出 ──

def print_table(title, headers, rows):
    """打印对齐表格"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    row_fmt = "  " + "  ".join(f"{{:<{w}}}" for w in widths)
    print(row_fmt.format(*headers))
    print("  " + "  ".join("-" * w for w in widths))
    for row in rows:
        print(row_fmt.format(*row))
    print()


def run_all(emb):
    """跑全部实验矩阵"""
    doc_paths = []
    for f in ["data/sample.txt", "data/Python入门.docx"]:
        if os.path.exists(f):
            doc_paths.append(f)
    if not doc_paths:
        print("❌ 没有找到测试文档")
        return

    print(f"\n📚 测试文档: {doc_paths}")
    print(f"🧪 实验总数: {len(CHUNK_CONFIGS)} × {len(RETRIEVAL_MODES)} = "
          f"{len(CHUNK_CONFIGS) * len(RETRIEVAL_MODES)} 组\n")

    # 表 1: chunk 对比（都在 RRF 模式下）
    print("⏳ 正在跑 chunk 对比实验...")
    chunk_rows = []
    for method, label, _ in CHUNK_CONFIGS:
        t0 = time.time()
        score, err = run_single_experiment(emb, doc_paths, method, "rrf_full")
        t1 = time.time()
        if err:
            chunk_rows.append([label, "-", "-", "-", err])
        else:
            chunk_rows.append([
                label,
                f"{score['recall@5']:.2%}",
                f"{score['mrr']:.4f}",
                f"{score['ndcg@5']:.4f}",
                f"{(t1-t0):.1f}s",
            ])

    print_table("表1: Chunk 策略对比 (RRF 检索)",
                ["策略", "Recall@5", "MRR", "NDCG@5", "耗时"],
                chunk_rows)

    # 表 2: 检索模式消融（最优 chunk 参数）
    print("⏳ 正在跑检索模式消融...")
    with open(os.path.join(os.path.dirname(__file__), "queries.json"),
              "r", encoding="utf-8") as f:
        queries = json.load(f)

    mode_rows = []
    # 用第一个 chunk config (sentence 5句) 做基础对比
    for mode in RETRIEVAL_MODES:
        t0 = time.time()
        score, err = run_single_experiment(emb, doc_paths, "sentence", mode)
        t1 = time.time()
        mode_rows.append([
            MODES[mode],
            f"{score['recall@5']:.2%}" if score else "-",
            f"{score['mrr']:.4f}" if score else "-",
            f"{score['ndcg@5']:.4f}" if score else "-",
            f"{(t1-t0):.1f}s",
        ])

    print_table("表2: 检索模式消融 (sentence 5句)",
                ["检索模式", "Recall@5", "MRR", "NDCG@5", "耗时"],
                mode_rows)

    # 表 3: 最佳组合推荐
    best = max(chunk_rows, key=lambda r: float(r[1].rstrip('%'))/100
               if r[1] != '-' else 0)
    best_mode = max(mode_rows, key=lambda r: float(r[1].rstrip('%'))/100
                    if r[1] != '-' else 0)

    print(f"{'='*70}")
    print(f"  🏆 推荐配置")
    print(f"{'='*70}")
    print(f"  最佳 chunk: {best[0]}  →  Recall@5: {best[1]}")
    print(f"  最佳检索:   {best_mode[0]}  →  Recall@5: {best_mode[1]}")
    print(f"{'='*70}\n")


def run_quick(emb):
    """快速模式：只跑一组验证流程"""
    doc_paths = []
    for f in ["data/sample.txt", "data/Python入门.docx"]:
        if os.path.exists(f):
            doc_paths.append(f)

    print(f"\n⚡ 快速模式: sentence(8句,1ov) + RRF")
    score, err = run_single_experiment(emb, doc_paths, "sentence", "rrf_full")
    if err:
        print(f"❌ {err}")
    else:
        print(f"  Recall@5: {score['recall@5']:.2%}")
        print(f"  MRR:      {score['mrr']:.4f}")
        print(f"  NDCG@5:   {score['ndcg@5']:.4f}")
        print(f"  Hit Rate: {score['hit_rate']:.2%}")


# ── 入口 ──

if __name__ == "__main__":
    print("=" * 70)
    print("  RAG 消融实验框架 v0.1")
    print("=" * 70)

    emb = EmbeddingService()
    print(f"[Init] Embedding: {emb.dimension}维")

    if "--quick" in sys.argv:
        run_quick(emb)
    else:
        run_all(emb)
