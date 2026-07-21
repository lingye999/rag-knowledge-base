"""检索评估指标：Recall@k, MRR, NDCG

每个指标都是无 LLM 的纯数学计算，基于标注的 must_contain 关键词。
"""

import math


def is_relevant(result_text: str, must_contain: list[str]) -> bool:
    """判断搜索结果是否命中正确答案"""
    return any(keyword in result_text for keyword in must_contain)


def recall_at_k(results: list[dict], must_contain: list[str], k: int) -> float:
    """Recall@k：前 k 条结果中至少有一条命中"""
    for r in results[:k]:
        if is_relevant(r["text"], must_contain):
            return 1.0
    return 0.0


def mrr(results: list[dict], must_contain: list[str]) -> float:
    """MRR (Mean Reciprocal Rank)：第一个正确答案排第几

    MRR = 1 / rank
      rank=1 → MRR=1.0（最佳）
      rank=5 → MRR=0.2
      rank=∞ → MRR=0.0（未找到）
    """
    for rank, r in enumerate(results, start=1):
        if is_relevant(r["text"], must_contain):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(results: list[dict], must_contain: list[str], k: int) -> float:
    """NDCG@k (Normalized Discounted Cumulative Gain)

    考虑排名位置的权重：排在前面的命中更有价值
    """
    # 理想 DCG: 正确答案都排在最前面（位置 1, 2, 3...）
    ideal_hits = min(len(must_contain), k)
    ideal_dcg = 0.0
    for i in range(ideal_hits):
        ideal_dcg += 1.0 / math.log2(i + 2)  # log2(position + 1)

    if ideal_dcg == 0:
        return 0.0

    # 实际 DCG: 按实际排名位置计算
    dcg = 0.0
    rank = 1
    for r in results[:k]:
        if is_relevant(r["text"], must_contain):
            dcg += 1.0 / math.log2(rank + 1)
        rank += 1

    return dcg / ideal_dcg


def evaluate(results: list[dict], query: dict, k: int = 5) -> dict:
    """对一个 query 的结果计算所有指标"""
    mc = query["must_contain"]
    return {
        "recall@5": recall_at_k(results, mc, k),
        "mrr": mrr(results, mc),
        "ndcg@5": ndcg_at_k(results, mc, k),
        "found": any(is_relevant(r["text"], mc) for r in results),
    }


def aggregate(all_scores: list[dict]) -> dict:
    """聚合多个 query 的指标（平均值）"""
    n = len(all_scores)
    if n == 0:
        return {}
    return {
        "recall@5": round(sum(s["recall@5"] for s in all_scores) / n, 4),
        "mrr": round(sum(s["mrr"] for s in all_scores) / n, 4),
        "ndcg@5": round(sum(s["ndcg@5"] for s in all_scores) / n, 4),
        "hit_rate": round(sum(1 for s in all_scores if s["found"]) / n, 4),
    }
