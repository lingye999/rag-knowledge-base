"""检索引擎：多路召回 → 三维度加权 → Cross-Encoder 重排序 → 阈值过滤"""
from .reranker import Reranker


class Retriever:
    """统一检索引擎

    用法:
        retriever = Retriever(db, reranker=reranker)
        results = retriever.search(query_text, query_vec, top_k=5)
    """

    # 三维度权重
    ALPHA = 0.7   # 相关性（Route A chunk 分数）
    BETA = 0.1    # 元数据（Route B 文档排名）
    GAMMA = 0.2   # 文档属性（质量，暂用默认值 0.8）

    RRF_K = 60        # RRF 常数
    CHUNK_DECAY = 0.2 # 多切片几何衰减

    def __init__(self, db, reranker: Reranker | None = None):
        self.db = db
        self.reranker = reranker

    def search(self, query_text: str, query_vec: list[float], top_k: int = 5,
               doc_filter: str | None = None,
               threshold: float = 0.3) -> list[dict]:
        """完整检索流程：多路召回 → 加权 → 精排 → 过滤

        参数:
            query_text: 原始查询文本（供 reranker 使用）
            query_vec: 查询向量（由调用方编码）
            top_k: 返回结果数
            doc_filter: 限定文档名
            threshold: 最终分数下限

        返回:
            [{"text": ..., "score": ..., "doc": ...}, ...]
        """
        # ── 0. 召回阶段：多拿一些候选供精排使用 ──
        expand = 5 if self.reranker else 1   # 有精排时多召回 5 倍
        route_k = top_k * 5 * expand
        route_a = self.db.search(query_vec, top_k=route_k)

        # 如果有文档过滤，先缩小范围
        if doc_filter:
            route_a = [r for r in route_a
                       if self.db.meta[r["index"]].get("doc") == doc_filter]

        if not route_a:
            return []

        # ── 1. Route B: 按文档聚合 ──
        doc_chunks: dict[str, list[dict]] = {}
        for r in route_a:
            doc = self.db.meta[r["index"]].get("doc", "未知")
            doc_chunks.setdefault(doc, []).append(r)

        doc_scores = {}
        for doc, chunks in doc_chunks.items():
            sorted_chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)
            doc_rrf = 0.0
            for rank, c in enumerate(sorted_chunks):
                decay = self.CHUNK_DECAY ** rank
                doc_rrf += c["score"] * decay
            doc_scores[doc] = doc_rrf

        # ── 2. 三维度加权 ──
        MAX_DOC_RRF = 1.0

        results = []
        for r in route_a:
            doc = self.db.meta[r["index"]].get("doc", "未知")

            alpha_score = min(max(r["score"], 0.0), 1.0)
            doc_raw = doc_scores.get(doc, 0)
            beta_score = min(doc_raw / MAX_DOC_RRF, 1.0)
            gamma_score = 0.8

            final_score = (self.ALPHA * alpha_score +
                           self.BETA * beta_score +
                           self.GAMMA * gamma_score)

            final_score = final_score * (1 - threshold) + threshold

            results.append({
                "text": r["text"],
                "score": round(final_score, 4),
                "doc": doc,
                "index": r["index"],
            })

        # ── 3. Cross-Encoder 精排（Day 3 新增） ──
        if self.reranker and results:
            # 先用加权分预排序，交给 reranker 精排
            results.sort(key=lambda x: x["score"], reverse=True)
            # 取 top_k * 10 个候选送精排（不要太多，cross-encoder 慢）
            pre_candidates = results[:min(len(results), top_k * 10)]
            results = self.reranker.rerank(query_text, pre_candidates, top_k=top_k)
            return results

        # ── 4. 无精排时：Threshold 过滤 + top_k 截断 ──
        results = [r for r in results if r["score"] >= threshold]
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

        return results
