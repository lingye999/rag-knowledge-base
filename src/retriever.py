"""检索引擎：两路召回 + 三维度加权 + 阈值过滤"""


class Retriever:
    """统一检索引擎

    用法:
        retriever = Retriever(db, hybrid, emb)
        results = retriever.search(query, top_k=5)
    """

    # 三维度权重
    ALPHA = 0.7   # 相关性（Route A chunk 分数）
    BETA = 0.1    # 元数据（Route B 文档排名）
    GAMMA = 0.2   # 文档属性（质量，暂用默认值 0.8）

    RRF_K = 60        # RRF 常数
    CHUNK_DECAY = 0.2 # 多切片几何衰减

    def __init__(self, db):
        self.db = db

    def search(self, query_vec: list[float], top_k: int = 5,
               doc_filter: str | None = None,
               threshold: float = 0.3) -> list[dict]:
        """完整检索流程：两路召回 → 加权 → 过滤

        参数:
            query_vec: 查询向量（由调用方编码）
            top_k: 返回结果数
            doc_filter: 限定文档名
            threshold: 最终分数下限

        返回:
            [{"text": ..., "score": ..., "doc": ...}, ...]
        """
        # ── 0. ──
        route_k = top_k * 5
        route_a = self.db.search(query_vec, top_k=route_k)

        # 如果有文档过滤，先缩小范围
        if doc_filter:
            route_a = [r for r in route_a
                       if self.db.meta[r["index"]].get("doc") == doc_filter]

        if not route_a:
            return []

        # ── 2. Route B: 按文档聚合 ──
        # 把 chunk 按文档分组，统计每个文档的命中情况
        doc_chunks: dict[str, list[dict]] = {}
        for r in route_a:
            doc = self.db.meta[r["index"]].get("doc", "未知")
            doc_chunks.setdefault(doc, []).append(r)

        # 对每个文档计算 RRF 聚合分
        doc_scores = {}
        for doc, chunks in doc_chunks.items():
            # 按原始 score 降序
            sorted_chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)
            # 几何衰减聚合
            doc_rrf = 0.0
            for rank, c in enumerate(sorted_chunks):
                decay = self.CHUNK_DECAY ** rank
                doc_rrf += c["score"] * decay
            doc_scores[doc] = doc_rrf

        # ── 3. Phase 2: 定向补搜 ──
        # 只在指定文档内补搜（如果有 doc_filter）
        phase2_chunks = []
        if doc_filter and doc_filter in doc_chunks:
            # 已经在 route_a 里搜过了，不需要再补搜
            pass

        # ── 4. 三维度加权（理论锚点归一化） ──
        # 不用当前批次的 min/max，避免微小差距被放大；
        # α 用余弦相似度原始值 [0, 1]；β 除以文档 RRF 理论锚点并截断

        MAX_DOC_RRF = 1.0  # 文档几何衰减的理论锚点

        results = []
        for r in route_a:
            doc = self.db.meta[r["index"]].get("doc", "未知")

            # α: chunk 相关性（余弦相似度天然在 [0, 1]）
            alpha_score = min(max(r["score"], 0.0), 1.0)

            # β: 文档元数据分（除以理论锚点，截断到 1.0）
            doc_raw = doc_scores.get(doc, 0)
            beta_score = min(doc_raw / MAX_DOC_RRF, 1.0)

            # γ: 文档属性（没有质量分，默认给 0.8）
            gamma_score = 0.8

            final_score = (self.ALPHA * alpha_score +
                           self.BETA * beta_score +
                           self.GAMMA * gamma_score)

            # 映射到 [threshold, 1.0]
            final_score = final_score * (1 - threshold) + threshold

            results.append({
                "text": r["text"],
                "score": round(final_score, 4),
                "doc": doc,
            })

        # ── 5. Threshold 过滤 + top_k 截断 ──
        results = [r for r in results if r["score"] >= threshold]
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

        return results
