"""检索引擎：多路召回 → 三维度加权 → Cross-Encoder 重排序 → 阈值过滤

用法:
    retriever = Retriever(db, reranker=reranker)
    retriever.add_texts(chunks)
    results = retriever.search(query_text, query_vec, top_k=5)
"""
import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from config import config as _cfg
from .reranker import Reranker

_cfg_r = _cfg["retrieval"]


class Retriever:
    """统一检索引擎（Dense + BM25 双路召回 → RRF → 文档聚合 → 3D → 精排）"""

    # 从配置中心读取，可运行时修改
    ALPHA = _cfg_r["alpha"]
    BETA = _cfg_r["beta"]
    GAMMA = _cfg_r["gamma"]
    RRF_K = _cfg_r["rrf_k"]
    CHUNK_DECAY = _cfg_r["chunk_decay"]

    RRF_K = 60
    CHUNK_DECAY = 0.2

    def __init__(self, db, reranker: Reranker | None = None):
        self.db = db
        self.reranker = reranker

        # BM25 内部索引（通过 add_texts 填充）
        self._tokenized: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    # ── BM25 构建 ──

    def add_texts(self, texts: list[str]):
        """增量添加文本到 BM25 索引"""
        self._tokenized.extend(jieba.lcut(t) for t in texts)
        self._bm25 = BM25Okapi(self._tokenized)

    def _rebuild_bm25(self, texts: list[str]):
        """全量替换 BM25 索引（/switch 时使用）"""
        self._tokenized = [jieba.lcut(t) for t in texts]
        self._bm25 = BM25Okapi(self._tokenized)

    # ── 核心检索 ──

    def search(self, query_text: str,
               query_vec: list[float],
               top_k: int = None,
               doc_filter: str | None = None,
               threshold: float = None) -> list[dict]:
        """完整检索流程

        步骤:
            1. Dense 召回 + BM25 召回 → RRF 融合
            2. Route B 文档聚合 → 几何衰减
            3. 三维度加权评分
            4. Cross-Encoder 精排（可选）或阈值过滤
        """
        if top_k is None:
            top_k = _cfg_r["top_k"]
        if threshold is None:
            threshold = _cfg_r["threshold"]
        expand = _cfg_r["recall_expand"] if self.reranker else 1
        route_k = top_k * 5 * expand
        scores: dict[int, float] = {}

        # ── 1a. Dense 召回 ──
        dense_result = self.db.search(query_vec, top_k=route_k)
        for rank, r in enumerate(dense_result):
            scores[r["index"]] = scores.get(r["index"], 0) + 1 / (self.RRF_K + rank)

        # ── 1b. BM25 召回 ──
        if self._bm25 is not None:
            query_tokens = jieba.lcut(query_text)
            bm25_scores = self._bm25.get_scores(query_tokens)
            top_bm25 = np.argsort(bm25_scores)[::-1][:route_k]
            for rank, idx in enumerate(top_bm25):
                scores[idx] = scores.get(idx, 0) + 1 / (self.RRF_K + rank)

        if not scores:
            return []

        # 按 RRF 总分排序
        sorted_idxs = sorted(scores, key=scores.get, reverse=True)
        fused = []
        for i in sorted_idxs:
            if i < len(self.db.texts):
                fused.append({
                    "text": self.db.texts[i],
                    "score": scores[i],
                    "index": i,
                })

        # 文档过滤
        if doc_filter:
            fused = [r for r in fused
                     if self.db.meta[r["index"]].get("doc") == doc_filter]

        if not fused:
            return []

        # ── 2. Route B：文档聚合 ──
        doc_chunks: dict[str, list[dict]] = {}
        for r in fused:
            doc = self.db.meta[r["index"]].get("doc", "未知")
            doc_chunks.setdefault(doc, []).append(r)

        doc_scores = {}
        for doc, chunks in doc_chunks.items():
            sorted_chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)
            doc_rrf = sum(c["score"] * (self.CHUNK_DECAY ** rank)
                          for rank, c in enumerate(sorted_chunks))
            doc_scores[doc] = doc_rrf

        # ── 3. 三维度加权 ──
        MAX_DOC_RRF = 1.0
        results = []
        for r in fused:
            doc = self.db.meta[r["index"]].get("doc", "未知")

            alpha_score = min(max(r["score"], 0.0), 1.0)
            beta_score = min(doc_scores.get(doc, 0) / MAX_DOC_RRF, 1.0)
            gamma_score = self.db.meta[r["index"]].get("quality", 0.5)  # 后续可改为动态质量分

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

        # ── 4a. Cross-Encoder 精排 ──
        if self.reranker and results:
            results.sort(key=lambda x: x["score"], reverse=True)
            pre_candidates = results[:min(len(results), top_k * 10)]
            return self.reranker.rerank(query_text, pre_candidates, top_k=top_k)

        # ── 4b. 阈值过滤 ──
        results = [r for r in results if r["score"] >= threshold]
        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
