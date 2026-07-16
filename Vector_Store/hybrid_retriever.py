from rank_bm25 import BM25Okapi
from .vector_store_faiss import FaissVectorStore
import jieba
import numpy as np


class HybridRetriever:
    """混合检索器：Dense（语义） + BM25（关键词）双路召回 + RRF 融合

    用法:
        hybrid = HybridRetriever(dimension=512)
        hybrid.add_texts(texts, vectors)   # 喂数据（只调一次）
        hybrid.search(query_vec, tokens)   # 搜
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        # Dense 索引：负责语义相似度检索
        self.dense_index = FaissVectorStore(dimension)
        # BM25 索引：负责关键词精确匹配，add_texts 时创建
        self.bm25 = None
        # 所有存储的文本列表（add_texts 内部给 dense_index 管理）
        # 搜索时通过 self.dense_index.texts[i] 按索引取文本

    def add_texts(self, texts: list[str], vectors: list[list[float]]):
        """喂数据给两条路

        注意: 只能调一次。再次调用会导致 BM25 和 Dense 的索引对不齐。
              如果需要加更多数据，一次性准备好再调。
        """
        # Dense 路：向量存 FAISS，文本追加到 self.dense_index.texts
        self.dense_index.add_batch(texts, vectors)
        # BM25 路：先分词，再建索引
        tokenized = [jieba.lcut(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query_vec: list[float], query_tokens: list[str], top_k: int = 5) -> list[dict]:
        """混合检索：Dense + BM25 并行，RRF 融合排名

        参数:
            query_vec: embed.encode() 输出的向量（给 Dense）
            query_tokens: jieba.lcut() 分词结果（给 BM25）

        返回:
            [{"text": ..., "score": ..., "index": ...}, ...]
            按 RRF 总分降序，最多 top_k 条
        """
        # 空库保护
        if self.bm25 is None or self.dense_index.count == 0:
            return []

        k = 60          # RRF 常数（防止分母为 0，控制排名衰减速度）
        KNN = top_k * 5 # 每条路多召回一些，给 RRF 融合留余地
        scores = {}     # 记分板: {文本索引: RRF总分}

        # ── 路 A: Dense（语义搜索） ──────────────────────
        dense_result = self.dense_index.search(query_vec, KNN)
        for rank, r in enumerate(dense_result):
            idx = r["index"]               # 文本在 self.dense_index.texts 里的位置
            scores[idx] = scores.get(idx, 0) + 1 / (k + rank)

        # ── 路 B: BM25（关键词搜索） ─────────────────────
        bm25_scores = self.bm25.get_scores(query_tokens)  # 所有文档的 BM25 分数
        top_bm25 = np.argsort(bm25_scores)[::-1][:KNN]    # 取分数最高的 KNN 个索引

        for rank, idx in enumerate(top_bm25):
            scores[idx] = scores.get(idx, 0) + 1 / (k + rank)

        # ── RRF 融合：按总分降序，取 top_k ──────────────
        sorted_idxs = sorted(scores, key=scores.get, reverse=True)[:top_k]

        return [
            {"text": self.dense_index.texts[i], "score": scores[i], "index": i}
            for i in sorted_idxs
        ]
