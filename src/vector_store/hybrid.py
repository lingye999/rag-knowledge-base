from rank_bm25 import BM25Okapi
from .faiss_store import FaissVectorStore
import jieba
import numpy as np


class HybridRetriever:
    """混合检索器：Dense（语义） + BM25（关键词）双路召回 + RRF 融合"""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.dense_index = FaissVectorStore(dimension)
        self.bm25 = None

    def add_texts(self, texts: list[str], vectors: list[list[float]]):
        self.dense_index.add_batch(texts, vectors)
        tokenized = [jieba.lcut(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query_vec: list[float], query_tokens: list[str], top_k: int = 5) -> list[dict]:
        if self.bm25 is None or self.dense_index.count == 0:
            return []

        k = 60
        KNN = top_k * 5
        scores = {}

        # 路 A: Dense
        dense_result = self.dense_index.search(query_vec, KNN)
        for rank, r in enumerate(dense_result):
            idx = r["index"]
            scores[idx] = scores.get(idx, 0) + 1 / (k + rank)

        # 路 B: BM25
        bm25_scores = self.bm25.get_scores(query_tokens)
        top_bm25 = np.argsort(bm25_scores)[::-1][:KNN]
        for rank, idx in enumerate(top_bm25):
            scores[idx] = scores.get(idx, 0) + 1 / (k + rank)

        sorted_idxs = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [
            {"text": self.dense_index.texts[i], "score": scores[i], "index": i}
            for i in sorted_idxs
        ]
