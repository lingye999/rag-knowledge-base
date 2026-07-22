from rank_bm25 import BM25Okapi
import jieba
import numpy as np
from ..chunk_repository import ChunkRepository


class HybridRetriever:
    """混合检索器：Dense（语义） + BM25（关键词）双路召回 + RRF 融合

    用法:
        hybrid = HybridRetriever(db)          # 共享 db，不重复存数据
        hybrid.add_texts(texts)               # 只维护 BM25
        hybrid.search(query_vec, tokens)      # 搜
    """

    def __init__(self, db):
        """接收外部 FaissVectorStore，不复建"""
        self.db = db
        self.repository = ChunkRepository(db)
        self._tokenized: list[list[str]] = []  # 分词结果，和 db.texts 对齐
        self.bm25 = None

    def add_texts(self, texts: list[str]):
        """添加文本到 BM25 索引（向量已在 db 中，不重复存）"""
        self._tokenized.extend(jieba.lcut(t) for t in texts)
        self.bm25 = BM25Okapi(self._tokenized)

    def search(self, query_vec: list[float], query_tokens: list[str],
               top_k: int = 5) -> list[dict]:
        """Dense + BM25 并行，RRF 融合"""
        if self.bm25 is None or self.db.count == 0:
            return []

        k = 60
        KNN = top_k * 5
        scores: dict[int, float] = {}

        # 路 A: Dense（复用的共享 db）
        dense_result = self.db.search(query_vec, KNN)
        for rank, r in enumerate(dense_result):
            scores[r["index"]] = scores.get(r["index"], 0) + 1 / (k + rank)

        # 路 B: BM25
        bm25_scores = self.bm25.get_scores(query_tokens)
        top_bm25 = np.argsort(bm25_scores)[::-1][:KNN]
        for rank, idx in enumerate(top_bm25):
            scores[idx] = scores.get(idx, 0) + 1 / (k + rank)

        sorted_idxs = sorted(scores, key=scores.get, reverse=True)[:top_k]
        results = []
        for index in sorted_idxs:
            record = self.repository.get(int(index))
            if record is None or record.deleted:
                continue
            results.append({
                "text": record.text,
                "score": scores[index],
                "index": int(index),
            })
        return results
