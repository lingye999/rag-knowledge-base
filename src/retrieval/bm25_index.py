"""BM25 关键词侧索引。"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from rank_bm25 import BM25Okapi

from .text_analyzer import TextAnalyzer


class BM25TextIndex:
    """只维护 BM25 需要的分词语料，不碰向量库数据。"""

    def __init__(self, settings: dict | None = None):
        settings = settings or {}
        self.analyzer = TextAnalyzer(settings)
        self.k1 = float(settings.get("k1", 1.5))
        self.b = float(settings.get("b", 0.75))
        self.min_score = float(settings.get("min_score", 0.0))
        self.tokenized: list[list[str]] = []
        self.bm25: BM25Okapi | None = None

    def add_texts(self, texts: list[str]):
        """增量加入文本，并重建 BM25 内部统计。"""
        self.tokenized.extend(self.analyzer.analyze(text) for text in texts)
        self._rebuild_model()

    def rebuild(self, texts: list[str]):
        """向量库切换或加载后，用完整文本重新建立 BM25。"""
        self.tokenized = [self.analyzer.analyze(text) for text in texts]
        self._rebuild_model()

    def _rebuild_model(self):
        """按配置重建 BM25 的词频统计。"""
        self.bm25 = (
            BM25Okapi(self.tokenized, k1=self.k1, b=self.b)
            if self.tokenized else None
        )

    def clear(self):
        """清空关键词索引。"""
        self.tokenized.clear()
        self.bm25 = None

    def search(
        self,
        query_text: str,
        top_k: int,
        is_deleted: Callable[[int], bool],
    ) -> list[tuple[int, int, float]]:
        """返回有效 BM25 命中的 `(rank, index, score)`，零分候选不参与融合。"""
        if self.bm25 is None:
            return []

        query_tokens = self.analyzer.analyze_query(query_text)
        if not query_tokens:
            return []
        bm25_scores = self.bm25.get_scores(query_tokens)
        hits = []
        for index in np.argsort(bm25_scores)[::-1]:
            index = int(index)
            score = float(bm25_scores[index])
            if score <= self.min_score or is_deleted(index):
                continue
            hits.append((len(hits) + 1, index, score))
            if len(hits) >= top_k:
                break
        return hits
