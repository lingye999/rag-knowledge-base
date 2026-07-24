"""检索外观类：编排召回、补充、扩展和最终选择。"""
from __future__ import annotations

from config import config as _cfg

from .bm25_index import BM25TextIndex
from .chunk_repository import ChunkRepository
from .document_recall import DocumentRecall
from .final_selector import FinalContextSelector
from .first_stage import FirstStageRecall
from .reranker import Reranker
from .result_utils import dedupe_by_index
from .text_matching import (
    anchor_overlap_score,
    anchor_terms,
    char_ngrams,
    content_tokens,
    definition_match_score,
    is_definition_query,
    is_near_duplicate,
    looks_like_definition_entry,
    normalize_compact,
    token_overlap_score,
)

_cfg_r = _cfg["retrieval"]


class Retriever:
    """统一检索入口，保留原有 `search` / `search_with_trace` 接口。"""

    ALPHA = _cfg_r["alpha"]
    BETA = _cfg_r["beta"]
    GAMMA = _cfg_r["gamma"]
    RRF_K = _cfg_r["rrf_k"]
    CHUNK_DECAY = _cfg_r["chunk_decay"]

    def __init__(
        self,
        db,
        reranker: Reranker | None = None,
        repository: ChunkRepository | None = None,
    ):
        self.db = db
        self.repository = repository or ChunkRepository(db)
        self.reranker = reranker
        self._bm25_index = BM25TextIndex(_cfg_r.get("bm25"))
        self._rebind_parts()

    @property
    def _tokenized(self) -> list[list[str]]:
        """兼容旧 CLI 清空逻辑，真实数据放在 BM25TextIndex。"""
        return self._bm25_index.tokenized

    @property
    def _bm25(self):
        """兼容旧代码读取 BM25 实例。"""
        return self._bm25_index.bm25

    @_bm25.setter
    def _bm25(self, value):
        self._bm25_index.bm25 = value

    def _rebind_parts(self):
        """仓储或索引变化后，刷新各阶段组件的依赖。"""
        self._first_stage = FirstStageRecall(
            self.db,
            self.repository,
            self._bm25_index,
            _cfg_r,
            has_reranker=lambda: self.reranker is not None,
        )
        self._document_recall = DocumentRecall(self.repository, _cfg_r)
        self._final_selector = FinalContextSelector(_cfg_r)

    def bind_store(self, db, repository: ChunkRepository | None = None):
        """绑定替换后的向量库，并重建 BM25 侧索引。"""
        self.db = db
        self.repository = repository or ChunkRepository(db)
        self._rebind_parts()
        self._rebuild_bm25(self.repository.all_texts())

    def add_texts(self, texts: list[str]):
        """增量添加文本到 BM25 关键词索引。"""
        self._bm25_index.add_texts(texts)

    def clear_text_index(self):
        """清空 BM25 关键词索引。"""
        self._bm25_index.clear()

    def _rebuild_bm25(self, texts: list[str]):
        """完整重建 BM25 关键词索引。"""
        self._bm25_index.rebuild(texts)

    def search(
        self,
        query_text: str,
        query_vec: list[float],
        top_k: int | None = None,
        doc_filter: str | None = None,
        threshold: float | None = None,
        retrieve_top: int | None = None,
        fusion_mode: str | None = None,
    ) -> list[dict]:
        """返回最终上下文列表，兼容旧版 list 接口。"""
        return self.search_with_trace(
            query_text=query_text,
            query_vec=query_vec,
            top_k=top_k,
            doc_filter=doc_filter,
            threshold=threshold,
            retrieve_top=retrieve_top,
            fusion_mode=fusion_mode,
        )["final"]

    def search_with_trace(
        self,
        query_text: str,
        query_vec: list[float],
        top_k: int | None = None,
        doc_filter: str | None = None,
        threshold: float | None = None,
        retrieve_top: int | None = None,
        fusion_mode: str | None = None,
    ) -> dict:
        """执行完整检索，并暴露每个阶段用于评测诊断。"""
        top_k = _cfg_r["top_k"] if top_k is None else top_k
        threshold = _cfg_r["threshold"] if threshold is None else threshold
        if retrieve_top is None:
            retrieve_top = _cfg_r.get("retrieve_top", max(top_k * 6, top_k))
        retrieve_top = max(top_k, retrieve_top)

        first_stage = self._first_stage_recall(
            query_text=query_text,
            query_vec=query_vec,
            retrieve_top=retrieve_top,
            doc_filter=doc_filter,
            fusion_mode=fusion_mode,
        )
        doc_internal = self._doc_internal_recall(query_text, first_stage)
        expanded = self._expand_neighbors(first_stage + doc_internal)
        final, final_selector = self._select_final_contexts(
            query_text=query_text,
            candidates=expanded,
            top_k=top_k,
            threshold=threshold,
            include_trace=True,
        )
        return {
            "first_stage": first_stage,
            "doc_internal": doc_internal,
            "expanded": expanded,
            "candidates": expanded,
            "final": final,
            "final_selector": final_selector,
        }

    def _first_stage_recall(
        self,
        query_text: str,
        query_vec: list[float],
        retrieve_top: int,
        doc_filter: str | None,
        fusion_mode: str | None = None,
    ) -> list[dict]:
        """兼容旧测试名：首轮 Dense + BM25 融合召回。"""
        return self._first_stage.run(
            query_text=query_text,
            query_vec=query_vec,
            retrieve_top=retrieve_top,
            doc_filter=doc_filter,
            fusion_mode=fusion_mode,
        )

    def _aggregate_doc_scores(self, candidates: list[dict]) -> dict[str, float]:
        """兼容旧测试名：按文档聚合候选分数。"""
        return self._first_stage.aggregate_doc_scores(candidates)

    def _doc_internal_recall(self, query_text: str, candidates: list[dict]) -> list[dict]:
        """兼容旧测试名：文档内部补召回。"""
        return self._document_recall.doc_internal_recall(query_text, candidates)

    def _top_docs_from_candidates(
        self,
        candidates: list[dict],
        limit: int,
    ) -> list[tuple[str, float]]:
        """兼容旧测试名：从候选中选高分文档。"""
        return self._document_recall.top_docs_from_candidates(candidates, limit)

    def _expand_neighbors(self, candidates: list[dict]) -> list[dict]:
        """兼容旧测试名：扩展相邻 chunk。"""
        return self._document_recall.expand_neighbors(candidates)

    def _select_final_contexts(
        self,
        query_text: str,
        candidates: list[dict],
        top_k: int,
        threshold: float,
        include_trace: bool = False,
    ) -> list[dict] | tuple[list[dict], dict]:
        """兼容旧测试名：重打分、可选精排、过滤并返回最终上下文。"""
        final, reranker_failed, selector_trace = self._final_selector.select(
            query_text=query_text,
            candidates=candidates,
            top_k=top_k,
            threshold=threshold,
            reranker=self.reranker,
        )
        if reranker_failed:
            self.reranker = None
        return (final, selector_trace) if include_trace else final

    def _rescore_final_candidates(
        self,
        query_text: str,
        candidates: list[dict],
    ) -> list[dict]:
        """兼容旧测试名：最终轻量重打分。"""
        return self._final_selector.rescore(query_text, candidates)

    def _diversify_and_filter(
        self,
        candidates: list[dict],
        top_k: int,
        threshold: float | None,
    ) -> list[dict]:
        """兼容旧测试名：阈值过滤、去近重复和页内限流。"""
        return self._final_selector.diversify_and_filter(candidates, top_k, threshold)

    def _dedupe_by_index(self, candidates: list[dict]) -> list[dict]:
        """兼容旧测试名：按 chunk index 去重。"""
        return dedupe_by_index(candidates)

    def _anchor_terms(self, query_text: str) -> list[str]:
        return anchor_terms(query_text)

    def _anchor_overlap_score(self, anchor_items: list[str], text: str) -> float:
        return anchor_overlap_score(anchor_items, text)

    def _is_definition_query(self, query_text: str) -> bool:
        return is_definition_query(query_text)

    def _definition_match_score(self, anchor_items: list[str], text: str) -> float:
        return definition_match_score(anchor_items, text)

    def _looks_like_definition_entry(self, compact_term: str, normalized: str) -> bool:
        return looks_like_definition_entry(compact_term, normalized)

    def _token_overlap_score(self, query_tokens: set[str], text: str) -> float:
        return token_overlap_score(query_tokens, text)

    def _content_tokens(self, text: str) -> set[str]:
        return content_tokens(text)

    def _is_near_duplicate(self, left: str, right: str) -> bool:
        threshold = _cfg_r.get("duplicate_threshold", 0.82)
        return is_near_duplicate(left, right, threshold)

    def _char_ngrams(self, text: str, n: int = 3) -> set[str]:
        return char_ngrams(text, n)

    def _normalize_compact(self, text: str) -> str:
        return normalize_compact(text)
