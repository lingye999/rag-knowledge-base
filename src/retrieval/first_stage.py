"""首轮 Dense + BM25 召回与 RRF 融合。"""
from __future__ import annotations

from collections import defaultdict
import re

from .bm25_index import BM25TextIndex
from .chunk_repository import ChunkRepository
from .result_utils import candidate_from_record


class FirstStageRecall:
    """负责第一阶段多路召回，不做邻居扩展和最终精排。"""

    def __init__(
        self,
        db,
        repository: ChunkRepository,
        bm25_index: BM25TextIndex,
        settings: dict,
        has_reranker,
    ):
        self.db = db
        self.repository = repository
        self.bm25_index = bm25_index
        self.settings = settings
        self.has_reranker = has_reranker

    def run(
        self,
        query_text: str,
        query_vec: list[float],
        retrieve_top: int,
        doc_filter: str | None,
        fusion_mode: str | None = None,
    ) -> list[dict]:
        """召回候选并转成统一评分结构。"""
        expand = self.settings["recall_expand"] if self.has_reranker() else 1
        route_k = retrieve_top * 5 * expand
        scores = self._fuse_dense_and_bm25(
            query_text,
            query_vec,
            route_k,
            fusion_mode=fusion_mode,
        )
        if not scores:
            return []

        fused = self._records_from_scores(scores, doc_filter)
        if not fused:
            return []

        return self._score_candidates(fused, retrieve_top)

    def _fuse_dense_and_bm25(
        self,
        query_text: str,
        query_vec: list[float],
        route_k: int,
        fusion_mode: str | None = None,
    ) -> dict[int, dict]:
        """按指定融合模式合并两路召回，并保留可诊断的路由信息。"""
        mode, profile, dense_weight, bm25_weight = self._fusion_profile(
            query_text,
            fusion_mode,
        )
        scores: dict[int, dict] = {}
        rrf_k = self.settings["rrf_k"]

        if mode != "bm25_only":
            dense_result = self.db.search(query_vec, top_k=route_k)
            for rank, result in enumerate(dense_result, start=1):
                self._add_route_score(
                    scores,
                    index=result["index"],
                    route="dense",
                    rank=rank,
                    raw_score=float(result.get("score", 0.0)),
                    contribution=dense_weight / (rrf_k + rank),
                )

        if mode != "dense_only":
            bm25_hits = self.bm25_index.search(
                query_text,
                route_k,
                is_deleted=self.db.is_deleted,
            )
            for rank, index, raw_score in bm25_hits:
                self._add_route_score(
                    scores,
                    index=index,
                    route="bm25",
                    rank=rank,
                    raw_score=raw_score,
                    contribution=bm25_weight / (rrf_k + rank),
                )

        for item in scores.values():
            item["fusion_mode"] = mode
            item["fusion_profile"] = profile
        return scores

    @staticmethod
    def _add_route_score(
        scores: dict[int, dict],
        index: int,
        route: str,
        rank: int,
        raw_score: float,
        contribution: float,
    ):
        """累加单个召回路的 RRF 贡献，并记录可解释字段。"""
        item = scores.setdefault(index, {"rrf_score": 0.0})
        item["rrf_score"] += contribution
        item[f"{route}_rank"] = rank
        item[f"{route}_score"] = raw_score
        item[f"{route}_rrf"] = contribution

    def _fusion_profile(
        self,
        query_text: str,
        requested_mode: str | None,
    ) -> tuple[str, str, float, float]:
        """根据查询特征选择融合模式和两路权重。"""
        mode = requested_mode or self.settings.get("fusion_mode", "rrf_equal")
        valid_modes = {"dense_only", "bm25_only", "rrf_equal", "rrf_weighted"}
        if mode not in valid_modes:
            raise ValueError(f"Unsupported fusion mode: {mode}")
        dense_weight = float(self.settings.get("rrf_dense_weight", 1.0))
        bm25_weight = float(self.settings.get("rrf_bm25_weight", 1.0))
        if mode != "rrf_weighted":
            return mode, mode, dense_weight, bm25_weight

        weighted = self.settings.get("weighted_rrf", {})
        if self._is_exact_query(query_text):
            return (
                mode,
                "exact",
                float(weighted.get("exact_dense_weight", 1.0)),
                float(weighted.get("exact_bm25_weight", 1.6)),
            )
        if self._is_semantic_query(query_text):
            return (
                mode,
                "semantic",
                float(weighted.get("semantic_dense_weight", 1.3)),
                float(weighted.get("semantic_bm25_weight", 1.0)),
            )
        return mode, "default", dense_weight, bm25_weight

    @staticmethod
    def _is_exact_query(query_text: str) -> bool:
        """识别型号、标准号、数值单位和引号术语等词面优先的查询。"""
        patterns = (
            r"[A-Za-z]{1,6}\s*/\s*[A-Za-z]{1,6}\s*\d+(?:\.\d+)*",
            r"[A-Za-z]+\s*-\s*[A-Za-z0-9]+",
            r"\d+(?:\.\d+)?\s*(?:ms|s|min|kv|v|ka|ma|a|hz|khz|mhz|mm|cm|m|毫秒|千伏|千安)",
            r"[\"'“”]",
        )
        return any(re.search(pattern, query_text, flags=re.IGNORECASE) for pattern in patterns)

    @staticmethod
    def _is_semantic_query(query_text: str) -> bool:
        """识别定义、解释、因果等更依赖语义泛化的查询。"""
        compact = "".join(query_text.split()).casefold()
        cues = ("是什么", "定义", "含义", "为什么", "如何", "怎么", "解释", "meaning", "define")
        return any(cue in compact for cue in cues)

    def _records_from_scores(
        self,
        scores: dict[int, dict],
        doc_filter: str | None,
    ) -> list[dict]:
        fused = []
        for index in sorted(scores, key=lambda item: scores[item]["rrf_score"], reverse=True):
            record = self.repository.get(index)
            if record is None or record.deleted:
                continue
            if doc_filter and record.doc != doc_filter:
                continue
            fused.append(candidate_from_record(
                record,
                score=scores[index]["rrf_score"],
                recall_stage="first_stage",
                **scores[index],
            ))
        return fused

    def _score_candidates(self, fused: list[dict], retrieve_top: int) -> list[dict]:
        doc_scores = self.aggregate_doc_scores(fused)
        max_chunk_score = max(item["score"] for item in fused) or 1.0
        max_doc_score = max(doc_scores.values(), default=0.0) or 1.0
        results = []

        for item in fused:
            record = self.repository.get(item["index"])
            if record is None:
                continue
            # RRF 原始值很小，按本轮最大值归一，保留同一次查询里的相对排序。
            alpha_score = min(max(item["score"] / max_chunk_score, 0.0), 1.0)
            beta_score = min(
                max(doc_scores.get(record.doc, 0.0) / max_doc_score, 0.0),
                1.0,
            )
            gamma_score = record.quality
            final_score = (
                self.settings["alpha"] * alpha_score
                + self.settings["beta"] * beta_score
                + self.settings["gamma"] * gamma_score
            )
            results.append(candidate_from_record(
                record,
                score=round(final_score, 4),
                recall_stage="first_stage",
                alpha_score=round(alpha_score, 4),
                beta_score=round(beta_score, 4),
                gamma_score=round(gamma_score, 4),
                rrf_score=round(item["rrf_score"], 6),
                dense_rank=item.get("dense_rank"),
                dense_score=item.get("dense_score"),
                dense_rrf=item.get("dense_rrf"),
                bm25_rank=item.get("bm25_rank"),
                bm25_score=item.get("bm25_score"),
                bm25_rrf=item.get("bm25_rrf"),
                fusion_mode=item.get("fusion_mode"),
                fusion_profile=item.get("fusion_profile"),
            ))

        return sorted(results, key=lambda item: item["score"], reverse=True)[:retrieve_top]

    def aggregate_doc_scores(self, candidates: list[dict]) -> dict[str, float]:
        """按文档聚合 chunk 分数，排名越靠后的 chunk 衰减越多。"""
        doc_chunks: dict[str, list[dict]] = defaultdict(list)
        for candidate in candidates:
            doc_chunks[candidate.get("doc", "")].append(candidate)

        doc_scores = {}
        for doc, chunks in doc_chunks.items():
            sorted_chunks = sorted(chunks, key=lambda item: item["score"], reverse=True)
            doc_scores[doc] = sum(
                chunk["score"] * (self.settings["chunk_decay"] ** rank)
                for rank, chunk in enumerate(sorted_chunks)
            )
        return doc_scores
