"""Retrieval engine: hybrid recall, document-aware supplementation, and final context selection."""
from __future__ import annotations

from collections import defaultdict

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from config import config as _cfg
from .chunk_repository import ChunkRepository
from .reranker import Reranker

_cfg_r = _cfg["retrieval"]


class Retriever:
    """Dense + BM25 retrieval with a second-stage evidence supplementation pass."""

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
        self._tokenized: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def bind_store(self, db, repository: ChunkRepository | None = None):
        """Bind a replacement index without exposing migration details."""
        self.db = db
        self.repository = repository or ChunkRepository(db)
        self._rebuild_bm25(self.repository.all_texts())

    def add_texts(self, texts: list[str]):
        """Incrementally add texts to the BM25 side index."""
        self._tokenized.extend(jieba.lcut(t) for t in texts)
        self._bm25 = BM25Okapi(self._tokenized)

    def _rebuild_bm25(self, texts: list[str]):
        """Fully rebuild the BM25 side index after store replacement."""
        self._tokenized = [jieba.lcut(t) for t in texts]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    def search(
        self,
        query_text: str,
        query_vec: list[float],
        top_k: int | None = None,
        doc_filter: str | None = None,
        threshold: float | None = None,
        retrieve_top: int | None = None,
    ) -> list[dict]:
        """Return the final contexts while keeping the legacy list interface."""
        return self.search_with_trace(
            query_text=query_text,
            query_vec=query_vec,
            top_k=top_k,
            doc_filter=doc_filter,
            threshold=threshold,
            retrieve_top=retrieve_top,
        )["final"]

    def search_with_trace(
        self,
        query_text: str,
        query_vec: list[float],
        top_k: int | None = None,
        doc_filter: str | None = None,
        threshold: float | None = None,
        retrieve_top: int | None = None,
    ) -> dict:
        """Run retrieval and expose each stage for evaluation diagnostics."""
        if top_k is None:
            top_k = _cfg_r["top_k"]
        if threshold is None:
            threshold = _cfg_r["threshold"]
        if retrieve_top is None:
            retrieve_top = _cfg_r.get("retrieve_top", max(top_k * 6, top_k))
        retrieve_top = max(top_k, retrieve_top)

        first_stage = self._first_stage_recall(
            query_text=query_text,
            query_vec=query_vec,
            retrieve_top=retrieve_top,
            doc_filter=doc_filter,
        )
        doc_internal = self._doc_internal_recall(query_text, first_stage)
        expanded = self._expand_neighbors(first_stage + doc_internal)
        final = self._select_final_contexts(
            query_text=query_text,
            candidates=expanded,
            top_k=top_k,
            threshold=threshold,
        )
        return {
            "first_stage": first_stage,
            "doc_internal": doc_internal,
            "expanded": expanded,
            "candidates": expanded,
            "final": final,
        }

    def _first_stage_recall(
        self,
        query_text: str,
        query_vec: list[float],
        retrieve_top: int,
        doc_filter: str | None,
    ) -> list[dict]:
        expand = _cfg_r["recall_expand"] if self.reranker else 1
        route_k = retrieve_top * 5 * expand
        scores: dict[int, float] = {}

        dense_result = self.db.search(query_vec, top_k=route_k)
        for rank, result in enumerate(dense_result):
            scores[result["index"]] = (
                scores.get(result["index"], 0.0) + 1 / (self.RRF_K + rank)
            )

        if self._bm25 is not None:
            query_tokens = jieba.lcut(query_text)
            bm25_scores = self._bm25.get_scores(query_tokens)
            top_bm25 = np.argsort(bm25_scores)[::-1][:route_k]
            for rank, idx in enumerate(top_bm25):
                idx = int(idx)
                if self.db.is_deleted(idx):
                    continue
                scores[idx] = scores.get(idx, 0.0) + 1 / (self.RRF_K + rank)

        if not scores:
            return []

        fused = []
        for index in sorted(scores, key=scores.get, reverse=True):
            record = self.repository.get(index)
            if record is None or record.deleted:
                continue
            if doc_filter and record.doc != doc_filter:
                continue
            fused.append({
                "text": record.text,
                "score": scores[index],
                "index": index,
                "doc": record.doc,
                "page": record.page,
                "source": record.source,
                "recall_stage": "first_stage",
            })

        if not fused:
            return []

        doc_scores = self._aggregate_doc_scores(fused)
        max_chunk_score = max(item["score"] for item in fused) or 1.0
        max_doc_score = max(doc_scores.values(), default=0.0) or 1.0
        results = []
        for item in fused:
            record = self.repository.get(item["index"])
            if record is None:
                continue
            # RRF values are small (roughly 1 / rrf_k). Multiplying by
            # rrf_k and clipping saturates nearly every candidate at 1,
            # which makes quality dominate and destroys query ordering.
            # Normalize within this query instead so the strongest chunk
            # and document receive 1 while rank differences remain useful.
            alpha_score = min(max(item["score"] / max_chunk_score, 0.0), 1.0)
            beta_score = min(
                max(doc_scores.get(record.doc, 0.0) / max_doc_score, 0.0), 1.0
            )
            gamma_score = record.quality
            final_score = (
                self.ALPHA * alpha_score
                + self.BETA * beta_score
                + self.GAMMA * gamma_score
            )
            results.append({
                "text": record.text,
                "score": round(final_score, 4),
                "doc": record.doc,
                "index": record.index,
                "page": record.page,
                "source": record.source,
                "recall_stage": "first_stage",
                "alpha_score": round(alpha_score, 4),
                "beta_score": round(beta_score, 4),
                "gamma_score": round(gamma_score, 4),
            })

        return sorted(results, key=lambda item: item["score"], reverse=True)[:retrieve_top]

    def _aggregate_doc_scores(self, candidates: list[dict]) -> dict[str, float]:
        doc_chunks: dict[str, list[dict]] = defaultdict(list)
        for candidate in candidates:
            doc_chunks[candidate.get("doc", "")].append(candidate)

        doc_scores = {}
        for doc, chunks in doc_chunks.items():
            sorted_chunks = sorted(chunks, key=lambda item: item["score"], reverse=True)
            doc_scores[doc] = sum(
                chunk["score"] * (self.CHUNK_DECAY ** rank)
                for rank, chunk in enumerate(sorted_chunks)
            )
        return doc_scores

    def _doc_internal_recall(self, query_text: str, candidates: list[dict]) -> list[dict]:
        if not candidates or not _cfg_r.get("enable_doc_recall", True):
            return []

        top_docs = self._top_docs_from_candidates(
            candidates,
            limit=_cfg_r.get("doc_recall_top_docs", 3),
        )
        query_tokens = self._content_tokens(query_text)
        if not query_tokens:
            return []

        existing = {candidate["index"] for candidate in candidates}
        supplements = []
        per_doc = _cfg_r.get("doc_recall_per_doc", 5)
        for doc, doc_score in top_docs:
            scored_records = []
            for record in self.repository.records_by_document(doc):
                if record.deleted or record.index in existing:
                    continue
                token_score = self._token_overlap_score(query_tokens, record.text)
                if token_score <= 0:
                    continue
                score = min(1.0, doc_score * 0.75 + token_score * 0.25)
                scored_records.append((score, token_score, record))

            scored_records.sort(key=lambda item: (item[0], item[1]), reverse=True)
            for score, token_score, record in scored_records[:per_doc]:
                supplements.append({
                    "text": record.text,
                    "score": round(score, 4),
                    "doc": record.doc,
                    "index": record.index,
                    "page": record.page,
                    "source": record.source,
                    "recall_stage": "doc_internal",
                    "token_overlap": round(token_score, 4),
                    "doc_base_score": round(doc_score, 4),
                })
                existing.add(record.index)
        return supplements

    def _top_docs_from_candidates(
        self,
        candidates: list[dict],
        limit: int,
    ) -> list[tuple[str, float]]:
        per_doc: dict[str, list[float]] = defaultdict(list)
        for candidate in candidates:
            doc = candidate.get("doc")
            if doc:
                per_doc[doc].append(float(candidate.get("score", 0.0)))

        doc_scores = []
        for doc, scores in per_doc.items():
            scores.sort(reverse=True)
            score = sum(
                value * (self.CHUNK_DECAY ** rank)
                for rank, value in enumerate(scores)
            )
            doc_scores.append((doc, min(1.0, score)))
        return sorted(doc_scores, key=lambda item: item[1], reverse=True)[:limit]

    def _expand_neighbors(self, candidates: list[dict]) -> list[dict]:
        if not candidates or not _cfg_r.get("enable_neighbor_expand", True):
            return self._dedupe_by_index(candidates)

        window = _cfg_r.get("neighbor_window", 1)
        expanded = list(candidates)
        seen = {candidate["index"] for candidate in candidates}

        for item in candidates:
            base = self.repository.get(item["index"])
            if base is None:
                continue
            for offset in range(-window, window + 1):
                if offset == 0:
                    continue
                index = item["index"] + offset
                if index in seen:
                    continue
                record = self.repository.get(index)
                if record is None or record.deleted or record.doc != base.doc:
                    continue
                expanded.append({
                    "text": record.text,
                    "score": round(float(item.get("score", 0.0)) * 0.92, 4),
                    "doc": record.doc,
                    "index": record.index,
                    "page": record.page,
                    "source": record.source,
                    "recall_stage": "neighbor",
                    "expanded_from": item["index"],
                })
                seen.add(index)
        return self._dedupe_by_index(expanded)

    def _select_final_contexts(
        self,
        query_text: str,
        candidates: list[dict],
        top_k: int,
        threshold: float,
    ) -> list[dict]:
        candidates = self._dedupe_by_index(candidates)
        if self.reranker and candidates:
            try:
                pre_candidates = sorted(
                    candidates, key=lambda item: item["score"], reverse=True
                )[:max(top_k * 10, top_k)]
                reranked = self.reranker.rerank(
                    query_text, pre_candidates, top_k=len(pre_candidates)
                )
                return self._diversify_and_filter(reranked, top_k, threshold=None)
            except Exception as exc:
                print(f"[Reranker] failed, falling back to base retrieval: {exc}")
                self.reranker = None

        return self._diversify_and_filter(candidates, top_k, threshold=threshold)

    def _diversify_and_filter(
        self,
        candidates: list[dict],
        top_k: int,
        threshold: float | None,
    ) -> list[dict]:
        sorted_candidates = sorted(
            candidates, key=lambda item: item["score"], reverse=True
        )
        selected = []
        doc_page_counts: dict[tuple[str, int | None], int] = defaultdict(int)
        for candidate in sorted_candidates:
            if threshold is not None and candidate["score"] < threshold:
                continue
            if any(self._is_near_duplicate(candidate["text"], item["text"]) for item in selected):
                continue
            key = (candidate.get("doc", ""), candidate.get("page"))
            if doc_page_counts[key] >= 3:
                continue
            selected.append(candidate)
            doc_page_counts[key] += 1
            if len(selected) >= top_k:
                break
        return selected

    def _dedupe_by_index(self, candidates: list[dict]) -> list[dict]:
        by_index = {}
        for candidate in candidates:
            index = candidate.get("index")
            if index is None:
                continue
            current = by_index.get(index)
            if current is None or candidate.get("score", 0.0) > current.get("score", 0.0):
                by_index[index] = candidate
        return list(by_index.values())

    def _token_overlap_score(self, query_tokens: set[str], text: str) -> float:
        text_tokens = self._content_tokens(text)
        if not text_tokens:
            return 0.0
        overlap = query_tokens & text_tokens
        return len(overlap) / max(len(query_tokens), 1)

    def _content_tokens(self, text: str) -> set[str]:
        return {
            token.strip().casefold()
            for token in jieba.lcut(text)
            if len(token.strip()) >= 2
        }

    def _is_near_duplicate(self, left: str, right: str) -> bool:
        threshold = _cfg_r.get("duplicate_threshold", 0.82)
        left_grams = self._char_ngrams(left)
        right_grams = self._char_ngrams(right)
        if not left_grams or not right_grams:
            return False
        jaccard = len(left_grams & right_grams) / len(left_grams | right_grams)
        return jaccard >= threshold

    def _char_ngrams(self, text: str, n: int = 3) -> set[str]:
        compact = "".join(text.split()).casefold()
        if len(compact) <= n:
            return {compact} if compact else set()
        return {compact[i:i + n] for i in range(len(compact) - n + 1)}
