"""文档内部补召回与邻居扩展。"""
from __future__ import annotations

from collections import defaultdict

from .chunk_repository import ChunkRepository
from .result_utils import candidate_from_record, dedupe_by_index
from .text_matching import content_tokens, token_overlap_score


class DocumentRecall:
    """在首轮候选文档内部找补充证据，并补齐相邻 chunk。"""

    def __init__(self, repository: ChunkRepository, settings: dict):
        self.repository = repository
        self.settings = settings

    def doc_internal_recall(self, query_text: str, candidates: list[dict]) -> list[dict]:
        """在高分文档内部，用词重叠召回首轮漏掉的证据 chunk。"""
        if not candidates or not self.settings.get("enable_doc_recall", True):
            return []

        top_docs = self.top_docs_from_candidates(
            candidates,
            limit=self.settings.get("doc_recall_top_docs", 3),
        )
        query_tokens = content_tokens(query_text)
        if not query_tokens:
            return []

        existing = {candidate["index"] for candidate in candidates}
        supplements = []
        per_doc = self.settings.get("doc_recall_per_doc", 5)
        for doc, doc_score in top_docs:
            scored_records = []
            for record in self.repository.records_by_document(doc):
                if record.deleted or record.index in existing:
                    continue
                token_score = token_overlap_score(query_tokens, record.text)
                if token_score <= 0:
                    continue
                score = min(1.0, doc_score * 0.75 + token_score * 0.25)
                scored_records.append((score, token_score, record))

            scored_records.sort(key=lambda item: (item[0], item[1]), reverse=True)
            for score, token_score, record in scored_records[:per_doc]:
                supplements.append(candidate_from_record(
                    record,
                    score=round(score, 4),
                    recall_stage="doc_internal",
                    token_overlap=round(token_score, 4),
                    doc_base_score=round(doc_score, 4),
                ))
                existing.add(record.index)
        return supplements

    def top_docs_from_candidates(
        self,
        candidates: list[dict],
        limit: int,
    ) -> list[tuple[str, float]]:
        """从候选里挑出最值得补召回的文档。"""
        per_doc: dict[str, list[float]] = defaultdict(list)
        for candidate in candidates:
            doc = candidate.get("doc")
            if doc:
                per_doc[doc].append(float(candidate.get("score", 0.0)))

        doc_scores = []
        for doc, scores in per_doc.items():
            scores.sort(reverse=True)
            score = sum(
                value * (self.settings["chunk_decay"] ** rank)
                for rank, value in enumerate(scores)
            )
            doc_scores.append((doc, min(1.0, score)))
        return sorted(doc_scores, key=lambda item: item[1], reverse=True)[:limit]

    def expand_neighbors(self, candidates: list[dict]) -> list[dict]:
        """补齐候选 chunk 前后的同文档邻居，支持跨 chunk 证据。"""
        if not candidates or not self.settings.get("enable_neighbor_expand", True):
            return dedupe_by_index(candidates)

        window = self.settings.get("neighbor_window", 1)
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
                expanded.append(candidate_from_record(
                    record,
                    score=round(float(item.get("score", 0.0)) * 0.92, 4),
                    recall_stage="neighbor",
                    expanded_from=item["index"],
                ))
                seen.add(index)
        return dedupe_by_index(expanded)
