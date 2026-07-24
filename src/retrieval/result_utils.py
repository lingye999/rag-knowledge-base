"""检索候选结果的轻量工具函数。"""
from __future__ import annotations

from .chunk_repository import ChunkRecord


def candidate_from_record(
    record: ChunkRecord,
    score: float,
    recall_stage: str,
    **extra,
) -> dict:
    """把仓储记录转换成 retriever 对外使用的候选字典。"""
    candidate = {
        "text": record.text,
        "score": score,
        "doc": record.doc,
        "index": record.index,
        "page": record.page,
        "source": record.source,
        "chunk_type": record.chunk_type,
        "recall_stage": recall_stage,
    }
    candidate.update(extra)
    return candidate


def dedupe_by_index(candidates: list[dict]) -> list[dict]:
    """同一个 chunk 只保留分数最高的一条候选。"""
    by_index = {}
    for candidate in candidates:
        index = candidate.get("index")
        if index is None:
            continue
        current = by_index.get(index)
        if current is None or candidate.get("score", 0.0) > current.get("score", 0.0):
            by_index[index] = candidate
    return list(by_index.values())
