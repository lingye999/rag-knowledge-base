from abc import ABC, abstractmethod
import faiss
import numpy as np


class BaseVectorStore(ABC):
    """所有 VectorStore 的统一接口规范"""

    # ── 工具方法（基类实现，子类可直接调） ──────────────

    def _to_normalized(self, vectors: list[list[float]]) -> np.ndarray:
        """转 np.float32 矩阵 + L2 归一化（所有子类完全一致的操作）"""
        vecs = np.array(vectors, dtype=np.float32)
        faiss.normalize_L2(vecs)
        return vecs

    def _run_search(self, index, query_vec: list[float], top_k: int = 5) -> list[dict]:
        """执行 FAISS 搜索，拼成统一格式的 dict 列表

        前提：子类必须维护 self.texts 列表，长度与 index.ntotal 一致
        """
        q = np.array([query_vec], dtype=np.float32)
        faiss.normalize_L2(q)

        scores, indices = index.search(q, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.texts):
                results.append({
                    "text": self.texts[idx],
                    "score": float(score),
                    "index": int(idx)
                })
        return results

    # ── 便捷方法（基类基于抽象方法组合） ────────────────

    def search_by_text(self, query: str, embedding_service, top_k: int = 5) -> list[dict]:
        """传入文本 → 自动 encode → 搜索"""
        vec = embedding_service.encode(query)
        return self.search(vec, top_k=top_k)

    # ── 契约：子类必须自己实现 ──────────────────────────

    @abstractmethod
    def add(self, text: str, vector: list[float]):
        ...

    @abstractmethod
    def add_batch(self, texts: list[str], vectors: list[list[float]]):
        ...

    @abstractmethod
    def add_from_file(self, file_path: str, embedding_service, chunk_method: str = "sentence"):
        ...

    @abstractmethod
    def search(self, query_vec: list[float], top_k: int = 5) -> list[dict]:
        ...

    @property
    @abstractmethod
    def count(self) -> int:
        ...

    @abstractmethod
    def save(self, path: str):
        ...

    @abstractmethod
    def load(self, path: str):
        ...
