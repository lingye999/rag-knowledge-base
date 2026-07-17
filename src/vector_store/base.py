from abc import ABC, abstractmethod
import faiss
import numpy as np


class BaseVectorStore(ABC):
    """所有 VectorStore 的统一接口规范"""

    def _to_normalized(self, vectors: list[list[float]]) -> np.ndarray:
        vecs = np.array(vectors, dtype=np.float32)
        faiss.normalize_L2(vecs)
        return vecs

    def _run_search(self, index, query_vec: list[float], top_k: int = 5) -> list[dict]:
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

    def search_by_text(self, query: str, embedding_service, top_k: int = 5) -> list[dict]:
        vec = embedding_service.encode(query)
        return self.search(vec, top_k=top_k)

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
