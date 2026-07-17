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

    def get_chunks_by_doc(self,doc_name:str)->list[str]:
        indices=self.doc_registry.get(doc_name,[])
        return [self.texts[i] for i in indices]

    def delete_doc(self,doc_name:str)->None:
        indices=self.doc_registry.pop(doc_name,[])
        self.deleted.update(indices)

    def _compact(self):
        if not self.deleted:
            return

        #提取出可用的位置信息并重建index
        alive=[i for i in range(len(self.texts)) if i not in self.deleted]
        all_vectors=self.index.reconstruct_n(0, self.index.ntotal)
        all_vectors=all_vectors[alive]
        vecs = self._to_normalized(all_vectors)

        # 根据索引类型建新索引（保留 compact 前的索引类型）
        idx_type = getattr(self, "_index_type", "flat")
        if idx_type == "hnsw":
            M = getattr(self, "_index_params", {}).get("M", 32)
            new_index = faiss.IndexHNSWFlat(self.dimension, M, faiss.METRIC_INNER_PRODUCT)
            new_index.hnsw.efConstruction = 128
            new_index.hnsw.efSearch = 64
            new_index.add(vecs)
        elif idx_type == "ivf":
            nlist = getattr(self, "_index_params", {}).get("nlist", 100)
            quantizer = faiss.IndexFlatIP(self.dimension)
            new_index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            new_index.nprobe = 10
            new_index.train(vecs)
            new_index.add(vecs)
        else:
            new_index = faiss.IndexFlatIP(self.dimension)
            new_index.add(vecs)

        self.index = new_index

        #同步meta和text里的文本内容
        self.meta = [self.meta[i] for i in alive]  # 取 alive 位置的 meta 字典
        self.texts = [self.texts[i] for i in alive]  # 取 alive 位置的文本

        #重建doc_name
        old_to_new={old:new for new ,old in enumerate(alive)}
        new_registry = {}  # 新增

        for doc_name,old_indices in self.doc_registry.items():
            new_indices=[old_to_new[i] for i in old_indices if i in old_to_new]
            if new_indices:  # 转换后还有位置才保留
                new_registry[doc_name] = new_indices  # 存到新字典

        self.doc_registry = new_registry  # 替换
        self.deleted.clear()  # 清空标记






    @abstractmethod
    def add(self, text: str, vector: list[float]):
        ...

    @abstractmethod
    def add_batch(self, texts: list[str], vectors: list[list[float]]):
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
