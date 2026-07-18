import faiss
import json
from .base import BaseVectorStore


class FaissVectorStore(BaseVectorStore):
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.texts = []
        self.doc_registry: dict[str, list[int]] = {}  # 文档名 → FAISS 位置列表
        self.meta: list[dict] = []  # 每个位置对应的来源信息
        self.deleted: set[int] = set()  # 标记删除的位置集合
        self._index_type = "flat"      # compact 时保持索引类型

    def add(self, text: str, vector: list[float]):
        vec = self._to_normalized([vector])
        self.index.add(vec)
        self.texts.append(text)

    def add_batch(self, texts: list[str], vectors: list[list[float]],doc_name: str | None = None):
        #记录下增加文档前的起始位置
        start=len(self.texts)

        vecs = self._to_normalized(vectors)
        self.index.add(vecs)
        self.texts.extend(texts)

        #如果传入了文档的这个名字，维护doc和meta
        if doc_name is not None:
            indices = list(range(start, start + len(texts)))  # 新 chunk 在 FAISS/texts 中的位置范围
            self.doc_registry[doc_name] = indices  # 文档名 → 位置范围映射
            for _ in texts:
                self.meta.append({"doc": doc_name})



    def search(self, query_vec: list[float], top_k: int = 5) -> list[dict]:
        return self._run_search(self.index, query_vec, top_k)

    @property
    def count(self) -> int:
        return self.index.ntotal

    def save(self, path: str):
        faiss.write_index(self.index, f"{path}.faiss")
        data = {
            "texts": self.texts,
            "doc_registry": self.doc_registry,
            "meta": self.meta,
        }
        with open(f"{path}_texts.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path: str):
        self.index = faiss.read_index(f"{path}.faiss")
        self.dimension = self.index.d
        with open(f"{path}_texts.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        # 兼容旧版本：旧格式是纯列表
        if isinstance(data, list):
            self.texts = data
        else:
            self.texts = data["texts"]
            self.doc_registry = data.get("doc_registry", {})
            self.meta = data.get("meta", [])
