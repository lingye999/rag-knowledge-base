import faiss
import json
from .base import BaseVectorStore


class IvfVectorStore(BaseVectorStore):
    def __init__(self, dimension: int, nlist: int = 100):
        self.dimension = dimension
        self.texts = []
        self.doc_registry: dict[str, list[int]] = {}  # 文档名 → FAISS 位置列表
        self.meta: list[dict] = []  # 每个位置对应的来源信息
        self.deleted: set[int] = set()  # 标记删除的位置集合
        self._index_type = "ivf"       # compact 时保持索引类型
        self._index_params = {"nlist": nlist}  # IVF 参数

        self.nlist = nlist
        quantizer = faiss.IndexFlatIP(dimension)
        self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        self.index.nprobe = 10
        self.is_trained = False

    def add(self, text: str, vector: list[float]):
        self.add_batch([text], [vector])

    def add_batch(self, texts: list[str], vectors: list[list[float]],doc_name: str | None = None):
        start = len(self.texts)

        vecs = self._to_normalized(vectors)
        if not self.is_trained:
            n_train = len(vecs)
            nlist = min(self.nlist, n_train)
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.nprobe = min(10, nlist)
            self.index.train(vecs)
            self.is_trained = True
        self.index.add(vecs)
        self.texts.extend(texts)

        # 如果传入了文档的这个名字，维护doc和meta
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
        self.is_trained = True
        with open(f"{path}_texts.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            self.texts = data
        else:
            self.texts = data["texts"]
            self.doc_registry = data.get("doc_registry", {})
            self.meta = data.get("meta", [])
