import faiss
import json
from .base_vector_store import BaseVectorStore
from document_loader import read_file, chunk_text


class IvfVectorStore(BaseVectorStore):
    def __init__(self, dimension: int, nlist: int = 100):
        self.dimension = dimension
        self.texts = []
        self.nlist = nlist
        quantizer = faiss.IndexFlatIP(dimension)
        self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        self.index.nprobe = 10
        self.is_trained = False

    def add(self, text: str, vector: list[float]):
        vec = self._to_normalized([vector])
        if not self.is_trained:
            nlist = 1
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.nprobe = 1
            self.index.train(vec)
            self.is_trained = True
        self.index.add(vec)
        self.texts.append(text)

    def add_batch(self, texts: list[str], vectors: list[list[float]]):
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

    def add_from_file(self, file_path: str, embedding_service, chunk_method: str = "sentence"):
        text = read_file(file_path)
        chunks = chunk_text(text, chunk_method)
        vectors = embedding_service.encode_batch(chunks)
        self.add_batch(chunks, vectors)
        print(f"文件 {file_path} 加载完成: {len(chunks)} 个文本块")

    def search(self, query_vec: list[float], top_k: int = 5) -> list[dict]:
        return self._run_search(self.index, query_vec, top_k)

    @property
    def count(self) -> int:
        return self.index.ntotal

    def save(self, path: str):
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}_texts.json", "w", encoding="utf-8") as f:
            json.dump(self.texts, f, ensure_ascii=False)

    def load(self, path: str):
        self.index = faiss.read_index(f"{path}.faiss")
        self.dimension = self.index.d
        self.is_trained = True
        with open(f"{path}_texts.json", "r", encoding="utf-8") as f:
            self.texts = json.load(f)
