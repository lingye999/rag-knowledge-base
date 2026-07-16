import faiss
import json
from .base_vector_store import BaseVectorStore
from document_loader import read_file, chunk_text


class HnswVectorStore(BaseVectorStore):
    def __init__(self, dimension: int, M: int = 32):
        self.dimension = dimension
        self.texts = []
        self.index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = 128
        self.index.hnsw.efSearch = 64

    def add(self, text: str, vector: list[float]):
        vec = self._to_normalized([vector])
        self.index.add(vec)
        self.texts.append(text)

    def add_batch(self, texts: list[str], vectors: list[list[float]]):
        vecs = self._to_normalized(vectors)
        self.index.add(vecs)
        self.texts.extend(texts)

    def add_from_file(self, file_path: str, embedding_service, chunk_method: str = "sentence"):
        text = read_file(file_path)
        chunks = chunk_text(text, chunk_method)
        if not chunks:
            print(f"[警告] 文件 {file_path} 未提取到有效文本块，已跳过")
            return
        vectors = embedding_service.encode_batch(chunks)
        if not vectors:
            print(f"[警告] 文件 {file_path} 向量化失败，已跳过")
            return
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
        with open(f"{path}_texts.json", "r", encoding="utf-8") as f:
            self.texts = json.load(f)
