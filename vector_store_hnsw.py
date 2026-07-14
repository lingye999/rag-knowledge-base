import faiss
import numpy as np
import json
from document_loader import read_file, chunk_by_sentence, chunk_by_paragraph, chunk_by_jieba


class HnswVectorStore:
    def __init__(self, dimension: int, M: int = 32):
        """HNSW 分层导航图索引"""
        self.dimension = dimension
        self.texts = []
        self.index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = 128
        self.index.hnsw.efSearch = 64

    def add(self, text: str, vector: list[float]):
        vec = np.array([vector], dtype=np.float32)
        faiss.normalize_L2(vec)
        self.index.add(vec)
        self.texts.append(text)

    def add_batch(self, texts: list[str], vectors: list[list[float]]):
        vecs = np.array(vectors, dtype=np.float32)
        faiss.normalize_L2(vecs)
        self.index.add(vecs)
        self.texts.extend(texts)

    def add_from_file(self, file_path: str, embedding_service, chunk_method="sentence"):
        text = read_file(file_path)

        if chunk_method == "sentence":
            chunks = chunk_by_sentence(text)
        elif chunk_method == "paragraph":
            chunks = chunk_by_paragraph(text)
        elif chunk_method == "jieba":
            chunks = chunk_by_jieba(text)
        else:
            raise ValueError(f"不支持的分块方法: {chunk_method}")

        vectors = embedding_service.encode_batch(chunks)
        self.add_batch(chunks, vectors)
        print(f"文件 {file_path} 加载完成: {len(chunks)} 个文本块")

    def search(self, query_vec: list[float], top_k: int = 5) -> list[dict]:
        q = np.array([query_vec], dtype=np.float32)
        faiss.normalize_L2(q)

        scores, indices = self.index.search(q, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.texts):
                results.append({
                    "text": self.texts[idx],
                    "score": float(score),
                    "index": int(idx)
                })
        return results

    @property
    def count(self) -> int:
        return self.index.ntotal

    def save(self, path: str):
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}_texts.json", "w", encoding="utf-8") as f:
            json.dump(self.texts, f, ensure_ascii=False)

    def load(self, path: str):
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}_texts.json", "r", encoding="utf-8") as f:
            self.texts = json.load(f)
