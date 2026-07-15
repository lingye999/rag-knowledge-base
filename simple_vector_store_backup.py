import numpy as np
from document_loader import read_file, chunk_by_sentence, chunk_by_paragraph, chunk_by_jieba
import json
import os


class SimpleVectorStore:
    def __init__(self, metric: str = "cosine"):
        self.vectors = []    # 存向量
        self.texts = []      # 存原始文本
        self.metric = metric

    def add(self, text: str, vector: list[float]):
        """添加一条"""
        # 1. 把 vector 存到 self.vectors
        self.vectors.append(vector.copy())
        # 2. 把 text 存到 self.texts
        self.texts.append(text)

    def add_batch(self, texts: list[str], vectors: list[list[float]]):
        """批量添加"""
        # 循环调 add 就行
        for text, vector in zip(texts, vectors):
            self.add(text,vector)

    def search(self, query_vec: list[float], top_k: int = 5) -> list[dict]:
        # 1. 遍历 self.vectors，计算每条和 query_vec 的相似度
        scores = []
        for i in range(len(self.vectors)):
            score=self._cosine_similarity(query_vec, self.vectors[i])
            scores.append({
                "text":self.texts[i],
                "score":score,
                "index":i
            })
        # 2. 按相似度从高到低排序
        scores.sort(key=lambda x:x["score"],reverse=True)
        # 3. 取前 top_k 条返回
        return scores[:top_k]

    def _cosine_similarity(self, v1, v2) -> float:
        """计算余弦相似度"""
        #点积
        dot=0
        for i in range(len(v1)):
            dot+=v1[i]*v2[i]

        #长度
        lengthV1=0
        for i in range(len(v1)):
            lengthV1+=v1[i]*v1[i]
        lengthV1=lengthV1**0.5

        lengthV2=0
        for i in range(len(v2)):
            lengthV2+=v2[i]*v2[i]
        lengthV2=lengthV2**0.5
        #返回值
        if lengthV1==0 or lengthV2==0:
            return 0
        return dot/(lengthV1*lengthV2)

    def add_from_file(self, file_path: str,embedding_service,chunk_method="sentence"):
        text=read_file(file_path)

        if chunk_method == "sentence":
            chunks = chunk_by_sentence(text)
        elif chunk_method == "paragraph":
            chunks = chunk_by_paragraph(text)
        elif chunk_method == "jieba":  # ← 加上这一支
            chunks = chunk_by_jieba(text)
        else:
            raise ValueError(f"不支持的分块方法: {chunk_method}")

        vectors=embedding_service.encode_batch(chunks)
        self.add_batch(chunks,vectors)
        print(f"文件 {file_path} 加载完成: {len(chunks)} 个文本块")

    @property
    def count(self) -> int:
        """返回向量数量"""
        return len(self.vectors)

    def save(self, path: str):
        """保存到硬盘"""
        # 转成 numpy 数组再保存
        np_vectors = np.array(self.vectors)
        np.save(f"{path}_vectors.npy", np_vectors)

        # 文本存为 JSON
        with open(f"{path}_texts.json", "w", encoding="utf-8") as f:
            json.dump(self.texts, f, ensure_ascii=False, indent=2)

        print(f"数据已保存到 {path}（{len(self.texts)} 条）")

    def load(self, path: str):
        """从硬盘加载"""
        # 加载向量
        np_vectors = np.load(f"{path}_vectors.npy")
        self.vectors = np_vectors.tolist()

        # 加载文本
        with open(f"{path}_texts.json", "r", encoding="utf-8") as f:
            self.texts = json.load(f)

        print(f"数据已加载（{len(self.texts)} 条）")
