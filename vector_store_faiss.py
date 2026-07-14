import faiss
import numpy as np
import json
from document_loader import read_file, chunk_by_sentence, chunk_by_paragraph,chunk_by_jieba

class FaissVectorStore:
    def __init__(self,dimension:int):
        """初始化索引"""
        self.dimension=dimension
        self.index=faiss.IndexFlatIP(dimension)
        self.texts=[]


    def add(self,text:str,vector:list[float]):
        vec=np.array([vector],dtype=np.float32)
        self.index.add(vec)
        self.texts.append(text)

    def add_batch(self,texts:list[str],vectors:list[list[float]]):
        for text,vector in zip(texts,vectors):
            self.add(text,vector)

    def add_from_file(self, file_path: str, embedding_service, chunk_method="sentence"):


        text = read_file(file_path)

        if chunk_method == "sentence":
            chunks = chunk_by_sentence(text)
        elif chunk_method == "paragraph":
            chunks = chunk_by_paragraph(text)
        elif chunk_method == "jieba":  # ← 加上这一支
            chunks = chunk_by_jieba(text)
        else:
            raise ValueError(f"不支持的分块方法: {chunk_method}")

        vectors = embedding_service.encode_batch(chunks)
        self.add_batch(chunks, vectors)
        print(f"文件 {file_path} 加载完成: {len(chunks)} 个文本块")


    def search(self,text:str,top_k=5)->list[dict]:
        #获取查询向量
        q=np.array([text],dtype=np.float32)
        faiss.normalize_L2(q) #归一化

        #用faiss去搜索对应的文本数据
        scores,indices=self.index.search(q,top_k)

        #转文本
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
        faiss.write_index(self.index, f"{path}.faiss")  # ← 存 FAISS 索引

        with open(f"{path}_texts.json","w",encoding="utf-8") as f:
            json.dump(self.texts,f)

    def load(self, path: str):
        self.index = faiss.read_index(f"{path}.faiss")  # ← 读 FAISS 索引

        with open(f"{path}_texts.json", "r", encoding="utf-8") as f:
            self.texts = json.load(f)  # ← 读文本
