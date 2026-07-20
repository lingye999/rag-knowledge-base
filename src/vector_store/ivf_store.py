import faiss
from .base import BaseVectorStore


class IvfVectorStore(BaseVectorStore):
    """IVF 倒排索引"""

    def __init__(self, dimension: int, nlist: int = 100):
        super().__init__(dimension, index_type="ivf",
                         index_params={"nlist": nlist})
        self.nlist = nlist
        self.is_trained = False

    def _build_index(self):
        """返回一个占位索引，真正训练在 _before_add 中完成"""
        quantizer = faiss.IndexFlatIP(self.dimension)
        idx = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist,
                                 faiss.METRIC_INNER_PRODUCT)
        idx.nprobe = 10
        return idx

    def _before_add(self, vectors):
        """首次添加时训练 IVF 索引"""
        if not self.is_trained:
            n_train = len(vectors)
            nlist = min(self.nlist, n_train)
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist,
                                            faiss.METRIC_INNER_PRODUCT)
            self.index.nprobe = min(10, nlist)
            self.index.train(vectors)
            self.is_trained = True
