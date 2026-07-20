import faiss
from .base import BaseVectorStore


class HnswVectorStore(BaseVectorStore):
    """HNSW 图索引"""

    def __init__(self, dimension: int, M: int = 32):
        super().__init__(dimension, index_type="hnsw", index_params={"M": M})

    def _build_index(self):
        M = self._index_params["M"]
        idx = faiss.IndexHNSWFlat(self.dimension, M,
                                   faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction = 128
        idx.hnsw.efSearch = 64
        return idx
