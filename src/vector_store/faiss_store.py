import faiss
from .base import BaseVectorStore


class FaissVectorStore(BaseVectorStore):
    """Flat IP 索引"""

    def __init__(self, dimension: int):
        super().__init__(dimension, index_type="flat")

    def _build_index(self):
        return faiss.IndexFlatIP(self.dimension)
