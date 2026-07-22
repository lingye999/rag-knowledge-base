"""Application service coordinating vector index and chunk records."""
from .chunk_repository import ChunkRepository


class IndexService:
    """Owns index lifecycle operations used by CLI and ingestion.

    Callers do not need to know how a FAISS index is rebuilt or persisted.
    """

    def __init__(self, vector_store, index_factory=None):
        self.vector_store = vector_store
        self.repository = ChunkRepository(vector_store)
        self._index_factory = index_factory
        self._save_path = None

    @property
    def count(self) -> int:
        return self.vector_store.count

    @property
    def has_save_path(self) -> bool:
        return self._save_path is not None

    def add_chunks(self, texts, vectors, doc_name=None, qualities=None):
        return self.repository.add_batch(
            texts,
            vectors,
            doc_name=doc_name,
            qualities=qualities,
        )

    def delete_document(self, doc_name: str):
        self.repository.delete_document(doc_name)

    def list_documents(self) -> list[dict]:
        result = []
        for doc_name in self.repository.document_names(include_deleted=True):
            chunks = self.repository.records_by_document(
                doc_name, include_deleted=True
            )
            result.append({
                "doc": doc_name,
                "count": len(chunks),
                "deleted": all(chunk.deleted for chunk in chunks),
            })
        return result

    def clear(self):
        self.vector_store.clear()

    def compact(self):
        self.vector_store.compact()

    def save(self, path=None):
        target = path or self._save_path
        if not target:
            return None
        self.vector_store.save(target)
        self._save_path = target
        return target

    def load(self, path):
        self.vector_store.load(path)
        self._save_path = path
        self.repository.refresh_store(self.vector_store)

    def switch_index(self, index_type: str):
        if self._index_factory is None:
            raise RuntimeError("IndexService requires an index factory to switch indexes")

        old_store = self.vector_store
        snapshot = old_store.export_state()
        new_store = self._index_factory(index_type, old_store.dimension)
        new_store.import_state(snapshot)

        self.vector_store = new_store
        self.repository.refresh_store(new_store)
        if self._save_path:
            new_store.save(self._save_path)

    def bind_consumers(self, retriever=None, ingestion=None):
        """Update collaborators after the underlying index is replaced."""
        if retriever is not None:
            retriever.bind_store(self.vector_store, self.repository)
        if ingestion is not None:
            ingestion.bind_store(self.vector_store, self.repository)

    def search(self, query_vec, top_k=5):
        return self.vector_store.search(query_vec, top_k=top_k)
