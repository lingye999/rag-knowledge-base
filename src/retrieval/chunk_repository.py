"""Access boundary for chunk text and metadata.

The current vector stores still keep legacy in-memory arrays for backward
compatibility. New code should use this repository instead of reading those
arrays directly.
"""
from dataclasses import dataclass
from typing import Any

from .chunk import Chunk


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    index: int
    text: str
    doc: str = ""
    quality: float = 0.5
    page: int | None = None
    source: str | None = None
    deleted: bool = False


class ChunkRepository:
    """Stable read/write interface for chunk records.

    This is deliberately an adapter around the existing store in the first
    migration step. Persistence can move here later without changing callers.
    """

    def __init__(self, vector_store):
        self._store = vector_store

    def get(self, index: int) -> ChunkRecord | None:
        if index < 0 or index >= self._store.count:
            return None
        metadata = self._store.get_metadata(index)
        return ChunkRecord(
            index=index,
            text=self._store.get_text(index),
            doc=metadata.get("doc", ""),
            quality=float(metadata.get("quality", 0.5)),
            page=metadata.get("page"),
            source=metadata.get("source"),
            deleted=self._store.is_deleted(index),
        )

    def get_text(self, index: int) -> str:
        record = self.get(index)
        return record.text if record else ""

    def get_metadata(self, index: int) -> dict[str, Any]:
        record = self.get(index)
        if record is None:
            return {}
        return {
            "doc": record.doc,
            "quality": record.quality,
            "page": record.page,
            "source": record.source,
        }

    def all_texts(self, include_deleted: bool = True) -> list[str]:
        return [
            self._store.get_text(i)
            for i in range(self._store.count)
            if include_deleted or not self._store.is_deleted(i)
        ]

    def records(self, include_deleted: bool = False) -> list[ChunkRecord]:
        result = []
        for index in range(self._store.count):
            record = self.get(index)
            if record and (include_deleted or not record.deleted):
                result.append(record)
        return result

    def records_by_document(self, doc_name: str,
                            include_deleted: bool = False) -> list[ChunkRecord]:
        return [
            record for record in self.records(include_deleted=True)
            if record.doc == doc_name and (include_deleted or not record.deleted)
        ]

    def document_names(self, include_deleted: bool = False) -> list[str]:
        names = []
        for record in self.records(include_deleted=True):
            if not record.doc or (record.deleted and not include_deleted):
                continue
            if record.doc not in names:
                names.append(record.doc)
        return names

    def add_batch(self, chunks, vectors, doc_name=None, qualities=None,
                  pages=None, sources=None):
        """Store legacy text lists or structured :class:`Chunk` instances."""
        if chunks and isinstance(chunks[0], Chunk):
            chunk_docs = {chunk.doc for chunk in chunks}
            if doc_name is None:
                if len(chunk_docs) != 1:
                    raise ValueError("A batch of Chunk objects must belong to one document")
                doc_name = chunk_docs.pop()
            texts = [chunk.text for chunk in chunks]
            pages = [chunk.page for chunk in chunks]
            sources = [chunk.source for chunk in chunks]
            qualities = qualities or [chunk.quality for chunk in chunks]
        else:
            texts = chunks
        return self._store.add_batch(
            texts,
            vectors,
            doc_name=doc_name,
            qualities=qualities,
            pages=pages,
            sources=sources,
        )

    def delete_document(self, doc_name: str):
        return self._store.delete_doc(doc_name)

    def clear(self):
        self._store.clear()

    def refresh_store(self, vector_store):
        self._store = vector_store
