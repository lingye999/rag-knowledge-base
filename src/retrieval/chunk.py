"""The structured unit passed from parsing to indexing and retrieval."""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Chunk:
    """A searchable piece of a document with source provenance."""

    text: str
    doc: str
    page: int | None = None
    source: str | None = None
    quality: float = 0.5
    chunk_type: str = "content"
