import os
import tempfile

from src.document import read_file, read_file_structured
from src.logger import get_logger, setup_logging
from src.retriever import Retriever
from src.chunk_repository import ChunkRepository
from src.index_service import IndexService
from src.vector_store.faiss_store import FaissVectorStore
from src.vector_store.ivf_store import IvfVectorStore


def test_logger_accepts_structured_keyword_fields():
    setup_logging(log_file="", fmt="structured")
    log = get_logger("test")

    log.info("event", count=1, doc="sample.txt")


def test_read_file_structured_preserves_text_interface_for_txt_files():
    fd, path = tempfile.mkstemp(suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("hello\nworld")

        parsed = read_file_structured(path)

        assert parsed.parser == "txt"
        assert parsed.text == "hello\nworld"
        assert parsed.blocks[0].source == "txt"
        assert read_file(path) == parsed.text
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_vector_store_search_filters_deleted_chunks():
    db = FaissVectorStore(2)
    db.add_batch(
        ["deleted chunk", "alive chunk"],
        [[1.0, 0.0], [0.9, 0.1]],
        doc_name="doc.txt",
    )

    db.delete_doc("doc.txt")

    assert db.search([1.0, 0.0], top_k=1) == []


def test_retriever_filters_deleted_chunks_from_dense_and_bm25_paths():
    db = FaissVectorStore(2)
    db.add_batch(["deleted apple", "alive apple"], [[1.0, 0.0], [0.9, 0.1]], doc_name="deleted.txt")
    db.add_batch(["surviving apple"], [[0.8, 0.2]], doc_name="alive.txt")
    retriever = Retriever(db)
    retriever.add_texts(db.texts)

    db.delete_doc("deleted.txt")
    results = retriever.search("apple", [1.0, 0.0], top_k=3, threshold=0.0)

    assert results
    assert all(result["doc"] != "deleted.txt" for result in results)


def test_ivf_vector_store_can_initialize_and_add_vectors():
    db = IvfVectorStore(2, nlist=1)

    db.add_batch(["a", "b"], [[1.0, 0.0], [0.0, 1.0]], doc_name="doc.txt")

    assert db.count == 2
    assert db.search([1.0, 0.0], top_k=1)[0]["text"] == "a"


def test_retriever_uses_chunk_repository_boundary():
    db = FaissVectorStore(2)
    db.add_batch(["repository text"], [[1.0, 0.0]], doc_name="repo.txt")
    repository = ChunkRepository(db)
    retriever = Retriever(db, repository=repository)
    retriever.add_texts(repository.all_texts())

    result = retriever.search("repository", [1.0, 0.0], top_k=1,
                              threshold=0.0)

    assert result[0]["text"] == "repository text"
    assert result[0]["doc"] == "repo.txt"


def test_index_service_hides_index_switch_state_migration():
    db = FaissVectorStore(2)
    db.add_batch(["switch me"], [[1.0, 0.0]], doc_name="switch.txt")
    service = IndexService(
        db,
        index_factory=lambda index_type, dimension: FaissVectorStore(dimension),
    )

    service.switch_index("flat")

    record = service.repository.get(0)
    assert service.count == 1
    assert record.text == "switch me"
    assert record.doc == "switch.txt"
