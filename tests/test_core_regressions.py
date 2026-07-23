import os
import tempfile

from src.parsing.document import read_file, read_file_structured
from src.parsing.parse_result import TextBlock
from src.retrieval.chunk import Chunk
from src.retrieval.chunk_repository import ChunkRepository
from src.retrieval.chunker import chunk_blocks, chunk_by_jieba
from src.retrieval.retriever import Retriever
from src.logger import get_logger, setup_logging
from src.services.index_service import IndexService
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


def test_retriever_falls_back_when_optional_reranker_fails():
    class FailingReranker:
        def rerank(self, query, candidates, top_k):
            raise RuntimeError("model unavailable")

    db = FaissVectorStore(2)
    db.add_batch(["reliable fallback"], [[1.0, 0.0]], doc_name="repo.txt")
    retriever = Retriever(db, reranker=FailingReranker())
    retriever.add_texts(db.texts)

    result = retriever.search("fallback", [1.0, 0.0], top_k=1, threshold=0.0)

    assert result[0]["text"] == "reliable fallback"
    assert retriever.reranker is None


def test_chunk_metadata_survives_chunking_storage_and_retrieval():
    chunks = chunk_blocks(
        [TextBlock(text="额定电压为 12kV。", source="pdf_hybrid", page=2)],
        method="sentence",
        doc="manual.pdf",
    )
    assert chunks == [
        Chunk(
            text="额定电压为 12kV",
            doc="manual.pdf",
            page=2,
            source="pdf_hybrid",
        )
    ]

    db = FaissVectorStore(2)
    repository = ChunkRepository(db)
    repository.add_batch(chunks, [[1.0, 0.0]])
    retriever = Retriever(db, repository=repository)
    retriever.add_texts(repository.all_texts())

    record = repository.get(0)
    result = retriever.search("额定电压", [1.0, 0.0], top_k=1, threshold=0.0)

    assert record.page == 2
    assert record.source == "pdf_hybrid"
    assert result[0]["page"] == 2
    assert result[0]["source"] == "pdf_hybrid"

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "index")
        db.save(path)
        loaded = FaissVectorStore(2)
        loaded.load(path)
        loaded_record = ChunkRepository(loaded).get(0)
        loaded.close()
        db.close()

    assert loaded_record.page == 2
    assert loaded_record.source == "pdf_hybrid"


def test_jieba_chunking_keeps_normal_prose_on_sentence_boundaries():
    text = (
        "第一段用于说明当前系统的处理流程和输入输出关系。"
        "第二段说明系统需要创建更加可持续的社会，并持续进行维护。"
        "第三段给出后续的配置要求和验收步骤。"
    )

    chunks = chunk_by_jieba(text, max_words=18, overlap=4)

    assert any("创建更加可持续的社会" in chunk for chunk in chunks)
    assert all(not chunk.startswith("的社会") for chunk in chunks)
    assert all(not chunk.startswith("的") for chunk in chunks)


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


def test_retriever_trace_supplements_chunks_inside_high_scoring_documents():
    db = FaissVectorStore(2)
    db.add_batch(
        [
            "alpha overview",
            "hidden evidence omega",
            "neighbor proof tail",
            "unrelated document",
        ],
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.9], [0.9, 0.1]],
        doc_name="manual.pdf",
    )
    db.meta[3]["doc"] = "other.pdf"
    db.doc_registry = {"manual.pdf": [0, 1, 2], "other.pdf": [3]}
    retriever = Retriever(db)

    trace = retriever.search_with_trace(
        "hidden evidence omega",
        [1.0, 0.0],
        top_k=2,
        retrieve_top=1,
        threshold=0.0,
    )

    assert trace["first_stage"][0]["index"] == 0
    assert any(item["index"] == 1 for item in trace["doc_internal"])
    assert any(item["index"] == 1 for item in trace["final"])
    assert isinstance(retriever.search("hidden evidence omega", [1.0, 0.0]), list)
