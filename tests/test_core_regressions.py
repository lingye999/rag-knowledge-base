import os
import tempfile

from src.parsing.document import read_file, read_file_structured
from src.parsing.parse_result import TextBlock
from src.parsing.quality_gate import assess_parse_quality, needs_ocr_fallback
from src.retrieval.chunk import Chunk
from src.retrieval.chunk_repository import ChunkRepository
from src.retrieval.chunker import chunk_blocks, chunk_by_jieba
from src.retrieval.bm25_index import BM25TextIndex
from src.retrieval.final_selector import FinalContextSelector
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import Retriever
from src.retrieval.text_analyzer import TextAnalyzer
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


def test_parse_quality_gate_requests_ocr_for_empty_or_garbled_text():
    settings = {
        "enabled": True,
        "min_total_chars": 80,
        "min_readable_ratio": 0.55,
        "max_garbled_ratio": 0.03,
    }
    clean = assess_parse_quality([
        TextBlock(text="额定电压为 12kV。" * 20, source="pdf_hybrid")
    ])
    garbled = assess_parse_quality([
        TextBlock(text="□�▯" * 40, source="pdf_hybrid")
    ])

    assert not needs_ocr_fallback(clean, settings)
    assert needs_ocr_fallback(garbled, settings)


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


def test_text_analyzer_normalizes_domain_tokens_consistently():
    analyzer = TextAnalyzer()

    document_tokens = analyzer.analyze("E VAC 的分闸时间为 20 至 50 毫秒，符合 GB/Z 185.4。")
    query_tokens = analyzer.analyze("E-VAC 分闸时间 20~50ms GB/Z185.4")

    for token in ("e-vac", "20~50ms", "gb/z185.4"):
        assert token in document_tokens
        assert token in query_tokens


def test_bm25_domain_dictionary_and_query_rewrite_align_aliases():
    settings = {
        "domain_terms": [
            {
                "canonical": "agentid",
                "aliases": ["agent id", "智能体id", "智能体身份码"],
            },
        ],
        "query_rewrites": [
            {"trigger": "跳闸", "expansions": ["分闸", "分闸时间"]},
        ],
    }
    analyzer = TextAnalyzer(settings)

    assert "agentid" in analyzer.analyze("agentId 必须唯一")
    assert "agentid" in analyzer.analyze_query("智能体 ID 的要求")
    rewrite_tokens = analyzer.analyze_query("断路器跳闸需要多久")
    assert {"分闸", "时间"} <= set(rewrite_tokens)

    index = BM25TextIndex(settings)
    index.rebuild([
        "断路器的分闸时间为 20~50ms",
        "agentId 必须唯一",
        "设备安装维护说明",
        "医疗器械市场分析",
    ])

    assert index.search("断路器跳闸需要多久", 2, is_deleted=lambda _: False)[0][1] == 0
    assert index.search("智能体 ID 的要求", 2, is_deleted=lambda _: False)[0][1] == 1


def test_bm25_query_rewrite_can_be_disabled_for_ablation():
    texts = [
        "断路器的分闸时间为 20~50ms",
        "设备安装维护说明",
        "医疗器械市场分析",
    ]
    rewritten = BM25TextIndex({
        "enable_query_rewrite": True,
        "query_rewrites": [{"trigger": "跳闸", "expansions": ["分闸"]}],
    })
    baseline = BM25TextIndex({
        "enable_query_rewrite": False,
        "query_rewrites": [{"trigger": "跳闸", "expansions": ["分闸"]}],
    })
    rewritten.rebuild(texts)
    baseline.rebuild(texts)

    rewritten_hit = rewritten.search("断路器跳闸需要多久", 3, lambda _: False)[0]
    baseline_hit = baseline.search("断路器跳闸需要多久", 3, lambda _: False)[0]

    assert rewritten_hit[1] == baseline_hit[1] == 0
    assert rewritten_hit[2] > baseline_hit[2]


def test_final_selector_reuses_bm25_query_rewrite_for_lexical_scoring():
    selector = FinalContextSelector({
        "bm25": {
            "enable_query_rewrite": True,
            "query_rewrites": [{"trigger": "跳闸", "expansions": ["分闸"]}],
        },
    })
    rescored = selector.rescore(
        "断路器跳闸需要多久",
        [{"text": "断路器分闸时间为 20~50ms", "score": 0.5, "index": 0}],
    )

    assert rescored[0]["lexical_score"] > 0


def test_bm25_discards_zero_score_candidates():
    index = BM25TextIndex()
    index.rebuild([
        "rated voltage 12kV",
        "installation notes",
        "maintenance guide",
    ])

    assert index.search("unmatched-token", 5, is_deleted=lambda _: False) == []
    hits = index.search("rated voltage", 5, is_deleted=lambda _: False)
    assert hits
    assert all(score > 0 for _, _, score in hits)


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
    db.add_batch(
        ["reliable fallback", "secondary candidate", "third candidate"],
        [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]],
        doc_name="repo.txt",
    )
    retriever = Retriever(db, reranker=FailingReranker())
    retriever.add_texts(db.texts)

    result = retriever.search("fallback是什么", [1.0, 0.0], top_k=1, threshold=0.0)

    assert result[0]["text"] == "reliable fallback"
    assert retriever.reranker is None


def test_reranker_preload_loads_local_model_once(monkeypatch):
    calls = []

    class StubCrossEncoder:
        def __init__(self, model_name, **kwargs):
            calls.append((model_name, kwargs))

    monkeypatch.setattr(
        "src.retrieval.reranker.CrossEncoder",
        StubCrossEncoder,
    )
    reranker = Reranker("D:/models/reranker", local_files_only=True)

    reranker.preload()
    reranker.preload()

    assert reranker.is_loaded
    assert calls == [
        ("D:/models/reranker", {"device": "cpu", "local_files_only": True})
    ]


def test_final_selector_only_runs_reranker_for_ambiguous_or_complex_queries():
    class RecordingReranker:
        def __init__(self):
            self.calls = 0

        def rerank(self, query, candidates, top_k):
            self.calls += 1
            return candidates[:top_k]

    selector = FinalContextSelector({
        "reranker": {
            "min_candidates": 2,
            "max_candidates": 2,
            "ambiguous_score_gap": 0.05,
            "complex_query_cues": ["是什么"],
        },
        "duplicate_threshold": 0.82,
    })
    candidates = [
        {"text": "first", "score": 0.9, "doc": "a", "index": 0},
        {"text": "second", "score": 0.7, "doc": "a", "index": 1},
        {"text": "third", "score": 0.6, "doc": "a", "index": 2},
    ]
    reranker = RecordingReranker()

    selector.select("额定电压是多少", candidates, top_k=1, threshold=0.0, reranker=reranker)
    selector.select("额定电压是什么", candidates, top_k=1, threshold=0.0, reranker=reranker)

    assert reranker.calls == 1


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
    assert loaded_record.chunk_type == "content"


def test_page_context_chunk_keeps_cross_chunk_evidence_together():
    text = "APG工艺" + "甲" * 420 + "真空灭弧室采用固封结构"
    chunks = chunk_blocks(
        [TextBlock(text=text, source="pdf_hybrid", page=5)],
        method="size",
        doc="manual.pdf",
        include_page_context=True,
        page_context_max_chars=900,
    )

    assert len(chunks) == 3
    assert any(
        all(term in chunk.text for term in ("APG工艺", "真空灭弧室", "固封"))
        for chunk in chunks
    )
    assert chunks[-1].chunk_type == "page_context"


def test_final_selector_limits_page_context_chunks_without_dropping_content():
    selector = FinalContextSelector({
        "page_context_max_final_chunks": 1,
        "duplicate_threshold": 0.82,
    })
    candidates = [
        {"text": "page context one", "score": 0.9, "doc": "a", "index": 0, "chunk_type": "page_context"},
        {"text": "page context two", "score": 0.8, "doc": "a", "index": 1, "chunk_type": "page_context"},
        {"text": "precise content", "score": 0.7, "doc": "a", "index": 2, "chunk_type": "content"},
    ]

    selected, trace = selector.diversify_and_filter(
        candidates,
        top_k=3,
        threshold=0.0,
        with_trace=True,
    )

    assert [item["index"] for item in selected] == [0, 2]
    assert trace["reason_counts"]["page_context_limit"] == 1


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


def test_final_rescore_promotes_definition_entry_from_candidate_pool():
    db = FaissVectorStore(2)
    retriever = Retriever(db)
    candidates = [
        {
            "text": "Table 2 Skill attributes skillId skillName description",
            "score": 0.95,
            "doc": "standard.pdf",
            "index": 0,
            "page": 9,
        },
        {
            "text": "This standard defines the interoperability framework.",
            "score": 0.94,
            "doc": "standard.pdf",
            "index": 1,
            "page": 6,
        },
        {
            "text": "Foreword and drafting rules for the standard.",
            "score": 0.93,
            "doc": "standard.pdf",
            "index": 2,
            "page": 5,
        },
        {
            "text": "3.4 Skill agent capability. Example: office automation QA.",
            "score": 0.90,
            "doc": "standard.pdf",
            "index": 3,
            "page": 7,
        },
    ]

    final = retriever._select_final_contexts(
        'What is "Skill"?',
        candidates,
        top_k=3,
        threshold=0.0,
    )

    assert any(item["index"] == 3 for item in final)
    definition = next(item for item in final if item["index"] == 3)
    assert definition["definition_score"] == 1.0
    assert definition["score"] > definition["base_score"]


def test_final_rescore_prefers_definition_entry_over_attribute_table():
    selector = FinalContextSelector({
        "final_lexical_weight": 0.06,
        "final_anchor_weight": 0.08,
        "final_definition_weight": 0.06,
        "final_definition_entry_weight": 0.12,
        "final_phrase_weight": 0.08,
        "duplicate_threshold": 0.82,
    })
    candidates = [
        {
            "text": "表 技能的属性：技能标识、技能名称、技能描述。",
            "score": 0.93,
            "doc": "standard.pdf",
            "index": 0,
            "page": 9,
        },
        {
            "text": "3.4 技能：智能体的功能。",
            "score": 0.85,
            "doc": "standard.pdf",
            "index": 1,
            "page": 7,
        },
    ]

    final, _, _ = selector.select(
        '标准中“技能”指的是什么？',
        candidates,
        top_k=1,
        threshold=0.0,
    )

    assert final[0]["index"] == 1
    assert final[0]["definition_entry_score"] == 1.0


def test_final_rescore_promotes_numeric_answer_with_matching_time_unit():
    selector = FinalContextSelector({
        "final_lexical_weight": 0.06,
        "final_anchor_weight": 0.08,
        "final_phrase_weight": 0.08,
        "final_numeric_weight": 0.18,
        "final_anchor_min_lexical_coverage": 0.25,
        "duplicate_threshold": 0.82,
    })
    candidates = [
        {
            "text": "E-VAC 断路器外形尺寸为 210mm。",
            "score": 0.98,
            "doc": "manual.pdf",
            "index": 0,
            "page": 1,
        },
        {
            "text": "E-VAC 断路器的分闸时间范围为 20~50ms。",
            "score": 0.90,
            "doc": "manual.pdf",
            "index": 1,
            "page": 2,
        },
    ]

    rescored = selector.rescore("E-VAC 断路器的分闸时间范围是多少？", candidates)
    ranked = sorted(rescored, key=lambda item: item["score"], reverse=True)

    assert ranked[0]["index"] == 1
    assert ranked[0]["numeric_score"] > ranked[1]["numeric_score"]
    assert ranked[0]["numeric_score"] > 0


def test_final_selector_trace_records_candidate_decisions():
    selector = FinalContextSelector({"duplicate_threshold": 0.82})
    candidates = [
        {"text": "rated voltage 12kV", "score": 0.99, "doc": "a.pdf", "index": 0, "page": 1},
        {"text": "rated voltage 12kV", "score": 0.98, "doc": "a.pdf", "index": 1, "page": 1},
        {"text": "rated voltage table", "score": 0.97, "doc": "a.pdf", "index": 2, "page": 1},
        {"text": "rated voltage note", "score": 0.96, "doc": "a.pdf", "index": 3, "page": 1},
        {"text": "rated voltage appendix", "score": 0.95, "doc": "a.pdf", "index": 4, "page": 1},
        {"text": "installation notes", "score": 0.94, "doc": "a.pdf", "index": 5, "page": 2},
    ]

    selected, trace = selector.diversify_and_filter(
        candidates,
        top_k=3,
        threshold=0.0,
        with_trace=True,
    )

    assert [item["index"] for item in selected] == [0, 2, 3]
    assert trace["reason_counts"] == {
        "near_duplicate": 1,
        "page_limit": 1,
        "selected": 3,
        "top_k_limit": 1,
    }
    duplicate = next(item for item in trace["decisions"] if item["index"] == 1)
    assert duplicate["duplicate_of"] == 0
