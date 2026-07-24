import os
import tempfile
from unittest.mock import patch

from eval.evaluation import (
    evaluate_answer,
    evaluate_citations,
    evaluate_context_coverage,
    evaluate_retrieval,
)
from eval.generation_judge import GenerationJudge
from eval.run_extraction_checks import find_forbidden_tokens
from eval.run_generation_eval import (
    QA_DATASETS,
    RETRIEVAL_DATASETS,
    _load_retrieval_contracts,
    _select_qa_rows,
)
from eval.run_retrieval_diagnostics import diagnose_trace
from src.parsing.document import read_file, read_file_structured


def test_retrieval_requires_all_terms_in_one_chunk():
    query = {
        "relevant_doc": "manual.pdf",
        "evidence": [{"must_contain": ["12kV", "rated voltage"]}],
    }

    partial = [{"doc": "manual.pdf", "text": "12kV"}]
    complete = [{"doc": "manual.pdf", "text": "rated voltage: 12kV"}]

    partial_score = evaluate_retrieval(query, partial, 5)
    assert not partial_score["hit"]
    assert partial_score["evidence_recall"] == 0.5
    assert evaluate_retrieval(query, complete, 5)["hit"]


def test_list_retrieval_can_opt_into_same_doc_multi_chunk_evidence():
    query = {
        "relevant_doc": "manual.pdf",
        "evidence": [{"must_contain": ["handcart", "fixed"]}],
        "evidence_policy": "multi_chunk_same_doc",
    }
    split_evidence = [
        {"doc": "manual.pdf", "text": "handcart type"},
        {"doc": "other.pdf", "text": "fixed type"},
        {"doc": "manual.pdf", "text": "fixed type"},
    ]

    score = evaluate_retrieval(query, split_evidence, 5)

    assert score["hit"]
    assert score["evidence_rank"] == 3
    assert score["matched"] == ["handcart", "fixed"]
    assert score["supporting_hits"] == [
        {"rank": 1, "matched": ["handcart"]},
        {"rank": 3, "matched": ["fixed"]},
    ]
    assert score["evidence_recall"] == 1.0
    assert score["context_precision"] == 2 / 3
    assert score["ndcg"] == 0.75


def test_retrieval_can_opt_into_same_page_evidence():
    query = {
        "relevant_doc": "manual.pdf",
        "evidence": [{"page": 2, "must_contain": ["rated voltage", "12kV"]}],
        "evidence_policy": "same_page",
    }
    split_page = [
        {"doc": "manual.pdf", "page": 2, "text": "rated voltage"},
        {"doc": "manual.pdf", "page": 3, "text": "12kV"},
        {"doc": "manual.pdf", "page": 2, "text": "12kV"},
    ]

    score = evaluate_retrieval(query, split_page, 5)

    assert score["hit"]
    assert score["evidence_rank"] == 3
    assert score["matched"] == ["rated voltage", "12kV"]


def test_retrieval_can_opt_into_neighbor_window_evidence():
    query = {
        "relevant_doc": "manual.pdf",
        "evidence": [{"page": 2, "window": 1, "must_contain": ["URL", "FQDN"]}],
        "evidence_policy": "window_chunk",
    }
    neighboring_hits = [
        {"doc": "manual.pdf", "page": 2, "index": 10, "text": "access by URL"},
        {"doc": "manual.pdf", "page": 2, "index": 11, "text": "or FQDN"},
    ]

    score = evaluate_retrieval(query, neighboring_hits, 5)

    assert score["hit"]
    assert score["evidence_rank"] == 1
    assert score["matched"] == ["URL", "FQDN"]


def test_retrieval_requires_the_expected_evidence_page_when_anchored():
    query = {
        "relevant_doc": "manual.pdf",
        "evidence": [{"page": 3, "must_contain": ["12kV"]}],
    }
    wrong_page = [{"doc": "manual.pdf", "page": 2, "text": "12kV"}]
    right_page = [{"doc": "manual.pdf", "page": 3, "text": "12kV"}]

    assert not evaluate_retrieval(query, wrong_page, 5)["hit"]
    assert evaluate_retrieval(query, wrong_page, 5)["evidence_recall"] == 0.0
    assert evaluate_retrieval(query, right_page, 5)["hit"]


def test_negative_retrieval_checks_unsupported_evidence_not_empty_results():
    query = {
        "relevant_doc": None,
        "forbidden_evidence": ["bluetooth remote control"],
    }

    unrelated = [{"doc": "manual.pdf", "text": "rated voltage: 12kV"}]
    unsafe = [{"doc": "manual.pdf", "text": "bluetooth remote control"}]

    assert evaluate_retrieval(query, unrelated, 5)["hit"]
    assert not evaluate_retrieval(query, unsafe, 5)["hit"]


def test_answer_check_requires_facts_and_citation_when_requested():
    golden = {
        "required_facts": ["12kV"],
        "forbidden_claims": ["24kV"],
        "citation_required": True,
    }

    assert not evaluate_answer("The answer is 12kV.", golden)["passed"]
    assert evaluate_answer("The answer is 12kV. [p. 5]", golden)["passed"]


def test_generation_context_coverage_and_citation_validation():
    golden = {
        "required_facts": ["12kV", "rated voltage"],
    }
    contexts = [
        {"id": "chunk-7", "text": "The rated voltage is 12kV."},
        {"id": "chunk-8", "text": "Installation information."},
    ]

    coverage = evaluate_context_coverage(golden, contexts)
    citations = evaluate_citations(
        "The voltage is 12kV. [chunk-7] [chunk-999]",
        contexts,
        citation_required=True,
    )

    assert coverage["passed"]
    assert coverage["required_fact_ratio"] == 1.0
    assert citations["valid_citation_ids"] == ["chunk-7"]
    assert citations["invalid_citation_ids"] == ["chunk-999"]
    assert not citations["passed"]


def test_generation_datasets_link_to_matching_retrieval_contracts():
    for name, qa_path in QA_DATASETS.items():
        rows = _select_qa_rows(qa_path, ids=None)
        contracts = _load_retrieval_contracts(RETRIEVAL_DATASETS[name], rows)

        assert contracts
        assert {row["retrieval_id"] for row in rows} <= set(contracts)


class _FakeJudgeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)

    def complete_json(self, system_prompt, user_prompt):
        return next(self.responses)


def test_faithfulness_rejects_judge_evidence_not_in_contexts():
    judge = GenerationJudge(_FakeJudgeLLM([{
        "claims": [
            {
                "claim": "The voltage is 12kV.",
                "supported": True,
                "supporting_chunk_ids": ["chunk-1"],
            },
            {
                "claim": "Bluetooth is supported.",
                "supported": True,
                "supporting_chunk_ids": ["chunk-404"],
            },
        ]
    }]))

    result = judge.judge_faithfulness(
        "The voltage is 12kV and Bluetooth is supported.",
        [{"id": "chunk-1", "text": "Rated voltage: 12kV."}],
    )

    assert result["score"] == 0.5
    assert result["unsupported_claims"] == ["Bluetooth is supported."]


def test_extraction_check_can_allow_form_boxes_without_ignoring_garbled_text():
    row = {
        "must_not_contain": ["□", "�"],
        "allow_form_boxes": True,
    }

    assert find_forbidden_tokens(row, "E-VAC □ handcart") == []
    assert find_forbidden_tokens(row, "E-VAC � handcart") == ["�"]


def test_structured_pdf_results_preserve_page_boundaries():
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    try:
        with patch("src.parsing.document.read_pdf_hybrid_pages", return_value=["one", "two"]):
            parsed = read_file_structured(path)

        assert [block.page for block in parsed.blocks] == [1, 2]
        assert parsed.text == "one\ntwo"
        with patch("src.parsing.document.read_pdf_hybrid_pages", return_value=["one", "two"]):
            assert read_file(path) == "one\ntwo"
    finally:
        os.remove(path)


def test_retrieval_diagnostics_marks_final_selector_drop():
    query = {
        "id": "diag_positive",
        "query": "rated voltage?",
        "relevant_doc": "manual.pdf",
        "evidence": [{"must_contain": ["rated voltage", "12kV"]}],
    }
    trace = {
        "first_stage": [
            {"doc": "manual.pdf", "text": "rated voltage: 12kV", "score": 0.9}
        ],
        "doc_internal": [],
        "expanded": [
            {"doc": "manual.pdf", "text": "rated voltage: 12kV", "score": 0.9}
        ],
        "final": [
            {"doc": "manual.pdf", "text": "installation notes", "score": 0.8}
        ],
    }

    result = diagnose_trace(query, trace, top_k=1)

    assert result["diagnosis"] == "final_selector_dropped"
    assert result["stage_scores"]["expanded"]["hit"]
    assert not result["stage_scores"]["final"]["hit"]


def test_retriever_trace_exposes_final_selector_decisions():
    from src.retrieval.retriever import Retriever
    from src.vector_store.faiss_store import FaissVectorStore

    db = FaissVectorStore(2)
    db.add_batch(
        ["rated voltage 12kV", "installation notes"],
        [[1.0, 0.0], [0.9, 0.1]],
        doc_name="manual.pdf",
    )
    retriever = Retriever(db)

    trace = retriever.search_with_trace(
        "rated voltage",
        [1.0, 0.0],
        top_k=1,
        threshold=0.0,
    )

    assert trace["final_selector"]["reason_counts"]["selected"] == 1
    assert all("reason" in item for item in trace["final_selector"]["decisions"])


def test_retriever_supports_fusion_ablation_modes_and_route_trace():
    from src.retrieval.retriever import Retriever
    from src.vector_store.faiss_store import FaissVectorStore

    db = FaissVectorStore(2)
    db.add_batch(
        ["E-VAC rated voltage 12kV", "installation notes", "maintenance guide"],
        [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]],
        doc_name="manual.pdf",
    )
    retriever = Retriever(db)
    retriever.add_texts(db.texts)

    dense_trace = retriever.search_with_trace(
        "E-VAC 12kV",
        [1.0, 0.0],
        top_k=1,
        threshold=0.0,
        fusion_mode="dense_only",
    )
    weighted_trace = retriever.search_with_trace(
        "E-VAC 12kV",
        [1.0, 0.0],
        top_k=1,
        threshold=0.0,
        fusion_mode="rrf_weighted",
    )

    assert dense_trace["first_stage"][0]["fusion_mode"] == "dense_only"
    assert dense_trace["first_stage"][0]["bm25_rank"] is None
    weighted = weighted_trace["first_stage"][0]
    assert weighted["fusion_mode"] == "rrf_weighted"
    assert weighted["fusion_profile"] == "exact"
    assert weighted["dense_rrf"] is not None
    assert weighted["bm25_rrf"] is not None


def test_retrieval_diagnostics_reports_negative_evidence_filtered_before_final():
    query = {
        "id": "diag_negative",
        "query": "does it support bluetooth?",
        "relevant_doc": None,
        "evidence": [],
        "forbidden_evidence": ["bluetooth remote control"],
    }
    trace = {
        "first_stage": [
            {"doc": "manual.pdf", "text": "bluetooth remote control", "score": 0.9}
        ],
        "doc_internal": [],
        "expanded": [
            {"doc": "manual.pdf", "text": "bluetooth remote control", "score": 0.9}
        ],
        "final": [
            {"doc": "manual.pdf", "text": "rated voltage: 12kV", "score": 0.8}
        ],
    }

    result = diagnose_trace(query, trace, top_k=1)

    assert result["diagnosis"] == "unsafe_filtered_before_final_from_first_stage"
    assert result["stage_scores"]["final"]["hit"]
