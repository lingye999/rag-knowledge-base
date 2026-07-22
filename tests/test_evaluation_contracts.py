import os
import tempfile
from unittest.mock import patch

from eval.evaluation import evaluate_answer, evaluate_retrieval
from eval.run_extraction_checks import find_forbidden_tokens
from src.document import read_file, read_file_structured


def test_retrieval_requires_all_terms_in_one_chunk():
    query = {
        "relevant_doc": "manual.pdf",
        "evidence": [{"must_contain": ["12kV", "rated voltage"]}],
    }

    partial = [{"doc": "manual.pdf", "text": "12kV"}]
    complete = [{"doc": "manual.pdf", "text": "rated voltage: 12kV"}]

    assert not evaluate_retrieval(query, partial, 5)["hit"]
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
        with patch("src.document.read_pdf_hybrid_pages", return_value=["one", "two"]):
            parsed = read_file_structured(path)

        assert [block.page for block in parsed.blocks] == [1, 2]
        assert parsed.text == "one\ntwo"
        with patch("src.document.read_pdf_hybrid_pages", return_value=["one", "two"]):
            assert read_file(path) == "one\ntwo"
    finally:
        os.remove(path)
