"""校验评测数据集文件的结构和引用文档。

用法：
    python eval/validate_dataset.py
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EVAL_DIR = ROOT / "eval"
DATASET_DIR = EVAL_DIR / "datasets"
MANIFEST_PATH = EVAL_DIR / "document_manifest.json"
STRICT_RETRIEVAL_DATASETS = {
    "smoke": DATASET_DIR / "retrieval_smoke.jsonl",
    "dev": DATASET_DIR / "retrieval_dev.jsonl",
    "test": DATASET_DIR / "retrieval_test.jsonl",
}
QA_DATASETS = {
    "smoke": DATASET_DIR / "qa_smoke.jsonl",
    "dev": DATASET_DIR / "qa_dev.jsonl",
    "test": DATASET_DIR / "qa_test.jsonl",
}


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            rows.append(item)
    return rows


def _require(item: dict, field: str, path: Path):
    if field not in item:
        raise ValueError(f"{path}: {item.get('id', '<missing id>')} missing {field}")
    return item[field]


def _check_ids(rows: list[dict], path: Path):
    seen = set()
    for item in rows:
        item_id = _require(item, "id", path)
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"{path}: id must be a non-empty string")
        if item_id in seen:
            raise ValueError(f"{path}: duplicate id {item_id}")
        seen.add(item_id)


def _check_doc_exists(doc: str | None, item_id: str, path: Path):
    if doc is None:
        return
    if not isinstance(doc, str) or not doc:
        raise ValueError(f"{path}: {item_id} doc must be string or null")
    if not (DATA_DIR / doc).exists():
        raise ValueError(f"{path}: {item_id} missing data file {doc}")


def _load_manifest() -> dict:
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _check_manifest(referenced_docs: set[str]):
    manifest = _load_manifest()
    for doc in sorted(referenced_docs):
        entry = manifest.get(doc)
        if entry is None:
            raise ValueError(f"{MANIFEST_PATH}: {doc} is missing")
        digest = hashlib.sha256((DATA_DIR / doc).read_bytes()).hexdigest()
        if digest != entry.get("sha256"):
            raise ValueError(f"{MANIFEST_PATH}: {doc} hash does not match")


def validate_extraction(path: Path) -> tuple[int, set[str]]:
    rows = _load_jsonl(path)
    _check_ids(rows, path)
    for item in rows:
        item_id = item["id"]
        _check_doc_exists(_require(item, "doc", path), item_id, path)
        must = _require(item, "must_contain", path)
        if not isinstance(must, list) or not must:
            raise ValueError(f"{path}: {item_id} must_contain must be non-empty list")
        _require(item, "parser_mode", path)
        _require(item, "type", path)
        page = item.get("page")
        if page is not None and (not isinstance(page, int) or page < 1):
            raise ValueError(f"{path}: {item_id} page must be a positive integer or null")
        allow_form_boxes = item.get("allow_form_boxes", False)
        if not isinstance(allow_form_boxes, bool):
            raise ValueError(f"{path}: {item_id} allow_form_boxes must be bool")
    return len(rows), {item["doc"] for item in rows}


def validate_retrieval(path: Path, require_page_anchors: bool = False) -> tuple[int, set[str]]:
    rows = _load_jsonl(path)
    _check_ids(rows, path)
    allowed_policies = {
        "same_chunk",
        "multi_chunk_same_doc",
        "same_page",
        "window_chunk",
    }
    for item in rows:
        item_id = item["id"]
        _require(item, "query", path)
        _check_doc_exists(_require(item, "relevant_doc", path), item_id, path)
        evidence = _require(item, "evidence", path)
        evidence_policy = item.get("evidence_policy", "same_chunk")
        if evidence_policy not in allowed_policies:
            raise ValueError(
                f"{path}: {item_id} evidence_policy must be one of {sorted(allowed_policies)}"
            )
        if item["relevant_doc"] is None:
            if evidence != []:
                raise ValueError(f"{path}: {item_id} negative query evidence must be []")
            if evidence_policy != "same_chunk":
                raise ValueError(f"{path}: {item_id} negative query cannot use {evidence_policy}")
            forbidden = _require(item, "forbidden_evidence", path)
            if not isinstance(forbidden, list) or not forbidden:
                raise ValueError(f"{path}: {item_id} negative query needs forbidden_evidence")
        elif not isinstance(evidence, list) or not evidence:
            raise ValueError(f"{path}: {item_id} evidence must be non-empty list")
        if item["relevant_doc"] is not None:
            for group in evidence:
                if not isinstance(group, dict):
                    raise ValueError(f"{path}: {item_id} evidence group must be object")
                terms = _require(group, "must_contain", path)
                if not isinstance(terms, list) or not terms:
                    raise ValueError(f"{path}: {item_id} evidence terms must be non-empty list")
                page = group.get("page")
                if page is not None and (not isinstance(page, int) or page < 1):
                    raise ValueError(f"{path}: {item_id} evidence page must be positive integer or null")
                if require_page_anchors and page is None:
                    raise ValueError(
                        f"{path}: {item_id} positive evidence requires a page anchor"
                    )
        _require(item, "query_type", path)
        _require(item, "difficulty", path)
        _require(item, "parser_mode", path)
    return len(rows), {
        item["relevant_doc"] for item in rows if item["relevant_doc"] is not None
    }


def _check_disjoint_queries(datasets: dict[str, list[dict]]):
    seen = {}
    for name in ("dev", "test"):
        for item in datasets[name]:
            query = item["query"]
            if query in seen:
                raise ValueError(
                    f"retrieval {name} duplicates query from {seen[query]}: {query}"
                )
            seen[query] = name


def validate_qa(path: Path, retrieval_ids: set[str]) -> int:
    rows = _load_jsonl(path)
    _check_ids(rows, path)
    for item in rows:
        item_id = item["id"]
        retrieval_id = _require(item, "retrieval_id", path)
        if not isinstance(retrieval_id, str) or not retrieval_id:
            raise ValueError(f"{path}: {item_id} retrieval_id must be a non-empty string")
        if retrieval_id not in retrieval_ids:
            raise ValueError(f"{path}: {item_id} references unknown retrieval_id {retrieval_id}")
        _require(item, "query", path)
        _require(item, "expected_answer", path)
        facts = _require(item, "required_facts", path)
        if not isinstance(facts, list) or not facts:
            raise ValueError(f"{path}: {item_id} required_facts must be non-empty list")
        forbidden = _require(item, "forbidden_claims", path)
        if not isinstance(forbidden, list):
            raise ValueError(f"{path}: {item_id} forbidden_claims must be list")
        citation_required = _require(item, "citation_required", path)
        if not isinstance(citation_required, bool):
            raise ValueError(f"{path}: {item_id} citation_required must be bool")
    return len(rows)


def main():
    extraction_count, extraction_docs = validate_extraction(
        EVAL_DIR / "extraction_checks.jsonl"
    )
    strict_rows = {}
    strict_counts = {}
    strict_docs = set()
    for name, path in STRICT_RETRIEVAL_DATASETS.items():
        count, docs = validate_retrieval(path, require_page_anchors=True)
        strict_counts[name] = count
        strict_docs.update(docs)
        strict_rows[name] = _load_jsonl(path)
    _check_disjoint_queries(strict_rows)
    _check_manifest(extraction_docs | strict_docs)
    counts = {
        "extraction": extraction_count,
    }
    for name, count in counts.items():
        print(f"{name}: {count} rows ok")
    for name, count in strict_counts.items():
        print(f"retrieval {name}: {count} rows ok (strict page anchors)")
        qa_count = validate_qa(
            QA_DATASETS[name], {row["id"] for row in strict_rows[name]}
        )
        print(f"qa {name}: {qa_count} rows ok")

    extraction_rows = _load_jsonl(EVAL_DIR / "extraction_checks.jsonl")
    extraction_anchored = sum(row.get("page") is not None for row in extraction_rows)
    evidence_groups = [
        group
        for rows in strict_rows.values()
        for row in rows
        for group in row.get("evidence", [])
    ]
    evidence_anchored = sum(group.get("page") is not None for group in evidence_groups)
    print(f"extraction page anchors: {extraction_anchored}/{len(extraction_rows)}")
    print(f"strict retrieval page anchors: {evidence_anchored}/{len(evidence_groups)}")


if __name__ == "__main__":
    main()
