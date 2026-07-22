"""Run deterministic parsing checks from extraction_checks.jsonl."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.evaluation import contains_all, normalize
from src.document import read_file_structured

DATASET = ROOT / "eval" / "extraction_checks.jsonl"
DATA_DIR = ROOT / "data"


def _load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read(path: Path, parser_mode: str, page_numbers: set[int] | None,
          allow_ocr: bool):
    return read_file_structured(
        str(path),
        force_ocr=parser_mode == "ocr",
        use_marker=parser_mode == "marker",
        page_numbers=page_numbers,
        allow_ocr=allow_ocr,
    )


def find_forbidden_tokens(row: dict, text: str) -> list[str]:
    tokens = row.get("must_not_contain", [])
    if row.get("allow_form_boxes", False):
        tokens = [token for token in tokens if token != "□"]
    normalized_text = normalize(text)
    return [token for token in tokens if normalize(token) in normalized_text]


def run(dataset: Path = DATASET, ids: set[str] | None = None,
        quick: bool = False, allow_ocr: bool = True) -> int:
    rows = _load_rows(dataset)
    if ids is not None:
        rows = [row for row in rows if row["id"] in ids]
    if quick:
        rows = [row for row in rows if row.get("page") is not None]
    if not rows:
        print("No extraction checks selected.")
        return 1
    requested_pages = defaultdict(set)
    full_documents = set()
    for row in rows:
        key = (row["doc"], row["parser_mode"])
        if row.get("page") is None:
            full_documents.add(key)
        else:
            requested_pages[key].add(row["page"])
    cache = {}
    results = []
    for row in rows:
        page = row.get("page")
        key = (row["doc"], row["parser_mode"])
        if key not in cache:
            pages = None if key in full_documents else requested_pages[key]
            label = "all" if pages is None else ",".join(map(str, sorted(pages)))
            print(f"PARSE {row['doc']} page={label}", flush=True)
            started = time.perf_counter()
            cache[key] = _read(
                DATA_DIR / row["doc"], row["parser_mode"], pages, allow_ocr
            )
            elapsed = time.perf_counter() - started
            print(f"DONE  {row['doc']} page={label} ({elapsed:.1f}s)", flush=True)
        parsed = cache[key]
        if page is None:
            text = parsed.text
        else:
            text = "\n".join(
                block.text for block in parsed.blocks if block.page == page
            )
        contains, matched = contains_all(text, row["must_contain"])
        forbidden = find_forbidden_tokens(row, text)
        result = {
            "id": row["id"],
            "passed": contains and not forbidden,
            "matched": matched,
            "missing": [token for token in row["must_contain"] if token not in matched],
            "forbidden": forbidden,
            "parser": parsed.parser,
            "page": page,
        }
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status:4s} {row['id']}")
        if not result["passed"]:
            print(f"      missing={result['missing']} forbidden={result['forbidden']}")

    grouped = defaultdict(list)
    for row, result in zip(rows, results):
        grouped[row["parser_mode"]].append(result)
    passed = sum(result["passed"] for result in results)
    print(f"\nExtraction pass rate: {passed}/{len(results)} ({passed / len(results):.1%})")
    for mode, group in sorted(grouped.items()):
        count = sum(result["passed"] for result in group)
        print(f"  {mode}: {count}/{len(group)} ({count / len(group):.1%})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--ids", help="Comma-separated extraction check IDs")
    parser.add_argument("--quick", action="store_true",
                        help="Run only checks with page anchors")
    parser.add_argument("--no-ocr", action="store_true",
                        help="Do not activate OCR fallback during this run")
    args = parser.parse_args()
    selected_ids = set(args.ids.split(",")) if args.ids else None
    raise SystemExit(run(
        args.dataset,
        ids=selected_ids,
        quick=args.quick,
        allow_ocr=not (args.quick or args.no_ocr),
    ))
