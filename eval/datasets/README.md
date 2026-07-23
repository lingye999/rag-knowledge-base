# Retrieval Dataset Policy

- `retrieval_smoke.jsonl`: fast regression checks after code changes.
- `retrieval_dev.jsonl`: parameter tuning for chunking, recall, fusion, and reranking.
- `retrieval_test.jsonl`: holdout checks. Do not use its results to choose parameters.

Positive retrieval rows require a document name, page anchor, and independent evidence phrases. Negative rows intentionally have no source page because they assert that unsupported evidence is absent.

Only source-supported, parser-visible evidence belongs in a strict dataset. Add new candidate questions outside these files, review them against the source PDF and parsed text, then promote them into `dev` or `test` deliberately.
