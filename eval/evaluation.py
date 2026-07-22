"""Shared, deterministic contracts for extraction, retrieval, and QA checks."""
from __future__ import annotations

import re
import math


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text).casefold()


def contains_all(text: str, phrases: list[str]) -> tuple[bool, list[str]]:
    normalized = normalize(text)
    matched = [phrase for phrase in phrases if normalize(phrase) in normalized]
    return len(matched) == len(phrases), matched


def match_evidence(hit: dict, relevant_doc: str, evidence: list[dict]) -> tuple[bool, list[str]]:
    """A positive hit must satisfy every phrase in one evidence group.

    Evidence groups are alternatives. Phrases within an individual group must
    come from the same retrieved chunk, preventing cross-chunk false passes.
    """
    if hit.get("doc") != relevant_doc:
        return False, []
    for group in evidence:
        passed, matched = contains_all(hit.get("text", ""), group["must_contain"])
        if passed:
            return True, matched
    return False, []


def match_evidence_across_chunks(
    hits: list[dict],
    relevant_doc: str,
    evidence: list[dict],
) -> tuple[bool, list[str], int | None, list[dict]]:
    """Allow list-style evidence to be covered by multiple chunks from one doc."""
    for group in evidence:
        required = group["must_contain"]
        matched = []
        supporting_hits = []
        for rank, hit in enumerate(hits, start=1):
            if hit.get("doc") != relevant_doc:
                continue
            _, terms = contains_all(hit.get("text", ""), required)
            new_terms = [term for term in terms if term not in matched]
            if not new_terms:
                continue
            matched.extend(new_terms)
            supporting_hits.append({"rank": rank, "matched": new_terms})
            if len(matched) == len(required):
                return True, matched, rank, supporting_hits
    return False, [], None, []


def evaluate_retrieval(query: dict, hits: list[dict], top_k: int) -> dict:
    """Score a structured retrieval query without model-based judging."""
    relevant_doc = query["relevant_doc"]
    inspected = hits[:top_k]

    if relevant_doc is None:
        forbidden = query["forbidden_evidence"]
        unsafe_hits = []
        for rank, hit in enumerate(inspected, start=1):
            present, matched = contains_all(hit.get("text", ""), forbidden)
            if present:
                unsafe_hits.append({"rank": rank, "matched": matched})
        safe = not unsafe_hits
        return {
            "positive": False,
            "hit": safe,
            "document_hit": safe,
            "evidence_rank": None,
            "mrr": 1.0 if safe else 0.0,
            "ndcg": 1.0 if safe else 0.0,
            "matched": [],
            "evidence_policy": query.get("evidence_policy", "same_chunk"),
            "supporting_hits": [],
            "unsafe_hits": unsafe_hits,
        }

    document_rank = None
    evidence_rank = None
    matched = []
    supporting_hits = []
    evidence_policy = query.get("evidence_policy", "same_chunk")
    for rank, hit in enumerate(inspected, start=1):
        if hit.get("doc") == relevant_doc and document_rank is None:
            document_rank = rank

    if evidence_policy == "multi_chunk_same_doc":
        evidence_hit, matched, evidence_rank, supporting_hits = (
            match_evidence_across_chunks(inspected, relevant_doc, query["evidence"])
        )
    else:
        for rank, hit in enumerate(inspected, start=1):
            evidence_hit, matched_terms = match_evidence(
                hit, relevant_doc, query["evidence"]
            )
            if evidence_hit:
                evidence_rank = rank
                matched = matched_terms
                supporting_hits = [{"rank": rank, "matched": matched_terms}]
                break

    return {
        "positive": True,
        "hit": evidence_rank is not None,
        "document_hit": document_rank is not None,
        "evidence_rank": evidence_rank,
        "mrr": 1.0 / evidence_rank if evidence_rank else 0.0,
        "ndcg": 1.0 / math.log2(evidence_rank + 1)
        if evidence_rank else 0.0,
        "matched": matched,
        "evidence_policy": evidence_policy,
        "supporting_hits": supporting_hits,
        "unsafe_hits": [],
    }


def evaluate_answer(answer: str, golden: dict) -> dict:
    """Check required facts and forbidden claims in a generated answer."""
    _, facts = contains_all(answer, golden["required_facts"])
    normalized = normalize(answer)
    forbidden = [
        claim for claim in golden["forbidden_claims"]
        if normalize(claim) in normalized
    ]
    citation_present = bool(re.search(r"\[[^\]]+\]", answer))
    required_ratio = len(facts) / len(golden["required_facts"])
    citation_ok = not golden["citation_required"] or citation_present
    return {
        "required_fact_ratio": required_ratio,
        "matched_facts": facts,
        "forbidden_claims": forbidden,
        "citation_ok": citation_ok,
        "passed": required_ratio == 1.0 and not forbidden and citation_ok,
    }
