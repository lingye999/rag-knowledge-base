"""Deterministic contracts for extraction, retrieval, and QA checks."""
from __future__ import annotations

import math
import re


_SOURCE_CITATION_RE = re.compile(r"\[(chunk-[^\]\s]+)\]")


def normalize(text: str) -> str:
    """Ignore whitespace and case when comparing evidence phrases."""
    return re.sub(r"\s+", "", text).casefold()


def contains_all(text: str, phrases: list[str]) -> tuple[bool, list[str]]:
    """Return whether every phrase occurs and the phrases that did occur."""
    normalized = normalize(text)
    matched = [phrase for phrase in phrases if normalize(phrase) in normalized]
    return len(matched) == len(phrases), matched


def _matches_source(hit: dict, relevant_doc: str, page: int | None) -> bool:
    """Match both document and page when the dataset supplies a page anchor."""
    if hit.get("doc") != relevant_doc:
        return False
    return page is None or hit.get("page") == page


def _matched_terms(hit: dict, relevant_doc: str, group: dict) -> list[str]:
    if not _matches_source(hit, relevant_doc, group.get("page")):
        return []
    _, matched = contains_all(hit.get("text", ""), group["must_contain"])
    return matched


def _score_evidence_group(
    hits: list[dict], relevant_doc: str, group: dict, policy: str,
) -> dict:
    """Score one alternative evidence group against ranked retrieved chunks."""
    if policy == "same_page":
        return _score_same_page(hits, relevant_doc, group)
    if policy == "window_chunk":
        return _score_window_chunk(hits, relevant_doc, group)

    required = group["must_contain"]
    seen_terms: list[str] = []
    supporting_hits = []
    evidence_rank = None
    best_single_terms: list[str] = []
    relevant_contexts = 0
    dcg = 0.0

    for rank, hit in enumerate(hits, start=1):
        terms = _matched_terms(hit, relevant_doc, group)
        if terms:
            relevant_contexts += 1
        if len(terms) > len(best_single_terms):
            best_single_terms = terms

        # A phrase contributes once, so duplicate chunks cannot inflate nDCG.
        new_terms = [term for term in terms if term not in seen_terms]
        if new_terms:
            seen_terms.extend(new_terms)
            supporting_hits.append({"rank": rank, "matched": new_terms})
            gain = len(new_terms) / len(required)
            dcg += gain / math.log2(rank + 1)

        if policy == "same_chunk":
            if len(terms) == len(required) and evidence_rank is None:
                evidence_rank = rank
        elif len(seen_terms) == len(required) and evidence_rank is None:
            evidence_rank = rank

    return {
        "hit": evidence_rank is not None,
        "evidence_rank": evidence_rank,
        "matched": best_single_terms if policy == "same_chunk" else seen_terms,
        "supporting_hits": supporting_hits,
        "evidence_recall": len(seen_terms) / len(required),
        "context_precision": relevant_contexts / len(hits) if hits else 0.0,
        "relevant_contexts": relevant_contexts,
        # All required evidence in rank 1 is the ideal and gives DCG = 1.
        "ndcg": min(dcg, 1.0),
    }


def _score_same_page(hits: list[dict], relevant_doc: str, group: dict) -> dict:
    """Allow evidence terms to be covered by multiple chunks on one page."""
    required = group["must_contain"]
    page_terms: dict[tuple[str, int | None], list[str]] = {}
    page_support: dict[tuple[str, int | None], list[dict]] = {}
    evidence_rank = None
    best_key = None
    best_terms: list[str] = []
    relevant_contexts = 0
    dcg = 0.0

    for rank, hit in enumerate(hits, start=1):
        terms = _matched_terms(hit, relevant_doc, group)
        if terms:
            relevant_contexts += 1
        key = (hit.get("doc"), hit.get("page"))
        current = page_terms.setdefault(key, [])
        support = page_support.setdefault(key, [])
        for term in terms:
            if term not in current:
                current.append(term)
                support.append({"rank": rank, "matched": [term]})
                gain = 1 / len(required)
                dcg += gain / math.log2(rank + 1)

        if len(current) > len(best_terms):
            best_terms = current[:]
            best_key = key
        if len(current) == len(required) and evidence_rank is None:
            evidence_rank = rank
            best_key = key

    return {
        "hit": evidence_rank is not None,
        "evidence_rank": evidence_rank,
        "matched": best_terms,
        "supporting_hits": page_support.get(best_key, []) if best_key else [],
        "evidence_recall": len(best_terms) / len(required),
        "context_precision": relevant_contexts / len(hits) if hits else 0.0,
        "relevant_contexts": relevant_contexts,
        "ndcg": min(dcg, 1.0),
    }


def _score_window_chunk(hits: list[dict], relevant_doc: str, group: dict) -> dict:
    """Allow evidence terms to be covered by retrieved neighboring chunks."""
    required = group["must_contain"]
    window = int(group.get("window", 1))
    best_terms: list[str] = []
    best_support = []
    evidence_rank = None
    relevant_contexts = 0
    dcg = 0.0

    for rank, hit in enumerate(hits, start=1):
        terms = _matched_terms(hit, relevant_doc, group)
        if terms:
            relevant_contexts += 1
        center_index = hit.get("index")
        if center_index is None or hit.get("doc") != relevant_doc:
            continue

        window_terms: list[str] = []
        support = []
        window_texts = []
        for other_rank, other in enumerate(hits, start=1):
            other_index = other.get("index")
            if other_index is None:
                continue
            if other.get("doc") != relevant_doc:
                continue
            if group.get("page") is not None and other.get("page") != group.get("page"):
                continue
            if abs(int(other_index) - int(center_index)) > window:
                continue
            window_texts.append(other.get("text", ""))
            _, matched = contains_all(other.get("text", ""), required)
            new_terms = [term for term in matched if term not in window_terms]
            if new_terms:
                window_terms.extend(new_terms)
                support.append({"rank": other_rank, "matched": new_terms})
                gain = len(new_terms) / len(required)
                dcg += gain / math.log2(other_rank + 1)

        combined = "\n".join(window_texts)
        _, combined_terms = contains_all(combined, required)
        if len(combined_terms) > len(best_terms):
            best_terms = combined_terms
            best_support = support
        if len(combined_terms) == len(required) and evidence_rank is None:
            evidence_rank = rank
            best_terms = combined_terms
            best_support = support

    return {
        "hit": evidence_rank is not None,
        "evidence_rank": evidence_rank,
        "matched": best_terms,
        "supporting_hits": best_support,
        "evidence_recall": len(best_terms) / len(required),
        "context_precision": relevant_contexts / len(hits) if hits else 0.0,
        "relevant_contexts": relevant_contexts,
        "ndcg": min(dcg, 1.0),
    }


def evaluate_retrieval(query: dict, hits: list[dict], top_k: int) -> dict:
    """Evaluate evidence recall, context precision, MRR, and nDCG at K."""
    relevant_doc = query["relevant_doc"]
    inspected = hits[:top_k]
    evidence_policy = query.get("evidence_policy", "same_chunk")

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
            "evidence_recall": None,
            "context_precision": None,
            "mrr": None,
            "ndcg": None,
            "matched": [],
            "evidence_policy": evidence_policy,
            "supporting_hits": [],
            "relevant_contexts": 0,
            "unsafe_hits": unsafe_hits,
        }

    document_rank = next(
        (rank for rank, hit in enumerate(inspected, start=1)
         if hit.get("doc") == relevant_doc),
        None,
    )
    group_scores = [
        _score_evidence_group(inspected, relevant_doc, group, evidence_policy)
        for group in query["evidence"]
    ]
    # Evidence groups are alternatives. Select the group with strongest proof.
    score = max(
        group_scores,
        key=lambda item: (
            item["hit"],
            item["evidence_recall"],
            -(item["evidence_rank"] or top_k + 1),
        ),
    )
    evidence_rank = score["evidence_rank"]

    return {
        "positive": True,
        "hit": score["hit"],
        "document_hit": document_rank is not None,
        "evidence_rank": evidence_rank,
        "evidence_recall": score["evidence_recall"],
        "context_precision": score["context_precision"],
        "mrr": 1.0 / evidence_rank if evidence_rank else 0.0,
        "ndcg": score["ndcg"],
        "matched": score["matched"],
        "evidence_policy": evidence_policy,
        "supporting_hits": score["supporting_hits"],
        "relevant_contexts": score["relevant_contexts"],
        "unsafe_hits": [],
    }


def evaluate_answer(answer: str, golden: dict) -> dict:
    """Check required facts, forbidden claims, and citation formatting."""
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


def evaluate_context_coverage(golden: dict, contexts: list[dict]) -> dict:
    """检查本轮检索上下文是否覆盖回答所需的黄金事实。

    该结果是生成前的证据覆盖率，不评价模型最终是否真的使用了这些证据。
    """
    context_text = "\n".join(context.get("text", "") for context in contexts)
    _, matched = contains_all(context_text, golden["required_facts"])
    required = golden["required_facts"]
    return {
        "required_fact_ratio": len(matched) / len(required),
        "matched_facts": matched,
        "missing_facts": [fact for fact in required if fact not in matched],
        "passed": len(matched) == len(required),
    }


def evaluate_citations(answer: str, contexts: list[dict],
                       citation_required: bool) -> dict:
    """检查答案引用的 chunk 是否来自本轮实际送入模型的上下文。"""
    citation_ids = _SOURCE_CITATION_RE.findall(answer)
    available_ids = {str(context.get("id")) for context in contexts}
    valid_ids = [chunk_id for chunk_id in citation_ids if chunk_id in available_ids]
    invalid_ids = [chunk_id for chunk_id in citation_ids if chunk_id not in available_ids]
    citation_precision = len(valid_ids) / len(citation_ids) if citation_ids else 0.0
    passed = (
        (not citation_required or bool(citation_ids))
        and not invalid_ids
    )
    return {
        "citation_ids": citation_ids,
        "valid_citation_ids": valid_ids,
        "invalid_citation_ids": invalid_ids,
        "citation_precision": citation_precision,
        "passed": passed,
    }
