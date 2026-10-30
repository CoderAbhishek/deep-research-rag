"""
metrics_v2.py - deterministic, page-labelled evaluation metrics.

Why these replace the old cross-encoder metrics (DR-02)
-------------------------------------------------------
The old context precision / recall asked a cross-encoder whether chunks were
*topically similar* to the reference answer. They scored perfect recall even
on questions the system refused to answer. These metrics instead use labels
written by a person:

  - relevant_pages: pages known to contain the answer. Any one of them counts.
  - must_include:   strings the answer must contain (with alternatives).
  - expected_behavior: "answer", "refuse" or "answer_or_refuse".

Every function here is pure (no models, no network), so it can be unit
tested and gives the same result every time.

Retrieval metrics are computed on ranked lists of chunks, mapped to pages:
  hit@k  - is at least one relevant page among the first k chunks?
  RR@k   - 1 / rank of the first relevant chunk within the first k (else 0).
           Averaged over questions this is MRR@k.

v2.1 addition: locate_loss returns "ok_unlabelled_source" when the answer is
correct but no labelled page reached the model, so label gaps show up as
label gaps instead of being blamed on retrieval.
Recall@k is deliberately NOT used: relevant_pages lists *alternative* pages
that each contain the answer (e.g. headcount appears on five pages), not
pages that are all required. Recall would wrongly penalise retrieving one.
"""

import re
import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

PageKey = Tuple[str, int]

# The fixed refusal sentence required by the generation prompt (generator.py).
REFUSAL_MARKER = "do not contain sufficient information"

# Pipeline checkpoints, in order, used to locate where a relevant page was lost.
STAGES = ("dense", "bm25", "rrf", "rerank", "final")


def normalize(text: Optional[str]) -> str:
    """Lower-case and drop whitespace, commas and rupee signs (₹ or the ` the PDF uses)."""
    return re.sub(r"[\s,₹`]", "", (text or "").lower())


def page_key(item: Dict[str, Any]) -> PageKey:
    """(file_name, page_number) from a source dict or a chunk summary."""
    meta = item.get("metadata", item)
    return (meta.get("file_name"), int(meta.get("page_number")))


def relevant_set(question: Dict[str, Any]) -> Set[PageKey]:
    return {(p["file_name"], int(p["page_number"])) for p in question.get("relevant_pages", [])}


def first_relevant_rank(pages: Sequence[PageKey], relevant: Set[PageKey]) -> Optional[int]:
    """1-based position of the first relevant page in a ranked list, or None."""
    for i, page in enumerate(pages, start=1):
        if page in relevant:
            return i
    return None


def hit_at_k(pages: Sequence[PageKey], relevant: Set[PageKey], k: int) -> Optional[bool]:
    """True/False; None when the question has no relevant pages (nothing to find)."""
    if not relevant:
        return None
    rank = first_relevant_rank(pages[:k], relevant)
    return rank is not None


def reciprocal_rank_at_k(pages: Sequence[PageKey], relevant: Set[PageKey], k: int) -> Optional[float]:
    if not relevant:
        return None
    rank = first_relevant_rank(pages[:k], relevant)
    return 1.0 / rank if rank else 0.0


def is_refusal(answer: Optional[str]) -> bool:
    return REFUSAL_MARKER in (answer or "").lower()


def check_must_include(answer: Optional[str], groups: List[List[str]]) -> Tuple[Optional[bool], List[List[str]]]:
    """
    Every group must be matched by at least one of its alternatives.
    Returns (ok, missing_groups). ok is None when there is nothing to check.
    """
    if not groups:
        return None, []
    a = normalize(answer)
    missing = [g for g in groups if not any(normalize(alt) in a for alt in g)]
    return (not missing), missing


def behaviour_correct(expected: str, refused: bool, answer_ok: Optional[bool]) -> bool:
    """Did the system do what the question expects: answer correctly, or decline?"""
    if expected == "refuse":
        return refused
    if expected == "answer_or_refuse":
        return refused or bool(answer_ok)
    return (not refused) and bool(answer_ok)


def citation_metrics(cited: Sequence[PageKey], relevant: Set[PageKey]) -> Dict[str, Any]:
    """
    citation_precision: share of cited pages that are labelled relevant.
    Caveat: relevant_pages may be incomplete, so a citation to an unlabelled
    page that also contains the answer is counted as wrong. Review by hand.
    """
    cited_unique = list(dict.fromkeys(cited))
    if not cited_unique or not relevant:
        return {"cited_count": len(cited_unique), "citation_precision": None, "any_citation_relevant": None}
    good = sum(1 for p in cited_unique if p in relevant)
    return {"cited_count": len(cited_unique),
            "citation_precision": good / len(cited_unique),
            "any_citation_relevant": good > 0}


def locate_loss(stage_pages: Dict[str, Sequence[PageKey]], relevant: Set[PageKey],
                answer_ok: Optional[bool], refused: bool) -> str:
    """
    Where did the answer get lost? Walks the pipeline checkpoints in order.

    not_retrieved        - no relevant page in dense top-20 or BM25 top-20
    fusion               - a retriever found it, but it fell out of the RRF pool of 20
    rerank               - it was in the RRF pool, but not in the reranked top 5
    dedup                - it was in the reranked top 5, but not in the final context
    generation           - it was in the final context, but the answer is wrong or a refusal
    ok                   - found and answered correctly
    ok_unlabelled_source - answered correctly, but no LABELLED page reached the model:
                           the answer came from a page missing from relevant_pages.
                           This is a label gap to fix, not a retrieval failure.
    n/a                  - question has no relevant pages (should-decline questions)
    """
    if not relevant:
        return "n/a"
    hits = {stage: bool(set(stage_pages.get(stage, [])) & relevant) for stage in STAGES}
    return locate_loss_from_hits(hits, answer_ok, refused)


def locate_loss_from_hits(hits: Dict[str, Optional[bool]], answer_ok: Optional[bool], refused: bool) -> str:
    """
    Same logic as locate_loss, from per-stage booleans: hits["dense"] and
    hits["bm25"] (top 20), hits["rrf"] (pool of 20), hits["rerank"] (top 5),
    hits["final"] (final context). Used when re-scoring a saved run.
    A value of None means "unknown"; if a needed stage is unknown the result
    is "unknown_stage".
    """
    answered = bool(answer_ok) and not refused
    if hits.get("final") is False and answered:
        return "ok_unlabelled_source"
    order = [("rrf", "fusion"), ("rerank", "rerank"), ("final", "dedup")]
    dense, bm25 = hits.get("dense"), hits.get("bm25")
    if dense is None or bm25 is None:
        if hits.get("final"):
            return "ok" if answered else "generation"
        return "unknown_stage"
    if not (dense or bm25):
        return "not_retrieved"
    for stage, label in order:
        if hits.get(stage) is None:
            return "unknown_stage"
        if not hits[stage]:
            return label
    return "ok" if answered else "generation"


def mean(values: Iterable[Any]) -> Optional[float]:
    """Mean of non-None values (booleans count as 0/1); None if there are none."""
    vals = [float(v) for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def percentile(values: Iterable[float], pct: float) -> Optional[float]:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return round(vals[0], 3)
    return round(statistics.quantiles(vals, n=100, method="inclusive")[int(pct) - 1], 3)
