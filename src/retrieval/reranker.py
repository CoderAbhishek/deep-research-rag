"""
reranker.py — Cross-encoder reranking over a candidate pool.

A cross-encoder reads a (query, document) pair as a single input and outputs
one relevance score. Because it sees both sides jointly, it can make fine-grained
relevance judgments that bi-encoders cannot: it detects when a chunk merely
contains some query tokens without actually answering the question.

Usage pattern:
    - Stage 1 (fast retrieval): HyDE dense + BM25 → RRF → top-20 candidates
    - Stage 2 (accurate reranking): cross-encoder scores all 20 pairs → top-5

Never use a cross-encoder directly on all 3439 chunks — no pre-computation
is possible, so every pair is scored fresh. 20 pairs ≈ fast; 3439 ≈ unusable.
"""

from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_reranker(model_name: str = DEFAULT_RERANKER) -> CrossEncoder:
    """
    Load a cross-encoder model from Hugging Face.
    Downloads ~80MB on first call; cached in ~/.cache/huggingface thereafter.
    """
    return CrossEncoder(model_name)


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    reranker: CrossEncoder,
    n_results: int = 5,
) -> List[Dict[str, Any]]:
    """
    Score each (query, candidate_text) pair with the cross-encoder,
    return the top n_results sorted by cross-encoder score.

    Parameters
    ----------
    query      : the user's original question (not the HyDE passage —
                 the cross-encoder needs the real question to judge relevance)
    candidates : output from reciprocal_rank_fusion or any search function;
                 each dict must have 'text' and 'metadata' keys
    reranker   : a loaded CrossEncoder instance from load_reranker()
    n_results  : number of top results to return

    Returns
    -------
    List of dicts with original keys plus 'rerank_score' (float) and
    updated 'rank' (1-indexed). Sorted by rerank_score descending.
    """
    if not candidates:
        return []

    # Build input pairs: list of (query, document_text) tuples
    # The cross-encoder tokenises these as a single sequence:
    # [CLS] query tokens [SEP] document tokens [SEP]
    pairs = [(query, c["text"]) for c in candidates]

    # Score all pairs in one forward pass (batched internally)
    # Returns a numpy array of floats, one per pair
    scores = reranker.predict(pairs)

    # Attach score to each candidate (copy so we don't mutate the input)
    scored = []
    for candidate, score in zip(candidates, scores):
        entry = dict(candidate)
        entry["rerank_score"] = float(score)
        scored.append(entry)

    # Sort descending by cross-encoder score
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)

    # Re-assign rank positions for the reranked list
    results = []
    for rank_idx, entry in enumerate(scored[:n_results]):
        entry["rank"] = rank_idx + 1
        results.append(entry)

    return results

def deduplicate_by_page(
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    After reranking, keep only the highest-scoring chunk per unique page.

    Why this is needed:
    The chunking pipeline can produce multiple overlapping chunks from the
    same PDF page. When both enter the candidate pool and get reranked,
    they receive different cross-encoder scores — because they contain
    different text. Without deduplication, two chunks from the same page
    can occupy two separate result slots, wasting capacity and confusing
    downstream generation (the LLM receives redundant, partially-overlapping
    text from the same source).

    Deduplication key: (file_name, page_number)
    — two chunks from the same file AND same page are considered duplicates.
    — chunks from different files with the same page number are not duplicates.

    The winner: the chunk with the lowest rank (= highest rerank_score),
    which is whichever chunk the cross-encoder found more relevant.

    Args:
        results: A list of chunk dicts sorted by rerank_score descending
                 (i.e. already the output of rerank()). Each dict must have
                 a 'metadata' key with 'file_name' and 'page_number'.

    Returns:
        A list of chunks, one per unique (file_name, page_number) pair,
        in the same relative order (lowest original rank wins).
        Ranks are reassigned 1..N after deduplication.
    """
    seen = set()
    deduplicated = []

    for chunk in results:
        # Build the deduplication key from file name + page number.
        # We use .get() with defaults so the function does not crash if
        # metadata is incomplete — a defensive pattern for production code.
        file_name  = chunk["metadata"].get("file_name",  "unknown")
        page_number = chunk["metadata"].get("page_number", -1)
        key = (file_name, page_number)

        if key not in seen:
            seen.add(key)
            deduplicated.append(chunk)

    # Reassign ranks 1..N after deduplication.
    # The original rank values from rerank() are now stale — rank 4 might
    # have become rank 2 if the intervening chunks were duplicates.
    for new_rank, chunk in enumerate(deduplicated, start=1):
        chunk["rank"] = new_rank

    return deduplicated