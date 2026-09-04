"""
hybrid.py

Hybrid retrieval via Reciprocal Rank Fusion (RRF).

Takes two or more ranked lists (one from dense, one from BM25 — or more
from multi-query in Session 7) and fuses them into a single ranked list.

RRF formula (Cormack, Clarke, Buettcher — SIGIR 2009):
    RRF_score(d) = Σ_i  1 / (rank_i(d) + k)

Where:
    rank_i(d) = the 0-indexed rank of document d in system i's list
                (rank 0 = best result from that system)
    k = 60     — damping constant; makes rank differences small enough
                that a document consistent across both systems beats
                one that was top-1 in only one system

The function is deliberately generic: it accepts a list of ranked lists,
so it works equally well with 2 systems (dense + BM25), 3 systems
(dense + BM25 + HyDE), or any other combination Session 7 introduces.
"""

from typing import List, Dict, Any


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = 60,
    n_results: int = 5,
) -> List[Dict[str, Any]]:
    """
    Fuse multiple ranked lists into one using Reciprocal Rank Fusion.

    Args:
        ranked_lists: A list of ranked lists. Each inner list is a list of
                      result dicts with at minimum "text" and "metadata" keys.
                      Metadata must contain "file_name", "page_number", and
                      "chunk_index" for document identity (the unique key we
                      use to match the same chunk across different systems).
        k:            RRF damping constant. Default 60 (the original paper's value).
        n_results:    Number of top results to return after fusion.

    Returns:
        List of result dicts sorted by RRF score (highest first).
        Each dict: rank, rrf_score, text, metadata.

    Identity key:
        We identify each unique chunk by composing:
            f"{file_name}_p{page_number}_c{chunk_index}"
        This matches the ID format we used in embed_and_store() in Session 4.
        Both dense and BM25 retrieve from the same ChromaDB-backed corpus,
        so the same chunk will have identical metadata in both result lists.
    """
    # rrf_scores: maps chunk_key → accumulated RRF score
    # doc_store:  maps chunk_key → the full chunk dict (text + metadata)
    rrf_scores: Dict[str, float] = {}
    doc_store: Dict[str, Dict[str, Any]] = {}

    for ranked_list in ranked_lists:
        for rank_idx, chunk in enumerate(ranked_list):
            m = chunk["metadata"]

            # Build the unique identity key for this chunk
            key = f"{m['file_name']}_p{m['page_number']}_c{m['chunk_index']}"

            # rank_idx is 0-based (rank_idx=0 → best result).
            # The formula uses 1-based rank, so: rank = rank_idx + 1.
            # → contribution = 1 / (rank_idx + 1 + k)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rank_idx + 1 + k)

            # Store the chunk the first time we see it.
            # Both systems return the same text and metadata for the same chunk,
            # so it doesn't matter which system's copy we keep.
            if key not in doc_store:
                doc_store[key] = chunk

    # Sort all keys by accumulated RRF score, descending
    sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    top_keys = sorted_keys[:n_results]

    results = []
    for rank_idx, key in enumerate(top_keys):
        results.append(
            {
                "rank": rank_idx + 1,
                "rrf_score": round(rrf_scores[key], 6),
                "text": doc_store[key]["text"],
                "metadata": doc_store[key]["metadata"],
            }
        )

    return results