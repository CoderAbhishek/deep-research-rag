import re
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Tuple


def tokenize(text: str) -> List[str]:
    """
    Convert a string of text into a list of lowercase word tokens.

    BM25 is a word-level algorithm — it needs a list of tokens, not
    a raw string. This function does the minimum necessary:
    - Lowercase everything (so "Revenue" and "revenue" are the same token)
    - Extract all sequences of word characters using regex
      (splits on spaces, punctuation, parentheses, slashes, etc.)

    Example:
        tokenize("Revenue from North America: $5.2B (FY26)")
        → ["revenue", "from", "north", "america", "5", "2b", "fy26"]

    Note: We deliberately keep numbers as tokens. "1,78,650" becomes
    ["1", "78", "650"] — not ideal for exact figure matching, but
    acceptable for our use case. A more sophisticated tokeniser could
    normalise numbers or handle compound terms like "North America" as
    a single token — that's a later optimisation.
    """
    return re.findall(r'\b\w+\b', text.lower())


def build_bm25_index(
    collection,
) -> Tuple[BM25Okapi, List[Dict[str, Any]]]:
    """
    Build a BM25 index from all chunks stored in a ChromaDB collection.

    BM25 needs the corpus to be tokenised upfront — it computes IDF
    (inverse document frequency) at index-build time, not at query time.
    That's why building the index is a separate step from querying.

    Args:
        collection: A ChromaDB Collection object. All documents and
                    metadata are fetched from it internally.

    Returns:
        (bm25, chunks): The BM25Okapi index and the list of chunk dicts.
        We return both together so the caller can map result indices back
        to the original chunk metadata. The index alone knows only positions
        (0, 1, 2, ...) — not document content or metadata.
    """
    result = collection.get(include=["documents", "metadatas"])
    chunks = [
        {"text": doc, "metadata": meta}
        for doc, meta in zip(result["documents"], result["metadatas"])
    ]

    print(f"Tokenising {len(chunks)} chunks...")
    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]

    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)

    print(f"BM25 index ready. Corpus: {len(chunks)} chunks.")
    return bm25, chunks


def search_bm25(
    query: str,
    bm25: BM25Okapi,
    chunks: List[Dict[str, Any]],
    n_results: int = 5,
) -> List[Dict[str, Any]]:
    """
    Query the BM25 index and return the top-n most relevant chunks.

    Args:
        query:     Natural-language question from the user.
        bm25:      Pre-built BM25Okapi index (from build_bm25_index).
        chunks:    The original chunks list — must be in the same order
                   as when the index was built. The index maps positions
                   back into this list.
        n_results: Number of results to return.

    Returns:
        List of result dicts sorted by BM25 score (highest first).
        Each dict contains: rank, score, text, metadata.
    """
    query_tokens = tokenize(query)
    print(f"Query tokens: {query_tokens}")

    scores = bm25.get_scores(query_tokens)

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:n_results]

    results = []
    for rank, idx in enumerate(top_indices):
        results.append({
            "rank": rank + 1,
            "score": float(scores[idx]),   # float() converts NumPy scalar to Python float
            "text": chunks[idx]["text"],
            "metadata": chunks[idx]["metadata"],
        })

    return results