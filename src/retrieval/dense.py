"""
dense.py

Dense vector retrieval using ChromaDB + sentence-transformers.

This module wraps the two operations we already know from Session 4:
  1. Embed the query with sentence-transformers
  2. Query ChromaDB for the nearest neighbours

We expose it as a clean function so hybrid.py can call it the same
way it calls search_bm25 — one function, one ranked list back.

Design note: we pass the model and collection in as arguments rather
than constructing them inside the function. This lets the caller
load both once and reuse them across many queries, instead of
reloading the 80 MB model on every call.
"""

from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict, Any


def search_dense(
    query: str,
    collection: chromadb.Collection,
    model: SentenceTransformer,
    n_results: int = 10,
) -> List[Dict[str, Any]]:
    """
    Run dense vector search against a ChromaDB collection.

    Args:
        query:      Natural-language question from the user.
        collection: A pre-loaded ChromaDB collection (PersistentClient).
        model:      A loaded SentenceTransformer model.
        n_results:  Number of candidates to retrieve. We default to 10
                    (not 5) so RRF has a wider pool to work with.

    Returns:
        List of result dicts sorted by cosine similarity (highest first).
        Each dict: rank, score (cosine similarity), text, metadata.

    ChromaDB distance note:
        ChromaDB stores the collection with metadata={"hnsw:space": "cosine"}.
        It returns cosine *distance* in the "distances" field, not similarity.
        Cosine distance = 1 - cosine similarity.
        So: similarity = 1 - distance.
        Distance 0.0 = identical. Distance 2.0 = opposite.
        We convert so that score=1.0 is a perfect match (consistent
        with what an interviewer expects "score" to mean).
    """
    # Embed the query — encode() returns a (1, 384) array; [0] gives the vector
    query_embedding = model.encode([query])[0]

    # ChromaDB returns results nested in lists-of-lists because it supports
    # batch queries. results["documents"][0] is the list for our single query.
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for rank_idx, (doc, meta, dist) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ):
        output.append(
            {
                "rank": rank_idx + 1,
                "score": round(1.0 - dist, 6),   # cosine distance → similarity
                "text": doc,
                "metadata": meta,
            }
        )

    return output