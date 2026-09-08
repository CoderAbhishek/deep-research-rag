"""
test_sparse_retrieval.py

BM25 Sparse Retrieval

Runs the same query as Session 4 ("What is Infosys revenue from North America?")
but through BM25 instead of dense vector search.

Key design decision: we load chunks from the existing ChromaDB collection
rather than re-running load + chunk + embed. ChromaDB already holds all
3439 chunks with their text and metadata — it is our source of truth for
the corpus. BM25 is a separate in-memory index built on top of the same data.
"""

import chromadb
from src.retrieval.sparse import build_bm25_index, search_bm25

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load chunks from ChromaDB
# ─────────────────────────────────────────────────────────────────────────────
# We already ran the full ingestion pipeline in Session 4.
# ChromaDB's PersistentClient reads the stored data from ./chroma_db on disk.
# collection.get() with include=["documents", "metadatas"] returns all stored
# text and metadata without the embeddings (we don't need them for BM25).

print("=" * 60)
print("Step 1: Loading chunks from ChromaDB")
print("=" * 60)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="documents")

result = collection.get(include=["documents", "metadatas"])

# Reconstruct the same chunk dict format our ingestion pipeline produces
chunks = [
    {"text": doc, "metadata": meta}
    for doc, meta in zip(result["documents"], result["metadatas"])
]

print(f"Loaded {len(chunks)} chunks from ChromaDB.")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Build the BM25 index
# ─────────────────────────────────────────────────────────────────────────────
# BM25 needs to tokenise every chunk and compute IDF across the whole corpus.
# This is fast — tokenisation is just regex, IDF is just counting.
# On 3439 chunks, expect under 3 seconds.

print("\n" + "=" * 60)
print("Step 2: Building BM25 index")
print("=" * 60)

bm25, chunks = build_bm25_index(chunks)

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Query — same question as Session 4 dense retrieval
# ─────────────────────────────────────────────────────────────────────────────
# We use exactly the same query so we can compare BM25 vs dense results
# side by side. This is how you evaluate retrieval methods rigorously —
# identical inputs, observe different outputs, understand why they differ.

print("\n" + "=" * 60)
print("Step 3: BM25 retrieval — same query as Session 4")
print("=" * 60)

query = "What is Infosys revenue from North America?"
print(f"Query: '{query}'\n")

results = search_bm25(query, bm25, chunks, n_results=5)

print()
for r in results:
    print(f"--- Result {r['rank']} (BM25 score: {r['score']:.4f}) ---")
    print(f"Source: {r['metadata']['file_name']}, Page {r['metadata']['page_number']}")
    print(f"Text preview:\n{r['text'][:500]}")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# Recap of what dense retrieval returned in Session 4 (for comparison)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Session 4 Dense Retrieval results (for comparison):")
print("=" * 60)
print("Result 1: additional-information-2025-26.pdf, Page 1 — ratio analysis table (WRONG)")
print("Result 2: infosys-ar-26.pdf, Page 46       — subsidiary with 'NA' abbreviation (WRONG)")
print("Result 3: infosys-ar-26.pdf, Page 287      — consolidated P&L total revenue (wrong level)")
print()
print("If BM25 results differ meaningfully — that's the improvement we expected.")
print("If BM25 shows the geographic segment note — BM25 fixed it.")
print("If neither method gets the right answer — hybrid + reranking is why we have Sessions 6 and 8.")