"""
test_hybrid_retrieval.py

Hybrid Search + Reciprocal Rank Fusion

Runs the same query ("What is Infosys revenue from North America?") through
three retrieval methods and prints them side by side:
  1. Dense-only (ChromaDB + sentence-transformers)
  2. BM25-only  (rank_bm25 over ChromaDB corpus)
  3. Hybrid     (Dense top-10 + BM25 top-10 → RRF → top-5)

Candidate pool is top-10 from each system (not top-5) so that RRF has
a wider pool. Chunks that rank consistently in both systems will rise;
chunks that only appeared in one system will be demoted.
"""

import chromadb
from sentence_transformers import SentenceTransformer

from src.retrieval.dense import search_dense
from src.retrieval.sparse import build_bm25_index, search_bm25
from src.retrieval.hybrid import reciprocal_rank_fusion

QUERY = "What is Infosys revenue from North America?"
N_CANDIDATES = 10   # retrieve this many from each system before fusing
N_FINAL = 5         # return this many after RRF

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load shared resources
# ─────────────────────────────────────────────────────────────────────────────
# We load ChromaDB, the embedding model, and build the BM25 index once.
# All three retrieval methods share the same underlying corpus — 3439 chunks
# stored in ChromaDB. BM25 is rebuilt from the ChromaDB text, same as Session 5.

print("=" * 60)
print("Step 1: Loading shared resources")
print("=" * 60)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="documents")

# Load all chunks for BM25 (text + metadata, no embeddings needed)
all_data = collection.get(include=["documents", "metadatas"])
chunks = [
    {"text": doc, "metadata": meta}
    for doc, meta in zip(all_data["documents"], all_data["metadatas"])
]
print(f"Loaded {len(chunks)} chunks from ChromaDB.")

# Embedding model — same one used at ingestion time
# Loading takes ~2 seconds; subsequent calls use the cached model
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded.")

# BM25 index — built on the same text corpus
bm25, chunks = build_bm25_index(chunks)

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Dense retrieval (top-10)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Step 2: Dense retrieval — top {N_CANDIDATES}")
print("=" * 60)
print(f"Query: '{QUERY}'\n")

dense_results = search_dense(QUERY, collection, model, n_results=N_CANDIDATES)

for r in dense_results:
    print(
        f"  Rank {r['rank']:2d} | score={r['score']:.4f} | "
        f"{r['metadata']['file_name']}, Page {r['metadata']['page_number']}"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: BM25 retrieval (top-10)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Step 3: BM25 retrieval — top {N_CANDIDATES}")
print("=" * 60)

bm25_results = search_bm25(QUERY, bm25, chunks, n_results=N_CANDIDATES)

for r in bm25_results:
    print(
        f"  Rank {r['rank']:2d} | score={r['score']:.4f} | "
        f"{r['metadata']['file_name']}, Page {r['metadata']['page_number']}"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Hybrid — RRF over the two candidate pools
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Step 4: Hybrid RRF — top {N_FINAL} from fused pool")
print("=" * 60)

hybrid_results = reciprocal_rank_fusion(
    ranked_lists=[dense_results, bm25_results],
    k=60,
    n_results=N_FINAL,
)

print()
for r in hybrid_results:
    print(f"--- Hybrid Rank {r['rank']} (RRF score: {r['rrf_score']:.6f}) ---")
    print(f"Source: {r['metadata']['file_name']}, Page {r['metadata']['page_number']}")
    print(f"Text preview:\n{r['text'][:500]}")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Side-by-side summary
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Side-by-side summary (page numbers only)")
print("=" * 60)
print(f"{'Rank':<6} {'Dense':<35} {'BM25':<35} {'Hybrid':<35}")
print("-" * 111)

max_rows = max(len(dense_results), len(bm25_results), len(hybrid_results))
for i in range(max_rows):
    d = f"p.{dense_results[i]['metadata']['page_number']}"  if i < len(dense_results)  else "—"
    b = f"p.{bm25_results[i]['metadata']['page_number']}"   if i < len(bm25_results)   else "—"
    h = f"p.{hybrid_results[i]['metadata']['page_number']}" if i < len(hybrid_results) else "—"
    print(f"{i+1:<6} {d:<35} {b:<35} {h:<35}")

print()
print("Target: Page 80 (infosys-ar-26.pdf) = correct North America revenue answer")
print("Dense MRR (Session 4):  0   (not in top 3)")
print("BM25  MRR (Session 5):  0.33 (correct at rank 3)")
# Compute actual hybrid MRR
hybrid_mrr = 0.0
for r in hybrid_results:
    if r["metadata"]["page_number"] == 80 and "infosys-ar-26" in r["metadata"]["file_name"]:
        hybrid_mrr = 1.0 / r["rank"]
        break
print(f"Hybrid MRR (Session 6): {hybrid_mrr:.2f} (correct at rank {int(1/hybrid_mrr) if hybrid_mrr else 'not found'})")