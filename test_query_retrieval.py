"""
test_query_retrieval_v2.py

Corrected version of test_query_retrieval.py.

Bug fixed: Technique 5 now uses dense_results_hyde (HyDE dense retrieval)
as the dense component, not dense_results_rewritten. This runs the actual
experiment: HyDE Dense + BM25 → RRF.

Run from project root with the .venv activated:
    python test_query_retrieval_v2.py
"""

import os
import sys

# ─── Path setup ──────────────────────────────────────────────────────────────
# Allows importing from src/ without installing as a package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import chromadb
from sentence_transformers import SentenceTransformer

from src.retrieval.sparse  import search_bm25, build_bm25_index
from src.retrieval.dense   import search_dense
from src.retrieval.hybrid  import reciprocal_rank_fusion
from src.retrieval.query   import rewrite_query, generate_multi_query, generate_hyde

# ─── Config ──────────────────────────────────────────────────────────────────
CHROMA_PATH   = "./chroma_db"
COLLECTION    = "documents"
EMBED_MODEL   = "all-MiniLM-L6-v2"
QUERY         = "What is the revenue breakdown by geography for Infosys?"
CORRECT_PAGE  = 80       # page_number in metadata for the expected answer
TOP_K_POOL    = 10       # candidates per system for RRF input
TOP_K_DISPLAY = 5        # results shown to user

# ─── Load resources ──────────────────────────────────────────────────────────
print("Loading ChromaDB and embedding model...")
client     = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION)
model      = SentenceTransformer(EMBED_MODEL)

print("Building BM25 index from ChromaDB corpus...")
result = collection.get(include=["documents", "metadatas"])
raw_chunks = [
    {"text": doc, "metadata": meta}
    for doc, meta in zip(result["documents"], result["metadatas"])
]
bm25, chunks = build_bm25_index(raw_chunks)
print(f"  BM25 index built over {len(chunks)} chunks.\n")


# ─── Helper: compute MRR@k ───────────────────────────────────────────────────
def compute_mrr(results: list, correct_page: int, k: int = 5) -> float:
    """
    MRR = 1 / rank of the first result whose page_number matches correct_page.
    Returns 0.0 if correct_page does not appear in the top-k results.
    """
    for result in results[:k]:
        if result["metadata"]["page_number"] == correct_page:
            return 1.0 / result["rank"]
    return 0.0


# ─── Helper: print results table ─────────────────────────────────────────────
def print_results(results: list, label: str, correct_page: int) -> None:
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    for r in results:
        page = r["metadata"]["page_number"]
        flag = " ← CORRECT" if page == correct_page else ""
        score_key = "rrf_score" if "rrf_score" in r else "score"
        print(f"  Rank {r['rank']:2d}  Page {page:4d}  score={r[score_key]:.6f}{flag}")
        # Print first 100 chars of text for context
        snippet = r["text"].replace("\n", " ")[:100]
        print(f"           {snippet}...")
    mrr = compute_mrr(results, correct_page)
    print(f"\n  MRR@5 = {mrr:.6f}")


# ─── Technique 1: Original query → dense ─────────────────────────────────────
print("\n" + "="*60)
print("TECHNIQUE 1: Original query → dense retrieval")
print("="*60)
print(f"Query: {QUERY}")

dense_results_original = search_dense(QUERY, collection, model, n_results=TOP_K_POOL)
print_results(dense_results_original[:TOP_K_DISPLAY], "Dense (original query)", CORRECT_PAGE)


# ─── Technique 2: Rewritten query → dense ────────────────────────────────────
print("\n" + "="*60)
print("TECHNIQUE 2: Rewritten query → dense retrieval")
print("="*60)

rewritten = rewrite_query(QUERY)
print(f"Rewritten: {rewritten}")

dense_results_rewritten = search_dense(rewritten, collection, model, n_results=TOP_K_POOL)
print_results(dense_results_rewritten[:TOP_K_DISPLAY], "Dense (rewritten query)", CORRECT_PAGE)


# ─── Technique 3: Multi-query → dense union → RRF ────────────────────────────
print("\n" + "="*60)
print("TECHNIQUE 3: Multi-query → dense union → RRF")
print("="*60)

variants = generate_multi_query(QUERY, n=3)
print("Generated variants:")
for i, v in enumerate(variants, 1):
    print(f"  {i}. {v}")

all_dense_lists = [dense_results_original]          # include original
for v in variants:
    vr = search_dense(v, collection, model, n_results=TOP_K_POOL)
    all_dense_lists.append(vr)

multi_query_results = reciprocal_rank_fusion(all_dense_lists, n_results=TOP_K_DISPLAY)
print_results(multi_query_results, "Multi-query dense → RRF", CORRECT_PAGE)


# ─── Technique 4: HyDE → dense ───────────────────────────────────────────────
print("\n" + "="*60)
print("TECHNIQUE 4: HyDE → dense retrieval")
print("="*60)

hyde_passage = generate_hyde(QUERY)
print(f"HyDE passage:\n{hyde_passage}\n")

# IMPORTANT: embed the PASSAGE, not the question
dense_results_hyde = search_dense(hyde_passage, collection, model, n_results=TOP_K_POOL)
print_results(dense_results_hyde[:TOP_K_DISPLAY], "Dense (HyDE passage)", CORRECT_PAGE)


# ─── Technique 5 (CORRECTED): HyDE Dense + BM25 → RRF ───────────────────────
print("\n" + "="*60)
print("TECHNIQUE 5 (CORRECTED): HyDE Dense + BM25 → RRF")
print("  (v1 bug: used rewritten dense instead of HyDE dense)")
print("  (v2 fix: using dense_results_hyde as the dense component)")
print("="*60)

bm25_results = search_bm25(QUERY, bm25, chunks, n_results=TOP_K_POOL)
print("\nBM25 top-5 (using original query):")
print_results(bm25_results[:TOP_K_DISPLAY], "BM25 (original query)", CORRECT_PAGE)

# Now fuse HyDE dense + BM25
hyde_bm25_results = reciprocal_rank_fusion(
    [dense_results_hyde, bm25_results],   # FIX: dense_results_hyde, not dense_results_rewritten
    n_results=TOP_K_DISPLAY,
)
print_results(hyde_bm25_results, "HyDE Dense + BM25 → RRF (CORRECTED)", CORRECT_PAGE)


# ─── Chunk-level overlap diagnostic ──────────────────────────────────────────
print("\n" + "="*60)
print("DIAGNOSTIC: Chunk-level overlap between HyDE Dense and BM25")
print("="*60)

def chunk_keys(results: list) -> dict:
    """Return {key: rank} for a result list."""
    keys = {}
    for r in results:
        m = r["metadata"]
        key = f"{m['file_name']}_p{m['page_number']}_c{m['chunk_index']}"
        keys[key] = r["rank"]
    return keys

hyde_keys = chunk_keys(dense_results_hyde)
bm25_keys = chunk_keys(bm25_results)
overlap   = set(hyde_keys.keys()) & set(bm25_keys.keys())

print(f"\n  HyDE dense pool size : {len(hyde_keys)} chunks")
print(f"  BM25 pool size       : {len(bm25_keys)} chunks")
print(f"  Chunk-level overlap  : {len(overlap)} chunks")

if overlap:
    print("\n  Overlapping chunks (got cross-system bonus):")
    for key in overlap:
        parts = key.split("_p")
        page  = parts[-1].split("_c")[0]
        cidx  = parts[-1].split("_c")[1]
        print(f"    key={key}  (HyDE rank {hyde_keys[key]}, BM25 rank {bm25_keys[key]})")
        # Compute what the RRF score would be
        k = 60
        rrf = 1.0 / (hyde_keys[key] + k) + 1.0 / (bm25_keys[key] + k)
        print(f"    Combined RRF score = 1/{hyde_keys[key]+k} + 1/{bm25_keys[key]+k} = {rrf:.6f}")
else:
    print("\n  Zero chunk-level overlap.")
    print("  Every RRF score came from exactly one system.")
    print("  No cross-system bonus fired.")
    print("  RRF was mathematically equivalent to interleaving, not fusion.")


# ─── Summary table ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY — MRR@5 across all techniques")
print("="*60)

t1_mrr  = compute_mrr(dense_results_original[:TOP_K_DISPLAY], CORRECT_PAGE)
t2_mrr  = compute_mrr(dense_results_rewritten[:TOP_K_DISPLAY], CORRECT_PAGE)
t3_mrr  = compute_mrr(multi_query_results, CORRECT_PAGE)
t4_mrr  = compute_mrr(dense_results_hyde[:TOP_K_DISPLAY], CORRECT_PAGE)
t5_mrr  = compute_mrr(hyde_bm25_results, CORRECT_PAGE)
bm25_mrr = compute_mrr(bm25_results[:TOP_K_DISPLAY], CORRECT_PAGE)

rows = [
    ("Dense (original)",              t1_mrr),
    ("Dense (rewritten)",             t2_mrr),
    ("Multi-query dense RRF",         t3_mrr),
    ("HyDE Dense",                    t4_mrr),
    ("BM25 (original query)",         bm25_mrr),
    ("HyDE Dense + BM25 RRF (FIXED)", t5_mrr),
]
print(f"\n  {'Technique':<40} MRR@5")
print(f"  {'-'*40} ------")
for label, mrr in rows:
    flag = " ← BEST" if mrr == max(r for _, r in rows) and mrr > 0 else ""
    print(f"  {label:<40} {mrr:.6f}{flag}")

print(f"\n  Correct answer: Page {CORRECT_PAGE}")
print(f"  Chunk-level overlap (HyDE Dense vs BM25): {len(overlap)} chunks")