"""
test_reranking.py

Two-stage retrieval pipeline:
  Stage 1: HyDE → dense top-20 + BM25 top-20 → RRF → 20 candidates
  Stage 2: cross-encoder reranker → top-5

Compares:
  - Hybrid top-5 (no reranking)
  - Reranked top-5 (with cross-encoder)

Run from project root with .venv activated:
    python test_reranking.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import chromadb
from sentence_transformers import SentenceTransformer

from src.retrieval.sparse   import search_bm25, build_bm25_index
from src.retrieval.dense    import search_dense
from src.retrieval.hybrid   import reciprocal_rank_fusion
from src.retrieval.query    import generate_hyde
from src.retrieval.reranker import load_reranker, rerank

# ── Config ───────────────────────────────────────────────────────────────────
CHROMA_PATH   = "./chroma_db"
COLLECTION    = "documents"
EMBED_MODEL   = "all-MiniLM-L6-v2"
QUERY         = "What is the revenue breakdown by geography for Infosys?"
CORRECT_PAGE  = 80
POOL_SIZE     = 20    # candidates retrieved per system for Stage 1
TOP_K         = 5     # final results shown to user

# ── Load resources ────────────────────────────────────────────────────────────
print("Loading ChromaDB and embedding model...")
client     = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION)
model      = SentenceTransformer(EMBED_MODEL)

print("Building BM25 index...")
bm25, chunks = build_bm25_index(collection)

print("Loading cross-encoder reranker...")
reranker = load_reranker()
print("All models loaded.\n")


# ── Helper: MRR@k ─────────────────────────────────────────────────────────────
def compute_mrr(results: list, correct_page: int, k: int = 5) -> float:
    for r in results[:k]:
        if r["metadata"]["page_number"] == correct_page:
            return 1.0 / r["rank"]
    return 0.0


# ── Helper: print results ─────────────────────────────────────────────────────
def print_results(results: list, label: str, correct_page: int,
                  score_key: str = "rrf_score") -> None:
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    for r in results:
        page = r["metadata"]["page_number"]
        flag = " ← CORRECT" if page == correct_page else ""
        score = r.get(score_key, r.get("rerank_score", r.get("score", 0)))
        print(f"  Rank {r['rank']:2d}  Page {page:4d}  score={score:.6f}{flag}")
        snippet = r["text"].replace("\n", " ")[:100]
        print(f"           {snippet}...")
    mrr = compute_mrr(results, correct_page)
    print(f"\n  MRR@5 = {mrr:.6f}")


# ── Stage 1: HyDE → dense top-20 + BM25 top-20 → RRF ─────────────────────────
print("=" * 60)
print("STAGE 1: HyDE Dense + BM25 → RRF (20-candidate pool)")
print("=" * 60)

print("\nGenerating HyDE passage...")
hyde_passage = generate_hyde(QUERY)
print(f"HyDE passage:\n{hyde_passage}\n")

print("Running dense retrieval on HyDE passage (top-20)...")
dense_candidates = search_dense(
    hyde_passage, collection, model, n_results=POOL_SIZE
)

print("Running BM25 retrieval on original query (top-20)...")
bm25_candidates = search_bm25(QUERY, bm25, chunks, n_results=POOL_SIZE)

print("Fusing with RRF...")
hybrid_pool = reciprocal_rank_fusion(
    [dense_candidates, bm25_candidates],
    n_results=POOL_SIZE,    # keep all 20 for reranking — don't cut to 5 yet
)

print_results(hybrid_pool[:TOP_K], "Hybrid top-5 (before reranking)", CORRECT_PAGE)

# Show where correct answer sits in the full pool of 20
print(f"\n  Full 20-candidate pool — correct page position:")
for r in hybrid_pool:
    if r["metadata"]["page_number"] == CORRECT_PAGE:
        print(f"    Page {CORRECT_PAGE} at pool position {r['rank']}"
              f"  rrf_score={r['rrf_score']:.6f}")


# ── Stage 2: Cross-encoder reranking ──────────────────────────────────────────
print("\n" + "=" * 60)
print("STAGE 2: Cross-encoder reranking of 20-candidate pool")
print(f"  Model: cross-encoder/ms-marco-MiniLM-L-6-v2")
print(f"  Query passed to reranker: original question (not HyDE passage)")
print("=" * 60)

reranked = rerank(QUERY, hybrid_pool, reranker, n_results=TOP_K)
print_results(reranked, "Reranked top-5", CORRECT_PAGE, score_key="rerank_score")


# ── Show rerank score for every candidate (diagnostic) ────────────────────────
print("\n" + "=" * 60)
print("DIAGNOSTIC: Cross-encoder scores for all 20 candidates")
print("=" * 60)

all_scored = rerank(QUERY, hybrid_pool, reranker, n_results=POOL_SIZE)
print(f"\n  {'Rank':>4}  {'Page':>5}  {'CE Score':>10}  {'RRF Score':>10}  Text snippet")
print(f"  {'─'*4}  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*30}")
for r in all_scored:
    page     = r["metadata"]["page_number"]
    flag     = " ← CORRECT" if page == CORRECT_PAGE else ""
    snippet  = r["text"].replace("\n", " ")[:40]
    print(f"  {r['rank']:>4}  {page:>5}  {r['rerank_score']:>10.4f}"
          f"  {r.get('rrf_score', 0):>10.6f}  {snippet}...{flag}")


# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
hybrid_mrr   = compute_mrr(hybrid_pool[:TOP_K],   CORRECT_PAGE)
reranked_mrr = compute_mrr(reranked,               CORRECT_PAGE)
print(f"\n  {'System':<40} MRR@5")
print(f"  {'─'*40} ──────")
print(f"  {'Hybrid top-5 (no reranking)':<40} {hybrid_mrr:.6f}")
print(f"  {'Hybrid + cross-encoder reranker':<40} {reranked_mrr:.6f}")
print(f"\n  Correct answer: Page {CORRECT_PAGE}")
print(f"  Candidate pool size fed to reranker: {len(hybrid_pool)}")