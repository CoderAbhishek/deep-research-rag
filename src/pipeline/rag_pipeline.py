"""
rag_pipeline.py — Full end-to-end RAG pipeline assembly.

This module wires together every component built in Sessions 3–9 into a
single callable function: run_pipeline(query) → answer + sources.

Pipeline stages (in order):
1. HyDE generation      — LLM generates a hypothetical document passage
2. Dense retrieval       — embed HyDE passage, query ChromaDB for top-N
3. BM25 retrieval        — keyword search using original query, top-N
4. RRF fusion            — combine dense + BM25 ranked lists
5. Cross-encoder rerank  — score top-K (query, chunk) pairs, sort by CE score
6. Page deduplication    — keep highest-CE chunk per unique page
7. LLM generation        — produce a grounded, cited answer from top chunks

LangSmith tracing:
Each stage is decorated with @traceable so it appears as a child run in
the LangSmith trace tree. This means when you open LangSmith and click on
a run, you see the input/output of every stage — not just the final answer.
This is essential for debugging: when the answer is wrong, you can tell
immediately whether retrieval returned the wrong chunks (retrieval problem)
or the LLM ignored the correct chunks (generation problem).

Environment variables required (all in .env):
    GROQ_API_KEY         — for HyDE generation and answer generation
    LANGSMITH_API_KEY    — for LangSmith tracing
    LANGCHAIN_PROJECT    — project name shown in LangSmith UI (optional)
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer
from langsmith import traceable

from src.retrieval.sparse  import build_bm25_index, search_bm25
from src.retrieval.dense   import search_dense
from src.retrieval.hybrid  import reciprocal_rank_fusion
from src.retrieval.query   import generate_hyde
from src.retrieval.reranker import load_reranker, rerank, deduplicate_by_page
from src.generation.generator import generate_answer

# ─── Config ──────────────────────────────────────────────────────────────────

# These match the values used throughout the test scripts.
# Centralising them here means the pipeline uses the same settings as
# the individual-component tests — no drift between experiment and production.

CHROMA_PATH   = "./chroma_db"
COLLECTION    = "documents"
EMBED_MODEL   = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

POOL_SIZE      = 20   # candidates retrieved from each system (dense, BM25)
TOP_K_RERANK   = 5    # candidates passed to cross-encoder (from RRF pool)
TOP_K_GENERATE = 5    # chunks passed to LLM for generation


# ─── Resource loading ─────────────────────────────────────────────────────────

def load_resources():
    """
    Load all heavy resources once at startup.

    ChromaDB client, embedding model, BM25 index, and cross-encoder are all
    loaded here — not inside run_pipeline(). Loading them inside the pipeline
    function would reload ~300MB of model weights on every query call, which
    is clearly wrong for a production system.

    This function returns a resource dict that is passed into run_pipeline().
    The caller loads resources once (on startup) and reuses the same dict
    for every query.

    Returns:
        dict with keys: 'collection', 'embed_model', 'bm25', 'chunks', 'reranker'
    """
    print("Loading ChromaDB...")
    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION)

    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL)

    print("Building BM25 index...")
    # build_bm25_index now accepts a Collection directly (Session 8 fix).
    # It calls collection.get() internally — the caller does not need to
    # pre-fetch anything.
    bm25, chunks = build_bm25_index(collection)
    print(f"  BM25 index: {len(chunks)} chunks")

    print("Loading cross-encoder reranker...")
    reranker = load_reranker(RERANKER_MODEL)

    return {
        "collection":  collection,
        "embed_model": embed_model,
        "bm25":        bm25,
        "chunks":      chunks,
        "reranker":    reranker,
    }


# ─── Individual pipeline stages (each traced separately in LangSmith) ─────────

@traceable(name="1. HyDE Generation")
def stage_hyde(query: str) -> str:
    """
    Stage 1: Generate a hypothetical document passage using the LLM.

    @traceable wraps this function so LangSmith logs:
    - Input: the original query string
    - Output: the generated HyDE passage
    - Latency: how long the LLM call took

    Why HyDE here: we use the HyDE passage as the dense retrieval query
    (not the original question). This closes the query-document embedding
    gap — a passage in "annual report language" lands closer to actual
    annual report chunks than a conversational question does.

    The original query is passed separately to BM25 (keyword matching)
    and to the generator (answering the actual question).
    """
    return generate_hyde(query)


@traceable(name="2. Dense Retrieval")
def stage_dense(hyde_passage: str, collection, embed_model, n_results: int) -> list:
    """
    Stage 2: Dense retrieval using the HyDE passage as the query.

    Note: we embed the PASSAGE, not the original question.
    search_dense handles the embedding internally.
    """
    return search_dense(hyde_passage, collection, embed_model, n_results=n_results)


@traceable(name="3. BM25 Retrieval")
def stage_bm25(query: str, bm25, chunks, n_results: int) -> list:
    """
    Stage 3: BM25 keyword retrieval using the ORIGINAL query.

    We use the original query for BM25 (not the HyDE passage) because:
    - BM25 scores exact token matches against the corpus vocabulary
    - The original query tokens ("revenue", "geography", "infosys") are
      the right terms to look for in exact match
    - The HyDE passage contains fabricated numbers that would confuse BM25
      (it would try to match "1,24,500 crore" as a literal token)
    """
    return search_bm25(query, bm25, chunks, n_results=n_results)


@traceable(name="4. RRF Fusion")
def stage_rrf(dense_results: list, bm25_results: list, n_results: int) -> list:
    """
    Stage 4: Reciprocal Rank Fusion — combine dense and BM25 ranked lists.

    reciprocal_rank_fusion accepts a list of ranked lists, so passing
    [dense_results, bm25_results] means we fuse exactly two systems.
    In Session 7, we demonstrated passing more lists (multi-query variants).
    """
    return reciprocal_rank_fusion([dense_results, bm25_results], n_results=n_results)


@traceable(name="5. Cross-Encoder Reranking")
def stage_rerank(query: str, candidates: list, reranker, n_results: int) -> list:
    """
    Stage 5: Cross-encoder reranking of the RRF candidate pool.

    We pass the ORIGINAL query to the cross-encoder (not the HyDE passage).
    The cross-encoder evaluates (query, chunk) pairs for relevance to the
    actual question — the HyDE passage is irrelevant at this stage.
    """
    return rerank(query, candidates, reranker, n_results=n_results)


@traceable(name="6. Page Deduplication")
def stage_dedup(reranked: list) -> list:
    """
    Stage 6: Remove duplicate chunks from the same page.

    After reranking, the same page may appear multiple times (different
    chunks with different CE scores). Keep only the highest-CE chunk
    per unique (file_name, page_number) pair.

    Session 8 finding: Page 80 appeared at pool positions 4 (CE +0.818)
    and 12 (CE -1.995). Without deduplication, both compete for top-5 slots.
    With deduplication, only the table chunk (CE +0.818) survives.
    """
    return deduplicate_by_page(reranked)


@traceable(name="7. LLM Generation")
def stage_generate(query: str, chunks: list) -> dict:
    """
    Stage 7: LLM answer generation from retrieved chunks.

    The top chunks (after deduplication) are formatted into a labelled
    context block and passed to the LLM with a grounding-discipline prompt.
    """
    return generate_answer(query, chunks)


# ─── Full pipeline ────────────────────────────────────────────────────────────

@traceable(name="RAG Pipeline")
def run_pipeline(
    query: str,
    resources: dict,
    pool_size:      int = POOL_SIZE,
    top_k_rerank:   int = TOP_K_RERANK,
    top_k_generate: int = TOP_K_GENERATE,
    verbose:        bool = True,
) -> dict:
    """
    Run the full RAG pipeline end-to-end.

    The @traceable decorator on this function creates the parent trace in
    LangSmith. Every @traceable stage function called inside here becomes
    a child run under this parent — you see the full tree in the LangSmith
    UI: RAG Pipeline → HyDE Generation → Dense Retrieval → ... → LLM Generation.

    Args:
        query:          User's question.
        resources:      Dict from load_resources() — collection, embed_model,
                        bm25, chunks, reranker.
        pool_size:      Candidates to retrieve from dense and BM25 each.
                        Both systems are called with n_results=pool_size,
                        then the results are fused. Default 20.
        top_k_rerank:   Candidates from the RRF pool to pass to the
                        cross-encoder. Cross-encoder sees top_k_rerank
                        (query, chunk) pairs. Default 5.
        top_k_generate: Chunks to pass to the LLM after deduplication.
                        Typically ≤ top_k_rerank. Default 5.
        verbose:        If True, print stage-by-stage progress to stdout.

    Returns:
        Dict with:
            'answer'           — the generated answer string
            'sources'          — list of {source_num, file_name, page_number}
            'query'            — the original query
            'num_chunks'       — chunks in the generation context
            'model'            — LLM used for generation
            'hyde_passage'     — the HyDE passage generated (for debugging)
            'rrf_pool_size'    — how many candidates entered the RRF pool
            'reranked_count'   — how many candidates went to the cross-encoder
            'final_chunk_count'— chunks after deduplication
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

    # Stage 1: HyDE
    if verbose: print("\n[Stage 1] Generating HyDE passage...")
    hyde_passage = stage_hyde(query)
    if verbose: print(f"  HyDE passage ({len(hyde_passage)} chars) generated.")

    # Stage 2: Dense retrieval on HyDE passage
    if verbose: print(f"\n[Stage 2] Dense retrieval (top-{pool_size})...")
    dense_results = stage_dense(
        hyde_passage,
        resources["collection"],
        resources["embed_model"],
        n_results=pool_size,
    )
    if verbose: print(f"  {len(dense_results)} dense candidates.")

    # Stage 3: BM25 retrieval on original query
    if verbose: print(f"\n[Stage 3] BM25 retrieval (top-{pool_size})...")
    bm25_results = stage_bm25(
        query,
        resources["bm25"],
        resources["chunks"],
        n_results=pool_size,
    )
    if verbose: print(f"  {len(bm25_results)} BM25 candidates.")

    # Stage 4: RRF fusion
    if verbose: print("\n[Stage 4] RRF fusion...")
    rrf_pool = stage_rrf(dense_results, bm25_results, n_results=pool_size)
    if verbose: print(f"  RRF pool: {len(rrf_pool)} candidates.")

    # Stage 5: Cross-encoder reranking
    if verbose: print(f"\n[Stage 5] Cross-encoder reranking (top-{top_k_rerank})...")
    reranked = stage_rerank(
        query,
        rrf_pool,
        resources["reranker"],
        n_results=top_k_rerank,
    )
    if verbose: print(f"  Reranked: {len(reranked)} candidates.")

    # Stage 6: Page deduplication
    if verbose: print("\n[Stage 6] Page-level deduplication...")
    deduped = stage_dedup(reranked)
    if verbose:
        print(f"  After dedup: {len(deduped)} unique pages.")
        for c in deduped:
            meta = c["metadata"]
            print(f"    Rank {c['rank']}  Page {meta.get('page_number')}  "
                  f"CE={c.get('rerank_score', 'n/a'):.4f}  "
                  f"{meta.get('file_name', '')}")

    # Stage 7: Generation — use top top_k_generate chunks after dedup
    generation_chunks = deduped[:top_k_generate]
    if verbose: print(f"\n[Stage 7] Generating answer from {len(generation_chunks)} chunks...")
    result = stage_generate(query, generation_chunks)

    # Add pipeline metadata to the result dict
    result["contexts"]          = [c["text"] for c in generation_chunks]   # Session 10
    result["hyde_passage"]      = hyde_passage
    result["rrf_pool_size"]     = len(rrf_pool)
    result["reranked_count"]    = len(reranked)
    result["final_chunk_count"] = len(deduped)

    if verbose:
        print(f"\n{'─'*60}")
        print("ANSWER:")
        print(f"{'─'*60}")
        print(result["answer"])
        print(f"\nSources cited:")
        for s in result["sources"]:
            print(f"  [SOURCE {s['source_num']}] {s['file_name']} — Page {s['page_number']}")

    return result