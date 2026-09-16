"""
rag_pipeline.py — Full end-to-end RAG pipeline assembly.

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

Logging:
Set LOG_LEVEL=DEBUG in your .env to enable per-chunk debug output at each
pipeline stage (chunk text, cross-encoder score, file name, page number).
Default level is INFO — stage-level progress only.

Environment variables required (all in .env):
    GROQ_API_KEY         — for HyDE generation and answer generation
    LANGSMITH_API_KEY    — for LangSmith tracing
    LANGCHAIN_PROJECT    — project name shown in LangSmith UI (optional)
    LOG_LEVEL            — logging verbosity: DEBUG | INFO | WARNING (default: INFO)
"""

import logging
import os
import time

import chromadb
from langsmith import traceable
from sentence_transformers import SentenceTransformer

from src.generation.generator import generate_answer
from src.retrieval.dense   import search_dense
from src.retrieval.hybrid  import reciprocal_rank_fusion
from src.retrieval.query   import generate_hyde
from src.retrieval.reranker import deduplicate_by_page, load_reranker, rerank
from src.retrieval.sparse  import build_bm25_index, search_bm25

logger = logging.getLogger(__name__)

CHROMA_PATH    = "./chroma_db"
COLLECTION     = "documents"
EMBED_MODEL    = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

POOL_SIZE      = 20   # candidates retrieved from each system (dense, BM25)
TOP_K_RERANK   = 5    # candidates passed to cross-encoder (from RRF pool)
TOP_K_GENERATE = 5    # chunks passed to LLM for generation


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
    logger.info("Loading ChromaDB...")
    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION)

    logger.info("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL)

    logger.info("Building BM25 index...")
    bm25, chunks = build_bm25_index(collection)
    logger.info("  BM25 index: %d chunks", len(chunks))

    logger.info("Loading cross-encoder reranker...")
    reranker = load_reranker(RERANKER_MODEL)

    return {
        "collection":  collection,
        "embed_model": embed_model,
        "bm25":        bm25,
        "chunks":      chunks,
        "reranker":    reranker,
    }


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


@traceable(name="RAG Pipeline")
def run_pipeline(
    query: str,
    resources: dict,
    pool_size:      int = POOL_SIZE,
    top_k_rerank:   int = TOP_K_RERANK,
    top_k_generate: int = TOP_K_GENERATE,
) -> dict:
    """
    Run the full RAG pipeline end-to-end.

    The @traceable decorator on this function creates the parent trace in
    LangSmith. Every @traceable stage function called inside here becomes
    a child run under this parent — you see the full tree in the LangSmith
    UI: RAG Pipeline → HyDE Generation → Dense Retrieval → ... → LLM Generation.

    Progress is logged at INFO level. Set LOG_LEVEL=DEBUG in .env to see
    per-chunk details (chunk text, cross-encoder score, file, page) at each
    stage — no code changes required.

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
            'timings'          — per-stage wall-clock seconds + total
    """
    t_pipeline_start = time.time()

    logger.info("=" * 60)
    logger.info("Query: %s", query)
    logger.info("=" * 60)

    # Stage 1 — HyDE
    logger.info("[Stage 1] Generating HyDE passage...")
    t0 = time.time()
    hyde_passage = stage_hyde(query)
    t_hyde = time.time() - t0
    logger.info("  HyDE passage (%d chars) generated in %.2fs.", len(hyde_passage), t_hyde)

    # Stage 2 — Dense retrieval
    logger.info("[Stage 2] Dense retrieval (top-%d)...", pool_size)
    t0 = time.time()
    dense_results = stage_dense(
        hyde_passage,
        resources["collection"],
        resources["embed_model"],
        n_results=pool_size,
    )
    t_dense = time.time() - t0
    logger.info("  %d dense candidates in %.2fs.", len(dense_results), t_dense)

    # Stage 3 — BM25 retrieval
    logger.info("[Stage 3] BM25 retrieval (top-%d)...", pool_size)
    t0 = time.time()
    bm25_results = stage_bm25(
        query,
        resources["bm25"],
        resources["chunks"],
        n_results=pool_size,
    )
    t_bm25 = time.time() - t0
    logger.info("  %d BM25 candidates in %.2fs.", len(bm25_results), t_bm25)

    # Stage 4 — RRF fusion
    logger.info("[Stage 4] RRF fusion...")
    t0 = time.time()
    rrf_pool = stage_rrf(dense_results, bm25_results, n_results=pool_size)
    t_rrf = time.time() - t0
    logger.info("  RRF pool: %d candidates in %.2fs.", len(rrf_pool), t_rrf)

    # Stage 5 — Cross-encoder reranking
    logger.info("[Stage 5] Cross-encoder reranking (top-%d)...", top_k_rerank)
    t0 = time.time()
    reranked = stage_rerank(
        query,
        rrf_pool,
        resources["reranker"],
        n_results=top_k_rerank,
    )
    t_rerank = time.time() - t0
    logger.info("  Reranked: %d candidates in %.2fs.", len(reranked), t_rerank)

    # Stage 6 — Deduplication
    logger.info("[Stage 6] Page-level deduplication...")
    t0 = time.time()
    deduped = stage_dedup(reranked)
    t_dedup = time.time() - t0
    logger.info("  After dedup: %d unique pages in %.2fs.", len(deduped), t_dedup)

    # DEBUG: per-chunk details after dedup (only emitted when LOG_LEVEL=DEBUG)
    for c in deduped:
        meta = c["metadata"]
        logger.debug(
            "    Rank %d  Page %s  CE=%.4f  %s | %.80s",
            c["rank"],
            meta.get("page_number"),
            c.get("rerank_score", float("nan")),
            meta.get("file_name", ""),
            c["text"].replace("\n", " "),
        )

    # Stage 7 — LLM generation
    generation_chunks = deduped[:top_k_generate]
    logger.info("[Stage 7] Generating answer from %d chunks...", len(generation_chunks))
    t0 = time.time()
    result = stage_generate(query, generation_chunks)
    t_generate = time.time() - t0
    logger.info("  Answer generated in %.2fs.", t_generate)

    t_total = time.time() - t_pipeline_start

    logger.info("─" * 60)
    logger.info("ANSWER: %s", result["answer"])
    logger.info("─" * 60)
    for s in result["sources"]:
        logger.info("  [SOURCE %d] %s — Page %s", s["source_num"], s["file_name"], s["page_number"])
    logger.info("Total pipeline time: %.2fs", t_total)

    result["contexts"]          = [c["text"] for c in generation_chunks]
    result["hyde_passage"]      = hyde_passage
    result["rrf_pool_size"]     = len(rrf_pool)
    result["reranked_count"]    = len(reranked)
    result["final_chunk_count"] = len(deduped)
    result["timings"] = {
        "hyde_s":     round(t_hyde,     3),
        "dense_s":    round(t_dense,    3),
        "bm25_s":     round(t_bm25,     3),
        "rrf_s":      round(t_rrf,      3),
        "rerank_s":   round(t_rerank,   3),
        "dedup_s":    round(t_dedup,    3),
        "generate_s": round(t_generate, 3),
        "total_s":    round(t_total,    3),
    }

    return result
