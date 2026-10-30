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
Each stage logs one INFO line (what it produced and how long it took).
LOG_LEVEL=DEBUG also logs the HyDE passage and every candidate chunk.
The same detail is returned in result["debug"].

Environment variables (all in .env):
    GROQ_API_KEY          - for HyDE generation and answer generation
    LANGCHAIN_TRACING_V2  - "true" turns LangSmith tracing on (optional)
    LANGSMITH_API_KEY     - for LangSmith tracing (optional)
    LANGCHAIN_PROJECT     - project name shown in LangSmith UI (optional)
    LOG_LEVEL             - DEBUG | INFO | WARNING (optional, default INFO)
"""

import logging
import time

import chromadb
from langsmith import get_current_run_tree, traceable
from sentence_transformers import SentenceTransformer

from src.retrieval.sparse  import build_bm25_index, search_bm25
from src.retrieval.dense   import search_dense
from src.retrieval.hybrid  import reciprocal_rank_fusion
from src.retrieval.query   import generate_hyde_result
from src.retrieval.reranker import load_reranker, rerank, deduplicate_by_page
from src.generation.generator import generate_answer

logger = logging.getLogger(__name__)


CHROMA_PATH   = "./chroma_db"
COLLECTION    = "documents"
EMBED_MODEL   = "all-MiniLM-L6-v2"
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
    t_start = time.perf_counter()
    logger.info("Loading ChromaDB from %s ...", CHROMA_PATH)
    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION)

    logger.info("Loading embedding model %s ...", EMBED_MODEL)
    embed_model = SentenceTransformer(EMBED_MODEL)

    bm25, chunks = build_bm25_index(collection)  # logs its own progress

    logger.info("Loading cross-encoder %s ...", RERANKER_MODEL)
    reranker = load_reranker(RERANKER_MODEL)
    logger.info("Resources loaded in %.1fs", time.perf_counter() - t_start)

    return {
        "collection":  collection,
        "embed_model": embed_model,
        "bm25":        bm25,
        "chunks":      chunks,
        "reranker":    reranker,
    }


@traceable(name="1. HyDE Generation")
def stage_hyde(query: str) -> dict:
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
    result = generate_hyde_result(query)
    return {"passage": result.text, "llm": result.stats()}


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

    Motivation (Session 8): two chunks from page 80 appeared in the same
    20-candidate pool with very different cross-encoder scores (+0.818 and
    -1.995). Deduplication guarantees one page cannot take two slots.

    Note: this runs on the already-truncated top-5 list, so when it removes
    a duplicate the generator receives fewer than 5 chunks. In the recorded
    test runs it removed nothing.
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

    Progress is logged (one INFO line per stage) instead of printed. Set
    LOG_LEVEL=DEBUG to also log the HyDE passage and every candidate.
    Callers can pass langsmith_extra={"metadata": {...}, "tags": [...]};
    @traceable consumes it and attaches it to this trace.

    Args:
        query:          User's question.
        resources:      Dict from load_resources() — collection, embed_model,
                        bm25, chunks, reranker.
        pool_size:      Candidates to retrieve from dense and BM25 each.
                        Both systems are called with n_results=pool_size,
                        then the results are fused. Default 20.
        top_k_rerank:   How many candidates to keep after reranking.
                        The cross-encoder scores every candidate in the
                        RRF pool (pool_size pairs) and keeps the best
                        top_k_rerank. Default 5.
        top_k_generate: Chunks to pass to the LLM after deduplication.
                        Typically ≤ top_k_rerank. Default 5.

    Returns:
        Dict with:
            'answer'           — the generated answer string
            'sources'          — sources the answer cites: {source_num, file_name, page_number}
            'retrieved_sources'— every chunk passed to the generator
            'query'            — the original query
            'num_chunks'       — chunks in the generation context
            'model'            — LLM used for generation
            'hyde_passage'     — the HyDE passage generated (for debugging)
            'rrf_pool_size'    — how many candidates entered the RRF pool
            'reranked_count'   — how many candidates went to the cross-encoder
            'final_chunk_count'— chunks after deduplication
            'trace_id'         - LangSmith trace ID (None when tracing is off)
            'debug'            - stage counts, candidates and scores (dense, BM25,
                                 RRF pool, reranked), per-stage timings, LLM usage
                                 and estimated cost
    """
    t_pipeline = time.perf_counter()
    timings = {}

    # When tracing is on, @traceable has already opened the root run for this
    # call. Its trace_id is the ID you search for in LangSmith.
    run_tree = get_current_run_tree()
    trace_id = str(run_tree.trace_id) if run_tree else None
    logger.info("Pipeline start: query=%r trace_id=%s", query[:200], trace_id or "-")

    # Stage 1 - HyDE
    t0 = time.perf_counter()
    hyde = stage_hyde(query)
    timings["hyde_s"] = time.perf_counter() - t0
    hyde_passage = hyde["passage"]
    logger.info("[1/7] HyDE: %d-char passage in %.2fs", len(hyde_passage), timings["hyde_s"])
    logger.debug("HyDE passage: %s", hyde_passage)

    # Stage 2 - Dense retrieval (embeds the HyDE passage)
    t0 = time.perf_counter()
    dense_results = stage_dense(
        hyde_passage,
        resources["collection"],
        resources["embed_model"],
        n_results=pool_size,
    )
    timings["dense_s"] = time.perf_counter() - t0
    logger.info("[2/7] Dense: %d candidates in %.2fs", len(dense_results), timings["dense_s"])

    # Stage 3 - BM25 (uses the original query)
    t0 = time.perf_counter()
    bm25_results = stage_bm25(
        query,
        resources["bm25"],
        resources["chunks"],
        n_results=pool_size,
    )
    timings["bm25_s"] = time.perf_counter() - t0
    logger.info("[3/7] BM25: %d candidates in %.2fs", len(bm25_results), timings["bm25_s"])

    # Stage 4 - RRF fusion
    t0 = time.perf_counter()
    rrf_pool = stage_rrf(dense_results, bm25_results, n_results=pool_size)
    timings["rrf_s"] = time.perf_counter() - t0
    # Overlap = chunks found by BOTH systems. Only these get a combined RRF
    # score; with zero overlap, RRF just interleaves the two lists.
    overlap = len(_chunk_keys(dense_results) & _chunk_keys(bm25_results))
    logger.info("[4/7] RRF: pool of %d (%d found by both dense and BM25) in %.3fs",
                len(rrf_pool), overlap, timings["rrf_s"])
    for c in rrf_pool:
        logger.debug("  RRF    #%-2d %-40s rrf=%.5f", c["rank"], _cite(c), c["rrf_score"])

    # Stage 5 - Cross-encoder reranking (scores every pool candidate)
    t0 = time.perf_counter()
    reranked = stage_rerank(
        query,
        rrf_pool,
        resources["reranker"],
        n_results=top_k_rerank,
    )
    timings["rerank_s"] = time.perf_counter() - t0
    # Snapshot now: deduplication renumbers 'rank' on these same dicts.
    reranked_summary = [_chunk_summary(c, "rerank_score") for c in reranked]
    top_score = reranked[0]["rerank_score"] if reranked else float("nan")
    logger.info("[5/7] Rerank: scored %d pairs, kept %d (top score %.3f) in %.2fs",
                len(rrf_pool), len(reranked), top_score, timings["rerank_s"])
    for c in reranked:
        logger.debug("  Rerank #%-2d %-40s ce=%+.3f", c["rank"], _cite(c), c["rerank_score"])

    # Stage 6 - Page-level deduplication
    t0 = time.perf_counter()
    deduped = stage_dedup(reranked)
    timings["dedup_s"] = time.perf_counter() - t0
    removed = len(reranked) - len(deduped)
    logger.info("[6/7] Dedup: %d unique pages (%d duplicate chunks removed)", len(deduped), removed)

    # Stage 7 - LLM generation
    generation_chunks = deduped[:top_k_generate]
    t0 = time.perf_counter()
    result = stage_generate(query, generation_chunks)
    timings["generate_s"] = time.perf_counter() - t0
    cited = [s["source_num"] for s in result["sources"]]
    logger.info("[7/7] Generate: %d chunks in, cited %s, in %.2fs",
                len(generation_chunks), cited or "none", timings["generate_s"])
    if not cited:
        logger.warning("Answer cites no sources (a refusal, or the model omitted citations).")

    timings["total_s"] = time.perf_counter() - t_pipeline

    llm_usage = {"hyde": hyde["llm"], "generation": result.get("llm")}
    total_tokens = sum((u or {}).get("total_tokens") or 0 for u in llm_usage.values())
    costs = [(u or {}).get("cost_usd") for u in llm_usage.values()]
    total_cost = sum(costs) if all(c is not None for c in costs) else None
    logger.info("Pipeline done in %.2fs: tokens=%d cost_usd=%s trace_id=%s",
                timings["total_s"], total_tokens,
                f"{total_cost:.6f}" if total_cost is not None else "n/a", trace_id or "-")

    result["contexts"]          = [c["text"] for c in generation_chunks]   # Session 10
    result["hyde_passage"]      = hyde_passage
    result["rrf_pool_size"]     = len(rrf_pool)
    result["reranked_count"]    = len(reranked)
    result["final_chunk_count"] = len(deduped)
    result["trace_id"]          = trace_id
    result["debug"] = {
        "trace_id": trace_id,
        "hyde_passage": hyde_passage,
        "counts": {
            "dense": len(dense_results),
            "bm25": len(bm25_results),
            "dense_bm25_overlap": overlap,
            "rrf_pool": len(rrf_pool),
            "reranked": len(reranked),
            "after_dedup": len(deduped),
            "dedup_removed": removed,
            "generation_chunks": len(generation_chunks),
            "cited": len(cited),
        },
        "dense": [_chunk_summary(c, "score") for c in dense_results],
        "bm25": [_chunk_summary(c, "score") for c in bm25_results],
        "rrf_pool": [_chunk_summary(c, "rrf_score") for c in rrf_pool],
        "reranked": reranked_summary,
        "timings_s": {k: round(v, 3) for k, v in timings.items()},
        "llm": llm_usage,
        "total_tokens": total_tokens,
        "total_cost_usd": total_cost,
    }
    return result


# --- small helpers for logs and the debug payload ---------------------------

def _chunk_keys(results: list) -> set:
    """Identity of each chunk: (file, page, chunk index), as used by RRF."""
    return {
        (r["metadata"].get("file_name"), r["metadata"].get("page_number"), r["metadata"].get("chunk_index"))
        for r in results
    }


def _cite(c: dict) -> str:
    """Short label for a chunk, e.g. 'infosys-ar-26.pdf p.80 c2'."""
    m = c["metadata"]
    return f"{m.get('file_name')} p.{m.get('page_number')} c{m.get('chunk_index')}"


def _chunk_summary(c: dict, score_key: str) -> dict:
    """JSON-safe summary of a chunk for the debug payload (no full text)."""
    m = c["metadata"]
    score = c.get(score_key)
    return {
        "rank": c.get("rank"),
        "file_name": m.get("file_name"),
        "page_number": m.get("page_number"),
        "chunk_index": m.get("chunk_index"),
        score_key: round(float(score), 6) if score is not None else None,
    }
