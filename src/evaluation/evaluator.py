"""
Local RAGAS-equivalent evaluation for the Deep Research RAG pipeline.

Computes all four RAGAS metrics plus MRR@K and Hit Rate@K using models
already loaded in the pipeline. Zero external API calls during the
scoring phase.

Metric implementations:
  Faithfulness      — cross-encoder: is each answer sentence entailed by context?
  Answer Relevancy  — SentenceTransformer cosine sim: does answer address question?
  Context Precision — cross-encoder: are retrieved chunks relevant to the reference?
  Context Recall    — cross-encoder: does context cover reference answer sentences?
  MRR@K             — 1/rank of first relevant chunk in top-K (soft: CE > threshold)
  Hit Rate@K        — 1 if any top-K chunk is relevant, else 0 (averaged = coverage)

Cost estimation:
  estimate_cost(model, prompt_tokens, completion_tokens) returns estimated
  USD cost using Groq's published pricing. Update GROQ_PRICING when rates change.
"""

import csv
import json
import logging
import os
import re
import time
from typing import Any, Dict, List

import numpy as np

from src.pipeline.rag_pipeline import load_resources, run_pipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Groq pricing (USD per 1M tokens).  Update when rates change.
# Source: https://console.groq.com/docs/openai
# ---------------------------------------------------------------------------
GROQ_PRICING: Dict[str, Dict[str, float]] = {
    "groq/compound-mini":   {"input": 0.40,  "output": 0.80},
    "compound-beta-mini":   {"input": 0.40,  "output": 0.80},
    "openai/gpt-oss-20b":   {"input": 0.90,  "output": 1.80},
    "openai/gpt-oss-120b":  {"input": 3.00,  "output": 6.00},
}


def setup_logging() -> None:
    """
    Configure the root logger from the LOG_LEVEL environment variable.

    Call once at application entry-point (run_evaluation).
    All module-level loggers inherit this configuration automatically.

    LOG_LEVEL=DEBUG  — per-chunk details at each pipeline stage
    LOG_LEVEL=INFO   — stage-level progress only (default)
    LOG_LEVEL=WARNING — errors and warnings only
    """
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s  %(name)-35s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> List[str]:
    """Split text into sentences on .!? boundaries."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 8]


# ---------------------------------------------------------------------------
# Core RAGAS metrics
# ---------------------------------------------------------------------------

def compute_faithfulness(
    answer: str,
    contexts: List[str],
    reranker,
    threshold: float = -1.0,
) -> float:
    """
    Faithfulness = fraction of answer sentences entailed by at least one context chunk.

    For each sentence in the generated answer, the cross-encoder scores it against
    every retrieved context chunk.  If the max score exceeds `threshold`, the sentence
    is considered supported.  Faithfulness = supported / total sentences.

    threshold=-1.0 is permissive (below-average relevance is still accepted).
    Tighten to 0.0 for stricter scoring.
    """
    if not answer or not contexts:
        return float("nan")
    sentences = split_sentences(answer)
    if not sentences:
        return float("nan")
    supported = 0
    for sentence in sentences:
        pairs  = [(sentence, ctx) for ctx in contexts]
        scores = reranker.predict(pairs)
        if float(max(scores)) >= threshold:
            supported += 1
    return supported / len(sentences)


def compute_answer_relevancy(
    question: str,
    answer: str,
    embed_model,
) -> float:
    """
    Answer Relevancy = cosine similarity between the question embedding and the
    answer embedding.

    Higher = the answer is topically aligned with the question.
    Uses the same all-MiniLM-L6-v2 model as the retrieval pipeline.
    L2-normalised embeddings → dot product = cosine similarity.
    """
    if not answer or not question:
        return float("nan")
    q_emb = embed_model.encode(question, normalize_embeddings=True)
    a_emb = embed_model.encode(answer,   normalize_embeddings=True)
    return float(np.dot(q_emb, a_emb))


def compute_context_precision(
    reference: str,
    contexts: List[str],
    reranker,
    threshold: float = -1.0,
) -> float:
    """
    Context Precision = weighted precision of relevant chunks in the ranked list.

    Each retrieved chunk is scored against the reference answer by the cross-encoder.
    A chunk is "relevant" if its score >= threshold.  The weighted precision formula
    rewards pipelines that put relevant chunks at high ranks.

    weighted_precision = sum(Precision@k * rel(k)) / total_relevant_chunks
    """
    if not reference or not contexts:
        return float("nan")
    pairs    = [(reference, ctx) for ctx in contexts]
    scores   = reranker.predict(pairs)
    relevant = [1 if float(s) >= threshold else 0 for s in scores]
    total    = sum(relevant)
    if total == 0:
        return 0.0
    weighted_sum    = 0.0
    running_relevant = 0
    for k, rel in enumerate(relevant, start=1):
        if rel:
            running_relevant += 1
            weighted_sum     += running_relevant / k
    return weighted_sum / total


def compute_context_recall(
    reference: str,
    contexts: List[str],
    reranker,
    threshold: float = -1.0,
) -> float:
    """
    Context Recall = fraction of reference answer sentences attributable to context.

    Each sentence of the human-written reference is scored against every retrieved
    chunk.  If the max cross-encoder score >= threshold, the sentence is considered
    covered by the retrieved context.
    Recall = covered / total reference sentences.
    """
    if not reference or not contexts:
        return float("nan")
    sentences = split_sentences(reference)
    if not sentences:
        return float("nan")
    supported = 0
    for sentence in sentences:
        pairs  = [(sentence, ctx) for ctx in contexts]
        scores = reranker.predict(pairs)
        if float(max(scores)) >= threshold:
            supported += 1
    return supported / len(sentences)


# ---------------------------------------------------------------------------
# Retrieval quality metrics: MRR@K and Hit Rate@K
# ---------------------------------------------------------------------------

def compute_mrr_at_k(
    contexts: List[str],
    reference: str,
    reranker,
    k: int = 5,
    threshold: float = -1.0,
) -> float:
    """
    MRR@K — Mean Reciprocal Rank at K.

    Returns 1/rank of the FIRST relevant context in the top-K list.
    A chunk is "relevant" if its cross-encoder score against the reference
    answer exceeds `threshold`.
    Returns 0.0 if no relevant context is found in top-K.

    Note: True MRR@K requires chunk-level annotations (which specific chunk
    IDs answer each question). This implementation uses the cross-encoder
    score against the reference answer as a soft relevance signal — a
    principled proxy given no chunk-level labels exist in our ground truth.

    Args:
        contexts:  Ordered list of retrieved context strings (top-K first).
        reference: The human-written reference answer for this question.
        reranker:  Loaded CrossEncoder instance.
        k:         Maximum rank to consider. Default 5.
        threshold: Minimum CE score to count as relevant. Default -1.0.

    Returns:
        float in [0, 1]. Higher is better.
    """
    top_k = contexts[:k]
    if not top_k or not reference:
        return 0.0
    pairs  = [(reference, ctx) for ctx in top_k]
    scores = reranker.predict(pairs)
    for rank, score in enumerate(scores, start=1):
        if float(score) >= threshold:
            return 1.0 / rank
    return 0.0


def compute_hit_rate_at_k(
    contexts: List[str],
    reference: str,
    reranker,
    k: int = 5,
    threshold: float = -1.0,
) -> float:
    """
    Hit Rate@K — 1 if at least one top-K context is relevant, else 0.

    Averaged over all questions, this is retrieval coverage:
    "For what fraction of questions did the pipeline retrieve at least
    one relevant chunk in the top K?"

    A chunk is "relevant" if its cross-encoder score against the reference
    answer exceeds `threshold`.

    Args:
        contexts:  Ordered list of retrieved context strings (top-K first).
        reference: The human-written reference answer for this question.
        reranker:  Loaded CrossEncoder instance.
        k:         Maximum rank to consider. Default 5.
        threshold: Minimum CE score to count as relevant. Default -1.0.

    Returns:
        1.0 or 0.0.
    """
    top_k = contexts[:k]
    if not top_k or not reference:
        return 0.0
    pairs  = [(reference, ctx) for ctx in top_k]
    scores = reranker.predict(pairs)
    return 1.0 if float(max(scores)) >= threshold else 0.0


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Estimate the cost in USD for a single Groq LLM call.

    Uses GROQ_PRICING constants defined at module level.
    Returns 0.0 for unknown models (with a WARNING log).

    Args:
        model:             Groq model identifier (e.g. "groq/compound-mini").
        prompt_tokens:     Input tokens consumed.
        completion_tokens: Output tokens generated.

    Returns:
        Estimated cost in USD.
    """
    pricing = GROQ_PRICING.get(model)
    if pricing is None:
        logger.warning(
            "No pricing data for model '%s'. Update GROQ_PRICING in evaluator.py.", model
        )
        return 0.0
    return (
        prompt_tokens     * pricing["input"]  +
        completion_tokens * pricing["output"]
    ) / 1_000_000


# ---------------------------------------------------------------------------
# Ground truth + pipeline runner
# ---------------------------------------------------------------------------

def load_ground_truth(path: str = "data/ground_truth.json") -> List[Dict[str, str]]:
    """Load hand-written QA pairs from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_pipeline_outputs(
    ground_truth: List[Dict[str, str]],
    resources: Dict[str, Any],
    sleep_seconds: int = 15,
) -> List[Dict[str, Any]]:
    """
    Run every question through the live RAG pipeline.

    sleep_seconds between calls avoids hitting Groq's TPM limit on the
    generation model (compound-beta-mini has 8,000 TPM).
    """
    rows = []
    for idx, item in enumerate(ground_truth, start=1):
        question  = item["question"]
        reference = item["reference"]
        logger.info("  [%d/%d] %s", idx, len(ground_truth), question[:70])
        result = run_pipeline(question, resources)
        rows.append({
            "question":  question,
            "answer":    result["answer"],
            "contexts":  result.get("contexts", []),
            "reference": reference,
            "timings":   result.get("timings", {}),
            "model":     result.get("model", "unknown"),
        })
        if idx < len(ground_truth):
            time.sleep(sleep_seconds)
    return rows


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_all(rows: List[Dict[str, Any]], resources: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Compute all six metrics for every row. No API calls.

    Metrics computed:
      - faithfulness
      - answer_relevancy
      - context_precision
      - context_recall
      - mrr_at_5
      - hit_rate_at_5
    """
    reranker    = resources["reranker"]
    embed_model = resources["embed_model"]
    scored = []
    for row in rows:
        scored.append({
            "question":          row["question"],
            "answer":            row["answer"],
            "faithfulness":      compute_faithfulness(
                                     row["answer"], row["contexts"], reranker),
            "answer_relevancy":  compute_answer_relevancy(
                                     row["question"], row["answer"], embed_model),
            "context_precision": compute_context_precision(
                                     row["reference"], row["contexts"], reranker),
            "context_recall":    compute_context_recall(
                                     row["reference"], row["contexts"], reranker),
            "mrr_at_5":          compute_mrr_at_k(
                                     row["contexts"], row["reference"], reranker, k=5),
            "hit_rate_at_5":     compute_hit_rate_at_k(
                                     row["contexts"], row["reference"], reranker, k=5),
            "timings":           row.get("timings", {}),
            "model":             row.get("model", "unknown"),
        })
    return scored


# ---------------------------------------------------------------------------
# Results display and export
# ---------------------------------------------------------------------------

METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "mrr_at_5",
    "hit_rate_at_5",
]


def _bar(val: float, width: int = 20) -> str:
    if val is None or val != val:
        return "(N/A)"
    filled = max(0, min(width, int(round(val * width))))
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def log_results(scored: List[Dict[str, Any]]) -> None:
    """Log per-question and aggregate scores at INFO level."""
    logger.info("=" * 80)
    logger.info("PER-QUESTION SCORES")
    logger.info("=" * 80)
    for row in scored:
        q = row["question"]
        logger.info("Q: %s", q[:55] + "..." if len(q) > 55 else q)
        for m in METRICS:
            val = row.get(m)
            if val is not None and val == val:
                logger.info("  %-22s %.3f  %s", m, val, _bar(val))
            else:
                logger.info("  %-22s N/A", m)

    logger.info("=" * 80)
    logger.info("AGGREGATE SCORES (mean across all questions)")
    logger.info("=" * 80)
    for m in METRICS:
        vals = [r[m] for r in scored if r.get(m) is not None and r.get(m) == r.get(m)]
        if vals:
            mean = sum(vals) / len(vals)
            logger.info("  %-22s %.3f  %s", m, mean, _bar(mean))
        else:
            logger.info("  %-22s N/A", m)

    # Per-stage latency summary (mean across all questions)
    timing_keys = ["hyde_s", "dense_s", "bm25_s", "rrf_s", "rerank_s", "dedup_s", "generate_s", "total_s"]
    timing_rows = [r["timings"] for r in scored if r.get("timings")]
    if timing_rows:
        logger.info("=" * 80)
        logger.info("MEAN PER-STAGE LATENCY (seconds)")
        logger.info("=" * 80)
        for key in timing_keys:
            vals = [t[key] for t in timing_rows if key in t]
            if vals:
                logger.info("  %-22s %.3fs", key, sum(vals) / len(vals))


# Backward-compatible alias for existing code that calls print_results()
print_results = log_results


def save_csv(scored: List[Dict[str, Any]], path: str = "data/ragas_results.csv") -> None:
    fields = ["question", "answer"] + METRICS + ["total_s", "model"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in scored:
            writer.writerow({
                **{k: row.get(k, "") for k in fields if k not in ("total_s",)},
                "total_s": row.get("timings", {}).get("total_s", ""),
            })
    logger.info("Results saved to %s", path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_evaluation(ground_truth_path: str = "data/ground_truth.json") -> None:
    setup_logging()

    logger.info("Loading pipeline resources...")
    resources = load_resources()

    logger.info("Loading ground truth from %s...", ground_truth_path)
    ground_truth = load_ground_truth(ground_truth_path)
    logger.info("Loaded %d questions.", len(ground_truth))

    logger.info("Running pipeline on all questions (15s sleep between calls)...")
    rows = collect_pipeline_outputs(ground_truth, resources, sleep_seconds=15)

    logger.info("Scoring with local metrics (no API calls)...")
    scored = score_all(rows, resources)

    log_results(scored)
    save_csv(scored)
