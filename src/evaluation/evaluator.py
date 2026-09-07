"""
Local RAGAS-equivalent evaluation for the Deep Research RAG pipeline.

Computes all four metrics using models already loaded in the pipeline.
Zero external API calls during the scoring phase.

Metric implementations:
  Faithfulness      — cross-encoder: is each answer sentence entailed by context?
  Answer Relevancy  — SentenceTransformer cosine sim: does answer address question?
  Context Precision — cross-encoder: are retrieved chunks relevant to the reference?
  Context Recall    — cross-encoder: does context cover reference answer sentences?
"""

import csv
import json
import re
import time
from typing import Any, Dict, List

import numpy as np

from src.pipeline.rag_pipeline import load_resources, run_pipeline


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> List[str]:
    """Split text into sentences on .!? boundaries."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 8]


# ---------------------------------------------------------------------------
# Metric 1: Faithfulness
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


# ---------------------------------------------------------------------------
# Metric 2: Answer Relevancy
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Metric 3: Context Precision
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Metric 4: Context Recall
# ---------------------------------------------------------------------------

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
# Ground truth
# ---------------------------------------------------------------------------

def load_ground_truth(path: str = "data/ground_truth.json") -> List[Dict[str, str]]:
    """Load hand-written QA pairs from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def collect_pipeline_outputs(
    ground_truth: List[Dict[str, str]],
    resources: Dict[str, Any],
    sleep_seconds: int = 15,
    verbose: bool = False,
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
        print(f"  [{idx}/{len(ground_truth)}] {question[:70]}")
        result = run_pipeline(question, resources, verbose=verbose)
        rows.append({
            "question":  question,
            "answer":    result["answer"],
            "contexts":  result.get("contexts", []),
            "reference": reference,
        })
        if idx < len(ground_truth):
            time.sleep(sleep_seconds)
    return rows


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------

def score_all(rows: List[Dict[str, Any]], resources: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compute all four metrics for every row. No API calls."""
    reranker    = resources["reranker"]
    embed_model = resources["embed_model"]
    scored = []
    for row in rows:
        scored.append({
            "question":          row["question"],
            "answer":            row["answer"],
            "faithfulness":      compute_faithfulness(row["answer"],    row["contexts"], reranker),
            "answer_relevancy":  compute_answer_relevancy(row["question"], row["answer"], embed_model),
            "context_precision": compute_context_precision(row["reference"], row["contexts"], reranker),
            "context_recall":    compute_context_recall(row["reference"],   row["contexts"], reranker),
        })
    return scored


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def _bar(val: float, width: int = 20) -> str:
    if val is None or val != val:
        return "(N/A)"
    filled = max(0, min(width, int(round(val * width))))
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def print_results(scored: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print("PER-QUESTION SCORES")
    print("=" * 80)
    for row in scored:
        q = row["question"]
        print(f"\nQ: {q[:55] + '...' if len(q) > 55 else q}")
        for m in METRICS:
            val = row.get(m)
            if val is not None and val == val:
                print(f"  {m:<22} {val:.3f}  {_bar(val)}")
            else:
                print(f"  {m:<22} N/A")

    print("\n" + "=" * 80)
    print("AGGREGATE SCORES (mean across all questions)")
    print("=" * 80)
    for m in METRICS:
        vals = [r[m] for r in scored if r.get(m) is not None and r.get(m) == r.get(m)]
        if vals:
            mean = sum(vals) / len(vals)
            print(f"  {m:<22} {mean:.3f}  {_bar(mean)}")
        else:
            print(f"  {m:<22} N/A")


def save_csv(scored: List[Dict[str, Any]], path: str = "data/ragas_results.csv") -> None:
    fields = ["question", "answer"] + METRICS
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in scored:
            writer.writerow({k: row.get(k, "") for k in fields})
    print(f"\nResults saved to {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_evaluation(ground_truth_path: str = "data/ground_truth.json") -> None:
    print("Loading pipeline resources...")
    resources = load_resources()

    print(f"\nLoading ground truth from {ground_truth_path}...")
    ground_truth = load_ground_truth(ground_truth_path)
    print(f"Loaded {len(ground_truth)} questions.")

    print(f"\nRunning pipeline on all questions (15s sleep between calls)...")
    rows = collect_pipeline_outputs(ground_truth, resources, sleep_seconds=15, verbose=False)

    print("\nScoring with local metrics (no API calls)...")
    scored = score_all(rows, resources)

    print_results(scored)
    save_csv(scored)