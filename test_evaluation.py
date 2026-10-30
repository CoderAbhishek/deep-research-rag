"""
LEGACY evaluation runner (local RAGAS-style metrics, 13-question set).
Use the v2 evaluation instead:  python -m src.evaluation.eval_v2

Evaluation runner (local RAGAS-style metrics).

Run from the project root:
    python test_evaluation.py

Prerequisites:
    GROQ_API_KEY set in .env (the pipeline calls Groq for HyDE and generation)
    ChromaDB populated at ./chroma_db
    The `ragas` library is NOT required: metrics are implemented locally in
    src/evaluation/evaluator.py.

What this script does:
    1. Loads pipeline resources once (warm startup)
    2. Loads the QA pairs from data/ground_truth.json
    3. Runs the full RAG pipeline on every question (15 s pause between calls)
    4. Scores the results locally (no API calls during scoring):
           faithfulness, answer_relevancy, context_precision, context_recall
    5. Prints a per-question breakdown and aggregate scores
    6. Saves results to data/ragas_results.csv

Expected runtime: at least 3 minutes for 13 questions (12 pauses of 15 s
plus 13 pipeline calls).

Known limitation: see the note at the top of src/evaluation/evaluator.py.
"""

import os
import sys
from dotenv import load_dotenv

# ── Load environment variables ──────────────────────────────────────────────
load_dotenv()

required = ["GROQ_API_KEY"]
missing  = [k for k in required if not os.environ.get(k)]
if missing:
    print(f"ERROR: Missing environment variables: {', '.join(missing)}")
    print("Check your .env file.")
    sys.exit(1)

# Set LangSmith tracing env vars
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]    = "deep-research-rag"

# ── Run evaluation ──────────────────────────────────────────────────────────
from src.evaluation.evaluator import run_evaluation

if __name__ == "__main__":
    print("=" * 80)
    print("SESSION 10 — RAGAS EVALUATION")
    print("=" * 80)

    run_evaluation(ground_truth_path="data/ground_truth.json")

    print("\nDone.")