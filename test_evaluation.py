"""
RAGAS evaluation test.

Run from the project root:
    python test_evaluation.py

Prerequisites:
    pip install ragas==0.2.14 datasets langchain-openai
    GROQ_API_KEY set in .env
    ChromaDB populated (run test_pipeline.py from Session 5 first if needed)

What this script does:
    1. Loads pipeline resources once (warm startup)
    2. Loads 13 hand-written QA triples from data/ground_truth.json
    3. Runs the full RAG pipeline on every question
    4. Scores the results with four RAGAS metrics:
           faithfulness, answer_relevancy, context_precision, context_recall
    5. Prints a per-question breakdown and aggregate scores
    6. Saves full results to data/ragas_results.csv

Expected runtime: ~5–8 minutes (13 pipeline calls + RAGAS LLM critic calls)
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