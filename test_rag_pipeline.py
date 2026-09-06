"""
test_rag_pipeline.py

End-to-end test of the full RAG pipeline:
  query → HyDE → dense + BM25 → RRF → cross-encoder → dedup → LLM answer

Run from project root with .venv activated:
    python test_rag_pipeline.py

Requires:
    GROQ_API_KEY in .env
    LANGSMITH_API_KEY in .env  (set LANGCHAIN_TRACING_V2=true to enable)
    ChromaDB at ./chroma_db populated from the ingestion pipeline
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

# Enable LangSmith tracing.
# LANGCHAIN_TRACING_V2=true tells the LangSmith SDK to send traces.
# If LANGSMITH_API_KEY is absent, tracing silently does nothing —
# the pipeline still runs, just without sending traces.
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT",    "deep-research-rag")

from src.pipeline.rag_pipeline import load_resources, run_pipeline

# ─── Test queries ─────────────────────────────────────────────────────────────
# Using two queries:
# 1. The primary test query from Sessions 5–8 (known correct answer: Page 80)
# 2. A second query to test cross-document generalisability

QUERIES = [
    "What is the revenue breakdown by geography for Infosys?",
    "What was Infosys's employee headcount at the end of FY26?",
]

# ─── Load resources once ──────────────────────────────────────────────────────
print("=" * 60)
print("Loading pipeline resources (one-time startup)...")
print("=" * 60)
resources = load_resources()
print("\nAll resources loaded. Running queries...\n")

# ─── Run pipeline for each query ─────────────────────────────────────────────
for query in QUERIES:
    result = run_pipeline(query, resources, verbose=True)
    print("\n" + "=" * 60 + "\n")