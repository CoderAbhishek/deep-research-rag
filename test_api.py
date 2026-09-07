"""
Session 11 — FastAPI layer test.

Run from the project root:
    python test_api.py

Prerequisites:
    pip install fastapi httpx uvicorn
    GROQ_API_KEY set in .env
    ChromaDB populated (run ingest first if not already done)

Tests:
    1. GET /health  → 200, pipeline_loaded=True
    2. POST /query  (valid question) → 200, structured response with sources
    3. POST /query  (empty question) → 422 Unprocessable Entity
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

required = ["GROQ_API_KEY"]
missing = [k for k in required if not os.environ.get(k)]
if missing:
    print(f"ERROR: Missing environment variables: {', '.join(missing)}")
    sys.exit(1)

from fastapi.testclient import TestClient  # noqa: E402  (after env check)
from src.api.main import app              # noqa: E402


def run_tests():
    print("=" * 80)
    print("SESSION 11 — FastAPI LAYER TEST")
    print("=" * 80)
    print("\nStarting app (loading pipeline resources — may take ~30 s)...")

    with TestClient(app) as client:

        # ── Test 1: Health ────────────────────────────────────────────────────
        print("\n[1/3] GET /health")
        r = client.get("/health")
        print(f"  Status : {r.status_code}")
        print(f"  Body   : {r.json()}")
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        body = r.json()
        assert body["status"] == "ok", f"Expected status='ok', got {body['status']}"
        assert body["pipeline_loaded"] is True, "pipeline_loaded should be True after startup"
        print("  PASS ✓")

        # ── Test 2: Valid query ────────────────────────────────────────────────
        print("\n[2/3] POST /query — valid question")
        r = client.post("/query", json={"question": "What is Infosys Cobalt?"})
        print(f"  Status : {r.status_code}")
        data = r.json()
        print(f"  Answer (first 120 chars): {data.get('answer', '')[:120]}")
        print(f"  Sources: {data.get('sources', [])}")
        print(f"  Model  : {data.get('model', '')}")
        print(f"  Chunks : {data.get('num_chunks', 0)}")
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        assert "answer" in data, "Response missing 'answer' key"
        assert len(data["sources"]) > 0, "Expected at least one source"
        print("  PASS ✓")

        # ── Test 3: Empty question → 422 ──────────────────────────────────────
        print("\n[3/3] POST /query — empty question (expect 422)")
        r = client.post("/query", json={"question": ""})
        print(f"  Status : {r.status_code}")
        print(f"  Body   : {r.json()}")
        assert r.status_code == 422, f"Expected 422, got {r.status_code}"
        print("  PASS ✓")

    print("\n" + "=" * 80)
    print("All 3 tests passed.")
    print("=" * 80)
    print("\nTo run the live server:")
    print("  uvicorn src.api.main:app --reload --port 8000")
    print("\nAuto-generated docs at: http://localhost:8000/docs")


if __name__ == "__main__":
    run_tests()