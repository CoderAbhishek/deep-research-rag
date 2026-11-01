"""
Tests for two run-safety guards added with change F. No models, no network.

Run:  python -m tests.test_eval_guards
"""

import sys
import types

from src.evaluation.eval_v2 import is_daily_rate_limit


class RateLimitError(Exception):
    """Same class name as groq.RateLimitError; the check uses the name, not an import."""


def test_daily_limit_stops_the_run_minute_limit_does_not():
    tpd = RateLimitError("Error code: 429 - Rate limit reached ... on tokens per day (TPD): Limit 200000")
    tpm = RateLimitError("Error code: 429 - Rate limit reached ... on tokens per minute (TPM): Limit 8000")
    assert is_daily_rate_limit(tpd) is True
    assert is_daily_rate_limit(tpm) is False
    assert is_daily_rate_limit(ValueError("tokens per day")) is False


def test_trace_inputs_drop_resources():
    # Import rag_pipeline with its heavy dependencies stubbed, then check the hook.
    for name, attrs in {"chromadb": {}, "sentence_transformers": {"SentenceTransformer": object},
                        "langsmith": {"traceable": lambda *a, **k: (lambda f: f), "get_current_run_tree": lambda: None},
                        "src.retrieval.sparse": {"build_bm25_index": None, "search_bm25": None},
                        "src.retrieval.dense": {"search_dense": None},
                        "src.retrieval.hybrid": {"reciprocal_rank_fusion": None},
                        "src.retrieval.query": {"generate_hyde_result": None},
                        "src.retrieval.reranker": {"load_reranker": None, "rerank": None, "deduplicate_by_page": None},
                        "src.generation.generator": {"generate_answer": None}}.items():
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__dict__.update(attrs)
            sys.modules[name] = mod
    from src.pipeline.rag_pipeline import _drop_inputs
    import json
    inputs = {"query": "q", "resources": {"page_index": {("a.pdf", 1): []}}, "pool_size": 20}
    kept = _drop_inputs("resources")(inputs)
    assert kept == {"query": "q", "pool_size": 20}
    json.dumps(kept)                      # serialisable now
    try:
        json.dumps(inputs)
    except TypeError:
        pass                              # the original failure: tuple keys
    else:
        raise AssertionError("expected the unfiltered inputs to fail JSON serialisation")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
