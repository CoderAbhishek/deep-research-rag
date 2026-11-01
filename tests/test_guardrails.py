"""
Tests for finish-plan step 2 guardrails. No models, no network, no API key.

Run:  python -m tests.test_guardrails
"""

import sys
import types

from src.guardrails.input_checks import MAX_QUESTION_CHARS, InvalidQuestion, validate_question
from src.guardrails.output_checks import extract_figures, unsupported_figures


def _raises(fn, *args):
    try:
        fn(*args)
    except InvalidQuestion as exc:
        return str(exc)
    raise AssertionError("expected InvalidQuestion")


# ---------- input ----------

def test_input_rules():
    assert validate_question("  What was revenue in FY26?  ") == "What was revenue in FY26?"
    assert "enter a question" in _raises(validate_question, "   ")
    assert "enter a question" in _raises(validate_question, None)
    assert str(MAX_QUESTION_CHARS) in _raises(validate_question, "a" * (MAX_QUESTION_CHARS + 1))
    assert "in words" in _raises(validate_question, "??? !!!")
    validate_question("a" * MAX_QUESTION_CHARS)  # exactly at the limit is fine


# ---------- output ----------

CTX = ["Revenue by geography (In ₹ crore) North America 1,00,167 Europe 48,595",
       "Interim dividend ₹23 per share, final dividend ₹25 per share, as of March 31, 2026. EPS 70.87"]


def test_supported_figures_pass_in_any_grouping():
    assert unsupported_figures("North America revenue was ₹1,00,167 crore [SOURCE 1].", CTX) == []
    assert unsupported_figures("North America revenue was ₹100,167 crore.", CTX) == []   # international grouping
    assert unsupported_figures("EPS was ₹70.87 as of March 31, 2026 [SOURCE 2].", CTX) == []


def test_citation_labels_are_not_checked():
    assert extract_figures("₹70.87 [SOURCE 12] 【SOURCE 3】") == ["70.87"]


def test_computed_and_partial_numbers_are_flagged():
    # q13 shape: a correct sum that the document never states.
    assert unsupported_figures("Total dividend ₹48 (₹23 interim + ₹25 final).", CTX) == ["48"]
    # A number that only appears inside a longer one is not support.
    assert unsupported_figures("Revenue was 167 crore.", CTX) == ["167"]
    # Single digits are not checked.
    assert unsupported_figures("Infosys has 5 segments.", CTX) == []


def test_dates_are_not_glued_into_one_number():
    assert extract_figures("as of March 31, 2026") == ["31", "2026"]


# ---------- Groq retries (SDK stubbed) ----------

class RateLimitError(Exception):
    def __init__(self, msg, retry_after=None):
        super().__init__(msg)
        self.response = types.SimpleNamespace(headers={"retry-after": retry_after} if retry_after else {})


class AuthenticationError(Exception):
    pass


def _groq_client():
    if "groq" not in sys.modules:
        sys.modules["groq"] = types.SimpleNamespace(Groq=object)
    if "langsmith" not in sys.modules:
        sys.modules["langsmith"] = types.SimpleNamespace(traceable=lambda *a, **k: (lambda f: f),
                                                         get_current_run_tree=lambda: None)
    from src.llm import groq_client as g
    return g


def _run(g, failures):
    """Make _traced_chat fail with each exception in `failures`, then succeed."""
    calls, sleeps = [], []
    def fake(*args, **kwargs):
        calls.append(1)
        if len(calls) <= len(failures):
            raise failures[len(calls) - 1]
        return {"content": "ok", "finish_reason": "stop", "latency_s": 0.5, "reasoning_chars": 0,
                "cost_usd": 0.0, "usage_metadata": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}}
    g._traced_chat, real_sleep = fake, g.time.sleep
    g.time.sleep = sleeps.append
    try:
        return g.chat([{"role": "user", "content": "q"}], model="m", temperature=0, max_tokens=10,
                      purpose="test"), calls, sleeps
    finally:
        g.time.sleep = real_sleep


def test_minute_limits_are_retried_and_recorded():
    g = _groq_client()
    res, calls, sleeps = _run(g, [RateLimitError("tokens per minute (TPM)", retry_after="3"),
                                  RateLimitError("tokens per minute (TPM)")])
    assert res.text == "ok" and len(calls) == 3
    assert sleeps == [3.0, 2.0]                     # advised wait, then backoff 2 s
    assert res.retries == 2 and res.retry_wait_s == 5.0
    assert res.latency_s == 0.5                     # model time only, waits excluded


def test_no_retry_for_daily_limit_wrong_key_or_long_wait():
    g = _groq_client()
    for exc in (RateLimitError("tokens per day (TPD): Limit 200000"),
                AuthenticationError("invalid api key"),
                RateLimitError("tokens per minute (TPM)", retry_after="120")):
        try:
            _run(g, [exc])
        except type(exc):
            continue
        raise AssertionError(f"expected {type(exc).__name__} to be raised without retry")


def test_gives_up_after_max_retries():
    g = _groq_client()
    try:
        _run(g, [RateLimitError("TPM")] * (g.GROQ_MAX_RETRIES + 1))
    except RateLimitError:
        return
    raise AssertionError("expected the last RateLimitError to be raised")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
