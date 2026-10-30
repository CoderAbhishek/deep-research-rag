"""
groq_client.py - the single place where this project calls the Groq API.

Both LLM calls in the pipeline go through chat():
  - Stage 1, HyDE (src/retrieval/query.py)
  - Stage 7, answer generation (src/generation/generator.py)

Instrumenting the call once, here, gives every LLM call the same observability:
  - a LangSmith run of type "llm" with the messages, the output and token usage
  - one log line with model, latency, tokens, finish_reason and estimated cost
  - a WARNING when the output is empty or was cut off (finish_reason != "stop")
  - an LLMResult the pipeline can return in its debug payload
"""

import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from groq import Groq
from langsmith import traceable

logger = logging.getLogger(__name__)

# Estimated price in USD per 1 million tokens.
# Source: computeprices.com (third-party listing of Groq prices), checked
# 2026-10-29. Groq's own pricing page could not be read at that time, so
# verify on console.groq.com before quoting any cost figure.
# Models not listed here get cost_usd = None (unknown), never a guess.
PRICING_USD_PER_MILLION: Dict[str, Dict[str, float]] = {
    "openai/gpt-oss-20b": {"input": 0.075, "output": 0.30},
}


@dataclass
class LLMResult:
    """Text plus the operational facts about one LLM call."""
    text: str
    model: str
    purpose: str
    finish_reason: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    latency_s: float
    cost_usd: Optional[float]
    reasoning_chars: int
    max_tokens: Optional[int] = None          # the ceiling this call ran with
    reasoning_effort: Optional[str] = None    # "low" / "medium" / "high", or None = model default

    def stats(self) -> Dict[str, Any]:
        """Everything except the text (the text is returned separately)."""
        d = asdict(self)
        d.pop("text")
        return d


def estimate_cost_usd(model: str, prompt_tokens: Optional[int],
                      completion_tokens: Optional[int]) -> Optional[float]:
    """Return estimated cost in USD, or None if the model or token counts are unknown."""
    price = PRICING_USD_PER_MILLION.get(model)
    if price is None or prompt_tokens is None or completion_tokens is None:
        return None
    return (prompt_tokens * price["input"] + completion_tokens * price["output"]) / 1_000_000


_client: Optional[Groq] = None


def _get_client() -> Groq:
    """Create the Groq client once and reuse it (keeps its HTTP connection pool)."""
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found. Add it to your .env file: GROQ_API_KEY=your_key_here")
        _client = Groq(api_key=api_key)
    return _client


@traceable(run_type="llm", name="groq.chat")
def _traced_chat(messages: List[Dict[str, str]], model: str,
                 temperature: float, max_tokens: int,
                 reasoning_effort: Optional[str] = None) -> Dict[str, Any]:
    """
    The actual API call, recorded by LangSmith as an "llm" run.

    The function's arguments become the run's inputs (so the trace shows the
    exact messages sent). The returned dict becomes the run's outputs. The
    'usage_metadata' key is the format LangSmith reads to display token counts
    and cost for an llm run.

    reasoning_effort: for reasoning models (gpt-oss) Groq accepts "low",
    "medium" or "high". groq==0.13.0 has no named argument for it, so it is
    sent in the request body via extra_body. None = the model's default.
    For a reasoning model, max_tokens caps reasoning AND visible output
    together (DR-26).
    """
    start = time.perf_counter()
    response = _get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"reasoning_effort": reasoning_effort} if reasoning_effort else None,
    )
    latency_s = time.perf_counter() - start

    choice = response.choices[0]
    text = choice.message.content or ""          # content can be None
    reasoning = getattr(choice.message, "reasoning", None) or ""  # reasoning models only
    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else None
    completion_tokens = usage.completion_tokens if usage else None
    total_tokens = usage.total_tokens if usage else None
    cost = estimate_cost_usd(model, prompt_tokens, completion_tokens)

    usage_metadata = None
    if usage:
        usage_metadata = {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        if cost is not None:
            price = PRICING_USD_PER_MILLION[model]
            usage_metadata["input_cost"] = prompt_tokens * price["input"] / 1_000_000
            usage_metadata["output_cost"] = completion_tokens * price["output"] / 1_000_000
            usage_metadata["total_cost"] = cost

    return {
        "content": text,
        "finish_reason": choice.finish_reason,
        "latency_s": latency_s,
        "reasoning_chars": len(reasoning),
        "cost_usd": cost,
        "usage_metadata": usage_metadata,
    }


def chat(messages: List[Dict[str, str]], *, model: str, temperature: float,
         max_tokens: int, purpose: str, reasoning_effort: Optional[str] = None) -> LLMResult:
    """
    Call Groq once, trace it, log it, and return an LLMResult.

    purpose: a short label ("hyde", "generation", "rewrite", ...). It names
             the LangSmith run and appears in the log line, so you can tell
             the pipeline's LLM calls apart.
    """
    start = time.perf_counter()
    try:
        out = _traced_chat(
            messages, model, temperature, max_tokens, reasoning_effort,
            # langsmith_extra is consumed by @traceable, not by _traced_chat.
            # ls_provider / ls_model_name are LangSmith's standard metadata
            # keys for identifying the model on an llm run.
            langsmith_extra={
                "name": f"groq.chat ({purpose})",
                "metadata": {"ls_provider": "groq", "ls_model_name": model, "purpose": purpose,
                             "max_tokens": max_tokens, "reasoning_effort": reasoning_effort},
            },
        )
    except Exception:
        logger.error("LLM %s call failed after %.2fs (model=%s)", purpose, time.perf_counter() - start, model)
        raise

    usage = out["usage_metadata"] or {}
    result = LLMResult(
        text=out["content"],
        model=model,
        purpose=purpose,
        finish_reason=out["finish_reason"],
        prompt_tokens=usage.get("input_tokens"),
        completion_tokens=usage.get("output_tokens"),
        total_tokens=usage.get("total_tokens"),
        latency_s=round(out["latency_s"], 3),
        cost_usd=out["cost_usd"],
        reasoning_chars=out["reasoning_chars"],
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
    )

    cost_text = f"{result.cost_usd:.6f}" if result.cost_usd is not None else "n/a"
    logger.info(
        "LLM %s: model=%s effort=%s latency=%.2fs tokens(prompt=%s, completion=%s/%d) finish_reason=%s cost_usd=%s",
        purpose, model, reasoning_effort or "default", result.latency_s, result.prompt_tokens,
        result.completion_tokens, max_tokens, result.finish_reason, cost_text,
    )
    if result.finish_reason != "stop" or not result.text:
        logger.warning(
            "LLM %s output may be truncated or empty: finish_reason=%s, %d answer chars, "
            "%d reasoning chars, max_tokens=%d",
            purpose, result.finish_reason, len(result.text), result.reasoning_chars, max_tokens,
        )
    return result
