"""
rate_limits.py - classify provider errors without importing the provider SDK.

Kept separate so evaluation tools (rescore, compare) can use it without the
groq package. Classification is by exception class NAME, which matches the
groq SDK's classes (RateLimitError, APITimeoutError, APIConnectionError,
InternalServerError).
"""

from typing import Optional

RETRYABLE = {"RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError"}


def is_daily_rate_limit(exc: BaseException) -> bool:
    """A 429 on a per-DAY quota (Groq: TPD / RPD). Waiting minutes will not help; fail fast."""
    text = str(exc).lower()
    return type(exc).__name__ == "RateLimitError" and ("per day" in text or "tpd" in text or "rpd" in text)


def is_retryable(exc: BaseException) -> bool:
    return type(exc).__name__ in RETRYABLE and not is_daily_rate_limit(exc)


def retry_after_seconds(exc: BaseException) -> Optional[float]:
    """The provider's Retry-After header in seconds, if the error carries one."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    try:
        value = headers.get("retry-after")
        return float(value) if value is not None else None
    except (TypeError, ValueError, AttributeError):
        return None
