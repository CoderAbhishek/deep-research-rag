"""
tracing.py - report whether LangSmith tracing is on, without logging secrets.

How tracing works in this project
---------------------------------
Every pipeline stage is decorated with @traceable (from the langsmith SDK).
When tracing is enabled, each decorated call becomes a "run" that the SDK
sends to LangSmith in a background thread. A decorated function called inside
another becomes its child run, so one question produces a tree:

    RAG Pipeline                      (chain run, the trace root)
    |- 1. HyDE Generation
    |    '- groq.chat (hyde)          (llm run: messages, output, tokens)
    |- 2. Dense Retrieval
    |- 3. BM25 Retrieval
    |- 4. RRF Fusion
    |- 5. Cross-Encoder Reranking
    |- 6. Page Deduplication
    '- 7. LLM Generation
         '- groq.chat (generation)   (llm run)

Tracing is controlled only by environment variables, read by the SDK:
    LANGSMITH_TRACING=true    or the older name LANGCHAIN_TRACING_V2=true
    LANGSMITH_API_KEY=...     or LANGCHAIN_API_KEY
    LANGSMITH_PROJECT=...     or LANGCHAIN_PROJECT (default project: "default")

When tracing is off, @traceable simply calls the function: no network
traffic, no errors, same results.
"""

import logging
from typing import Dict, Optional

from langsmith import utils as ls_utils


def tracing_status() -> Dict[str, object]:
    """Return {'enabled', 'api_key_set', 'project'} using the SDK's own checks."""
    enabled = bool(ls_utils.tracing_is_enabled())
    api_key_set = bool(ls_utils.get_env_var("API_KEY", default=None))
    project = ls_utils.get_env_var("PROJECT", default=None) or "default"
    return {"enabled": enabled, "api_key_set": api_key_set, "project": project}


def log_tracing_status(logger: Optional[logging.Logger] = None) -> Dict[str, object]:
    """Log one line at startup saying whether traces will reach LangSmith."""
    logger = logger or logging.getLogger(__name__)
    status = tracing_status()
    if status["enabled"] and status["api_key_set"]:
        logger.info("LangSmith tracing ON: project=%r", status["project"])
    elif status["enabled"]:
        logger.warning("LangSmith tracing is enabled but no API key is set: traces will not upload.")
    else:
        logger.info("LangSmith tracing OFF (set LANGCHAIN_TRACING_V2=true and LANGSMITH_API_KEY to enable).")
    return status
