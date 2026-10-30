"""
logging_config.py - configure logging once for every entry point.

Two rules this module follows:

1. Libraries emit, applications configure.
   Every module in src/ only creates a logger with logging.getLogger(__name__)
   and calls logger.info(...), logger.debug(...), etc. It never decides where
   logs go or what they look like. Entry points (app.py, src/api/main.py and
   the scripts) call setup_logging() once at startup to make that decision.

2. Every log line carries a request ID.
   Each HTTP request (or evaluation question) gets a short ID. It is stored in
   a context variable, stamped onto every log record by RequestIdFilter,
   returned to the client, and attached to the LangSmith trace as metadata.
   One ID therefore links the log lines, the API response and the trace for
   the same request.

Output format:
    14:02:11 INFO    [3f9c1a2b7d4e] src.pipeline.rag_pipeline: [5/7] Rerank: ...
    time     level   request ID     logger (module) name       message
Lines logged outside a request (startup, shutdown) show [-].
"""

import contextvars
import logging
import os
import uuid
from typing import Optional

# A ContextVar holds a separate value per thread / async task. FastAPI runs each
# sync request handler in its own worker-thread context, so concurrent requests
# never see each other's ID.
_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

LOG_FORMAT = "%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s: %(message)s"
DATE_FORMAT = "%H:%M:%S"

# Third-party libraries that log a lot at DEBUG. Kept at WARNING so that
# LOG_LEVEL=DEBUG shows *our* detail, not HTTP connection chatter.
_NOISY_LOGGERS = (
    "httpx", "httpcore", "urllib3", "groq", "chromadb", "sentence_transformers",
    "huggingface_hub", "filelock", "langsmith", "asyncio", "multipart", "python_multipart",
)

_configured = False


def new_request_id() -> str:
    """Create a 12-character request ID and make it current for this context."""
    request_id = uuid.uuid4().hex[:12]
    _request_id.set(request_id)
    return request_id


def get_request_id() -> str:
    """Return the current request ID, or '-' outside a request."""
    return _request_id.get()


class RequestIdFilter(logging.Filter):
    """Adds the current request ID to every log record as record.request_id."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True  # never drops a record; it only adds a field


def setup_logging(level: Optional[str] = None) -> None:
    """
    Configure the root logger. Safe to call more than once (later calls do nothing).

    Level comes from the `level` argument, else the LOG_LEVEL environment
    variable, else INFO. Call it AFTER load_dotenv() so LOG_LEVEL from .env
    is visible.
    """
    global _configured
    if _configured:
        return

    level_name = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    numeric_level = getattr(logging, level_name, None)
    invalid_level = not isinstance(numeric_level, int)
    if invalid_level:
        numeric_level = logging.INFO

    handler = logging.StreamHandler()  # writes to stderr, i.e. the terminal
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    # The filter sits on the HANDLER, not on a logger: records from every
    # module propagate up to the root logger's handlers, so one filter here
    # stamps the request ID on all of them.
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    root.setLevel(numeric_level)
    root.handlers[:] = [handler]  # replace, so re-imports never print each line twice

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _configured = True
    if invalid_level:
        logging.getLogger(__name__).warning("Unknown LOG_LEVEL %r; using INFO.", level_name)
