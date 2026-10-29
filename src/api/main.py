"""
FastAPI application for the Deep Research RAG pipeline.

Routes:
  POST /query   — run the full 7-stage pipeline and return a grounded answer
  GET  /health  — pipeline readiness probe
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from src.observability.logging_config import new_request_id, setup_logging
from src.observability.tracing import log_tracing_status
from src.pipeline.rag_pipeline import load_resources, run_pipeline

load_dotenv()      # first, so LOG_LEVEL and the LangSmith variables are visible
setup_logging()    # configure logging once, before anything logs

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    """Incoming request body for POST /query."""
    question: str = Field(..., min_length=1, description="Research question to answer")
    verbose: bool = Field(False, description="If true, include the pipeline debug payload (stage counts, candidates and scores, timings, token usage)")


class SourceRef(BaseModel):
    """A single source chunk cited in the answer."""
    source_num: int
    file_name: str
    page_number: Union[int, str]


class QueryResponse(BaseModel):
    """Structured response returned by POST /query."""
    answer: str
    sources: List[SourceRef] = Field(description="Sources the answer actually cites (empty if it cites none)")
    retrieved_sources: List[SourceRef] = Field(default_factory=list, description="Every chunk passed to the model")
    query: str
    num_chunks: int
    model: str
    request_id: str = Field(description="ID for this request; appears in every server log line and in the LangSmith trace metadata")
    trace_id: Optional[str] = Field(default=None, description="LangSmith trace ID (null when tracing is off)")
    debug: Optional[Dict[str, Any]] = Field(default=None, description="Pipeline debug payload, included only when verbose=true")


class HealthResponse(BaseModel):
    """Response returned by GET /health."""
    status: str
    pipeline_loaded: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources once at startup; yield; teardown on shutdown."""
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY not set — check .env")
    log_tracing_status(logger)
    t0 = time.perf_counter()
    app.state.resources = load_resources()
    logger.info("API ready in %.1fs", time.perf_counter() - t0)
    yield


app = FastAPI(
    title="Deep Research RAG API",
    description=(
        "7-stage RAG pipeline: HyDE → dense retrieval → BM25 → "
        "RRF fusion → cross-encoder rerank → dedup → LLM generation"
    ),
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    Readiness probe.

    Returns 200 with pipeline_loaded=True once startup has finished loading
    models and the vector store.  Safe to call before startup completes —
    pipeline_loaded will be False until load_resources() returns.
    """
    loaded = hasattr(app.state, "resources") and app.state.resources is not None
    return HealthResponse(status="ok", pipeline_loaded=loaded)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, response: Response) -> QueryResponse:
    """
    Run the full 7-stage RAG pipeline for a research question.

    Pydantic enforces min_length=1 on `question`, so an empty string
    automatically returns HTTP 422 before this function is called.

    Observability:
    - A request ID is created per call. It is returned in the body and in the
      X-Request-ID header, stamped on every log line, and attached to the
      LangSmith trace as metadata.
    - With verbose=true the response includes the pipeline debug payload.
    - On failure the full traceback is logged; the client gets HTTP 500 with
      a generic message and the request ID, never the raw exception text.
    """
    request_id = new_request_id()
    response.headers["X-Request-ID"] = request_id
    logger.info("API request: %d-char question, verbose=%s", len(request.question), request.verbose)
    try:
        result = run_pipeline(
            request.question,
            app.state.resources,
            langsmith_extra={
                "metadata": {"request_id": request_id, "entrypoint": "api"},
                "tags": ["api"],
            },
        )
    except Exception as exc:
        logger.exception("Pipeline failed")
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error. Quote request id {request_id} when reporting it.",
        ) from exc

    return QueryResponse(
        answer=result["answer"],
        sources=[SourceRef(**s) for s in result["sources"]],
        retrieved_sources=[SourceRef(**s) for s in result.get("retrieved_sources", [])],
        query=result["query"],
        num_chunks=result["num_chunks"],
        model=result["model"],
        request_id=request_id,
        trace_id=result.get("trace_id"),
        debug=result.get("debug") if request.verbose else None,
    )
