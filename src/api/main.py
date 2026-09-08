"""
FastAPI application for the Deep Research RAG pipeline.

Routes:
  POST /query   — run the full 7-stage pipeline and return a grounded answer
  GET  /health  — pipeline readiness probe
"""

import os
from contextlib import asynccontextmanager
from typing import List, Union

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.pipeline.rag_pipeline import load_resources, run_pipeline

load_dotenv()


class QueryRequest(BaseModel):
    """Incoming request body for POST /query."""
    question: str = Field(..., min_length=1, description="Research question to answer")
    verbose: bool = Field(False, description="Include pipeline debug info in response")


class SourceRef(BaseModel):
    """A single source chunk cited in the answer."""
    source_num: int
    file_name: str
    page_number: Union[int, str]


class QueryResponse(BaseModel):
    """Structured response returned by POST /query."""
    answer: str
    sources: List[SourceRef]
    query: str
    num_chunks: int
    model: str


class HealthResponse(BaseModel):
    """Response returned by GET /health."""
    status: str
    pipeline_loaded: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources once at startup; yield; teardown on shutdown."""
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY not set — check .env")
    app.state.resources = load_resources()
    yield


app = FastAPI(
    title="Deep Research RAG API",
    description=(
        "7-stage RAG pipeline: HyDE → dense retrieval → BM25 → "
        "RRF fusion → cross-encoder rerank → dedup → LLM generation"
    ),
    version="0.1.0",
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
def query(request: QueryRequest) -> QueryResponse:
    """
    Run the full 7-stage RAG pipeline for a research question.

    Pydantic enforces min_length=1 on `question`, so an empty string
    automatically returns HTTP 422 before this function is called.
    Any internal pipeline error is surfaced as HTTP 500 with the
    exception message in `detail`.
    """
    try:
        result = run_pipeline(
            request.question,
            app.state.resources,
            verbose=request.verbose,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return QueryResponse(
        answer=result["answer"],
        sources=[SourceRef(**s) for s in result["sources"]],
        query=result["query"],
        num_chunks=result["num_chunks"],
        model=result["model"],
    )