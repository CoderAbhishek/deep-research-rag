# Deep Research RAG

A Retrieval-Augmented Generation (RAG) system for dense business and research document corpora. Designed for research analysts and strategy consultants who need evidence-grounded, cited answers across multi-document collections — annual reports, DRHPs, industry reports, expert transcripts.

**Current corpus:** four Infosys FY2025-26 documents (integrated annual report, AGM transcript, additional information, CSR projects): 444 pages of text, 3,439 chunks.

**Live demo:** currently offline. Run it locally with the Quickstart below.

---

## Problem

Analysts spend hours manually reading documents to locate specific data points, cross-reference claims across sources, and build evidence-grounded views. Keyword search (Ctrl+F) is shallow. A basic chatbot is unreliable — it hallucinates and cannot cite sources.

This system solves that with a 7-stage pipeline that retrieves the most relevant context before generating an answer. The model is instructed to cite each claim to a source document and page, and the response lists only the sources the answer actually cites.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│  Stage 1: HyDE                      │
│  Generate a hypothetical answer     │
│  to improve dense retrieval recall  │
└───────────────┬─────────────────────┘
                │
    ┌───────────┴───────────┐
    ▼                       ▼
┌──────────────┐    ┌──────────────────┐
│  Stage 2     │    │  Stage 3         │
│  Dense       │    │  BM25 Sparse     │
│  Retrieval   │    │  Retrieval       │
│  (ChromaDB + │    │  (rank-bm25)     │
│  MiniLM-L6)  │    │                  │
└──────┬───────┘    └────────┬─────────┘
       │                     │
       └──────────┬──────────┘
                  ▼
┌─────────────────────────────────────┐
│  Stage 4: RRF Fusion                │
│  Reciprocal Rank Fusion merges      │
│  dense and sparse ranked lists      │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  Stage 5: Cross-Encoder Reranking   │
│  ms-marco-MiniLM-L-6-v2 rescores    │
│  (query, chunk) pairs directly      │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  Stage 6: Page-level deduplication  │
│  Keep the highest-scoring chunk     │
│  per page before generation         │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  Stage 7: LLM Generation            │
│  Groq openai/gpt-oss-20b            │
│  Grounded answer + inline citations │
└───────────────┬─────────────────────┘
                │
                ▼
     Answer + Source Citations
     (file name + page number)
```

---

## Tech Stack

| Component | Library / Service |
|---|---|
| API framework | FastAPI + Uvicorn |
| Vector store | ChromaDB |
| Dense embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Sparse retrieval | rank-bm25 |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM (HyDE + generation) | Groq API (openai/gpt-oss-20b) |
| Containerisation | Docker + Docker Compose |
| Deployment | Docker image (previously deployed on Railway; currently offline) |
| Evaluation | Custom RAGAS-style metrics (not the `ragas` library); being rebuilt, see below |
| Observability | Python `logging` with request IDs, LangSmith tracing, per-stage timings, token and cost logging |

---

## Evaluation

**Status: being rebuilt.**

The first evaluation used 13 hand-written questions and locally implemented RAGAS-style metrics (faithfulness, answer relevancy, context precision, context recall). A later audit found it could not support conclusions about quality:

- Most reference answers refer to FY2023, while the corpus covers FY2025-26, and some reference figures do not appear in the documents at all.
- Context precision and recall compared each chunk with the reference answer using a cross-encoder and a lenient threshold. They measured topical similarity, not whether the answer was present, and scored perfect recall even on questions the system declined to answer.
- Faithfulness scored 0 on the 8 questions where the system declined to answer, which explains the low average (0.385).

**Evaluation v2 (in progress).** `data/eval/questions_v2.json` holds 41 seed questions written from the FY2025-26 documents. Every reference figure was checked against the indexed text, and each question records:

- the pages that contain the answer (`relevant_pages`; any one of them counts),
- strings the answer must contain (`must_include`, with alternatives),
- what the system should do (`answer`, `refuse` or `answer_or_refuse`),
- its type (numbers, percentages, tables, comparisons, definitions, people, policy, cross-document conflicts, unanswerable, false premise, prompt injection, off-topic) and phrasing (document vocabulary, paraphrase, abbreviation, conversational).

The set is split 24 dev / 17 test (stratified, fixed seed) before any results exist. Error analysis and tuning use the dev split only; the test split is run only to report numbers.

```bash
python -m src.evaluation.eval_v2                 # dev split
python -m src.evaluation.eval_v2 --limit 3       # quick smoke run
python -m tests.test_metrics_v2                  # metric unit tests, no API key needed
```

Comparing runs and fixing labels (no API calls needed):

```bash
python -m src.evaluation.rescore data/eval/runs/<run_id>          # re-score a saved run with the current labels
python -m src.evaluation.compare_runs --baseline RUN1 RUN2 --candidate RUN3 RUN4 --name change-a \
    --regression data/eval/regression_ids.json                    # per-question fixed / regressed report
```

Every change is run twice and compared question by question against two baseline runs: between two identical baseline runs, 7 of 24 questions flipped pass / fail, so a single before / after run cannot show whether a change helped.

Each run writes `data/eval/runs/<run_id>/results.csv` (one row per question: answer checks, whether the relevant page was found at each stage, where it was lost, citations, latency, tokens, cost, trace ID, and empty columns for manual review) and `summary.json` (configuration and aggregate metrics). Metrics are deterministic: hit@k and MRR@k from page labels, answer correctness from `must_include`, correct and false refusal rates, citation precision. Baseline results will be published here after the first dev run and error analysis; the set will grow to 60-80 questions based on the failures observed.

## Known Limitations

- **Tables and charts:** PDF text extraction flattens tables, so column headings can end up separated from their numbers. Numeric questions over tables are the weakest area.
- **Reranker domain:** the cross-encoder (`ms-marco-MiniLM-L-6-v2`) was trained on web-search data, not financial reports, and can over-rank pages dense with matching terms (for example, a page listing office locations for a geography question).
- **Output checks:** grounding and citation rules are enforced through the system prompt. Answers are not yet validated against the source text in code.
- **Access:** no authentication or rate limiting. This is a portfolio project, not a hardened service.

---

## Observability

**Logging.** Every module logs through Python's `logging`; entry points call `setup_logging()` once at startup. Set `LOG_LEVEL` in `.env` (`INFO` by default; `DEBUG` adds the HyDE passage and every candidate chunk). Every line carries the request ID:

```
14:02:11 INFO    [3f9c1a2b7d4e] src.pipeline.rag_pipeline: [5/7] Rerank: scored 20 pairs, kept 5 (top score 2.702) in 0.84s
```

Each query logs one line per stage (what it produced and how long it took), one line per LLM call (model, latency, prompt and completion tokens, `finish_reason`, estimated cost), a warning if an answer cites no sources or an LLM output looks truncated, and a final summary with total time, tokens and the LangSmith trace ID.

**Tracing (LangSmith).** Set `LANGCHAIN_TRACING_V2=true`, `LANGSMITH_API_KEY` and `LANGCHAIN_PROJECT` in `.env`. The startup log says whether tracing is on. Each question produces one trace:

```
RAG Pipeline
├── 1. HyDE Generation
│   └── groq.chat (hyde)          LLM run: messages, output, tokens
├── 2. Dense Retrieval
├── 3. BM25 Retrieval
├── 4. RRF Fusion
├── 5. Cross-Encoder Reranking
├── 6. Page Deduplication
└── 7. LLM Generation
    └── groq.chat (generation)    LLM run: messages, output, tokens
```

Each trace carries `request_id` and `entrypoint` (`api`, `demo-ui` or `evaluation`) metadata and a matching tag, so a log line, an API response and a trace can be matched by the same ID.

**Debug payload.** `POST /query` with `"verbose": true` returns a `debug` object: stage counts, the RRF candidate pool, the reranked top 5 with cross-encoder scores, per-stage timings, and LLM usage and estimated cost.

## Project Structure

```
deep-research-rag/
├── app.py               # Portfolio demo UI (FastAPI + dark-mode HTML, port 7860)
├── src/
│   ├── ingestion/       # loader.py (PDF parsing), chunker.py, embedder.py (ChromaDB ingest)
│   ├── retrieval/       # dense.py, sparse.py (BM25), hybrid.py (RRF), query.py (HyDE), reranker.py
│   ├── generation/      # generator.py (grounded answer + citation parsing)
│   ├── pipeline/        # rag_pipeline.py (7-stage orchestration)
│   ├── evaluation/      # evaluator.py (local RAGAS-style metrics)
│   ├── llm/             # groq_client.py (one traced, logged wrapper for every Groq call)
│   ├── observability/   # logging_config.py (logging + request IDs), tracing.py (LangSmith status)
│   └── api/             # main.py (FastAPI application)
├── chroma_db/           # Pre-populated vector store (committed to repo)
├── data/
│   └── ground_truth.json
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quickstart — Portfolio Demo UI

The fastest way to run the system locally:

```bash
# 1. Clone and install (Python 3.11)
git clone https://github.com/CoderAbhishek/deep-research-rag.git
cd deep-research-rag
pip install -r requirements.txt

# 2. Set your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# 3. Run the demo UI (ChromaDB is pre-populated in the repo)
python app.py
```

Then open [http://localhost:7860](http://localhost:7860) for the dark-mode browser interface with example questions, inline answers, and page-level source citations.

---

## Quickstart — Production API

```bash
# After steps 1–2 above
uvicorn src.api.main:app --reload --port 8000

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Infosys Cobalt?"}'
```

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Quickstart — Docker

```bash
# 1. Build the image
docker compose build

# 2. Start
docker compose up

# 3. Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Infosys Cobalt?"}'

# 4. Stop
docker compose down
```

The ChromaDB vector store is baked into the Docker image. HuggingFace model weights are cached in a named Docker volume (`huggingface_cache`) so they download once and persist across container restarts.

---

## API Reference

### `GET /health`

Returns pipeline readiness status.

```json
{"status": "ok", "pipeline_loaded": true}
```

### `POST /query`

Runs the full 7-stage pipeline.

**Request body:**
```json
{
  "question": "What is Infosys Cobalt?",
  "verbose": false
}
```

**Response:**
```json
{
  "answer": "Infosys Cobalt is a set of services, solutions and platforms... [SOURCE 1]",
  "sources": [
    {"source_num": 1, "file_name": "infosys-ar-26.pdf", "page_number": 22}
  ],
  "retrieved_sources": [
    {"source_num": 1, "file_name": "infosys-ar-26.pdf", "page_number": 22},
    {"source_num": 2, "file_name": "agm-2026-transcript.pdf", "page_number": 12}
  ],
  "query": "What is Infosys Cobalt?",
  "num_chunks": 4,
  "model": "openai/gpt-oss-20b",
  "request_id": "3f9c1a2b7d4e",
  "trace_id": "1f0b6c2e-0000-0000-0000-000000000000",
  "debug": null
}
```

`sources` lists only the sources the answer cites (empty if the system declines to answer). `retrieved_sources` lists every chunk passed to the model. `request_id` is also sent as the `X-Request-ID` header and appears in every server log line for the request. `trace_id` is the LangSmith trace ID (null when tracing is off). With `\"verbose\": true`, `debug` contains the pipeline debug payload.

**Errors:**
- `422` — empty or missing `question` field
- `500` — pipeline error (check `detail` field)