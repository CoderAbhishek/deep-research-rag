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

---

## Evaluation

**Status: being rebuilt.**

The first evaluation used 13 hand-written questions and locally implemented RAGAS-style metrics (faithfulness, answer relevancy, context precision, context recall). A later audit found it could not support conclusions about quality:

- Most reference answers refer to FY2023, while the corpus covers FY2025-26, and some reference figures do not appear in the documents at all.
- Context precision and recall compared each chunk with the reference answer using a cross-encoder and a lenient threshold. They measured topical similarity, not whether the answer was present, and scored perfect recall even on questions the system declined to answer.
- Faithfulness scored 0 on the 8 questions where the system declined to answer, which explains the low average (0.385).

A new evaluation set is in progress: 60-80 questions with page-level labels, built from observed failures and split into a development set and a held-out test set. Results will be published here.

## Known Limitations

- **Tables and charts:** PDF text extraction flattens tables, so column headings can end up separated from their numbers. Numeric questions over tables are the weakest area.
- **Reranker domain:** the cross-encoder (`ms-marco-MiniLM-L-6-v2`) was trained on web-search data, not financial reports, and can over-rank pages dense with matching terms (for example, a page listing office locations for a geography question).
- **Output checks:** grounding and citation rules are enforced through the system prompt. Answers are not yet validated against the source text in code.
- **Access:** no authentication or rate limiting. This is a portfolio project, not a hardened service.

---

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
  "model": "openai/gpt-oss-20b"
}
```

`sources` lists only the sources the answer cites (empty if the system declines to answer). `retrieved_sources` lists every chunk passed to the model. The `verbose` request field is reserved and currently has no effect.

**Errors:**
- `422` — empty or missing `question` field
- `500` — pipeline error (check `detail` field)