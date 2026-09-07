# Deep Research RAG

A production-grade Retrieval-Augmented Generation system for dense business and research document corpora. Designed for research analysts and strategy consultants who need evidence-grounded, cited answers across multi-document collections — annual reports, DRHPs, industry reports, expert transcripts.

---

## Problem

Analysts spend hours manually reading documents to locate specific data points, cross-reference claims across sources, and build evidence-grounded views. Keyword search (Ctrl+F) is shallow. A basic chatbot is unreliable — it hallucinates and cannot cite sources.

This system solves that with a 7-stage pipeline that retrieves the most relevant context before generating an answer, and cites every claim to a specific source document and page number.

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
│  Stage 6: Deduplication             │
│  Remove near-duplicate chunks       │
│  before passing to generation       │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  Stage 7: LLM Generation            │
│  Groq compound-beta-mini            │
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
| LLM generation | Groq API (compound-beta-mini) |
| Containerisation | Docker + Docker Compose |
| Evaluation | Local RAGAS-equivalent metrics (no API calls) |

---

## Evaluation Results

Evaluated on 13 ground-truth question-answer pairs across the ingested corpus.

| Metric | Score | What it means |
|---|---|---|
| Context Precision | **0.991** | Retrieved chunks are almost entirely relevant |
| Context Recall | **1.000** | Every reference sentence is covered by the retrieved context |
| Faithfulness | 0.385 | See note below |
| Answer Relevancy | 0.462 | Correlates with faithfulness — same root cause |

**Note on Faithfulness:** The 0.385 is a measurement artefact, not hallucination. The cross-encoder faithfulness metric penalises terse one-liner answers (produced for numerical fact queries) because short answer sentences score in a different logit range than full sentences when paired with long context paragraphs. Conceptual questions — where the model produces multi-sentence answers — all scored 1.000. Fix: modify the generation prompt to require full-sentence answers for all query types.

---

## Project Structure

```
deep-research-rag/
├── src/
│   ├── ingestion/       # PDF parsing, chunking, embedding, ChromaDB ingest
│   ├── retrieval/       # Dense retriever, BM25 retriever, RRF fusion
│   ├── reranking/       # Cross-encoder reranker
│   ├── generation/      # HyDE, LLM generation, citation extraction
│   ├── pipeline/        # Full 7-stage pipeline orchestration
│   ├── evaluation/      # Local RAGAS-equivalent metrics
│   └── api/             # FastAPI application
├── data/
│   ├── chroma_db/       # Vector store (not in git)
│   └── ground_truth.json
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quickstart — Local

```bash
# 1. Clone and install
git clone https://github.com/CoderAbhishek/deep-research-rag.git
cd deep-research-rag
pip install -r requirements.txt

# 2. Set your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# 3. Ingest documents (place PDFs in documents/)
python -m src.ingestion.ingest

# 4. Run the API
uvicorn src.api.main:app --reload --port 8000

# 5. Query
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

# 2. Start (ChromaDB must already be populated on the host)
docker compose up

# 3. Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Infosys Cobalt?"}'

# 4. Stop
docker compose down
```

The `data/` directory is volume-mounted — the container reads the ChromaDB database that your local ingest script populated. HuggingFace model weights are cached in a named Docker volume (`huggingface_cache`) so they download once and persist across container restarts.

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
  "answer": "Infosys Cobalt is a set of services, solutions and platforms...",
  "sources": [
    {"source_num": 1, "file_name": "infosys-ar-26.pdf", "page_number": 22}
  ],
  "query": "What is Infosys Cobalt?",
  "num_chunks": 4,
  "model": "compound-beta-mini"
}
```

**Errors:**
- `422` — empty or missing `question` field
- `500` — pipeline error (check `detail` field)