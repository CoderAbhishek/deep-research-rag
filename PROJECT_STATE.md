# PROJECT_STATE.md — Deep Research RAG Engineering

> **Persistent source of truth for Project 1.**
> A fresh Claude session must read this file before doing anything else.
> Update the relevant sections after every session milestone.

---

## 1. Project Identity

| Field | Value |
|---|---|
| Project name | Deep Research RAG Engineering |
| Portfolio label | Project 1 |
| GitHub repo (planned) | `deep-research-rag` |
| Domain | Market research / consulting documents |
| Target audience | Portfolio — AI Engineer / AI Consultant interviews |

---

## 2. Project Objective

Build a production-grade RAG system that a research analyst or consultant could use to
query a multi-document corpus of dense business and research documents
(annual reports, DRHPs, industry reports, expert transcripts, consulting reports)
and receive evidence-grounded, cited answers.

The word "deep" signals the scope: this is NOT a basic PDF chatbot.
It covers the full stack of retrieval engineering — sparse retrieval, hybrid search,
query rewriting, multi-query, HyDE, reranking, systematic evaluation — as well as
a proper API layer, Docker deployment, and a portfolio-ready README.

---

## 3. Problem Definition (Step 0 — complete)

### Who is the user?
A research analyst or strategy consultant who works with dense document corpora:
annual reports, DRHPs, industry reports, expert interview transcripts,
competitor analyses, consulting deliverables.

### What is the problem?
Analysts spend significant hours reading full documents to:
- Locate specific data points or claims
- Cross-reference information across multiple documents
- Build evidence-grounded views (with citations) on companies/markets
- Compare companies or sectors

### What is currently inefficient?
- Manual reading of entire documents (Ctrl+F is keyword-only)
- No cross-document semantic search
- No automatic evidence linking
- Repetitive reading across similar document types
- No structured recall of information from a growing document corpus

### What should AI improve?
- Instant semantic search across a multi-document corpus
- Evidence-grounded answers with source citations and page numbers
- Cross-document comparison on a specific question
- Structured analyst-style output (not just a chat response)

### What should NOT be automated?
- Final analytical judgment and interpretation
- Business decisions based on retrieved evidence
- Contextual reasoning that requires domain expertise the system does not have

### Inputs
- One or more PDF documents (consulting domain)
- A natural-language research question

### Outputs
- A direct answer grounded in document evidence
- Source citations: document name + page number + relevant chunk
- Confidence/limitation flags where appropriate

### Success criteria (measurable)
- Retrieval quality improves over naive baseline (Recall@5, Precision@5, MRR)
- Answer faithfulness score (RAGAS or manual) is measurably above random
- System can handle a corpus of at least 5 documents simultaneously
- All major components are explainable at interview depth

### Explicit limitations (by design)
- Not a real-time system (batch document ingestion, not live feeds)
- Does not scrape or access live URLs
- Does not replace analyst judgment
- Local model inference may be slow without GPU

---

## 4. What the Baseline Already Covers (prior RAG PDF Chat)

Abhi has already built and understands:
- PDF text extraction (manual)
- Fixed-size chunking (manual)
- Sentence transformer embeddings
- ChromaDB vector store
- Basic dense similarity search
- LangChain RetrievalQA chain (and comparison with manual pipeline)
- Groq as LLM inference provider
- Streamlit UI
- Basic .env / secret management

**This project does NOT re-teach those basics from scratch.**
It establishes a controlled baseline version in Session 2, then progressively
replaces each component with a deeper engineering approach.

---

## 5. Architecture (Target — End of Project 1)

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│          Query Understanding Layer           │
│  - Query rewriting                           │
│  - Multi-query generation                    │
│  - HyDE (hypothetical document embedding)    │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│           Hybrid Retrieval Layer             │
│  Dense vector search (sentence-transformers) │
│           +                                  │
│  Sparse keyword search (BM25)                │
│           ↓                                  │
│  Reciprocal Rank Fusion (RRF)                │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│            Reranking Layer                   │
│  Cross-encoder reranker                      │
│  (retrieve 20 → rerank → top 5)              │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│          Context Construction                │
│  Chunk text + source metadata + page number  │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│              LLM Generation                  │
│  Ollama (local) or Groq (hosted open-weight) │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│         Evaluation + Tracing Layer           │
│  LangSmith tracing                           │
│  RAGAS / custom evaluation metrics           │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│              API Layer (FastAPI)             │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│              UI (Streamlit / Gradio)         │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│              Deployment (Docker)             │
└─────────────────────────────────────────────┘
```

**Document ingestion pipeline (separate from query pipeline):**
```
PDF document(s)
    │
    ▼
Text extraction (PyMuPDF or pdfplumber)
    │
    ▼
Chunking strategy (recursive, parent-child)
    │
    ▼
Embedding (sentence-transformers, local)
    │
    ▼
ChromaDB (dense store) + BM25 index (sparse store)
```

---

## 6. Planned Project Structure

```
deep-research-rag/
├── README.md
├── ARCHITECTURE.md
├── .gitignore
├── LICENSE
├── .env.example
├── requirements.txt
│
├── src/
│   ├── ingestion/
│   │   ├── loader.py           # PDF loading + text extraction
│   │   ├── chunker.py          # Chunking strategies
│   │   └── embedder.py         # Embedding pipeline
│   │
│   ├── retrieval/
│   │   ├── dense.py            # Dense vector search
│   │   ├── sparse.py           # BM25 search
│   │   ├── hybrid.py           # Hybrid search + RRF
│   │   ├── reranker.py         # Cross-encoder reranking
│   │   └── query.py            # Query rewriting, multi-query, HyDE
│   │
│   ├── generation/
│   │   └── rag_chain.py        # Full RAG pipeline
│   │
│   ├── evaluation/
│   │   ├── retrieval_eval.py   # Recall@k, Precision@k, MRR, NDCG
│   │   └── answer_eval.py      # Faithfulness, relevance, correctness
│   │
│   └── api/
│       └── main.py             # FastAPI application
│
├── experiments/
│   ├── chunking/
│   ├── dense_vs_sparse/
│   ├── hybrid_search/
│   ├── reranking/
│   ├── query_rewriting/
│   └── evaluation/
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_rag_pipeline.py
│
├── docs/
│   └── architecture.png
│
├── data/
│   └── documents/              # Your PDF corpus (gitignored)
│
├── app.py                      # Streamlit or Gradio UI
├── Dockerfile
└── docker-compose.yml
```

---

## 7. Technologies & Frameworks

| Category | Technology | When introduced |
|---|---|---|
| PDF parsing | PyMuPDF (fitz) or pdfplumber | Session 2 |
| Chunking | LangChain text splitters (recursive) | Session 2 |
| Embeddings | sentence-transformers (local, free) | Session 2 |
| Dense vector store | ChromaDB | Session 2 |
| LLM inference | Ollama (local) or Groq (hosted, free tier) | Session 2 |
| Sparse retrieval | rank-bm25 | Session 5 |
| Hybrid fusion | Custom RRF implementation | Session 6 |
| Query engineering | LangChain / custom | Session 7 |
| Reranking | sentence-transformers cross-encoder | Session 8 |
| LLM orchestration | LangChain (compared vs manual) | Session 9 |
| Tracing | LangSmith (free tier) | Session 9 |
| Evaluation | RAGAS or custom metrics | Session 10 |
| API | FastAPI | Session 11 |
| UI | Streamlit | Session 2 (baseline), refined later |
| Containerisation | Docker | Session 12 |
| Version control | Git + GitHub | Session 1 |

---

## 8. Session Plan

| Session | Title | Key concepts | Deliverables |
|---|---|---|---|
| 0 | Problem Definition + Planning | Problem framing, architecture overview, session map | PROJECT_STATE.md (this file) |
| 1 | Architecture + GitHub | Git fundamentals, repo setup, README structure | GitHub repo, first commit |
| 2 | Environment + Baseline RAG | VS Code, venv, .env, baseline RAG (build quickly from prior knowledge) | Controlled baseline working |
| 3 | Multi-Document Corpus + Advanced Ingestion | Corpus management, metadata, recursive chunking, parent-child concept | Ingestion pipeline v2 |
| 4 | Advanced Chunking + Embedding Deep Dive | Semantic chunking, embedding model trade-offs, chunk quality | Chunker module, embedding module |
| 5 | Sparse Retrieval — BM25 | TF-IDF → BM25, why keywords still matter, rank_bm25 | sparse.py, first experiment |
| 6 | Hybrid Search + RRF | Dense + sparse fusion, Reciprocal Rank Fusion | hybrid.py, experiment: dense vs sparse vs hybrid |
| 7 | Query Engineering | Query rewriting, multi-query, HyDE | query.py, experiment: query strategies |
| 8 | Reranking | Cross-encoder vs bi-encoder, retrieve-then-rerank | reranker.py, experiment: with/without reranker |
| 9 | Pipeline Assembly + LangSmith | Full advanced RAG chain, LangChain vs manual, tracing | rag_chain.py, LangSmith traces |
| 10 | Evaluation | Retrieval metrics, answer metrics, evaluation dataset | evaluation/, results documented |
| 11 | FastAPI | API layer, routes, schemas, validation, error handling | api/main.py, testable API |
| 12 | Docker + Portfolio Prep | Dockerfile, Docker Compose, ARCHITECTURE.md, final README | Deployable repo, portfolio-ready |

---

## 9. Experiment Log

| Session | Experiment | Question | Baseline | Result | Conclusion |
|---|---|---|---|---|---|
| — | None yet | — | — | — | — |

---

## 10. Implementation Milestones

- [ ] Session 0: Problem definition + planning complete
- [ ] Session 1: GitHub repo live, first commit pushed
- [ ] Session 2: Baseline RAG working end-to-end
- [ ] Session 3: Multi-document ingestion pipeline
- [ ] Session 4: Advanced chunking + embedding module
- [ ] Session 5: BM25 sparse retrieval working
- [ ] Session 6: Hybrid retrieval + RRF working
- [ ] Session 7: Query rewriting + multi-query + HyDE working
- [ ] Session 8: Cross-encoder reranker integrated
- [ ] Session 9: Full pipeline assembled + LangSmith traces visible
- [ ] Session 10: Evaluation dataset + metrics computed
- [ ] Session 11: FastAPI layer working
- [ ] Session 12: Dockerised + portfolio-ready

---

## 11. Key Concepts Learned

*(Populated progressively as sessions complete)*

---

## 12. Architectural / Design Decisions

*(Populated progressively as decisions are made and reasoned through)*

---

## 13. Errors / Debugging Log

*(Populated progressively)*

---

## 14. Evaluation Results

*(Populated in Session 10)*

---

## 15. Git Commits / Checkpoints

*(Populated from Session 1 onward)*

---

## 16. Unresolved Questions

*(Populated as questions arise during build)*

---

## 17. Concepts Needing Revisiting

*(Populated as sessions complete)*

---

## 18. Dependencies & Environment

*(Populated in Session 2)*

| Package | Version pinned | Purpose |
|---|---|---|
| — | — | — |

---

## 19. Commands Reference

*(Populated in Session 2)*

---

## 20. Current State

| Field | Value |
|---|---|
| Last completed session | Session 0 — Problem Definition + Planning |
| Current session | — |
| Last commit | None yet |
| Working baseline | Not yet established |

---

## 21. Exact Next Step

**Session 1 — Architecture + GitHub**

Actions:
1. Draw the full system architecture on paper (or in the session)
2. Create GitHub repository: `deep-research-rag`
3. Learn Git fundamentals (what Git is, repo, working tree, staging, commit, branch, remote, push, .gitignore)
4. Create: README.md (initial), .gitignore (Python), LICENSE (MIT)
5. Initialise Git locally, connect to remote, push first commit
6. Commit message: `chore: initialise project`

Do not start installing Python dependencies yet. That is Session 2.