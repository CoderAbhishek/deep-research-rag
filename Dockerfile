# ── Base image ────────────────────────────────────────────────────────────────
# python:3.11-slim is Debian-based with only Python installed — no extras.
# "slim" keeps the image small (~50 MB vs ~900 MB for the full image).
FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
# build-essential: gcc and g++ for compiling C extensions (rank-bm25, numpy)
# curl: useful for healthcheck scripts and debugging
# Clean up apt cache immediately to keep the layer small.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
# All subsequent COPY, RUN, and CMD instructions resolve paths relative to /app.
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements.txt BEFORE the source code.
# Docker caches each instruction as a layer. If requirements.txt hasn't changed,
# the pip install layer is reused — rebuilds after code changes take seconds,
# not minutes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Source code ───────────────────────────────────────────────────────────────
# Copied after deps so code changes don't invalidate the pip cache layer.
COPY src/ ./src/

# ── Portfolio demo UI ─────────────────────────────────────────────────────────
# app.py is the browser-accessible demo UI.
# It is separate from the production API in src/api/main.py.
COPY app.py .

# ── ChromaDB vector store ─────────────────────────────────────────────────────
# Bake the pre-populated ChromaDB into the image so hosted deployments have it
# at runtime without a volume mount.
# rag_pipeline.py reads from ./chroma_db (CHROMA_PATH constant).
# For local Docker use, docker-compose.yml mounts ./chroma_db:/app/chroma_db
# over this, so re-ingestion on the host is picked up without rebuilding.
COPY chroma_db/ ./chroma_db/

# ── Runtime port ──────────────────────────────────────────────────────────────
# EXPOSE is documentation — it tells Docker which port the process listens on.
# app.py listens on $PORT when the hosting platform sets it, otherwise on 7860.
# docker-compose.yml overrides CMD to run the production API on port 8000.
EXPOSE 7860

# ── Default command ───────────────────────────────────────────────────────────
# Default: runs the portfolio demo UI (app.py), used for hosted deployments.
# For local production API use, docker-compose.yml overrides this with uvicorn
# on src.api.main:app at port 8000.
CMD ["python", "app.py"]