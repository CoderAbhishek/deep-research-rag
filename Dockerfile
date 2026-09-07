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

# ── Runtime port ──────────────────────────────────────────────────────────────
# EXPOSE is documentation — it tells Docker which port the process listens on.
# It does NOT publish the port to the host; that happens via -p or compose ports:.
EXPOSE 8000

# ── Default command ───────────────────────────────────────────────────────────
# Runs the FastAPI app with Uvicorn.
# --host 0.0.0.0    bind to all interfaces (not just localhost) so requests
#                   from outside the container can reach the server
# --port 8000       match EXPOSE above
# --workers 1       single worker; the pipeline holds large models in memory —
#                   multiple workers would each load their own copy (several GB)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]