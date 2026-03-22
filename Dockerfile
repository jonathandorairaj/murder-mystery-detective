FROM python:3.12-slim

# System deps needed by chromadb (sqlite3 + hnswlib compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (cached layer — only invalidated when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source (excludes everything listed in .dockerignore)
COPY . .

EXPOSE 7777

# Default entrypoint: FastAPI server.
# Provider/model are configured via environment variables:
#   LLM_PROVIDER=ollama|openai|anthropic
#   LLM_MODEL=<model name>          (optional, falls back to per-provider default)
#   OPENAI_API_KEY / ANTHROPIC_API_KEY  (required for those providers)
CMD ["python", "server.py"]
