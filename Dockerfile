FROM python:3.13-slim

# Install system dependencies for asyncpg
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq5 && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create non-root user
RUN useradd -m -s /bin/bash appuser

WORKDIR /app

# Copy dependency files first (layer caching)
COPY pyproject.toml uv.lock README.md ./

# Install dependencies (no dev deps) and build the package
RUN uv sync --no-dev --frozen

# Copy source code and rebuild with source included
COPY weft/ weft/
RUN uv sync --no-dev --frozen

# Pre-download the fastembed model so cold starts don't download it at runtime
ENV FASTEMBED_CACHE_PATH=/app/.cache/fastembed
RUN uv run python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

# Make everything accessible to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Default to streamable HTTP transport in production
ENV WEFT_ENV=production \
    WEFT_TRANSPORT=streamable-http \
    PORT=8000

EXPOSE 8000

CMD ["uv", "run", "python", "-m", "weft.mcp"]
