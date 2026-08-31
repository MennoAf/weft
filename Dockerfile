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
COPY pyproject.toml uv.lock README.md docker-compose.weft.yml ./

# Install dependencies (no dev deps) and build the package
RUN uv sync --no-dev --frozen

# Copy source code and rebuild with source included
COPY weft/ weft/
COPY capability_registry/ capability_registry/
RUN find /app -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null; uv sync --no-dev --frozen

# Make everything accessible to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Default to streamable HTTP transport in production
ENV WEFT_ENV=production \
    WEFT_TRANSPORT=streamable-http \
    PORT=8000

EXPOSE 8000

CMD ["/app/.venv/bin/python", "-m", "weft.mcp"]
