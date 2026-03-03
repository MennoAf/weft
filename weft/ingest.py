"""Codebase ingestion pipeline — discovers, summarizes, and stores project knowledge."""

from __future__ import annotations

import asyncio
import logging
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import asyncpg
from anthropic import AsyncAnthropic

from weft.embeddings.base import EmbeddingProvider
from weft.models import MemorySource, MemoryType
from weft.store import upsert_by_topic

logger = logging.getLogger(__name__)

# Extensions that are never worth summarizing
_BINARY_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".bmp", ".svg", ".webp",
    ".pyc", ".pyo", ".so", ".dylib",
    ".whl", ".tar", ".gz", ".zip", ".bz2", ".xz",
})

# Config/metadata extensions — not source code
_CONFIG_EXTENSIONS = frozenset({
    ".yml", ".yaml", ".toml", ".cfg", ".ini", ".json",
})

# Lock files by exact name
_LOCK_FILES = frozenset({
    "uv.lock", "package-lock.json", "poetry.lock", "Pipfile.lock",
    "yarn.lock", "pnpm-lock.yaml", "composer.lock", "Gemfile.lock",
    "Cargo.lock",
})

# Config/metadata files by exact name
_CONFIG_FILES = frozenset({
    "pyproject.toml", "setup.cfg", "setup.py",
    ".gitignore", ".flake8",
    "Makefile", "Dockerfile",
})

# Haiku model for summarization
_MODEL = "claude-haiku-4-5-20251001"

# Max concurrent API calls
_MAX_CONCURRENCY = 5

# Max lines to send per file
_MAX_LINES = 500

# Min lines for a file to be worth summarizing
_MIN_LINES = 10


def discover_files(path: Path) -> list[str]:
    """Run ``git ls-files`` in *path* and return the list of tracked files.

    Raises ``RuntimeError`` if *path* is not inside a git repository.
    """
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=str(path),
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"{path} does not appear to be a git repository: {exc.stderr.strip()}"
        ) from exc

    return [line for line in result.stdout.splitlines() if line.strip()]


def filter_files(files: list[str], base_path: Path) -> list[str]:
    """Remove files that are not useful to summarize."""
    kept: list[str] = []

    for rel in files:
        p = Path(rel)

        # Hidden files/directories (any component starting with '.')
        if any(part.startswith(".") for part in p.parts):
            continue

        # __init__.py at any nesting level
        if p.name == "__init__.py":
            continue

        # Migration directories
        parts_lower = [part.lower() for part in p.parts]
        if "migrations" in parts_lower or "alembic" in parts_lower:
            continue

        # Lock files
        if p.name in _LOCK_FILES:
            continue

        # Config/metadata files by name
        if p.name in _CONFIG_FILES:
            continue

        # Binary extensions
        if p.suffix.lower() in _BINARY_EXTENSIONS:
            continue

        # Config extensions (not source code)
        if p.suffix.lower() in _CONFIG_EXTENSIONS:
            continue

        # Files shorter than _MIN_LINES
        full = base_path / rel
        try:
            line_count = sum(1 for _ in full.open("r", errors="replace"))
        except (OSError, UnicodeDecodeError):
            continue

        if line_count < _MIN_LINES:
            continue

        kept.append(rel)

    return kept


def build_file_tree(files: list[str]) -> str:
    """Build an ASCII directory tree from a flat list of relative paths."""
    # Build a nested dict representing the directory structure
    tree: dict = {}
    for f in sorted(files):
        parts = Path(f).parts
        node = tree
        for part in parts:
            node = node.setdefault(part, {})

    lines: list[str] = []

    def _render(node: dict, prefix: str) -> None:
        entries = sorted(node.keys())
        for i, name in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{name}")
            child_prefix = prefix + ("    " if is_last else "│   ")
            if node[name]:
                _render(node[name], child_prefix)

    _render(tree, "")
    return "\n".join(lines)


async def summarize_file(
    relative_path: str,
    content: str,
    client: AsyncAnthropic,
) -> str:
    """Ask Claude Haiku for a 1-2 sentence summary of a source file."""
    # Truncate to first _MAX_LINES lines
    lines = content.splitlines(keepends=True)
    if len(lines) > _MAX_LINES:
        truncated = "".join(lines[:_MAX_LINES])
        truncated += f"\n\n[... truncated at {_MAX_LINES} lines out of {len(lines)} ...]"
    else:
        truncated = content

    response = await client.messages.create(
        model=_MODEL,
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Summarize the following source file in 1-2 sentences. "
                    f"Describe what it does, its role in the project, and any key "
                    f"exports or interfaces.\n\n"
                    f"File: {relative_path}\n\n"
                    f"```\n{truncated}\n```"
                ),
            }
        ],
    )
    return response.content[0].text


async def generate_architecture_overview(
    tree: str,
    summaries: dict[str, str],
    client: AsyncAnthropic,
) -> str:
    """Generate a concise project architecture overview from file tree and summaries."""
    summary_text = "\n\n".join(
        f"### {path}\n{summary}" for path, summary in sorted(summaries.items())
    )

    response = await client.messages.create(
        model=_MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Based on the following project file tree and file summaries, "
                    f"write a concise architecture overview. Cover:\n"
                    f"- What the project does\n"
                    f"- How it's organized (key directories/modules)\n"
                    f"- Key modules and their relationships\n"
                    f"- Notable conventions or patterns\n\n"
                    f"## File Tree\n```\n{tree}\n```\n\n"
                    f"## File Summaries\n{summary_text}"
                ),
            }
        ],
    )
    return response.content[0].text


async def run_ingest(
    path: Path,
    project_id: str,
    *,
    depth: str = "full",
    pool: asyncpg.Pool,
    client: AsyncAnthropic,
    embedding_provider: EmbeddingProvider | None = None,
) -> dict:
    """Orchestrate the full codebase ingestion pipeline.

    Parameters
    ----------
    path:
        Root directory of the git repository to ingest.
    project_id:
        Weft project identifier for scoping stored memories.
    depth:
        ``"full"`` to summarize every file with an LLM call, or any other
        value to skip per-file summarization (tree + architecture only).
    pool:
        asyncpg connection pool for database writes.
    client:
        Anthropic async client for LLM calls.

    Returns
    -------
    dict
        Summary with keys ``files_discovered``, ``files_summarized``,
        and ``architecture_stored``.
    """
    # 1. Discover tracked files
    all_files = discover_files(path)

    # 2. Filter to summarizable source files
    filtered = filter_files(all_files, path)

    # 3. Summarize each file (if depth == "full")
    summaries: dict[str, str] = {}
    if depth == "full":
        sem = asyncio.Semaphore(_MAX_CONCURRENCY)

        async def _summarize_one(rel: str) -> tuple[str, str | None]:
            async with sem:
                full_path = path / rel
                try:
                    content = full_path.read_text(errors="replace")
                except OSError as exc:
                    logger.warning("Could not read %s: %s", rel, exc)
                    return rel, None
                try:
                    summary = await summarize_file(rel, content, client)
                    return rel, summary
                except Exception as exc:
                    logger.warning("Failed to summarize %s: %s", rel, exc)
                    return rel, None

        results = await asyncio.gather(*[_summarize_one(f) for f in filtered])
        for rel, summary in results:
            if summary is not None:
                summaries[rel] = summary

    # 4. Build file tree
    tree = build_file_tree(filtered)

    # 5. Generate architecture overview
    overview = await generate_architecture_overview(tree, summaries, client)

    # 6. Store architecture overview
    arch_embedding = None
    if embedding_provider:
        try:
            arch_embedding = await embedding_provider.embed(overview)
        except Exception as exc:
            logger.warning("Failed to embed architecture overview: %s", exc)

    await upsert_by_topic(
        pool,
        topic=["codebase-map", project_id],
        project_id=project_id,
        content=overview,
        memory_type=MemoryType.architecture,
        source=MemorySource.ingest,
        confidence=0.7,
        embedding=arch_embedding,
    )

    # 7. Store each file summary
    review_at = datetime.now(timezone.utc) + timedelta(days=30)
    for rel, summary in summaries.items():
        file_embedding = None
        if embedding_provider:
            try:
                file_embedding = await embedding_provider.embed(summary)
            except Exception as exc:
                logger.warning("Failed to embed summary for %s: %s", rel, exc)

        await upsert_by_topic(
            pool,
            topic=[f"file:{rel}", project_id],
            project_id=project_id,
            content=summary,
            memory_type=MemoryType.fact,
            source=MemorySource.ingest,
            confidence=0.7,
            review_after=review_at,
            embedding=file_embedding,
        )

    # 8. Return summary
    return {
        "files_discovered": len(all_files),
        "files_summarized": len(summaries),
        "architecture_stored": True,
    }
