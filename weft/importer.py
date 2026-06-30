"""MEMORY.md importer — parses flat markdown memory files into structured MemoryCreate objects.

Reads a markdown file, splits it by headers (## or ###), infers memory type
and confidence from header text, extracts topic tags, and produces a list
of MemoryCreate objects ready for storage.

This module is pure: no DB access, no embeddings, just parsing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import asyncpg

from weft.models import MemoryCreate, MemorySource, MemoryType

# Stop words to exclude from topic extraction
STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "their", "its", "this", "that", "these", "those",
    "and", "or", "but", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more", "most",
    "some", "no", "only", "same", "than", "too", "very",
    "of", "in", "to", "for", "with", "on", "at", "from", "by", "about",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "up", "down", "out", "off", "over", "under", "again", "then", "once",
    "here", "there", "when", "where", "why", "how", "what", "which", "who",
    "whom", "if", "because", "until", "while",
})

# Header keywords → MemoryType mapping (checked in order)
_TYPE_RULES: list[tuple[list[str], MemoryType]] = [
    (["preference", "prefer", "always", "never"], MemoryType.preference),
    (["pattern", "learned"], MemoryType.pattern),
    (["architecture", "arch", "structure"], MemoryType.architecture),
    (["solution", "fix", "workaround", "debug"], MemoryType.solution),
    (["relationship", "owner", "who"], MemoryType.relationship),
]

# Default confidence by type
_CONFIDENCE: dict[MemoryType, float] = {
    MemoryType.preference: 1.0,
    MemoryType.solution: 0.8,
    MemoryType.architecture: 0.8,
    MemoryType.pattern: 0.7,
    MemoryType.fact: 0.7,
    MemoryType.relationship: 0.9,
}

# Regex to match ## or ### headers
_HEADER_RE = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)


@dataclass
class ParseResult:
    """Result of parsing a MEMORY.md file."""

    memories: list[MemoryCreate] = field(default_factory=list)
    skipped: int = 0


def _infer_type(header: str) -> MemoryType:
    """Infer memory type from header text using keyword matching."""
    lower = header.lower()
    for keywords, mem_type in _TYPE_RULES:
        if any(kw in lower for kw in keywords):
            return mem_type
    return MemoryType.fact


def _extract_topics(header: str) -> list[str]:
    """Extract topic tags from header text by splitting into significant words."""
    # Remove markdown/punctuation, lowercase, split
    cleaned = re.sub(r"[^\w\s]", " ", header.lower())
    words = cleaned.split()
    # Filter stop words and very short words
    return [w for w in words if w not in STOP_WORDS and len(w) > 1]


def _clean_content(header: str, body: str) -> str:
    """Combine header and body into clean content string."""
    # Strip excessive whitespace from body while preserving structure
    lines = [line.rstrip() for line in body.strip().splitlines()]
    # Collapse runs of blank lines into a single blank line
    cleaned_lines: list[str] = []
    prev_blank = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank:
            if not prev_blank:
                cleaned_lines.append("")
            prev_blank = True
        else:
            cleaned_lines.append(line)
            prev_blank = False
    # Remove trailing blank line
    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()

    body_clean = "\n".join(cleaned_lines)
    if body_clean:
        return f"{header}\n\n{body_clean}"
    return header


def parse_memory_md(path: str | Path) -> ParseResult:
    """Parse a MEMORY.md file into structured MemoryCreate objects.

    Splits the file by ## or ### REDACTED Each section becomes a memory
    with type inferred from the header, topics extracted from header words,
    and confidence set by type.

    Args:
        path: Path to the markdown file to parse.

    Returns:
        ParseResult with list of MemoryCreate objects and count of skipped sections.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return parse_memory_md_text(text)


def parse_memory_md_text(text: str) -> ParseResult:
    """Parse MEMORY.md content from a string (useful for testing).

    Same logic as parse_memory_md but takes raw text instead of a file path.
    """
    result = ParseResult()

    # Find all headers and their positions
    headers: list[tuple[int, int, str]] = []  # (start, end, header_text)
    for match in _HEADER_RE.finditer(text):
        REDACTEDappend((match.start(), match.end(), match.group(2).strip()))

    if not headers:
        # No headers found — treat entire text as a single section if non-empty
        stripped = text.strip()
        if stripped:
            result.memories.append(
                MemoryCreate(
                    type=MemoryType.fact,
                    content=stripped,
                    topic=[],
                    source=MemorySource.documentation,
                    confidence=_CONFIDENCE[MemoryType.fact],
                )
            )
        return result

    for i, (start, end, header_text) in enumerate(headers):
        # Body is everything between this header's end and the next header's start
        body_start = end
        body_end = headers[i + 1][0] if i + 1 < len(headers) else len(text)
        body = text[body_start:body_end]

        # Skip empty sections
        body_stripped = body.strip()
        if not body_stripped:
            result.skipped += 1
            continue

        mem_type = _infer_type(header_text)
        topics = _extract_topics(header_text)
        confidence = _CONFIDENCE.get(mem_type, 0.7)
        content = _clean_content(header_text, body_stripped)

        result.memories.append(
            MemoryCreate(
                type=mem_type,
                content=content,
                topic=topics,
                source=MemorySource.documentation,
                confidence=confidence,
            )
        )

    return result


# --- Import with embeddings and dedup ---


@dataclass
class ImportReport:
    """Result of importing memories into the store."""

    stored: int = 0
    skipped_duplicate: int = 0
    skipped_empty: int = 0
    errors: list[str] = field(default_factory=list)


async def import_memories(
    pool: asyncpg.Pool,
    provider: "EmbeddingProvider",  # noqa: F821 — string annotation to avoid circular import
    creates: list[MemoryCreate],
    *,
    project_id: str | None = None,
    dry_run: bool = False,
    similarity_threshold: float = 0.95,
) -> ImportReport:
    """Import parsed memories into the store with embedding generation and dedup.

    For each MemoryCreate:
    1. Generate embedding
    2. Search for near-duplicates (similarity > threshold)
    3. Skip if duplicate found
    4. Otherwise store the memory with its embedding

    Args:
        pool: Database connection pool
        provider: Embedding provider for vectorization
        creates: List of MemoryCreate objects from parser
        project_id: Optional project_id to assign to all memories
        dry_run: If True, report what would happen without storing
        similarity_threshold: Cosine similarity threshold for dedup (default 0.95)

    Returns:
        ImportReport with counts and any errors
    """
    from weft.store import embed_text_for_memory, search_by_vector, store_memory

    report = ImportReport()

    for create in creates:
        if not create.content.strip():
            report.skipped_empty += 1
            continue

        try:
            # Override project_id if provided
            if project_id is not None:
                create = MemoryCreate(
                    type=create.type,
                    content=create.content,
                    topic=create.topic,
                    source=create.source,
                    confidence=create.confidence,
                    project_id=project_id,
                    agent_id=create.agent_id,
                )

            # Generate embedding (content + topics, RC2)
            embedding = await provider.embed(
                embed_text_for_memory(create.content, create.topic)
            )

            # Check for near-duplicates
            dupes = await search_by_vector(
                pool,
                embedding,
                limit=1,
                threshold=similarity_threshold,
            )
            if dupes:
                report.skipped_duplicate += 1
                continue

            if not dry_run:
                await store_memory(pool, create, embedding=embedding)
            report.stored += 1

        except Exception as e:
            report.errors.append(f"Error importing '{create.content[:50]}...': {e}")

    return report
