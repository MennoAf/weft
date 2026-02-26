"""Memory candidate extraction — heuristic identification of memorable content."""

from __future__ import annotations

import re
from typing import Any


# Pattern definitions: (regex, memory_type, base_confidence)
_PREFERENCE_PATTERNS = [
    (re.compile(r"(?:I |[Uu]ser )prefer[s]?\s+(.+)", re.IGNORECASE), "preference", 0.85),
    (re.compile(r"(?:I )?always\s+(.+)", re.IGNORECASE), "preference", 0.8),
    (re.compile(r"(?:I )?never\s+(.+)", re.IGNORECASE), "preference", 0.8),
    (re.compile(r"(?:I )?(?:don't|do not) (?:like|want|use)\s+(.+)", re.IGNORECASE), "preference", 0.8),
    (re.compile(r"(?:I )?(?:like|want) to\s+(.+)", re.IGNORECASE), "preference", 0.7),
]

_FACT_PATTERNS = [
    (re.compile(r"(?:(?:project|app|system|service|codebase) )?uses?\s+(.+)", re.IGNORECASE), "fact", 0.75),
    (re.compile(r"(?:runs?|running) on\s+(.+)", re.IGNORECASE), "fact", 0.75),
    (re.compile(r"version\s+(\d[\d.]*\S*)", re.IGNORECASE), "fact", 0.7),
    (re.compile(r"(?:deployed|hosted|stored) (?:on|in|at)\s+(.+)", re.IGNORECASE), "fact", 0.7),
    (re.compile(r"(?:database|db) is\s+(.+)", re.IGNORECASE), "fact", 0.75),
    (re.compile(r"(?:written|built|implemented) (?:in|with)\s+(.+)", re.IGNORECASE), "fact", 0.75),
]

_PATTERN_PATTERNS = [
    (re.compile(r"(?:the |our )?(?:pattern|convention|rule|practice) (?:is|for)\s+(.+)", re.IGNORECASE), "pattern", 0.7),
    (re.compile(r"(?:we |I )?(?:typically|usually|generally)\s+(.+)", re.IGNORECASE), "pattern", 0.65),
    (re.compile(r"(?:the |our )?(?:approach|workflow|process) (?:is|for)\s+(.+)", re.IGNORECASE), "pattern", 0.7),
]

_ARCHITECTURE_PATTERNS = [
    (re.compile(r"(?:the )?architecture\s+(?:is|uses|follows)\s+(.+)", re.IGNORECASE), "architecture", 0.8),
    (re.compile(r"(?:the )?(?:design|stack|infrastructure)\s+(?:is|uses|includes)\s+(.+)", re.IGNORECASE), "architecture", 0.8),
    (re.compile(r"(?:three|two|multi)[- ](?:tier|layer|stage)\s+(.+)", re.IGNORECASE), "architecture", 0.75),
]

ALL_PATTERNS = _PREFERENCE_PATTERNS + _FACT_PATTERNS + _PATTERN_PATTERNS + _ARCHITECTURE_PATTERNS


def _extract_topics(text: str) -> list[str]:
    """Extract likely topic keywords from text."""
    # Common tech terms that make good topics
    tech_terms = [
        "python", "javascript", "typescript", "rust", "go", "java",
        "postgres", "postgresql", "redis", "mongodb", "mysql", "sqlite",
        "docker", "kubernetes", "aws", "gcp", "azure",
        "react", "vue", "angular", "fastapi", "django", "flask",
        "git", "ci", "cd", "testing", "deployment",
        "api", "rest", "graphql", "grpc",
        "cache", "queue", "logging", "monitoring",
        "frontend", "backend", "database", "infrastructure",
    ]
    text_lower = text.lower()
    found = [t for t in tech_terms if t in text_lower]
    return found[:3]  # Cap at 3 topics


def extract_candidates(
    text: str,
    *,
    min_confidence: float = 0.5,
) -> list[dict[str, Any]]:
    """Extract memory candidates from a text block.

    Scans each sentence/line for patterns that suggest memorable content.
    Returns a list of candidate dicts (NOT stored — for agent review).

    Each candidate has: content, type, confidence, topic, source_line
    """
    if not text or not text.strip():
        return []

    candidates: list[dict[str, Any]] = []
    seen_content: set[str] = set()

    # Split into lines/sentences
    lines = text.splitlines()

    for line in lines:
        line = line.strip()
        if not line or len(line) < 10:
            continue

        # Strip markdown bullets
        clean = re.sub(r"^[-*]\s+", "", line)
        clean = re.sub(r"^\d+\.\s+", "", clean)

        for pattern, mem_type, confidence in ALL_PATTERNS:
            match = pattern.search(clean)
            if match:
                # Use the full clean line as content (more context than just the capture group)
                content = clean.strip()

                # Skip duplicates
                content_key = content.lower()
                if content_key in seen_content:
                    continue
                seen_content.add(content_key)

                if confidence >= min_confidence:
                    candidates.append({
                        "content": content,
                        "type": mem_type,
                        "confidence": round(confidence, 2),
                        "topic": _extract_topics(content),
                        "source_line": line,
                    })
                break  # One match per line

    return candidates
