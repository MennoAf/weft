"""Shared types for primer section builders.

Defines PrimerContext (the bag of state passed to every section builder)
and SectionResult (the uniform return type).  Also houses SECTION_BUDGETS
so the orchestrator owns token allocation — individual sections never
hardcode their own limits.

Reference: weft/primer.py lines 1-100 (constants) and 166-202 (build_primer
signature and shared setup).  Line numbers are as-of commit 2955aed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import asyncpg


@dataclass
class SectionResult:
    """Uniform return type for all section builders.

    Every section returns this shape so the orchestrator can budget,
    log, and apply progressive disclosure uniformly.
    """

    items: list[dict]
    """Rendered items for inclusion in the primer response."""

    tokens_used: int
    """Tokens consumed by this section (content + overhead)."""

    skipped: bool
    """True if the section had nothing to render (e.g., no project_id for grounding)."""

    skip_reason: str | None = None
    """Human-readable reason when skipped=True."""


# Per-section token caps — sourced from primer.py constants.
# The orchestrator reads these; sections do not hardcode their own limits.
SECTION_BUDGETS: dict[str, int] = {
    "grounding": 50,
    "rules": 100,
    "behaviors": 150,
    "handoff": 800,
    "recent_work": 150,
    "issues": 200,
    "anti_patterns": 150,
    "decisions": 250,
    "entities": 150,
}

# Hard caps on items shown per section.
SECTION_MAX_ITEMS: dict[str, int] = {
    "decisions": 5,
    "behaviors": 5,
    "entities": 10,
    "anti_patterns": 3,
    "recent_work": 3,
}


@dataclass
class PrimerContext:
    """Bag of state passed to every section builder.

    Created once at the start of build_primer and threaded through all
    section functions.  Mutable fields (used_tokens, seen_ids, excluded)
    are updated by the orchestrator between section calls.
    """

    # --- Identity ---
    user_id: str
    project_id: str | None
    agent_id: str | None

    # --- Infrastructure ---
    pool: asyncpg.Pool

    # --- Budget ---
    budget_tokens: int

    # --- Query bias ---
    query: str | None
    query_vec: list[float] | None

    # --- Display ---
    disclosure: str  # "progressive" or "full"
    mode: str | None

    # --- Mutable orchestrator state (set by orchestrator, read by sections) ---
    now: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    used_tokens: int = 0
    seen_ids: set[str] = field(default_factory=set)
    excluded: int = 0
    section_tokens: dict[str, int] = field(default_factory=dict)

    # --- Mode weights (resolved by orchestrator before sections run) ---
    behavior_boost: float = 1.0
    entity_boost: float = 1.0
    recency_bias: float = 0.0

    # --- Scope dict (convenience for passing to list_memories / search_by_vector) ---
    scope: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.query is not None and self.query_vec is None:
            raise ValueError(
                "query is set but query_vec is None — the caller must embed "
                "the query before constructing PrimerContext"
            )
        # Build scope from identity fields.
        if not self.scope:
            if self.project_id:
                self.scope["project_id"] = self.project_id
            if self.agent_id:
                self.scope["agent_id"] = self.agent_id

    @property
    def biased(self) -> bool:
        """True when query-biased recall is active."""
        return self.query_vec is not None
