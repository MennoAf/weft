"""Consolidation pipeline — maintenance for memory quality.

Subsystems:
1. Decay scoring: archive stale, low-value memories
2. Near-duplicate detection: merge memories with embedding similarity > 0.95
3. Contradiction detection: flag memories with conflicting content
4. Orchestrator: run all subsystems and produce a ConsolidationReport
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

import asyncpg

from weft.models import (
    Memory,
    MemoryStatus,
    MemoryType,
    RelationType,
)
from weft.store import (
    add_relationship,
    list_memories,
    search_by_vector,
    update_memory,
)

logger = logging.getLogger(__name__)

# Types that should never be decayed
IMMORTAL_TYPES = frozenset({MemoryType.preference, MemoryType.user_model})


@dataclass
class DecayConfig:
    """Configuration for the decay scoring system."""

    half_life_days: float = 30.0
    floor_score: float = 0.1
    min_confidence: float = 0.3


@dataclass
class ConsolidationConfig:
    """Configuration for the full consolidation pipeline."""

    decay: DecayConfig = field(default_factory=DecayConfig)
    duplicate_threshold: float = 0.95
    contradiction_similarity_min: float = 0.7
    contradiction_similarity_max: float = 0.95
    max_candidates: int = 100


@dataclass
class ConsolidationReport:
    """Results from a consolidation run."""

    decayed: list[str] = field(default_factory=list)
    duplicates_merged: list[tuple[str, str]] = field(default_factory=list)
    contradictions_flagged: list[tuple[str, str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def total_actions(self) -> int:
        return len(self.decayed) + len(self.duplicates_merged) + len(self.contradictions_flagged)

    def to_dict(self) -> dict:
        return {
            "decayed_count": len(self.decayed),
            "decayed": self.decayed,
            "duplicates_merged_count": len(self.duplicates_merged),
            "duplicates_merged": [
                {"kept": k, "archived": a} for k, a in self.duplicates_merged
            ],
            "contradictions_flagged_count": len(self.contradictions_flagged),
            "contradictions_flagged": [
                {"memory_a": a, "memory_b": b}
                for a, b in self.contradictions_flagged
            ],
            "total_actions": self.total_actions,
            "errors": self.errors,
        }


# --- Decay Scoring ---


def compute_decay_score(
    memory: Memory,
    *,
    now: datetime | None = None,
    config: DecayConfig | None = None,
) -> float:
    """Compute a decay score for a memory. Lower = more likely to be archived.

    Score combines: recency, access frequency, and confidence.
    Immortal types (preference, user_model) always return 1.0.
    """
    if memory.type in IMMORTAL_TYPES:
        return 1.0

    cfg = config or DecayConfig()
    now = now or datetime.now(timezone.utc)

    accessed_at = memory.accessed_at
    if accessed_at.tzinfo is None:
        accessed_at = accessed_at.replace(tzinfo=timezone.utc)

    # Recency factor: exponential decay
    age_days = max(0.0, (now - accessed_at).total_seconds() / 86400)
    if cfg.half_life_days > 0:
        recency = math.pow(0.5, age_days / cfg.half_life_days)
    else:
        recency = 1.0

    # Frequency factor: log-scaled access count
    freq = min(1.0, math.log1p(memory.access_count) / math.log1p(20))

    # Combine: weighted average favoring recency
    score = (0.5 * recency) + (0.2 * freq) + (0.3 * memory.confidence)

    return max(cfg.floor_score, score)


async def run_decay(
    pool: asyncpg.Pool,
    *,
    config: DecayConfig | None = None,
    dry_run: bool = False,
    now: datetime | None = None,
) -> list[str]:
    """Archive memories with decay scores below the floor.

    Returns list of archived memory IDs.
    """
    cfg = config or DecayConfig()
    now = now or datetime.now(timezone.utc)
    archived: list[str] = []

    memories = await list_memories(pool, status=MemoryStatus.active, limit=1000)

    for mem in memories:
        if mem.type in IMMORTAL_TYPES:
            continue
        score = compute_decay_score(mem, now=now, config=cfg)
        if score <= cfg.floor_score and mem.confidence < cfg.min_confidence:
            if not dry_run:
                await update_memory(pool, mem.id, status=MemoryStatus.decayed)
            archived.append(mem.id)
            logger.debug("Decayed memory %s (score=%.3f)", mem.id, score)

    return archived


# --- Near-Duplicate Detection ---


async def find_duplicates(
    pool: asyncpg.Pool,
    *,
    threshold: float = 0.95,
    dry_run: bool = False,
) -> list[tuple[str, str]]:
    """Find and merge near-duplicate memories.

    For each pair with similarity > threshold, keep the one with higher
    confidence (or more recent), archive the other, and create a
    supersedes relationship.

    Returns list of (kept_id, archived_id) tuples.
    """
    merged: list[tuple[str, str]] = []
    seen_archived: set[str] = set()

    memories = await list_memories(pool, status=MemoryStatus.active, limit=1000)

    for mem in memories:
        if mem.id in seen_archived:
            continue

        # Get this memory's embedding from DB
        row = await pool.fetchrow(
            "SELECT embedding FROM memories WHERE id = $1 AND embedding IS NOT NULL",
            mem.id,
        )
        if not row or not row["embedding"]:
            continue

        # Parse the stored vector string back to list[float]
        embedding = _parse_pgvector(row["embedding"])

        # Search for similar memories
        from weft.models import MemoryRecall

        similar = await search_by_vector(
            pool, embedding, limit=10, threshold=threshold, status=MemoryStatus.active,
        )

        for result in similar:
            other = result.memory
            if other.id == mem.id or other.id in seen_archived:
                continue
            if result.similarity >= threshold:
                # Keep the one with higher confidence, or more recent
                if mem.confidence > other.confidence or (
                    mem.confidence == other.confidence
                    and mem.updated_at >= other.updated_at
                ):
                    keep, archive = mem, other
                else:
                    keep, archive = other, mem

                if not dry_run:
                    await add_relationship(
                        pool, keep.id, archive.id, RelationType.supersedes,
                    )
                    await update_memory(pool, archive.id, status=MemoryStatus.archived)

                merged.append((keep.id, archive.id))
                seen_archived.add(archive.id)
                logger.debug(
                    "Merged duplicate: kept %s, archived %s (sim=%.3f)",
                    keep.id, archive.id, result.similarity,
                )

    return merged


# --- Contradiction Detection ---


async def find_contradictions(
    pool: asyncpg.Pool,
    *,
    sim_min: float = 0.7,
    sim_max: float = 0.95,
    dry_run: bool = False,
) -> list[tuple[str, str]]:
    """Find memories that may contradict each other.

    Looks for memory pairs with moderate-to-high similarity (likely same topic)
    but potentially conflicting content (detected via heuristics).

    Returns list of (memory_a_id, memory_b_id) tuples.
    """
    flagged: list[tuple[str, str]] = []
    checked_pairs: set[tuple[str, str]] = set()

    memories = await list_memories(pool, status=MemoryStatus.active, limit=500)

    for mem in memories:
        row = await pool.fetchrow(
            "SELECT embedding FROM memories WHERE id = $1 AND embedding IS NOT NULL",
            mem.id,
        )
        if not row or not row["embedding"]:
            continue

        embedding = _parse_pgvector(row["embedding"])

        similar = await search_by_vector(
            pool, embedding, limit=20, threshold=sim_min, status=MemoryStatus.active,
        )

        for result in similar:
            other = result.memory
            if other.id == mem.id:
                continue

            # Normalize pair ordering to avoid checking both directions
            pair = tuple(sorted([mem.id, other.id]))
            if pair in checked_pairs:
                continue
            checked_pairs.add(pair)

            # Only check moderate similarity (not near-duplicates)
            if result.similarity > sim_max:
                continue

            if _content_conflicts(mem.content, other.content):
                if not dry_run:
                    await add_relationship(
                        pool, mem.id, other.id, RelationType.contradicts,
                    )
                flagged.append((mem.id, other.id))
                logger.debug(
                    "Contradiction flagged: %s vs %s (sim=%.3f)",
                    mem.id, other.id, result.similarity,
                )

    return flagged


def _content_conflicts(content_a: str, content_b: str) -> bool:
    """Heuristic check for conflicting content.

    Looks for negation patterns, different numbers/versions, and
    opposing statements.
    """
    a_lower = content_a.lower()
    b_lower = content_b.lower()

    # Negation patterns: one has "not", "never", "don't", the other doesn't
    negation_words = {"not", "never", "no", "don't", "doesn't", "shouldn't", "cannot", "can't", "without"}
    a_negations = negation_words & set(a_lower.split())
    b_negations = negation_words & set(b_lower.split())
    if bool(a_negations) != bool(b_negations):
        return True

    # Version/number conflicts: different numbers in similar contexts
    import re

    a_numbers = set(re.findall(r'\d+\.?\d*', a_lower))
    b_numbers = set(re.findall(r'\d+\.?\d*', b_lower))
    if a_numbers and b_numbers and a_numbers != b_numbers:
        # Only flag if they share significant non-numeric words
        a_words = set(a_lower.split()) - a_numbers
        b_words = set(b_lower.split()) - b_numbers
        overlap = a_words & b_words
        if len(overlap) >= 3:
            return True

    return False


# --- Orchestrator ---


async def consolidate(
    pool: asyncpg.Pool,
    *,
    config: ConsolidationConfig | None = None,
    dry_run: bool = False,
) -> ConsolidationReport:
    """Run the full consolidation pipeline.

    Order: decay → dedup → contradiction detection.
    Returns a ConsolidationReport summarizing all actions taken.
    """
    cfg = config or ConsolidationConfig()
    report = ConsolidationReport()

    try:
        report.decayed = await run_decay(pool, config=cfg.decay, dry_run=dry_run)
    except Exception as e:
        report.errors.append(f"Decay failed: {e}")
        logger.exception("Decay subsystem failed")

    try:
        report.duplicates_merged = await find_duplicates(
            pool, threshold=cfg.duplicate_threshold, dry_run=dry_run,
        )
    except Exception as e:
        report.errors.append(f"Duplicate detection failed: {e}")
        logger.exception("Duplicate detection subsystem failed")

    try:
        report.contradictions_flagged = await find_contradictions(
            pool,
            sim_min=cfg.contradiction_similarity_min,
            sim_max=cfg.contradiction_similarity_max,
            dry_run=dry_run,
        )
    except Exception as e:
        report.errors.append(f"Contradiction detection failed: {e}")
        logger.exception("Contradiction detection subsystem failed")

    logger.info(
        "Consolidation complete: %d decayed, %d merged, %d contradictions",
        len(report.decayed),
        len(report.duplicates_merged),
        len(report.contradictions_flagged),
    )

    return report


# --- Proactive check for weft_remember ---


async def check_contradictions_on_store(
    pool: asyncpg.Pool,
    memory_id: str,
    embedding: list[float],
    *,
    sim_min: float = 0.7,
    sim_max: float = 0.95,
) -> list[dict]:
    """Check if a newly stored memory contradicts existing ones.

    Called from weft_remember when check_contradictions=True.
    Returns list of contradiction warnings.
    """
    warnings: list[dict] = []

    similar = await search_by_vector(
        pool, embedding, limit=10, threshold=sim_min, status=MemoryStatus.active,
    )

    # Get the new memory's content
    new_row = await pool.fetchrow("SELECT content FROM memories WHERE id = $1", memory_id)
    if not new_row:
        return warnings
    new_content = new_row["content"]

    for result in similar:
        other = result.memory
        if other.id == memory_id:
            continue
        if result.similarity > sim_max:
            continue

        if _content_conflicts(new_content, other.content):
            await add_relationship(pool, memory_id, other.id, RelationType.contradicts)
            warnings.append({
                "type": "contradiction",
                "memory_id": other.id,
                "content_preview": other.content[:100],
                "similarity": round(result.similarity, 4),
            })

    return warnings


# --- Helpers ---


def _parse_pgvector(vec_str: str) -> list[float]:
    """Parse pgvector string format '[1.0,2.0,3.0]' to list[float]."""
    if isinstance(vec_str, (list, tuple)):
        return list(vec_str)
    cleaned = vec_str.strip("[]")
    return [float(x) for x in cleaned.split(",")]
