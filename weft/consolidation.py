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

from weft.auth import current_user_id
from weft.models import (
    ContradictionWarning,
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

_UNSET = object()  # sentinel: distinguish "not provided" from explicit None

# Types that should never be decayed
IMMORTAL_TYPES = frozenset({MemoryType.preference, MemoryType.user_model, MemoryType.decision})


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
    contradiction_similarity_max: float = 0.99
    max_candidates: int = 100


@dataclass
class ConsolidationReport:
    """Results from a consolidation run."""

    decayed: list[str] = field(default_factory=list)
    duplicates_merged: list[tuple[str, str]] = field(default_factory=list)
    contradictions_flagged: list[tuple[str, str]] = field(default_factory=list)
    access_logs_pruned: int = 0
    errors: list[str] = field(default_factory=list)
    skipped: bool = False

    @property
    def total_actions(self) -> int:
        return len(self.decayed) + len(self.duplicates_merged) + len(self.contradictions_flagged)

    def to_dict(self) -> dict:
        d = {
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
            "access_logs_pruned": self.access_logs_pruned,
            "total_actions": self.total_actions,
            "errors": self.errors,
        }
        if self.skipped:
            d["skipped"] = True
        return d


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
    if memory.type in IMMORTAL_TYPES or memory.pinned:
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
        if mem.type in IMMORTAL_TYPES or mem.pinned:
            continue
        score = compute_decay_score(mem, now=now, config=cfg)
        if score <= cfg.floor_score and mem.confidence < cfg.min_confidence:
            if not dry_run:
                await update_memory(pool, mem.id, status=MemoryStatus.decayed)
            archived.append(mem.id)
            logger.debug("Decayed memory %s (score=%.3f)", mem.id, score)

    return archived


# --- Near-Duplicate Detection ---


async def _batch_fetch_embeddings(
    pool: asyncpg.Pool,
    *,
    status: MemoryStatus = MemoryStatus.active,
) -> dict[str, list[float]]:
    """Batch-fetch all embeddings for memories with a given status.

    Returns a dict mapping memory ID to its embedding vector.
    Single query instead of N per-memory fetches.
    """
    uid = current_user_id.get(None)
    if uid is not None:
        rows = await pool.fetch(
            "SELECT id, embedding FROM memories"
            " WHERE status = $1 AND embedding IS NOT NULL"
            " AND (user_id = $2 OR user_id IS NULL)",
            status.value,
            uid,
        )
    else:
        rows = await pool.fetch(
            "SELECT id, embedding FROM memories"
            " WHERE status = $1 AND embedding IS NOT NULL",
            status.value,
        )
    return {row["id"]: row["embedding"] for row in rows}


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
    embedding_map = await _batch_fetch_embeddings(pool, status=MemoryStatus.active)

    for mem in memories:
        if mem.id in seen_archived:
            continue

        embedding = embedding_map.get(mem.id)
        if not embedding:
            continue

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
                # Pinned memories are always kept
                if mem.pinned and not other.pinned:
                    keep, archive = mem, other
                elif other.pinned and not mem.pinned:
                    keep, archive = other, mem
                elif mem.pinned and other.pinned:
                    continue  # don't merge two pinned memories
                elif mem.confidence > other.confidence or (
                    mem.confidence == other.confidence
                    and mem.updated_at >= other.updated_at
                ):
                    keep, archive = mem, other
                else:
                    keep, archive = other, mem

                if not dry_run:
                    uid = current_user_id.get(None)
                    async with pool.acquire() as conn:
                        async with conn.transaction():
                            await conn.execute(
                                """
                                INSERT INTO memory_relationships
                                    (source_id, target_id, relation, created_at, user_id)
                                VALUES ($1, $2, $3, now(), nullif(current_setting('app.user_id', true), ''))
                                ON CONFLICT (source_id, target_id, relation) DO NOTHING
                                """,
                                keep.id,
                                archive.id,
                                RelationType.supersedes.value,
                            )
                            if uid is not None:
                                await conn.execute(
                                    "UPDATE memories SET status = $1, updated_at = now()"
                                    " WHERE id = $2 AND (user_id = $3 OR user_id IS NULL)",
                                    MemoryStatus.archived.value,
                                    archive.id,
                                    uid,
                                )
                            else:
                                await conn.execute(
                                    "UPDATE memories SET status = $1, updated_at = now() WHERE id = $2",
                                    MemoryStatus.archived.value,
                                    archive.id,
                                )

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
    sim_max: float = 0.99,
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
    embedding_map = await _batch_fetch_embeddings(pool, status=MemoryStatus.active)

    for mem in memories:
        embedding = embedding_map.get(mem.id)
        if not embedding:
            continue

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


_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "and", "but", "or", "if", "then", "else", "when", "while",
    "that", "this", "these", "those", "it", "its",
    "i", "we", "you", "he", "she", "they", "me", "us",
})

_NEGATION_WORDS = frozenset({
    "not", "never", "no", "don't", "doesn't", "shouldn't",
    "cannot", "can't", "without", "won't", "isn't", "aren't",
})


def _content_words(text: str, exclude: frozenset[str] = frozenset()) -> set[str]:
    """Extract significant content words, excluding stop words and extras."""
    return set(text.lower().split()) - _STOP_WORDS - exclude


def _subject_overlap_ratio(content_a: str, content_b: str, exclude: frozenset[str] = frozenset()) -> float:
    """Ratio of shared content words to the smaller set's size.

    High ratio (>= 0.5) means both texts discuss the same specific subject.
    Low ratio means they cover different aspects of a broader topic.
    """
    a = _content_words(content_a, exclude)
    b = _content_words(content_b, exclude)
    smaller = min(len(a), len(b))
    if smaller == 0:
        return 0.0
    return len(a & b) / smaller


def _negated_phrases(text: str, window: int = 3) -> list[str]:
    """Extract phrases that follow negation words (the negated subject).

    Returns a list of lowercase phrases like "support hnsw indexing".
    """
    import re
    words = re.findall(r'[a-z0-9]+(?:\.[a-z0-9]+)*', text.lower())
    phrases: list[str] = []
    for i, w in enumerate(words):
        if w in _NEGATION_WORDS:
            phrase_words = words[i + 1: i + 1 + window]
            # Filter out stop words from the phrase to get the semantic core
            core = [pw for pw in phrase_words if pw not in _STOP_WORDS]
            if core:
                phrases.append(" ".join(core))
    return phrases


def _overlap_threshold(content_a: str, content_b: str) -> float:
    """Length-aware overlap threshold.

    Short memories (< 30 words) that overlap at 50% are likely about the
    same specific claim. Long memories about the same system will
    incidentally share many words without contradicting each other.
    """
    shorter = min(len(content_a.split()), len(content_b.split()))
    if shorter < 30:
        return 0.5
    if shorter < 60:
        return 0.65
    return 0.8


def _number_context(text: str, window: int = 2) -> dict[str, str]:
    """Map each number in text to its surrounding context words.

    Returns {"0.8.1": "pgvector version", "30": "half life days", ...}.
    """
    import re
    words = text.lower().split()
    contexts: dict[str, str] = {}
    for i, w in enumerate(words):
        nums = re.findall(r'\d+\.?\d*', w)
        for n in nums:
            ctx_words = []
            for j in range(max(0, i - window), i):
                if words[j] not in _STOP_WORDS:
                    ctx_words.append(words[j])
            contexts[n] = " ".join(ctx_words)
    return contexts


def _content_conflicts(content_a: str, content_b: str) -> bool:
    """Heuristic check for conflicting content.

    Two checks:
    1. Negation: one memory negates a phrase that the other asserts.
       We extract the *negated subject* and check if it appears in the
       other memory, not just whether negation words exist.
    2. Numbers: different numbers in similar surrounding context
       (e.g., "version 0.7" vs "version 0.8"), not just any number diff.

    Both checks require sufficient subject overlap, with the threshold
    scaling by content length to avoid false positives on long memories.
    """
    a_lower = content_a.lower()
    b_lower = content_b.lower()

    threshold = _overlap_threshold(a_lower, b_lower)

    # --- Negation check ---
    # Extract what each memory negates, then check if the negated
    # subject appears in the other memory's content.
    a_neg_phrases = _negated_phrases(a_lower)
    b_neg_phrases = _negated_phrases(b_lower)

    # Only flag if exactly one side uses negation (asymmetric)
    a_has_neg = len(a_neg_phrases) > 0
    b_has_neg = len(b_neg_phrases) > 0

    if a_has_neg != b_has_neg:
        # Check if the negated subject appears in the non-negated memory
        neg_phrases = a_neg_phrases if a_has_neg else b_neg_phrases
        other_text = b_lower if a_has_neg else a_lower

        for phrase in neg_phrases:
            phrase_words = set(phrase.split())
            other_words = _content_words(other_text)
            # The negated subject must overlap with the other memory's content.
            # Use prefix matching to handle conjugation (support/supports, use/uses).
            matched = sum(
                1 for pw in phrase_words
                if any(ow.startswith(pw) or pw.startswith(ow) for ow in other_words)
            )
            if phrase_words and matched >= len(phrase_words) * 0.6:
                if _subject_overlap_ratio(a_lower, b_lower, _NEGATION_WORDS) >= threshold:
                    return True

    # --- Version/number check ---
    # Only flag when numbers appear in similar surrounding context
    # (e.g., both say "version X" but with different X).
    import re

    a_num_ctx = _number_context(a_lower)
    b_num_ctx = _number_context(b_lower)

    if a_num_ctx and b_num_ctx:
        # Find numbers that differ but share context
        a_nums = set(a_num_ctx.keys())
        b_nums = set(b_num_ctx.keys())
        if a_nums != b_nums:
            # Check if any differing numbers share the same context
            for a_num, a_ctx in a_num_ctx.items():
                if a_num in b_nums or not a_ctx:
                    continue
                a_ctx_words = set(a_ctx.split())
                for b_num, b_ctx in b_num_ctx.items():
                    if b_num in a_nums or not b_ctx:
                        continue
                    b_ctx_words = set(b_ctx.split())
                    # Context words must overlap (same metric, different value)
                    if a_ctx_words and a_ctx_words & b_ctx_words:
                        exclude = frozenset(a_nums | b_nums)
                        if _subject_overlap_ratio(a_lower, b_lower, exclude) >= threshold:
                            return True

    return False


# --- Orchestrator ---

# Advisory lock ID for serializing consolidation runs across processes
_CONSOLIDATION_LOCK_ID = 839272  # distinct from migration lock (839271)


async def consolidate(
    pool: asyncpg.Pool,
    *,
    config: ConsolidationConfig | None = None,
    dry_run: bool = False,
) -> ConsolidationReport:
    """Run the full consolidation pipeline.

    Order: decay → dedup → contradiction detection.
    Uses a Postgres advisory lock to prevent concurrent runs.
    Returns a ConsolidationReport summarizing all actions taken.
    """
    cfg = config or ConsolidationConfig()
    report = ConsolidationReport()

    async with pool.acquire() as lock_conn:
        locked = await lock_conn.fetchval(
            "SELECT pg_try_advisory_lock($1)", _CONSOLIDATION_LOCK_ID,
        )
        if not locked:
            logger.info("Consolidation already running, skipping")
            report.skipped = True
            return report

        try:
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
        finally:
            await lock_conn.execute(
                "SELECT pg_advisory_unlock($1)", _CONSOLIDATION_LOCK_ID,
            )

    # Pruning runs outside the advisory lock — it's independent of
    # decay/dedup/contradiction and is idempotent if two runs race.
    try:
        from weft.session_tracking import prune_old_access_logs

        if not dry_run:
            report.access_logs_pruned = await prune_old_access_logs(pool)
    except Exception as e:
        report.errors.append(f"Access log pruning failed: {e}")
        logger.warning("Access log pruning failed: %s", e)

    logger.info(
        "Consolidation complete: %d decayed, %d merged, %d contradictions",
        len(report.decayed),
        len(report.duplicates_merged),
        len(report.contradictions_flagged),
    )

    return report


# --- Proactive check for weft_remember ---


_MIN_CONTENT_LENGTH_FOR_CONTRADICTION = 50


async def check_contradictions_on_store(
    pool: asyncpg.Pool,
    memory_id: str,
    embedding: list[float],
    *,
    memory_type: MemoryType | None = None,
    project_id: str | object = _UNSET,
    sim_min: float = 0.8,
    sim_max: float = 0.99,
) -> list[dict]:
    """Check if a newly stored memory contradicts existing ones.

    Called from weft_remember when check_contradictions=True.
    Scoped by memory_type and project_id when provided.
    Returns list of contradiction warnings (dicts with ContradictionWarning fields).

    Short content (< 50 chars) is skipped to avoid false positives on noisy
    cosine similarity with short texts.
    """
    warnings: list[dict] = []

    # Get the new memory's content
    uid = current_user_id.get(None)
    if uid is not None:
        new_row = await pool.fetchrow(
            "SELECT content FROM memories WHERE id = $1 AND (user_id = $2 OR user_id IS NULL)",
            memory_id,
            uid,
        )
    else:
        new_row = await pool.fetchrow("SELECT content FROM memories WHERE id = $1", memory_id)
    if not new_row:
        return warnings
    new_content = new_row["content"]

    # Short content guard — cosine similarity on short texts is noisy
    if len(new_content) < _MIN_CONTENT_LENGTH_FOR_CONTRADICTION:
        return warnings

    # Build search kwargs for type/project scoping
    search_kwargs: dict = {}
    if memory_type is not None:
        search_kwargs["memory_type"] = memory_type
    if project_id is not _UNSET:
        search_kwargs["project_id"] = project_id

    similar = await search_by_vector(
        pool, embedding, limit=10, threshold=sim_min, status=MemoryStatus.active,
        **search_kwargs,
    )

    for result in similar:
        other = result.memory
        if other.id == memory_id:
            continue
        if result.similarity > sim_max:
            continue

        if _content_conflicts(new_content, other.content):
            await add_relationship(pool, memory_id, other.id, RelationType.contradicts)
            preview = other.content[:100] + ("…" if len(other.content) > 100 else "")
            warning = ContradictionWarning(
                memory_id=other.id,
                content_preview=preview,
                similarity=round(result.similarity, 4),
            )
            warnings.append(warning.to_dict())

    return warnings


# --- Auto-consolidation scheduling ---

_META_KEY = "last_consolidation_run"
_DEFAULT_INTERVAL_HOURS = 24


async def should_consolidate(
    pool: asyncpg.Pool,
    *,
    interval_hours: int = _DEFAULT_INTERVAL_HOURS,
) -> bool:
    """Check if consolidation is due based on last run timestamp."""
    from weft.store import get_metadata

    meta = await get_metadata(pool, _META_KEY)
    if meta is None:
        return True  # Never run before
    ran_at = meta.get("ran_at")
    if not ran_at:
        return True
    last_run = datetime.fromisoformat(ran_at)
    elapsed = (datetime.now(timezone.utc) - last_run).total_seconds() / 3600
    return elapsed >= interval_hours


async def record_consolidation_run(
    pool: asyncpg.Pool,
    *,
    memories_processed: int = 0,
    status: str = "completed",
) -> None:
    """Record that consolidation ran (optimistic — call before starting)."""
    from weft.store import set_metadata

    await set_metadata(pool, _META_KEY, {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "memories_processed": memories_processed,
        "status": status,
    })


async def consolidate_if_due(
    pool: asyncpg.Pool,
    *,
    interval_hours: int = _DEFAULT_INTERVAL_HOURS,
    dry_run: bool = False,
) -> dict:
    """Run consolidation if it hasn't run within interval_hours.

    Returns a summary dict. Safe to call from background tasks —
    all exceptions are caught and logged.
    """
    try:
        due = await should_consolidate(pool, interval_hours=interval_hours)
        if not due:
            return {"ran": False, "skipped_reason": "consolidation ran recently"}

        # Optimistic write — prevents concurrent triggers from both firing
        await record_consolidation_run(pool, status="running")

        report = await consolidate(pool, dry_run=dry_run)

        processed = len(report.decayed) + len(report.duplicates_merged) + len(report.contradictions_flagged)
        await record_consolidation_run(pool, memories_processed=processed, status="completed")

        logger.info(
            "Auto-consolidation complete: %d decayed, %d merged, %d contradictions",
            len(report.decayed), len(report.duplicates_merged), len(report.contradictions_flagged),
        )
        return {
            "ran": True,
            "decayed": len(report.decayed),
            "duplicates_merged": len(report.duplicates_merged),
            "contradictions_flagged": len(report.contradictions_flagged),
            "errors": report.errors,
        }
    except Exception as e:
        logger.warning("Auto-consolidation failed: %s", e)
        return {"ran": False, "skipped_reason": f"error: {e}"}


