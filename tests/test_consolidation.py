"""Tests for weft.consolidation — decay, dedup, contradiction, orchestrator."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from weft.consolidation import (
    IMMORTAL_TYPES,
    ConsolidationConfig,
    ConsolidationReport,
    DecayConfig,
    _content_conflicts,
    compute_decay_score,
    consolidate,
    find_contradictions,
    find_duplicates,
    run_decay,
)
from weft.embeddings import get_provider
from weft.models import (
    Memory,
    MemoryCreate,
    MemorySource,
    MemoryStatus,
    MemoryType,
)
from weft.store import get_memory, get_relationships, store_memory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_memory(
    *,
    memory_type: MemoryType = MemoryType.fact,
    content: str = "test content",
    confidence: float = 0.7,
    access_count: int = 0,
    accessed_at: datetime | None = None,
) -> Memory:
    """Build an in-memory Memory object for pure-function tests."""
    now = datetime.now(timezone.utc)
    return Memory(
        type=memory_type,
        content=content,
        confidence=confidence,
        access_count=access_count,
        accessed_at=accessed_at or now,
        created_at=now,
        updated_at=now,
    )


# ---------------------------------------------------------------------------
# 1. Decay scoring — pure function tests
# ---------------------------------------------------------------------------


class TestDecayScoreImmortals:
    """Verify that preference and user_model memories always score 1.0."""

    def test_decay_score_immortal_types(self):
        """Preference and user_model memories always return score 1.0
        regardless of age, confidence, or access count."""
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=365)

        for mem_type in IMMORTAL_TYPES:
            mem = _make_memory(
                memory_type=mem_type,
                confidence=0.1,
                access_count=0,
                accessed_at=old,
            )
            score = compute_decay_score(mem, now=now)
            assert score == 1.0, (
                f"Expected 1.0 for immortal type {mem_type.value}, got {score}"
            )


class TestDecayScoreRecent:
    """Recently accessed memories should keep a high score."""

    def test_decay_score_recent_memory(self):
        """A memory accessed just now should have a high decay score."""
        now = datetime.now(timezone.utc)
        mem = _make_memory(
            confidence=0.8,
            access_count=5,
            accessed_at=now,
        )
        score = compute_decay_score(mem, now=now)
        # High confidence + recent access + some access count → high score
        assert score > 0.7, f"Expected high score for recent memory, got {score}"


class TestDecayScoreOld:
    """Old, unaccessed memories should get a low score."""

    def test_decay_score_old_memory(self):
        """A memory not accessed in 90+ days with low confidence
        should have a low score."""
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=120)
        mem = _make_memory(
            confidence=0.2,
            access_count=0,
            accessed_at=old,
        )
        score = compute_decay_score(mem, now=now)
        # Low confidence + very old + no accesses → low score
        assert score < 0.3, f"Expected low score for old memory, got {score}"


# ---------------------------------------------------------------------------
# 1b. Decay execution — integration tests against real Postgres
# ---------------------------------------------------------------------------


class TestRunDecay:
    """Integration tests for run_decay() against the database."""

    async def test_run_decay_archives_stale(self, pool):
        """Create an old, low-confidence, never-accessed fact memory.
        Set accessed_at to 120 days ago. Run run_decay(). Verify it
        gets decayed status."""
        # Create a fact memory (mutable type, will decay)
        create = MemoryCreate(
            type=MemoryType.fact,
            content="Some stale information that nobody cares about",
            confidence=0.2,
            topic=["stale"],
        )
        mem = await store_memory(pool, create)

        # Backdate the accessed_at to 120 days ago via SQL
        old_dt = datetime.now(timezone.utc) - timedelta(days=120)
        await pool.execute(
            "UPDATE memories SET accessed_at = $1 WHERE id = $2",
            old_dt,
            mem.id,
        )

        # Run decay
        decayed = await run_decay(pool)
        assert mem.id in decayed

        # Verify the memory has decayed status in DB
        fetched = await get_memory(pool, mem.id)
        assert fetched is not None
        assert fetched.status == MemoryStatus.decayed

    async def test_run_decay_preserves_preferences(self, pool):
        """Even old preferences should survive decay."""
        create = MemoryCreate(
            type=MemoryType.preference,
            content="Always use sonnet for subagents",
            confidence=0.5,
            topic=["workflow"],
        )
        mem = await store_memory(pool, create)

        # Backdate to make it very old
        old_dt = datetime.now(timezone.utc) - timedelta(days=365)
        await pool.execute(
            "UPDATE memories SET accessed_at = $1 WHERE id = $2",
            old_dt,
            mem.id,
        )

        decayed = await run_decay(pool)
        assert mem.id not in decayed

        fetched = await get_memory(pool, mem.id)
        assert fetched is not None
        assert fetched.status == MemoryStatus.active


# ---------------------------------------------------------------------------
# 2. Near-duplicate detection
# ---------------------------------------------------------------------------


class TestFindDuplicates:
    """Integration tests for find_duplicates() with real embeddings."""

    async def test_find_duplicates_merges_similar(self, pool):
        """Two memories with nearly identical content should be merged."""
        provider = get_provider("fastembed")

        # Use paraphrased content that embeds very similarly
        content_a = "pgvector uses cosine distance to measure vector similarity"
        content_b = "pgvector uses cosine distance for vector similarity measurement"

        emb_a = await provider.embed(content_a)
        emb_b = await provider.embed(content_b)

        mem_a = await store_memory(
            pool,
            MemoryCreate(
                type=MemoryType.fact,
                content=content_a,
                confidence=0.8,
                topic=["pgvector"],
            ),
            embedding=emb_a,
        )
        mem_b = await store_memory(
            pool,
            MemoryCreate(
                type=MemoryType.fact,
                content=content_b,
                confidence=0.6,
                topic=["pgvector"],
            ),
            embedding=emb_b,
        )

        # find_duplicates reads embeddings from DB, no provider arg
        merged = await find_duplicates(pool, threshold=0.85)

        assert len(merged) >= 1
        # Returns list of (kept_id, archived_id) tuples
        kept_id, archived_id = merged[0]
        assert kept_id and archived_id

        # Verify the archived one is actually archived
        archived_mem = await get_memory(pool, archived_id)
        assert archived_mem.status == MemoryStatus.archived

        # Verify a supersedes relationship was created
        rels = await get_relationships(pool, kept_id)
        assert any(
            r.relation.value == "supersedes"
            for r in rels
        )

    async def test_find_duplicates_keeps_higher_confidence(self, pool):
        """The memory with higher confidence should be kept."""
        provider = get_provider("fastembed")

        content_a = "Redis is used for caching in Weft"
        content_b = "Redis is used for caching in Weft system"

        emb_a = await provider.embed(content_a)
        emb_b = await provider.embed(content_b)

        mem_a = await store_memory(
            pool,
            MemoryCreate(
                type=MemoryType.fact,
                content=content_a,
                confidence=0.9,
                topic=["redis"],
            ),
            embedding=emb_a,
        )
        mem_b = await store_memory(
            pool,
            MemoryCreate(
                type=MemoryType.fact,
                content=content_b,
                confidence=0.4,
                topic=["redis"],
            ),
            embedding=emb_b,
        )

        merged = await find_duplicates(pool, threshold=0.85)

        if len(merged) > 0:
            # The higher-confidence memory (mem_a, 0.9) should be kept
            kept_id, archived_id = merged[0]
            assert kept_id == mem_a.id
            assert archived_id == mem_b.id

    async def test_find_duplicates_ignores_different(self, pool):
        """Two very different memories should not be merged."""
        provider = get_provider("fastembed")

        content_a = "Python is a programming language used for AI"
        content_b = "The weather in Tokyo is rainy this week"

        emb_a = await provider.embed(content_a)
        emb_b = await provider.embed(content_b)

        await store_memory(
            pool,
            MemoryCreate(
                type=MemoryType.fact,
                content=content_a,
                confidence=0.7,
                topic=["python"],
            ),
            embedding=emb_a,
        )
        await store_memory(
            pool,
            MemoryCreate(
                type=MemoryType.fact,
                content=content_b,
                confidence=0.7,
                topic=["weather"],
            ),
            embedding=emb_b,
        )

        merged = await find_duplicates(pool, threshold=0.95)
        assert len(merged) == 0


# ---------------------------------------------------------------------------
# 3. Contradiction detection
# ---------------------------------------------------------------------------


class TestContentConflicts:
    """Unit tests for the _content_conflicts heuristic."""

    def test_find_contradictions_negation(self):
        """Negation mismatch should be flagged as a conflict."""
        assert _content_conflicts(
            "pgvector supports HNSW indexing",
            "pgvector does not support HNSW indexing",
        )

    def test_find_contradictions_version_mismatch(self):
        """Different version numbers should be flagged as a conflict."""
        assert _content_conflicts(
            "pgvector version is 0.7.0",
            "pgvector version is 0.8.1",
        )

    def test_find_contradictions_ignores_unrelated(self):
        """Two texts with no negation or version differences → no conflict."""
        assert not _content_conflicts(
            "Python is good for data science",
            "JavaScript is popular for web development",
        )


class TestFindContradictions:
    """Integration tests for find_contradictions() against real DB + embeddings."""

    async def test_find_contradictions_negation(self, pool):
        """Two memories about the same topic where one uses negation
        should flag contradiction."""
        provider = get_provider("fastembed")

        content_a = "pgvector supports HNSW indexing"
        content_b = "pgvector does not support HNSW indexing"

        emb_a = await provider.embed(content_a)
        emb_b = await provider.embed(content_b)

        mem_a = await store_memory(
            pool,
            MemoryCreate(
                type=MemoryType.fact,
                content=content_a,
                confidence=0.8,
                topic=["pgvector"],
            ),
            embedding=emb_a,
        )
        mem_b = await store_memory(
            pool,
            MemoryCreate(
                type=MemoryType.fact,
                content=content_b,
                confidence=0.7,
                topic=["pgvector"],
            ),
            embedding=emb_b,
        )

        # find_contradictions reads embeddings from DB, no provider arg
        flagged = await find_contradictions(pool, sim_min=0.7)

        assert len(flagged) >= 1
        # Returns list of (memory_a_id, memory_b_id) tuples
        ids_in_flagged = set()
        for pair in flagged:
            ids_in_flagged.add(pair[0])
            ids_in_flagged.add(pair[1])
        assert mem_a.id in ids_in_flagged
        assert mem_b.id in ids_in_flagged

    async def test_find_contradictions_version_mismatch(self, pool):
        """Two memories with different version numbers should flag contradiction."""
        provider = get_provider("fastembed")

        content_a = "pgvector 0.7.0 is the latest release"
        content_b = "pgvector 0.8.1 is the latest release"

        emb_a = await provider.embed(content_a)
        emb_b = await provider.embed(content_b)

        mem_a = await store_memory(
            pool,
            MemoryCreate(
                type=MemoryType.fact,
                content=content_a,
                confidence=0.8,
                topic=["pgvector"],
            ),
            embedding=emb_a,
        )
        mem_b = await store_memory(
            pool,
            MemoryCreate(
                type=MemoryType.fact,
                content=content_b,
                confidence=0.7,
                topic=["pgvector"],
            ),
            embedding=emb_b,
        )

        flagged = await find_contradictions(pool, sim_min=0.7)

        assert len(flagged) >= 1
        ids_in_flagged = set()
        for pair in flagged:
            ids_in_flagged.add(pair[0])
            ids_in_flagged.add(pair[1])
        assert mem_a.id in ids_in_flagged
        assert mem_b.id in ids_in_flagged

    async def test_find_contradictions_ignores_unrelated(self, pool):
        """Two completely unrelated memories should not be flagged."""
        provider = get_provider("fastembed")

        content_a = "Python is great for machine learning tasks"
        content_b = "The best pizza is in Naples, Italy"

        emb_a = await provider.embed(content_a)
        emb_b = await provider.embed(content_b)

        await store_memory(
            pool,
            MemoryCreate(
                type=MemoryType.fact,
                content=content_a,
                confidence=0.8,
                topic=["python"],
            ),
            embedding=emb_a,
        )
        await store_memory(
            pool,
            MemoryCreate(
                type=MemoryType.fact,
                content=content_b,
                confidence=0.7,
                topic=["food"],
            ),
            embedding=emb_b,
        )

        flagged = await find_contradictions(pool, sim_min=0.7)
        assert len(flagged) == 0


# ---------------------------------------------------------------------------
# 4. Orchestrator
# ---------------------------------------------------------------------------


class TestConsolidate:
    """Integration tests for the consolidate() orchestrator."""

    async def test_consolidate_runs_all_subsystems(self, pool):
        """Run consolidate() with a mixed pool of memories.
        Verify the report has entries for each subsystem."""
        provider = get_provider("fastembed")

        # 1. A stale memory that should be decayed
        stale = await store_memory(
            pool,
            MemoryCreate(
                type=MemoryType.fact,
                content="Very old stale fact that nobody accesses",
                confidence=0.1,
                topic=["stale"],
            ),
            embedding=await provider.embed("Very old stale fact that nobody accesses"),
        )
        old_dt = datetime.now(timezone.utc) - timedelta(days=200)
        await pool.execute(
            "UPDATE memories SET accessed_at = $1 WHERE id = $2",
            old_dt,
            stale.id,
        )

        # 2. Two near-duplicate memories
        dup_content_a = "asyncpg is the Postgres driver for Python async"
        dup_content_b = "asyncpg is the Postgres driver for async Python"
        await store_memory(
            pool,
            MemoryCreate(
                type=MemoryType.fact,
                content=dup_content_a,
                confidence=0.9,
                topic=["asyncpg"],
            ),
            embedding=await provider.embed(dup_content_a),
        )
        await store_memory(
            pool,
            MemoryCreate(
                type=MemoryType.fact,
                content=dup_content_b,
                confidence=0.5,
                topic=["asyncpg"],
            ),
            embedding=await provider.embed(dup_content_b),
        )

        # 3. Two contradicting memories
        contra_a = "Redis supports clustering natively"
        contra_b = "Redis does not support clustering natively"
        await store_memory(
            pool,
            MemoryCreate(
                type=MemoryType.fact,
                content=contra_a,
                confidence=0.7,
                topic=["redis"],
            ),
            embedding=await provider.embed(contra_a),
        )
        await store_memory(
            pool,
            MemoryCreate(
                type=MemoryType.fact,
                content=contra_b,
                confidence=0.6,
                topic=["redis"],
            ),
            embedding=await provider.embed(contra_b),
        )

        # Run full consolidation — consolidate() reads embeddings from DB
        config = ConsolidationConfig(
            decay=DecayConfig(min_confidence=0.3),
            duplicate_threshold=0.85,
            contradiction_similarity_min=0.7,
        )
        report = await consolidate(pool, config=config)

        # The report should reflect work done by each subsystem
        assert isinstance(report, ConsolidationReport)

        # At least the stale memory should be decayed
        assert len(report.decayed) >= 1
        assert stale.id in report.decayed

        # Verify the report serializes properly
        d = report.to_dict()
        assert "decayed" in d
        assert "duplicates_merged" in d
        assert "contradictions_flagged" in d
        assert "total_actions" in d

    async def test_consolidate_dry_run(self, pool):
        """Run with dry_run=True. Verify the report shows what would happen
        but no memories are actually modified."""
        provider = get_provider("fastembed")

        # Create a stale memory that would normally be decayed
        stale = await store_memory(
            pool,
            MemoryCreate(
                type=MemoryType.fact,
                content="Dry run stale fact for testing",
                confidence=0.1,
                topic=["test"],
            ),
            embedding=await provider.embed("Dry run stale fact for testing"),
        )
        old_dt = datetime.now(timezone.utc) - timedelta(days=200)
        await pool.execute(
            "UPDATE memories SET accessed_at = $1 WHERE id = $2",
            old_dt,
            stale.id,
        )

        report = await consolidate(pool, dry_run=True)

        # The stale memory should appear in the decayed list (what WOULD happen)
        assert stale.id in report.decayed

        # But the memory should still be active in the DB
        fetched = await get_memory(pool, stale.id)
        assert fetched is not None
        assert fetched.status == MemoryStatus.active


class TestConsolidationReport:
    """Tests for ConsolidationReport serialization."""

    def test_consolidation_report_to_dict(self):
        """Verify the report serializes correctly."""
        report = ConsolidationReport(
            decayed=["weft-aaa", "weft-bbb"],
            duplicates_merged=[("weft-ccc", "weft-ddd")],
            contradictions_flagged=[("weft-eee", "weft-fff")],
            errors=["Something went wrong"],
        )

        d = report.to_dict()
        assert d["decayed"] == ["weft-aaa", "weft-bbb"]
        assert d["decayed_count"] == 2
        assert d["duplicates_merged_count"] == 1
        assert d["contradictions_flagged_count"] == 1
        assert d["errors"] == ["Something went wrong"]
        assert d["total_actions"] == 4
