"""Tests for weft.consolidation — decay, dedup, contradiction, orchestrator."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from weft.consolidation import (
    IMMORTAL_TYPES,
    ConsolidationConfig,
    ConsolidationReport,
    DecayConfig,
    DedupResult,
    _content_conflicts,
    check_dedup_on_store,
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
from weft.store import get_memory, get_relationships, store_memory, update_memory


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
# 2b. Pre-insert dedup (check_dedup_on_store)
# ---------------------------------------------------------------------------


class TestCheckDedupOnStore:
    """Tests for the pre-insert dedup check in check_dedup_on_store."""

    async def test_no_match_stores_normally(self, pool):
        """When no similar memory exists, is_duplicate should be False."""
        provider = get_provider("fastembed")
        content = "Weft uses PostgreSQL with pgvector for semantic search"
        embedding = await provider.embed(content)

        result = await check_dedup_on_store(
            pool, content, embedding, new_confidence=0.7,
        )

        assert not result.is_duplicate
        assert result.action == "stored"
        assert result.existing_memory is None

    async def test_near_duplicate_deduplicates(self, pool):
        """Effectively identical content should return deduplicated action."""
        provider = get_provider("fastembed")
        content_a = "pgvector uses cosine distance to measure vector similarity in PostgreSQL"
        content_b = "pgvector uses cosine distance to measure vector similarity in PostgreSQL databases"

        emb_a = await provider.embed(content_a)
        emb_b = await provider.embed(content_b)

        # Store the first memory
        mem_a = await store_memory(
            pool,
            MemoryCreate(type=MemoryType.fact, content=content_a, confidence=0.7),
            embedding=emb_a,
        )

        # Check dedup with very similar content (same confidence, not longer)
        result = await check_dedup_on_store(
            pool, content_b, emb_b, new_confidence=0.7, threshold=0.85,
        )

        assert result.is_duplicate
        assert result.action == "deduplicated"
        assert result.existing_memory.id == mem_a.id

    async def test_substantive_update_revises(self, pool):
        """New content that is substantively longer should revise existing."""
        provider = get_provider("fastembed")
        content_short = "Redis is used for caching in the Weft memory system"
        content_long = (
            "Redis is used for caching in the Weft memory system. "
            "It provides L1 cache with TTL-based expiration and serves as the "
            "primary cache layer before falling back to PostgreSQL queries."
        )

        emb_short = await provider.embed(content_short)
        emb_long = await provider.embed(content_long)

        mem_short = await store_memory(
            pool,
            MemoryCreate(type=MemoryType.fact, content=content_short, confidence=0.6),
            embedding=emb_short,
        )

        result = await check_dedup_on_store(
            pool, content_long, emb_long, new_confidence=0.7, threshold=0.75,
        )

        assert result.is_duplicate
        assert result.action == "revised"
        assert result.existing_memory.id == mem_short.id
        # The existing memory should now have the longer content
        updated = await get_memory(pool, mem_short.id)
        assert updated.content == content_long
        assert updated.confidence == pytest.approx(0.7, abs=0.01)  # max(0.6, 0.7)

    async def test_higher_confidence_revises(self, pool):
        """Higher confidence new content should revise even if similar length."""
        provider = get_provider("fastembed")
        content_a = "FastEmbed generates embeddings locally using ONNX runtime models"
        content_b = "FastEmbed generates embeddings locally using ONNX runtime engine"

        emb_a = await provider.embed(content_a)
        emb_b = await provider.embed(content_b)

        mem_a = await store_memory(
            pool,
            MemoryCreate(type=MemoryType.fact, content=content_a, confidence=0.5),
            embedding=emb_a,
        )

        result = await check_dedup_on_store(
            pool, content_b, emb_b, new_confidence=0.8, threshold=0.85,
        )

        assert result.is_duplicate
        assert result.action == "revised"
        updated = await get_memory(pool, mem_a.id)
        assert updated.confidence == pytest.approx(0.8, abs=0.01)

    async def test_pinned_memory_not_revised(self, pool):
        """Pinned memories should never be revised — new content stores separately."""
        provider = get_provider("fastembed")
        content_a = "Weft project is owned by Jason Bauman and uses PostgreSQL"
        content_b = "Weft project is owned by Jason Bauman and uses PostgreSQL sixteen"

        emb_a = await provider.embed(content_a)
        emb_b = await provider.embed(content_b)

        await store_memory(
            pool,
            MemoryCreate(type=MemoryType.fact, content=content_a, confidence=0.9, pinned=True),
            embedding=emb_a,
        )

        result = await check_dedup_on_store(
            pool, content_b, emb_b, new_confidence=0.7, threshold=0.85,
        )

        assert not result.is_duplicate
        assert result.action == "stored"

    async def test_short_content_skips_check(self, pool):
        """Content shorter than 50 chars should skip dedup check."""
        provider = get_provider("fastembed")
        content = "short"
        embedding = await provider.embed(content)

        result = await check_dedup_on_store(
            pool, content, embedding, new_confidence=0.7,
        )

        assert not result.is_duplicate
        assert result.action == "stored"

    async def test_different_content_no_match(self, pool):
        """Very different content should not trigger dedup."""
        provider = get_provider("fastembed")
        content_a = "Python is a programming language used for artificial intelligence"
        content_b = "The weather forecast for Tokyo shows rain throughout the entire week"

        emb_a = await provider.embed(content_a)
        emb_b = await provider.embed(content_b)

        await store_memory(
            pool,
            MemoryCreate(type=MemoryType.fact, content=content_a, confidence=0.7),
            embedding=emb_a,
        )

        result = await check_dedup_on_store(
            pool, content_b, emb_b, new_confidence=0.7,
        )

        assert not result.is_duplicate
        assert result.action == "stored"

    async def test_dedup_result_to_dict(self):
        """DedupResult.to_dict() should include all relevant fields."""
        result = DedupResult(is_duplicate=False)
        d = result.to_dict()
        assert d == {"action": "stored", "is_duplicate": False}

        mem = _make_memory(content="test")
        result2 = DedupResult(
            is_duplicate=True,
            existing_memory=mem,
            similarity=0.95,
            action="revised",
        )
        d2 = result2.to_dict()
        assert d2["action"] == "revised"
        assert d2["is_duplicate"] is True
        assert d2["similarity"] == 0.95
        assert "existing_memory_id" in d2


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

    def test_complementary_memories_not_flagged(self):
        """Complementary memories about the same system should NOT be
        flagged as contradictions, even if one uses negation words
        or they contain different numbers."""
        # Module contracts (has "NEVER") vs key modules list (no negation)
        assert not _content_conflicts(
            "Module Contracts: graph/store.py — ONLY writer to Postgres. "
            "graph/cache.py — ONLY reader from Redis. "
            "db/migrations/ — NEVER modify existing files, only add new ones.",
            "Key Modules by Area: Orchestration: sweeper.py (claim TTL), "
            "retry.py (retry + DLQ), loop.py (main loop). "
            "Cloud: connections.py (Cloud SQL), secrets.py (GCP).",
        )

    def test_complementary_with_different_numbers_not_flagged(self):
        """Different phases/metrics in long memories should not conflict."""
        assert not _content_conflicts(
            "Phase 6 Dogfooding: Complete — 71 tasks done across 8 rounds. "
            "Loom built itself using 3 agents per round.",
            "Phase 7 Orchestration Reliability: Complete — 739 tests passing. "
            "6 streams merged: reset, escalation dedup, merge reliability.",
        )

    def test_same_metric_different_value_flagged(self):
        """Same metric with different values should be flagged."""
        assert _content_conflicts(
            "pgvector version is 0.7.0",
            "pgvector version is 0.8.1",
        )

    def test_short_negation_same_subject_flagged(self):
        """Short, focused claims with opposing negation should be flagged."""
        assert _content_conflicts(
            "Redis supports clustering natively",
            "Redis does not support clustering natively",
        )

    def test_negation_as_constraint_not_flagged(self):
        """Negation used as a rule/constraint should NOT conflict with
        a description that simply omits the constraint."""
        # "NEVER modify existing files" is a rule, not a contradiction of
        # something the other memory asserts
        assert not _content_conflicts(
            "Weft store.py is the ONLY Postgres writer. NEVER bypass store.py for writes.",
            "Weft architecture: store.py handles persistence, cache.py handles Redis, "
            "primer.py assembles context, consolidation.py runs maintenance.",
        )

    def test_architecture_and_conventions_not_flagged(self):
        """Architecture overview and coding conventions about the same system
        should NOT be flagged, even with high topic overlap."""
        assert not _content_conflicts(
            "Weft uses pgvector for semantic search. Embeddings are 384-dimensional "
            "vectors from fastembed. Redis caches memories with 1 hour TTL.",
            "Weft coding conventions: always use fastembed as default provider. "
            "store.py is the only module that writes to Postgres. "
            "Use testcontainers for integration tests.",
        )

    def test_different_aspects_with_incidental_negation_not_flagged(self):
        """Two memories about different aspects where one incidentally has
        negation should NOT be flagged."""
        assert not _content_conflicts(
            "The consolidation pipeline cannot run during active writes "
            "to prevent lock contention.",
            "The consolidation pipeline has three phases: decay scoring, "
            "near-duplicate detection, and contradiction detection.",
        )

    def test_true_contradiction_with_same_predicate(self):
        """Memories making opposite claims about the same predicate should be flagged."""
        assert _content_conflicts(
            "Weft uses OpenAI embeddings for semantic search",
            "Weft does not use OpenAI embeddings for semantic search",
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


# ---------------------------------------------------------------------------
# 5. Advisory lock
# ---------------------------------------------------------------------------


class TestConsolidationLock:
    """Tests for advisory-lock-based consolidation serialization."""

    async def test_lock_released_after_consolidation(self, pool):
        """Advisory lock should be released after consolidation completes."""
        from weft.consolidation import _CONSOLIDATION_LOCK_ID

        await consolidate(pool, dry_run=True)

        # Lock should be available now
        async with pool.acquire() as conn:
            locked = await conn.fetchval(
                "SELECT pg_try_advisory_lock($1)", _CONSOLIDATION_LOCK_ID,
            )
            assert locked, "Advisory lock should be available after consolidation"
            await conn.execute(
                "SELECT pg_advisory_unlock($1)", _CONSOLIDATION_LOCK_ID,
            )

    async def test_concurrent_consolidation_skips(self, pool):
        """A second concurrent consolidation should return skipped=True."""
        from weft.consolidation import _CONSOLIDATION_LOCK_ID

        # Hold the lock manually
        async with pool.acquire() as lock_conn:
            await lock_conn.execute(
                "SELECT pg_advisory_lock($1)", _CONSOLIDATION_LOCK_ID,
            )
            try:
                # Try to consolidate while lock is held
                report = await consolidate(pool, dry_run=True)
                assert report.skipped is True
                assert report.total_actions == 0
                assert report.to_dict()["skipped"] is True
            finally:
                await lock_conn.execute(
                    "SELECT pg_advisory_unlock($1)", _CONSOLIDATION_LOCK_ID,
                )

    async def test_concurrent_consolidation_no_conflict(self, pool):
        """Two concurrent consolidation calls should not raise."""
        results = await asyncio.gather(
            consolidate(pool, dry_run=True),
            consolidate(pool, dry_run=True),
        )
        # One should run, the other should be skipped
        skipped_count = sum(1 for r in results if r.skipped)
        assert skipped_count >= 1, "At least one concurrent run should be skipped"
