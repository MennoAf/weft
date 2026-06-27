"""Tests for re-ask detection (weft/reask.py).

Coverage:
- (a) Duplicate within window: flagged as re-ask
- (b) Distinct query: NOT flagged
- (c) Duplicate outside window: NOT flagged
- Variants: time window, turn window, similarity threshold, dict vs QueryRow input
- DB integration: apply_reask_feedback raises usefulness_score + stamps miss row
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from weft.reask import QueryRow, compute_reask_rate, detect_reasked_queries


@pytest.fixture
def base_time() -> datetime:
    """Reference time for test queries."""
    return datetime(2026, 6, 17, 14, 0, 0)


class TestDetectReaskedQueriesBasic:
    """Core functionality: within-window duplicates, distinct queries, outside-window."""

    def test_duplicate_within_window_is_flagged(self, base_time):
        """Query nearly identical to an earlier one, within time window, should be flagged."""
        rows = [
            QueryRow(
                query_id="q1",
                query_text="what is the status of the deployment",
                created_at=base_time,
                tool_name="recall",
            ),
            QueryRow(
                query_id="q2",
                query_text="what is the status of the deployment",  # Exact duplicate
                created_at=base_time + timedelta(minutes=5),
                tool_name="recall",
            ),
        ]

        result = detect_reasked_queries(rows)
        assert len(result) == 1
        original, reask = result[0]
        assert original.query_id == "q1"
        assert reask.query_id == "q2"

    def test_near_duplicate_within_window_is_flagged(self, base_time):
        """Slightly different phrasing of the same query should be flagged."""
        rows = [
            QueryRow(
                query_id="q1",
                query_text="what is the deployment status",
                created_at=base_time,
                tool_name="recall",
            ),
            QueryRow(
                query_id="q2",
                query_text="what is the deployment status?",
                created_at=base_time + timedelta(minutes=10),
                tool_name="recall",
            ),
        ]

        result = detect_reasked_queries(rows, similarity_threshold=0.70)
        assert len(result) == 1
        assert result[0][0].query_id == "q1"
        assert result[0][1].query_id == "q2"

    def test_distinct_query_not_flagged(self, base_time):
        """Semantically different query should NOT be flagged as re-ask."""
        rows = [
            QueryRow(
                query_id="q1",
                query_text="what is the deployment status",
                created_at=base_time,
                tool_name="recall",
            ),
            QueryRow(
                query_id="q2",
                query_text="how do I configure the database",
                created_at=base_time + timedelta(minutes=5),
                tool_name="recall",
            ),
        ]

        result = detect_reasked_queries(rows)
        assert len(result) == 0

    def test_duplicate_outside_window_not_flagged(self, base_time):
        """Duplicate that is outside the time window should NOT be flagged."""
        rows = [
            QueryRow(
                query_id="q1",
                query_text="what is the deployment status",
                created_at=base_time,
                tool_name="recall",
            ),
            QueryRow(
                query_id="q2",
                query_text="what is the deployment status",
                created_at=base_time + timedelta(minutes=45),  # > 30 min default window
                tool_name="recall",
            ),
        ]

        result = detect_reasked_queries(rows)
        assert len(result) == 0

    def test_multiple_duplicates_detected(self, base_time):
        """Multiple re-ask pairs should all be detected."""
        rows = [
            QueryRow(
                query_id="q1",
                query_text="deployment status",
                created_at=base_time,
                tool_name="recall",
            ),
            QueryRow(
                query_id="q2",
                query_text="deployment status",
                created_at=base_time + timedelta(minutes=5),
                tool_name="recall",
            ),
            QueryRow(
                query_id="q3",
                query_text="migration issues",
                created_at=base_time + timedelta(minutes=10),
                tool_name="recall",
            ),
            QueryRow(
                query_id="q4",
                query_text="migration issues",
                created_at=base_time + timedelta(minutes=15),
                tool_name="recall",
            ),
        ]

        result = detect_reasked_queries(rows)
        assert len(result) == 2
        # Check pairs
        pair_ids = [(orig.query_id, reask.query_id) for orig, reask in result]
        assert ("q1", "q2") in pair_ids
        assert ("q3", "q4") in pair_ids


class TestDetectReaskedQueriesWindowParameters:
    """Test custom time and turn windows."""

    def test_custom_time_window(self, base_time):
        """Custom time_window_minutes should be respected."""
        rows = [
            QueryRow(
                query_id="q1",
                query_text="deployment status",
                created_at=base_time,
                tool_name="recall",
            ),
            QueryRow(
                query_id="q2",
                query_text="deployment status",
                created_at=base_time + timedelta(minutes=25),
                tool_name="recall",
            ),
        ]

        # With default 30-min window, should be flagged.
        result = detect_reasked_queries(rows)
        assert len(result) == 1

        # With 20-min window, should NOT be flagged.
        result = detect_reasked_queries(rows, time_window_minutes=20)
        assert len(result) == 0

    def test_turn_window_respects_max_turn_delta(self, base_time):
        """Queries with turn_index > max_turn_delta apart should NOT be flagged."""
        rows = [
            QueryRow(
                query_id="q1",
                query_text="deployment status",
                created_at=base_time,
                tool_name="recall",
                turn_index=10,
            ),
            QueryRow(
                query_id="q2",
                query_text="deployment status",
                created_at=base_time + timedelta(minutes=5),
                tool_name="recall",
                turn_index=15,
            ),
        ]

        # With max_turn_delta=3, they are 5 turns apart, so NOT flagged.
        result = detect_reasked_queries(rows, max_turn_delta=3)
        assert len(result) == 0

        # With max_turn_delta=10, they are 5 turns apart, so flagged.
        result = detect_reasked_queries(rows, max_turn_delta=10)
        assert len(result) == 1

    def test_turn_window_with_missing_turn_index(self, base_time):
        """Rows without turn_index are skipped when max_turn_delta is set."""
        rows = [
            QueryRow(
                query_id="q1",
                query_text="deployment status",
                created_at=base_time,
                tool_name="recall",
                turn_index=10,
            ),
            QueryRow(
                query_id="q2",
                query_text="deployment status",
                created_at=base_time + timedelta(minutes=5),
                tool_name="recall",
                turn_index=None,  # Missing turn_index
            ),
        ]

        # Should NOT be flagged because one row lacks turn_index.
        result = detect_reasked_queries(rows, max_turn_delta=10)
        assert len(result) == 0


class TestDetectReaskedQueriesSimilarityThreshold:
    """Test similarity_threshold parameter."""

    def test_high_similarity_threshold(self, base_time):
        """High threshold requires very similar text."""
        rows = [
            QueryRow(
                query_id="q1",
                query_text="what is the deployment status",
                created_at=base_time,
                tool_name="recall",
            ),
            QueryRow(
                query_id="q2",
                query_text="what is the status of deployment",
                created_at=base_time + timedelta(minutes=5),
                tool_name="recall",
            ),
        ]

        # With threshold=0.95, slight word order change not flagged.
        result = detect_reasked_queries(rows, similarity_threshold=0.95)
        assert len(result) == 0

        # With threshold=0.70, should be flagged.
        result = detect_reasked_queries(rows, similarity_threshold=0.70)
        assert len(result) == 1

    def test_case_insensitive_similarity(self, base_time):
        """Similarity check should be case-insensitive."""
        rows = [
            QueryRow(
                query_id="q1",
                query_text="What Is The Deployment Status",
                created_at=base_time,
                tool_name="recall",
            ),
            QueryRow(
                query_id="q2",
                query_text="what is the deployment status",
                created_at=base_time + timedelta(minutes=5),
                tool_name="recall",
            ),
        ]

        result = detect_reasked_queries(rows)
        assert len(result) == 1


class TestDetectReaskedQueriesDictInput:
    """Test that dicts (from asyncpg Records, ORM objects) are handled correctly."""

    def test_dict_input_converted_to_query_row(self, base_time):
        """Dicts should be converted to QueryRow internally."""
        rows = [
            {
                "query_id": "q1",
                "query_text": "deployment status",
                "created_at": base_time,
                "tool_name": "recall",
                "project_id": "proj-123",
            },
            {
                "query_id": "q2",
                "query_text": "deployment status",
                "created_at": base_time + timedelta(minutes=5),
                "tool_name": "recall",
                "project_id": "proj-123",
            },
        ]

        result = detect_reasked_queries(rows)
        assert len(result) == 1
        assert isinstance(result[0][0], QueryRow)
        assert isinstance(result[0][1], QueryRow)

    def test_dict_with_missing_optional_fields(self, base_time):
        """Dicts with only required fields should work."""
        rows = [
            {
                "query_id": "q1",
                "query_text": "deployment status",
                "created_at": base_time,
                "tool_name": "recall",
            },
            {
                "query_id": "q2",
                "query_text": "deployment status",
                "created_at": base_time + timedelta(minutes=5),
                "tool_name": "recall",
            },
        ]

        result = detect_reasked_queries(rows)
        assert len(result) == 1


class TestDetectReaskedQueriesEdgeCases:
    """Edge cases: empty input, single row, ordering."""

    def test_empty_input(self):
        """Empty list should return empty result."""
        result = detect_reasked_queries([])
        assert result == []

    def test_single_row(self, base_time):
        """Single row cannot be a re-ask."""
        rows = [
            QueryRow(
                query_id="q1",
                query_text="deployment status",
                created_at=base_time,
                tool_name="recall",
            ),
        ]

        result = detect_reasked_queries(rows)
        assert result == []

    def test_unordered_input_is_sorted(self, base_time):
        """Input not sorted by time should still work (function will sort internally)."""
        rows = [
            QueryRow(
                query_id="q2",
                query_text="deployment status",
                created_at=base_time + timedelta(minutes=5),
                tool_name="recall",
            ),
            QueryRow(
                query_id="q1",
                query_text="deployment status",
                created_at=base_time,
                tool_name="recall",
            ),
        ]

        result = detect_reasked_queries(rows)
        # Should detect re-ask even though input was out of order.
        assert len(result) == 1
        # Original should be q1 (earlier time).
        assert result[0][0].query_id == "q1"
        assert result[0][1].query_id == "q2"

    def test_three_queries_chain(self, base_time):
        """Query 3 duplicates query 2, which duplicates query 1."""
        rows = [
            QueryRow(
                query_id="q1",
                query_text="deployment status",
                created_at=base_time,
                tool_name="recall",
            ),
            QueryRow(
                query_id="q2",
                query_text="deployment status",
                created_at=base_time + timedelta(minutes=5),
                tool_name="recall",
            ),
            QueryRow(
                query_id="q3",
                query_text="deployment status",
                created_at=base_time + timedelta(minutes=10),
                tool_name="recall",
            ),
        ]

        result = detect_reasked_queries(rows)
        # All three are duplicates within window, so we should detect:
        # (q1, q2), (q1, q3), (q2, q3)
        assert len(result) == 3
        pair_ids = [(orig.query_id, reask.query_id) for orig, reask in result]
        assert ("q1", "q2") in pair_ids
        assert ("q1", "q3") in pair_ids
        assert ("q2", "q3") in pair_ids


# ---------------------------------------------------------------------------
# DB integration tests — require testcontainers (real Postgres via pool fixture)
# ---------------------------------------------------------------------------


class TestComputeReaskRate:
    """Test the PROOF metric: same-session re-ask rate computation.

    done_when for loom-6ce14333: compute_reask_rate() takes a seeded set
    of recall-query rows and returns the fraction of queries involved in
    re-asks. Pure function, reuses detect_reasked_queries() (L1 logic).
    """

    def test_single_reask_rate_two_queries(self, base_time):
        """1 re-ask out of 2 queries => 1.0 rate (both in pair)."""
        rows = [
            QueryRow(
                query_id="q1",
                query_text="deployment status",
                created_at=base_time,
                tool_name="recall",
            ),
            QueryRow(
                query_id="q2",
                query_text="deployment status",
                created_at=base_time + timedelta(minutes=5),
                tool_name="recall",
            ),
        ]

        rate = compute_reask_rate(rows)
        assert rate == 1.0, f"Expected 1.0, got {rate}"

    def test_single_reask_rate_four_queries(self, base_time):
        """1 re-ask pair out of 4 queries => 0.5 rate (2 of 4 in pair)."""
        rows = [
            QueryRow(
                query_id="q1",
                query_text="deployment status",
                created_at=base_time,
                tool_name="recall",
            ),
            QueryRow(
                query_id="q2",
                query_text="deployment status",
                created_at=base_time + timedelta(minutes=5),
                tool_name="recall",
            ),
            QueryRow(
                query_id="q3",
                query_text="migration issues",
                created_at=base_time + timedelta(minutes=10),
                tool_name="recall",
            ),
            QueryRow(
                query_id="q4",
                query_text="bug report",
                created_at=base_time + timedelta(minutes=15),
                tool_name="recall",
            ),
        ]

        rate = compute_reask_rate(rows)
        assert rate == 0.5, f"Expected 0.5, got {rate}"

    def test_no_reasks_rate_zero(self, base_time):
        """No re-asks => 0.0 rate."""
        rows = [
            QueryRow(
                query_id="q1",
                query_text="deployment status",
                created_at=base_time,
                tool_name="recall",
            ),
            QueryRow(
                query_id="q2",
                query_text="migration issues",
                created_at=base_time + timedelta(minutes=5),
                tool_name="recall",
            ),
            QueryRow(
                query_id="q3",
                query_text="bug report",
                created_at=base_time + timedelta(minutes=10),
                tool_name="recall",
            ),
        ]

        rate = compute_reask_rate(rows)
        assert rate == 0.0, f"Expected 0.0, got {rate}"

    def test_multiple_reask_pairs(self, base_time):
        """Multiple re-ask pairs: 2 pairs out of 4 queries => 1.0 rate."""
        rows = [
            QueryRow(
                query_id="q1",
                query_text="deployment status",
                created_at=base_time,
                tool_name="recall",
            ),
            QueryRow(
                query_id="q2",
                query_text="deployment status",
                created_at=base_time + timedelta(minutes=5),
                tool_name="recall",
            ),
            QueryRow(
                query_id="q3",
                query_text="migration issues",
                created_at=base_time + timedelta(minutes=10),
                tool_name="recall",
            ),
            QueryRow(
                query_id="q4",
                query_text="migration issues",
                created_at=base_time + timedelta(minutes=15),
                tool_name="recall",
            ),
        ]

        rate = compute_reask_rate(rows)
        assert rate == 1.0, f"Expected 1.0 (all 4 queries in 2 pairs), got {rate}"

    def test_empty_rows_rate_zero(self):
        """Empty input => 0.0 rate."""
        rate = compute_reask_rate([])
        assert rate == 0.0

    def test_single_row_rate_zero(self, base_time):
        """Single row => 0.0 rate (no re-ask possible)."""
        rows = [
            QueryRow(
                query_id="q1",
                query_text="deployment status",
                created_at=base_time,
                tool_name="recall",
            ),
        ]

        rate = compute_reask_rate(rows)
        assert rate == 0.0

    def test_reask_outside_window_not_counted(self, base_time):
        """Re-ask outside time window should not be counted."""
        rows = [
            QueryRow(
                query_id="q1",
                query_text="deployment status",
                created_at=base_time,
                tool_name="recall",
            ),
            QueryRow(
                query_id="q2",
                query_text="migration issues",
                created_at=base_time + timedelta(minutes=10),
                tool_name="recall",
            ),
            QueryRow(
                query_id="q3",
                query_text="deployment status",  # Duplicate of q1, but outside window
                created_at=base_time + timedelta(minutes=45),
                tool_name="recall",
            ),
        ]

        rate = compute_reask_rate(rows, time_window_minutes=30)
        assert rate == 0.0, f"Expected 0.0 (re-ask outside window), got {rate}"

    def test_dict_input_support(self, base_time):
        """Dicts should work as input (like asyncpg Records)."""
        rows = [
            {
                "query_id": "q1",
                "query_text": "deployment status",
                "created_at": base_time,
                "tool_name": "recall",
            },
            {
                "query_id": "q2",
                "query_text": "deployment status",
                "created_at": base_time + timedelta(minutes=5),
                "tool_name": "recall",
            },
            {
                "query_id": "q3",
                "query_text": "other query",
                "created_at": base_time + timedelta(minutes=10),
                "tool_name": "recall",
            },
        ]

        rate = compute_reask_rate(rows)
        # q1 and q2 form a pair, so 2/3 queries are in re-asks
        assert abs(rate - 2.0 / 3.0) < 1e-6, f"Expected ~0.667, got {rate}"

    def test_custom_similarity_threshold(self, base_time):
        """Custom threshold should affect rate."""
        rows = [
            QueryRow(
                query_id="q1",
                query_text="what is the deployment status",
                created_at=base_time,
                tool_name="recall",
            ),
            QueryRow(
                query_id="q2",
                query_text="what is the status of deployment",
                created_at=base_time + timedelta(minutes=5),
                tool_name="recall",
            ),
            QueryRow(
                query_id="q3",
                query_text="other query",
                created_at=base_time + timedelta(minutes=10),
                tool_name="recall",
            ),
        ]

        # With high threshold, no re-ask detected => 0.0
        rate_high = compute_reask_rate(rows, similarity_threshold=0.95)
        assert rate_high == 0.0

        # With lower threshold, re-ask detected => 2/3
        rate_low = compute_reask_rate(rows, similarity_threshold=0.70)
        assert abs(rate_low - 2.0 / 3.0) < 1e-6


class TestReaskFeedbackIntegration:
    """Wire re-ask detection into usefulness feedback (DB-backed).

    done_when for loom-c5a6e9e3: when a re-ask is detected, the
    usefulness_score of the memory that satisfied the second query rises
    (via record_feedback EMA), and the miss is recorded as a hard-negative.
    """

    async def test_usefulness_score_rises_on_satisfying_memory(self, pool):
        """Core done_when assertion: satisfying memory usefulness_score increases
        after apply_reask_feedback is called, and the missed query row is stamped
        as is_reask_miss=TRUE with the satisfying_memory_id recorded.
        """
        from weft.models import MemoryCreate, MemoryType
        from weft.store import (
            apply_reask_feedback,
            get_memory,
            log_recall_query,
            store_memory,
        )

        # 1. Create a memory that will be the "satisfying" answer.
        mem = await store_memory(
            pool,
            MemoryCreate(
                type=MemoryType.fact,
                content="The deployment runs on Fly.io",
                topic=["deployment", "infra"],
                confidence=0.8,
            ),
        )
        initial_score = (await get_memory(pool, mem.id)).usefulness_score

        # 2. Log the "original" (missed) query and the re-ask query.
        await log_recall_query(pool, tool_name="recall", query_text="where does the app run")
        rows = await pool.fetch(
            "SELECT query_id FROM weft_recall_queries WHERE query_text = 'where does the app run'"
        )
        assert len(rows) == 1
        missed_query_id = rows[0]["query_id"]

        await log_recall_query(pool, tool_name="recall", query_text="where does the app run")

        # 3. Apply re-ask feedback: boost satisfying memory, stamp miss row.
        result = await apply_reask_feedback(pool, missed_query_id, mem.id)

        # 4. usefulness_score on the satisfying memory must have risen.
        updated_mem = await get_memory(pool, mem.id)
        assert updated_mem.usefulness_score > initial_score, (
            f"Expected usefulness_score > {initial_score}, got {updated_mem.usefulness_score}"
        )

        # result dict from record_feedback should reflect the new score
        assert result["memory_id"] == mem.id
        assert result["usefulness_score"] > initial_score

        # 5. The missed query row must be stamped as a hard-negative.
        miss_row = await pool.fetchrow(
            "SELECT is_reask_miss, reask_satisfying_memory_id "
            "FROM weft_recall_queries WHERE query_id = $1",
            missed_query_id,
        )
        assert miss_row["is_reask_miss"] is True
        assert miss_row["reask_satisfying_memory_id"] == mem.id

    async def test_ema_alpha_applied_correctly(self, pool):
        """EMA formula: new = 0.3 * 1.0 + 0.7 * old (alpha=0.3, helpful=True).

        Verifies the math matches record_feedback's EMA contract so we know
        the boost came from the right path, not a bespoke reimplementation.
        """
        from weft.models import MemoryCreate, MemoryType
        from weft.store import apply_reask_feedback, get_memory, log_recall_query, store_memory

        mem = await store_memory(
            pool,
            MemoryCreate(
                type=MemoryType.fact,
                content="Weft is built on asyncpg",
                topic=["weft"],
                confidence=0.5,
            ),
        )
        old_score = (await get_memory(pool, mem.id)).usefulness_score

        await log_recall_query(pool, tool_name="recall", query_text="what does weft use for db")
        rows = await pool.fetch(
            "SELECT query_id FROM weft_recall_queries "
            "WHERE query_text = 'what does weft use for db'"
        )
        missed_query_id = rows[0]["query_id"]

        result = await apply_reask_feedback(pool, missed_query_id, mem.id)

        expected_score = 0.3 * 1.0 + 0.7 * old_score
        expected_score = max(0.0, min(1.0, expected_score))
        assert abs(result["usefulness_score"] - expected_score) < 1e-6, (
            f"Expected EMA score {expected_score}, got {result['usefulness_score']}"
        )

    async def test_apply_reask_feedback_is_idempotent(self, pool):
        """Calling apply_reask_feedback twice on the same pair boosts the EMA exactly once.

        done_when for loom-16adf67a (P1): the second call must be a no-op —
        usefulness_score after call 2 must equal usefulness_score after call 1.
        """
        from weft.models import MemoryCreate, MemoryType
        from weft.store import apply_reask_feedback, get_memory, log_recall_query, store_memory

        mem = await store_memory(
            pool,
            MemoryCreate(
                type=MemoryType.fact,
                content="Idempotency test memory",
                topic=["test"],
                confidence=0.7,
            ),
        )

        await log_recall_query(pool, tool_name="recall", query_text="idempotency test query abc")
        rows = await pool.fetch(
            "SELECT query_id FROM weft_recall_queries "
            "WHERE query_text = 'idempotency test query abc'"
        )
        missed_query_id = rows[0]["query_id"]

        # First call: should apply the EMA boost and return a result dict.
        result1 = await apply_reask_feedback(pool, missed_query_id, mem.id)
        assert result1 is not None, "First call should return a result dict"
        score_after_call1 = (await get_memory(pool, mem.id)).usefulness_score

        # Second call on the SAME (missed_query_id, satisfying_memory_id) pair:
        # must be a no-op — returns None and leaves the score unchanged.
        result2 = await apply_reask_feedback(pool, missed_query_id, mem.id)
        assert result2 is None, "Second call should return None (idempotent no-op)"

        score_after_call2 = (await get_memory(pool, mem.id)).usefulness_score
        assert score_after_call2 == score_after_call1, (
            f"Score must not change on second call: "
            f"after call 1 = {score_after_call1}, after call 2 = {score_after_call2}"
        )

    async def test_get_recent_recall_queries_excludes_already_stamped(self, pool):
        """get_recent_recall_queries should not return rows already marked as misses.

        Once a miss is stamped, future scheduler passes should not re-process it.
        """
        from weft.models import MemoryCreate, MemoryType
        from weft.store import (
            apply_reask_feedback,
            get_recent_recall_queries,
            log_recall_query,
            store_memory,
        )

        mem = await store_memory(
            pool,
            MemoryCreate(
                type=MemoryType.fact,
                content="Sentinel memory for exclusion test",
                topic=["test"],
                confidence=0.7,
            ),
        )

        # Log two identical queries (re-ask scenario).
        await log_recall_query(pool, tool_name="recall", query_text="sentinel query xyz")
        rows = await pool.fetch(
            "SELECT query_id FROM weft_recall_queries WHERE query_text = 'sentinel query xyz'"
        )
        missed_query_id = rows[0]["query_id"]

        # Before stamping: both rows are in the recent window.
        before = await get_recent_recall_queries(pool, window_minutes=60)
        before_ids = [r["query_id"] for r in before]
        assert missed_query_id in before_ids

        # Stamp the miss.
        await apply_reask_feedback(pool, missed_query_id, mem.id)

        # After stamping: the missed row should be excluded.
        after = await get_recent_recall_queries(pool, window_minutes=60)
        after_ids = [r["query_id"] for r in after]
        assert missed_query_id not in after_ids


class TestReaskFeedbackLoopBody:
    """Tests for the scheduler loop body (_run_reask_feedback_pass).

    The loop body is factored as a standalone coroutine so it can be called
    in isolation here without an infinite asyncio loop. These tests seed the
    database with a re-ask scenario and assert that calling one pass of the
    loop boosts the satisfying memory's usefulness_score.
    """

    async def test_loop_pass_boosts_satisfying_memory(self, pool):
        """One pass of the re-ask feedback loop boosts the satisfying memory.

        done_when assertion (loom-7d8e4d69): after _run_reask_feedback_pass
        runs over a seeded re-ask pair, the satisfying memory's usefulness_score
        must be higher than before the pass ran.
        """
        import asyncio

        from weft.models import MemoryCreate, MemoryType
        from weft.scheduler import _run_reask_feedback_pass
        from weft.store import get_memory, log_recall_query, store_memory

        # 1. Create a memory and touch it so accessed_at is fresh.
        #    This makes it the "most recently accessed" memory for the
        #    satisfying_memory_id proxy lookup.
        mem = await store_memory(
            pool,
            MemoryCreate(
                type=MemoryType.fact,
                content="The scheduler loop test memory — loop pass boost",
                topic=["test", "scheduler"],
                confidence=0.8,
            ),
        )
        # Touch accessed_at so this memory is "recent" at query time.
        await pool.execute(
            "UPDATE memories SET accessed_at = now() WHERE id = $1",
            mem.id,
        )
        initial_score = (await get_memory(pool, mem.id)).usefulness_score

        # 2. Seed two nearly-identical queries (original + re-ask) so
        #    detect_reasked_queries fires on this pass.
        await log_recall_query(
            pool,
            tool_name="recall",
            query_text="loop pass test query scheduler unique",
        )
        # Tiny sleep so created_at strictly orders the two rows.
        await asyncio.sleep(0.01)
        await log_recall_query(
            pool,
            tool_name="recall",
            query_text="loop pass test query scheduler unique",
        )

        # 3. Run one pass of the loop body.
        processed = await _run_reask_feedback_pass(pool)

        # 4. At least one pair should have been processed this pass.
        assert processed >= 1, (
            f"Expected at least 1 re-ask pair processed, got {processed}"
        )

        # 5. The satisfying memory's usefulness_score must have risen.
        updated_score = (await get_memory(pool, mem.id)).usefulness_score
        assert updated_score > initial_score, (
            f"Expected usefulness_score > {initial_score} after loop pass, "
            f"got {updated_score}"
        )

    async def test_loop_pass_is_idempotent(self, pool):
        """Running _run_reask_feedback_pass twice on the same re-ask is a no-op on second pass.

        apply_reask_feedback is idempotent (stamps is_reask_miss=TRUE), so the
        second pass must return 0 newly processed pairs and leave the score unchanged.
        """
        import asyncio

        from weft.models import MemoryCreate, MemoryType
        from weft.scheduler import _run_reask_feedback_pass
        from weft.store import get_memory, log_recall_query, store_memory

        mem = await store_memory(
            pool,
            MemoryCreate(
                type=MemoryType.fact,
                content="Idempotent scheduler loop test memory unique abc",
                topic=["test"],
                confidence=0.7,
            ),
        )
        await pool.execute(
            "UPDATE memories SET accessed_at = now() WHERE id = $1",
            mem.id,
        )

        await log_recall_query(
            pool,
            tool_name="recall",
            query_text="idempotent loop pass test unique zyx",
        )
        await asyncio.sleep(0.01)
        await log_recall_query(
            pool,
            tool_name="recall",
            query_text="idempotent loop pass test unique zyx",
        )

        # First pass: should process the pair and boost the score.
        processed_first = await _run_reask_feedback_pass(pool)
        assert processed_first >= 1

        score_after_first = (await get_memory(pool, mem.id)).usefulness_score

        # Second pass: original query row is now stamped, so detect_reasked_queries
        # won't see it (get_recent_recall_queries excludes is_reask_miss=TRUE rows).
        # Processed count should be 0 for the already-processed pair.
        processed_second = await _run_reask_feedback_pass(pool)
        score_after_second = (await get_memory(pool, mem.id)).usefulness_score

        assert score_after_second == score_after_first, (
            f"Score changed on second pass: {score_after_first} → {score_after_second}"
        )

    async def test_reask_pass_does_not_bleed_across_users(self, pool):
        """Per-user isolation (loom-fdd9282a): a system-context re-ask pass must
        never attribute one user's re-ask to another user's memory.

        Setup is adversarial: user B owns the globally-most-recently-accessed
        memory, so a single unscoped global pass (the old behavior) would boost
        B's memory for A's re-ask. The fan-out scopes every lookup to the owning
        user, so A's own memory is boosted and B's is left untouched.
        """
        from weft.models import MemoryCreate, MemoryType
        from weft.scheduler import _run_reask_feedback_pass
        from weft.store import get_memory, store_memory

        USER_A = "user-aaaa-bleed"
        USER_B = "user-bbbb-bleed"

        # A's memory, accessed slightly in the past (a valid satisfying proxy).
        mem_a = await store_memory(
            pool,
            MemoryCreate(
                type=MemoryType.fact,
                content="User A memory for cross-user bleed test",
                topic=["test"],
                confidence=0.8,
            ),
        )
        await pool.execute(
            "UPDATE memories SET user_id = $1, accessed_at = now() - interval '6 seconds' WHERE id = $2",
            USER_A,
            mem_a.id,
        )

        # B's memory, accessed MORE recently than A's — the bait. A global
        # ORDER BY accessed_at DESC lookup would wrongly select this for A.
        mem_b = await store_memory(
            pool,
            MemoryCreate(
                type=MemoryType.fact,
                content="User B memory for cross-user bleed test",
                topic=["test"],
                confidence=0.8,
            ),
        )
        await pool.execute(
            "UPDATE memories SET user_id = $1, accessed_at = now() WHERE id = $2",
            USER_B,
            mem_b.id,
        )

        score_a_before = (await get_memory(pool, mem_a.id)).usefulness_score
        score_b_before = (await get_memory(pool, mem_b.id)).usefulness_score

        # A re-asks: two identical queries owned by user A, within the window.
        await pool.execute(
            "INSERT INTO weft_recall_queries (query_id, user_id, tool_name, query_text, created_at) "
            "VALUES ($1, $2, 'recall', $3, now() - interval '10 seconds')",
            "rq-bleed-a-1",
            USER_A,
            "cross user bleed test query unique",
        )
        await pool.execute(
            "INSERT INTO weft_recall_queries (query_id, user_id, tool_name, query_text, created_at) "
            "VALUES ($1, $2, 'recall', $3, now() - interval '5 seconds')",
            "rq-bleed-a-2",
            USER_A,
            "cross user bleed test query unique",
        )

        processed = await _run_reask_feedback_pass(pool)
        assert processed >= 1, "A's re-ask pair should have been processed"

        score_a_after = (await get_memory(pool, mem_a.id)).usefulness_score
        score_b_after = (await get_memory(pool, mem_b.id)).usefulness_score

        assert score_a_after > score_a_before, (
            "User A's own memory should be boosted by A's re-ask"
        )
        assert score_b_after == score_b_before, (
            "User B's memory must NOT be touched by A's re-ask — cross-user "
            f"bleed detected: {score_b_before} → {score_b_after}"
        )

    async def test_get_recent_recall_queries_scope_to_user_isolates(self, pool):
        """get_recent_recall_queries(scope_to_user=True) returns only the named
        user's rows even in a system context that may bypass RLS."""
        from weft.store import get_recent_recall_queries

        await pool.execute(
            "INSERT INTO weft_recall_queries (query_id, user_id, tool_name, query_text) "
            "VALUES ($1, $2, 'recall', $3)",
            "rq-scope-a",
            "user-scope-a",
            "scope isolation query a",
        )
        await pool.execute(
            "INSERT INTO weft_recall_queries (query_id, user_id, tool_name, query_text) "
            "VALUES ($1, $2, 'recall', $3)",
            "rq-scope-b",
            "user-scope-b",
            "scope isolation query b",
        )

        a_rows = await get_recent_recall_queries(
            pool, window_minutes=60, user_id="user-scope-a", scope_to_user=True
        )
        a_ids = {r["query_id"] for r in a_rows}
        assert "rq-scope-a" in a_ids
        assert "rq-scope-b" not in a_ids, "user A's scoped fetch leaked user B's row"
