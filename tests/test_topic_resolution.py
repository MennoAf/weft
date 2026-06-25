"""Integration tests for weft/topic_resolution.py — L1 Resolution Ratchet.

done_when assertions:
  (1) naive resolve('weft') returns {'weft', 'entity:Weft'}.
  (2) when a topic_resolution_aliases row exists for a token, resolve()
        returns its resolved_tags and the alias path is consulted BEFORE
        naive normalization.
  (3) consulting an alias bumps hit_count and increments a counter named
        topic_resolution.alias_hits.
  (4) record_alias(token, tags, source='manual') upserts a row.

Tests are self-isolating: each seeds rows under a UNIQUE per-test user_id
(uuid-derived string) and asserts on those rows only.  topic_resolution_aliases
is NOT in the conftest TRUNCATE list, so we clean up ourselves or use
unique user_ids to avoid cross-test interference.
"""

from __future__ import annotations

import uuid

import pytest

from weft.counters import get_counter
from weft.topic_resolution import (
    COUNTER_TOPIC_RESOLUTION_ALIAS_HITS,
    record_alias,
    resolve_topic,
)


def _test_user() -> str:
    """Return a per-test unique user_id (safe for app.user_id SET LOCAL)."""
    return f"test-tr-{uuid.uuid4().hex}"


# ---------------------------------------------------------------------------
# (1) Naive normalization: resolve('weft') → {'weft', 'entity:Weft'}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_naive_resolve_weft(pool):
    """resolve_topic('weft', ...) with no alias row returns naive tags.

    Expected: {'weft', 'entity:Weft'} — lower + Title-case entity tag.
    """
    user_id = _test_user()
    result = await resolve_topic("weft", user_id, pool)
    assert set(result) == {"weft", "entity:Weft"}, (
        f"Naive resolve('weft') should return {{'weft', 'entity:Weft'}}, got {result}"
    )


# ---------------------------------------------------------------------------
# (2) Alias path consulted BEFORE naive normalization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_alias_path_consulted_before_naive(pool):
    """When an alias row exists for a token, resolve() returns resolved_tags.

    Inserts an alias row for token='weft' with a custom resolved_tags set,
    then calls resolve_topic('weft', ...) and asserts the alias tags are
    returned — NOT the naive normalization.
    """
    user_id = _test_user()
    custom_tags = ["custom:weft", "tool:weft", "entity:Weft"]

    # Insert alias row directly so we can verify the lookup path
    await record_alias("weft", custom_tags, "manual", user_id, pool)

    result = await resolve_topic("weft", user_id, pool)
    assert set(result) == set(custom_tags), (
        f"Expected alias tags {custom_tags!r}, got {result!r}"
    )
    # Confirm the alias result is NOT the naive normalization
    naive = {"weft", "entity:Weft"}
    # The result has additional tags beyond naive — proves alias path was taken
    assert set(result) != naive or len(result) != len(naive), (
        "resolve_topic returned only naive tags, alias path was not taken"
    )


@pytest.mark.asyncio
async def test_alias_returned_not_naive_when_alias_exists(pool):
    """Alias row with tags that differ from naive normalization wins.

    Uses tags that differ completely from ['weft', 'entity:Weft'] so the
    assertion is unambiguous: if naive were returned the test would fail.
    """
    user_id = _test_user()
    overriding_tags = ["project:weft-memory", "arch:backend"]

    await record_alias("weft", overriding_tags, "manual", user_id, pool)

    result = await resolve_topic("weft", user_id, pool)
    assert set(result) == set(overriding_tags), (
        f"Expected alias-override tags {overriding_tags!r}, got {result!r}"
    )


# ---------------------------------------------------------------------------
# (3) Alias hit bumps hit_count AND increments counter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_alias_hit_bumps_hit_count(pool):
    """Calling resolve_topic() when an alias exists increments hit_count by 1."""
    user_id = _test_user()
    tags = ["weft", "entity:Weft"]

    await record_alias("weft", tags, "manual", user_id, pool)

    # Verify initial hit_count is 0
    initial = await pool.fetchval(
        """
        SELECT hit_count FROM topic_resolution_aliases
         WHERE user_id = $1 AND topic_token = 'weft'
        """,
        user_id,
    )
    assert initial == 0, f"Expected hit_count=0 before first resolve, got {initial}"

    await resolve_topic("weft", user_id, pool)

    after_one = await pool.fetchval(
        """
        SELECT hit_count FROM topic_resolution_aliases
         WHERE user_id = $1 AND topic_token = 'weft'
        """,
        user_id,
    )
    assert after_one == 1, f"Expected hit_count=1 after one resolve, got {after_one}"

    # A second resolve bumps to 2
    await resolve_topic("weft", user_id, pool)
    after_two = await pool.fetchval(
        """
        SELECT hit_count FROM topic_resolution_aliases
         WHERE user_id = $1 AND topic_token = 'weft'
        """,
        user_id,
    )
    assert after_two == 2, f"Expected hit_count=2 after two resolves, got {after_two}"


@pytest.mark.asyncio
async def test_alias_hit_increments_counter(pool):
    """Calling resolve_topic() when an alias exists increments topic_resolution.alias_hits."""
    user_id = _test_user()
    tags = ["weft", "entity:Weft"]

    await record_alias("weft", tags, "manual", user_id, pool)

    before = await get_counter(pool, COUNTER_TOPIC_RESOLUTION_ALIAS_HITS)

    await resolve_topic("weft", user_id, pool)

    after = await get_counter(pool, COUNTER_TOPIC_RESOLUTION_ALIAS_HITS)
    assert after == before + 1, (
        f"Expected counter to increment by 1 ({before} → {before + 1}), got {after}"
    )


@pytest.mark.asyncio
async def test_naive_resolve_does_not_increment_counter(pool):
    """resolve_topic() with NO alias row does NOT increment the alias_hits counter."""
    user_id = _test_user()

    before = await get_counter(pool, COUNTER_TOPIC_RESOLUTION_ALIAS_HITS)

    # No alias row for this user — must fall through to naive
    result = await resolve_topic("weft", user_id, pool)

    after = await get_counter(pool, COUNTER_TOPIC_RESOLUTION_ALIAS_HITS)

    # Counter must not change
    assert after == before, (
        f"Counter should not change on naive resolve (was {before}, now {after})"
    )
    # And the result is naive
    assert set(result) == {"weft", "entity:Weft"}


# ---------------------------------------------------------------------------
# (4) record_alias upserts a row
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_alias_inserts_row(pool):
    """record_alias(token, tags, source='manual') creates a row in the table."""
    user_id = _test_user()
    tags = ["weft", "entity:Weft", "product:weft"]

    await record_alias("weft", tags, "manual", user_id, pool)

    row = await pool.fetchrow(
        """
        SELECT topic_token, resolved_tags, source, hit_count
          FROM topic_resolution_aliases
         WHERE user_id = $1 AND topic_token = 'weft'
        """,
        user_id,
    )
    assert row is not None, "record_alias did not insert a row"
    assert row["topic_token"] == "weft"
    assert set(row["resolved_tags"]) == set(tags)
    assert row["source"] == "manual"
    assert row["hit_count"] == 0


@pytest.mark.asyncio
async def test_record_alias_upserts_on_conflict(pool):
    """record_alias on the same (user_id, token) updates resolved_tags and source."""
    user_id = _test_user()
    initial_tags = ["weft"]
    updated_tags = ["weft", "entity:Weft", "added:later"]

    await record_alias("weft", initial_tags, "manual", user_id, pool)
    await record_alias("weft", updated_tags, "learned", user_id, pool)

    row = await pool.fetchrow(
        """
        SELECT resolved_tags, source
          FROM topic_resolution_aliases
         WHERE user_id = $1 AND topic_token = 'weft'
        """,
        user_id,
    )
    assert row is not None
    assert set(row["resolved_tags"]) == set(updated_tags), (
        f"Expected updated_tags {updated_tags!r}, got {list(row['resolved_tags'])!r}"
    )
    assert row["source"] == "learned", (
        f"Expected source='learned' after upsert, got {row['source']!r}"
    )


@pytest.mark.asyncio
async def test_record_alias_normalizes_token_case(pool):
    """record_alias normalizes the token to lowercase before storing."""
    user_id = _test_user()

    # Insert with mixed-case token
    await record_alias("Weft", ["weft", "entity:Weft"], "manual", user_id, pool)

    row = await pool.fetchrow(
        """
        SELECT topic_token FROM topic_resolution_aliases
         WHERE user_id = $1 AND topic_token = 'weft'
        """,
        user_id,
    )
    assert row is not None, (
        "record_alias should store token 'Weft' normalized to 'weft'"
    )
    assert row["topic_token"] == "weft"


@pytest.mark.asyncio
async def test_record_alias_different_users_isolated(pool):
    """Two users can have alias rows for the same token without conflict."""
    user_a = _test_user()
    user_b = _test_user()

    tags_a = ["a:tag"]
    tags_b = ["b:tag"]

    await record_alias("shared-token", tags_a, "manual", user_a, pool)
    await record_alias("shared-token", tags_b, "manual", user_b, pool)

    row_a = await pool.fetchrow(
        "SELECT resolved_tags FROM topic_resolution_aliases WHERE user_id=$1 AND topic_token='shared-token'",
        user_a,
    )
    row_b = await pool.fetchrow(
        "SELECT resolved_tags FROM topic_resolution_aliases WHERE user_id=$1 AND topic_token='shared-token'",
        user_b,
    )

    assert row_a is not None and set(row_a["resolved_tags"]) == set(tags_a)
    assert row_b is not None and set(row_b["resolved_tags"]) == set(tags_b)
