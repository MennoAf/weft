"""Integration tests for the episode_turns store layer."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from weft.episode_turns import (
    append_turn,
    delete_turns_after_graduation,
    delete_turns_below_importance,
    delete_turns_for_graduated_episode,
    get_turn,
    list_turns,
    list_turns_in_range,
)
from weft.episodes import create_episode
from weft.models import EpisodeCreate, EpisodeTurnCreate, TurnRole


async def _make_episode(pool, *, title="test", **kwargs):
    return await create_episode(pool, EpisodeCreate(title=title, **kwargs))


# --- append_turn ---


async def test_append_first_turn_starts_at_index_zero(pool):
    ep = await _make_episode(pool)
    turn = await append_turn(
        pool, EpisodeTurnCreate(episode_id=ep.id, role=TurnRole.user, content="hi"),
    )
    assert turn.id.startswith("et-")
    assert turn.episode_id == ep.id
    assert turn.turn_index == 0
    assert turn.role == TurnRole.user
    assert turn.content == "hi"
    assert turn.token_count > 0
    assert turn.created_at is not None


async def test_append_assigns_sequential_indexes(pool):
    ep = await _make_episode(pool)
    t0 = await append_turn(
        pool, EpisodeTurnCreate(episode_id=ep.id, role=TurnRole.user, content="a"),
    )
    t1 = await append_turn(
        pool, EpisodeTurnCreate(episode_id=ep.id, role=TurnRole.assistant, content="b"),
    )
    t2 = await append_turn(
        pool, EpisodeTurnCreate(episode_id=ep.id, role=TurnRole.user, content="c"),
    )
    assert (t0.turn_index, t1.turn_index, t2.turn_index) == (0, 1, 2)


async def test_append_different_episodes_have_independent_indexes(pool):
    ep_a = await _make_episode(pool, title="A")
    ep_b = await _make_episode(pool, title="B")
    a0 = await append_turn(
        pool, EpisodeTurnCreate(episode_id=ep_a.id, role=TurnRole.user, content="hello A"),
    )
    b0 = await append_turn(
        pool, EpisodeTurnCreate(episode_id=ep_b.id, role=TurnRole.user, content="hello B"),
    )
    assert a0.turn_index == 0
    assert b0.turn_index == 0


async def test_append_with_embedding(pool):
    ep = await _make_episode(pool)
    embedding = [0.1] * 768
    t = await append_turn(
        pool,
        EpisodeTurnCreate(episode_id=ep.id, role=TurnRole.user, content="hi"),
        embedding=embedding,
    )
    fetched = await get_turn(pool, t.id)
    assert fetched is not None


async def test_append_with_explicit_occurred_at(pool):
    ep = await _make_episode(pool)
    when = datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)
    t = await append_turn(
        pool,
        EpisodeTurnCreate(
            episode_id=ep.id,
            role=TurnRole.user,
            content="historical",
            occurred_at=when,
        ),
    )
    assert t.occurred_at == when


async def test_append_with_trace_id(pool):
    ep = await _make_episode(pool)
    t = await append_turn(
        pool,
        EpisodeTurnCreate(
            episode_id=ep.id,
            role=TurnRole.assistant,
            content="response",
            trace_id="run-abc123",
        ),
    )
    assert t.trace_id == "run-abc123"


async def test_append_concurrent_writers_no_index_collision(pool):
    """Race-safety: 10 concurrent appenders to the same episode should yield
    10 distinct turn_indexes 0..9 with no UniqueViolationError leaking."""
    ep = await _make_episode(pool)
    creates = [
        EpisodeTurnCreate(episode_id=ep.id, role=TurnRole.user, content=f"msg {i}")
        for i in range(10)
    ]
    results = await asyncio.gather(*(append_turn(pool, c) for c in creates))
    indexes = sorted(t.turn_index for t in results)
    assert indexes == list(range(10))


# --- list_turns ---


async def test_list_turns_orders_by_index(pool):
    ep = await _make_episode(pool)
    for i in range(3):
        await append_turn(
            pool,
            EpisodeTurnCreate(
                episode_id=ep.id,
                role=TurnRole.user if i % 2 == 0 else TurnRole.assistant,
                content=f"turn {i}",
            ),
        )
    turns = await list_turns(pool, ep.id)
    assert [t.turn_index for t in turns] == [0, 1, 2]
    assert [t.content for t in turns] == ["turn 0", "turn 1", "turn 2"]


async def test_list_turns_respects_limit(pool):
    ep = await _make_episode(pool)
    for i in range(5):
        await append_turn(
            pool,
            EpisodeTurnCreate(
                episode_id=ep.id, role=TurnRole.user, content=f"t{i}",
            ),
        )
    turns = await list_turns(pool, ep.id, limit=2)
    assert len(turns) == 2
    assert turns[0].turn_index == 0


async def test_list_turns_empty_episode(pool):
    ep = await _make_episode(pool)
    turns = await list_turns(pool, ep.id)
    assert turns == []


# --- list_turns_in_range ---


async def test_list_turns_in_range_filters_by_time(pool):
    ep = await _make_episode(pool)
    base = datetime.now(timezone.utc)
    for i, offset_min in enumerate([-120, -60, -1]):
        await append_turn(
            pool,
            EpisodeTurnCreate(
                episode_id=ep.id,
                role=TurnRole.user,
                content=f"t{i}",
                occurred_at=base + timedelta(minutes=offset_min),
            ),
        )

    turns = await list_turns_in_range(
        pool,
        since=base - timedelta(minutes=90),
        until=base,
    )
    contents = [t.content for t in turns]
    assert "t0" not in contents  # -120 min is before since
    assert "t1" in contents      # -60 min is in range
    assert "t2" in contents      # -1 min is in range


async def test_list_turns_in_range_scopes_by_project(pool):
    ep_a = await _make_episode(pool, project_id="proj-a")
    ep_b = await _make_episode(pool, project_id="proj-b")
    when = datetime.now(timezone.utc) - timedelta(minutes=5)
    await append_turn(
        pool,
        EpisodeTurnCreate(
            episode_id=ep_a.id, role=TurnRole.user, content="A", occurred_at=when,
        ),
    )
    await append_turn(
        pool,
        EpisodeTurnCreate(
            episode_id=ep_b.id, role=TurnRole.user, content="B", occurred_at=when,
        ),
    )
    turns = await list_turns_in_range(
        pool,
        project_id="proj-a",
        since=when - timedelta(minutes=1),
        until=when + timedelta(minutes=1),
    )
    assert [t.content for t in turns] == ["A"]


# --- delete_turns_below_importance ---


async def test_delete_turns_below_importance_skips_null_scores(pool):
    """NULL importance_score (Face offline) is NOT pruned by this function."""
    ep = await _make_episode(pool)
    t = await append_turn(
        pool, EpisodeTurnCreate(episode_id=ep.id, role=TurnRole.user, content="x"),
    )
    assert t.importance_score is None
    await pool.execute(
        "UPDATE episodes SET status = 'graduated', "
        "ended_at = now() - interval '60 days' WHERE id = $1",
        ep.id,
    )
    deleted = await delete_turns_below_importance(pool, threshold=0.7, older_than_days=30)
    assert deleted == 0
    assert (await list_turns(pool, ep.id))  # still present


async def test_delete_turns_below_importance_drops_low_scores(pool):
    ep = await _make_episode(pool)
    keep = await append_turn(
        pool, EpisodeTurnCreate(episode_id=ep.id, role=TurnRole.user, content="keep"),
    )
    drop = await append_turn(
        pool, EpisodeTurnCreate(episode_id=ep.id, role=TurnRole.user, content="drop"),
    )
    await pool.execute(
        "UPDATE episode_turns SET importance_score = 0.9 WHERE id = $1", keep.id,
    )
    await pool.execute(
        "UPDATE episode_turns SET importance_score = 0.2 WHERE id = $1", drop.id,
    )
    await pool.execute(
        "UPDATE episodes SET status = 'graduated', "
        "ended_at = now() - interval '60 days' WHERE id = $1",
        ep.id,
    )
    deleted = await delete_turns_below_importance(pool, threshold=0.7, older_than_days=30)
    assert deleted == 1
    surviving = [t.id for t in await list_turns(pool, ep.id)]
    assert keep.id in surviving
    assert drop.id not in surviving


async def test_delete_turns_below_importance_respects_age(pool):
    """Recently-graduated episode shouldn't have turns pruned yet."""
    ep = await _make_episode(pool)
    drop = await append_turn(
        pool, EpisodeTurnCreate(episode_id=ep.id, role=TurnRole.user, content="drop"),
    )
    await pool.execute(
        "UPDATE episode_turns SET importance_score = 0.1 WHERE id = $1", drop.id,
    )
    # Graduated only 5 days ago — under the 30-day TTL.
    await pool.execute(
        "UPDATE episodes SET status = 'graduated', "
        "ended_at = now() - interval '5 days' WHERE id = $1",
        ep.id,
    )
    deleted = await delete_turns_below_importance(pool, threshold=0.7, older_than_days=30)
    assert deleted == 0


# --- delete_turns_for_graduated_episode (age-only fallback) ---


async def test_delete_turns_for_graduated_episode_drops_regardless_of_score(pool):
    ep = await _make_episode(pool)
    for i in range(3):
        await append_turn(
            pool,
            EpisodeTurnCreate(
                episode_id=ep.id, role=TurnRole.user, content=f"t{i}",
            ),
        )
    await pool.execute(
        "UPDATE episodes SET status = 'graduated', "
        "ended_at = now() - interval '60 days' WHERE id = $1",
        ep.id,
    )
    deleted = await delete_turns_for_graduated_episode(pool, older_than_days=30)
    assert deleted == 3
    assert await list_turns(pool, ep.id) == []


async def test_delete_turns_for_graduated_episode_preserves_open_episodes(pool):
    ep = await _make_episode(pool)
    await append_turn(
        pool, EpisodeTurnCreate(episode_id=ep.id, role=TurnRole.user, content="alive"),
    )
    deleted = await delete_turns_for_graduated_episode(pool, older_than_days=30)
    assert deleted == 0
    assert len(await list_turns(pool, ep.id)) == 1


# --- delete_turns_after_graduation ---


async def _graduate_episode_aged(pool, ep_id: str, *, days_ago: int) -> None:
    """Mark an episode graduated with ended_at set N days in the past."""
    await pool.execute(
        "UPDATE episodes SET status = 'graduated', "
        "ended_at = now() - ($1 || ' days')::interval "
        "WHERE id = $2",
        str(days_ago),
        ep_id,
    )


async def test_delete_turns_after_graduation_score_path(pool):
    """Scored path: low-score turns deleted, high-score turns retained after TTL."""
    ep = await _make_episode(pool)

    low = await append_turn(
        pool, EpisodeTurnCreate(episode_id=ep.id, role=TurnRole.user, content="low score"),
    )
    high = await append_turn(
        pool, EpisodeTurnCreate(episode_id=ep.id, role=TurnRole.assistant, content="high score"),
    )

    # Set importance scores.
    await pool.execute(
        "UPDATE episode_turns SET importance_score = 0.3 WHERE id = $1", low.id,
    )
    await pool.execute(
        "UPDATE episode_turns SET importance_score = 0.9 WHERE id = $1", high.id,
    )

    # Graduate the episode far enough in the past to exceed ttl_days_scored=90.
    await _graduate_episode_aged(pool, ep.id, days_ago=100)

    result = await delete_turns_after_graduation(
        pool,
        high_threshold=0.7,
        ttl_days_scored=90,
        ttl_days_no_score=30,
    )

    assert result["scored_deleted"] == 1
    assert result["no_score_deleted"] == 0

    surviving = [t.id for t in await list_turns(pool, ep.id)]
    assert high.id in surviving
    assert low.id not in surviving


async def test_delete_turns_after_graduation_no_score_fallback(pool):
    """No-score fallback: NULL-score turns deleted after ttl_days_no_score."""
    ep = await _make_episode(pool)

    t1 = await append_turn(
        pool, EpisodeTurnCreate(episode_id=ep.id, role=TurnRole.user, content="null score 1"),
    )
    t2 = await append_turn(
        pool, EpisodeTurnCreate(episode_id=ep.id, role=TurnRole.assistant, content="null score 2"),
    )
    # importance_score remains NULL (Face offline, default state).

    # Graduate the episode far enough in the past to exceed ttl_days_no_score=30.
    await _graduate_episode_aged(pool, ep.id, days_ago=40)

    result = await delete_turns_after_graduation(
        pool,
        high_threshold=0.7,
        ttl_days_scored=90,
        ttl_days_no_score=30,
    )

    assert result["no_score_deleted"] == 2
    assert result["scored_deleted"] == 0
    assert await list_turns(pool, ep.id) == []


async def test_delete_turns_after_graduation_mixed_scores(pool):
    """Mixed episode: scored-low + NULL each swept by their own policy; high-score retained."""
    ep = await _make_episode(pool)

    scored_low = await append_turn(
        pool, EpisodeTurnCreate(episode_id=ep.id, role=TurnRole.user, content="low"),
    )
    scored_high = await append_turn(
        pool, EpisodeTurnCreate(episode_id=ep.id, role=TurnRole.assistant, content="high"),
    )
    no_score = await append_turn(
        pool, EpisodeTurnCreate(episode_id=ep.id, role=TurnRole.user, content="null"),
    )

    await pool.execute(
        "UPDATE episode_turns SET importance_score = 0.2 WHERE id = $1", scored_low.id,
    )
    await pool.execute(
        "UPDATE episode_turns SET importance_score = 0.95 WHERE id = $1", scored_high.id,
    )
    # no_score.importance_score remains NULL.

    # Graduated 100 days ago — past both TTLs (scored=90, no-score=30).
    await _graduate_episode_aged(pool, ep.id, days_ago=100)

    result = await delete_turns_after_graduation(
        pool,
        high_threshold=0.7,
        ttl_days_scored=90,
        ttl_days_no_score=30,
    )

    assert result["scored_deleted"] == 1
    assert result["no_score_deleted"] == 1

    surviving = [t.id for t in await list_turns(pool, ep.id)]
    assert scored_high.id in surviving
    assert scored_low.id not in surviving
    assert no_score.id not in surviving
