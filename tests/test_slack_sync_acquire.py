"""Regression tests for slack/sync's acquire() integration.

Before this fix, ``_store_message_memory`` and ``_archive_memory``
called ``store_memory`` / ``pool.execute`` directly without an
``acquire()`` wrapper. Slack sync runs from a background scheduler
task, so the request-scoped ``current_user_id`` contextvar was never
set by HTTP middleware — the migration-34 NOT NULL on
``memories.user_id`` tripped on every Slack-ingested message and the
errors only surfaced as per-message ``logger.warning`` lines in prod.

These tests pin the fix: the scheduler loop sets ``current_user_id``
to ``WEFT_DEFAULT_USER_ID`` once at startup, and the per-message
write/archive helpers wrap their DB ops in ``acquire()``.
"""

from __future__ import annotations

import pytest

from weft.auth import current_user_id
from weft.slack.config import ChannelMapping
from weft.slack.parser import SlackMessage
from weft.slack.sync import ChannelInfo, _archive_memory, _store_message_memory
from weft.models import MemoryType


_DEFAULT_USER = "slack-sync-default-user"


def _msg(ts: str = "1700000000.000001", text: str = "hello world") -> SlackMessage:
    return SlackMessage(text=text, user="U001", ts=ts)


@pytest.fixture
def channel():
    return ChannelInfo(id="C001", name="general")


@pytest.fixture
def mapping():
    return ChannelMapping(
        memory_type=MemoryType.fact, topics=["slack"], confidence=0.7,
    )


@pytest.mark.asyncio
async def test_store_message_memory_stamps_user_id(pool, channel, mapping):
    """The fix: acquire() inside _store_message_memory issues SET LOCAL
    so the row's user_id matches the contextvar (which the scheduler
    loop sets to WEFT_DEFAULT_USER_ID at startup)."""
    tok = current_user_id.set(_DEFAULT_USER)
    try:
        ids = await _store_message_memory(
            _msg(ts="1700000000.000123", text="slack regression note"),
            channel,
            mapping,
            pool,
            embedding_provider=None,
            user_names={"U001": "alice"},
        )
    finally:
        current_user_id.reset(tok)

    assert len(ids) == 1
    row = await pool.fetchrow(
        "SELECT user_id, status FROM memories WHERE id = $1", ids[0],
    )
    assert row["user_id"] == _DEFAULT_USER
    assert row["status"] == "active"


@pytest.mark.asyncio
async def test_archive_memory_runs_under_acquire_scope(pool, channel, mapping):
    """_archive_memory used to run pool.execute directly. With RLS
    enforced under a non-superuser app role (production), the WHERE
    clause would silently match zero rows. acquire() puts the right
    app.user_id GUC in place so the UPDATE finds the row."""
    tok = current_user_id.set(_DEFAULT_USER)
    try:
        ids = await _store_message_memory(
            _msg(ts="1700000000.000456", text="to-be-archived"),
            channel,
            mapping,
            pool,
            embedding_provider=None,
            user_names={"U001": "alice"},
        )
        await _archive_memory(pool, ids[0])
    finally:
        current_user_id.reset(tok)

    row = await pool.fetchrow(
        "SELECT status FROM memories WHERE id = $1", ids[0],
    )
    assert row["status"] == "archived"
