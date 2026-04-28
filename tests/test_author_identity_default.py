"""Tests for the author_identity column default added in migration 37.

After migration 37, INSERTs into ``memories`` that don't specify
``author_identity`` get a structured value derived from the session's
``app.user_id``: ``local_user`` for normal sessions, ``system`` for the
SYSTEM_GLOBAL sentinel, ``unknown`` if no user_id is set.
"""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_default_author_identity_for_normal_user(pool):
    """The pool fixture sets app.user_id = 'test-user-default'."""
    await pool.execute(
        """INSERT INTO memories (id, type, topic, content, source, confidence,
           token_count, created_at, updated_at, accessed_at, access_count,
           status, pinned)
           VALUES ('mem-author-1', 'fact', '{}', 'hello', 'conversation',
           0.7, 1, now(), now(), now(), 0, 'active', false)"""
    )
    row = await pool.fetchrow(
        "SELECT author_identity FROM memories WHERE id = 'mem-author-1'"
    )
    payload = row["author_identity"]
    if isinstance(payload, str):
        import json
        payload = json.loads(payload)
    assert payload == {"kind": "local_user", "user_id": "test-user-default"}


@pytest.mark.asyncio
async def test_default_author_identity_for_system_global(pool):
    async with pool.acquire() as conn:
        await conn.execute("SET app.user_id = '__system_global_zathras__'")
        await conn.execute(
            """INSERT INTO memories (id, type, topic, content, source, confidence,
               token_count, created_at, updated_at, accessed_at, access_count,
               status, pinned)
               VALUES ('mem-author-sys', 'fact', '{}', 'seed', 'seed',
               0.9, 1, now(), now(), now(), 0, 'active', false)"""
        )
        row = await conn.fetchrow(
            "SELECT author_identity FROM memories WHERE id = 'mem-author-sys'"
        )
    payload = row["author_identity"]
    if isinstance(payload, str):
        import json
        payload = json.loads(payload)
    assert payload == {"kind": "system", "component": "runtime"}


@pytest.mark.asyncio
async def test_explicit_author_identity_overrides_default(pool):
    """Call sites can supply their own author_identity (e.g. an agent
    acting on a user's behalf) and the default is bypassed."""
    import json
    custom = {"kind": "agent", "agent_id": "wick", "on_behalf_of": "test-user-default"}
    await pool.execute(
        """INSERT INTO memories (id, type, topic, content, source, confidence,
           token_count, created_at, updated_at, accessed_at, access_count,
           status, pinned, author_identity)
           VALUES ('mem-author-agent', 'fact', '{}', 'agent wrote', 'conversation',
           0.7, 2, now(), now(), now(), 0, 'active', false, $1::jsonb)""",
        json.dumps(custom),
    )
    row = await pool.fetchrow(
        "SELECT author_identity FROM memories WHERE id = 'mem-author-agent'"
    )
    payload = row["author_identity"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    assert payload == custom
