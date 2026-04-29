"""Tests for weft/credentials.py — token issuance, lookup, listing,
and revocation. L2 of Phase 2.5 (credential-bound caller mode).

Goes through the real DB (function-scoped ``pool`` fixture) so the
DB-side CHECK constraint and partial indexes participate in the
contract. No middleware integration here — that's L4.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from weft.credentials import (
    TOKEN_PREFIX,
    TokenRow,
    issue_token,
    list_tokens,
    lookup_token,
    revoke_token,
)


@pytest.mark.asyncio
async def test_issue_token_returns_plaintext_and_row(pool):
    plaintext, row = await issue_token(
        pool, user_id="u-1", caller_mode="supervisor", label="laptop",
    )

    assert plaintext.startswith(TOKEN_PREFIX), plaintext
    assert isinstance(row, TokenRow)
    assert row.user_id == "u-1"
    assert row.caller_mode == "supervisor"
    assert row.label == "laptop"
    assert row.revoked_at is None
    assert row.expires_at is None
    assert row.last_used_at is None
    # The hash on the row is sha256 of the plaintext, hex.
    assert row.token_hash == hashlib.sha256(plaintext.encode()).hexdigest()


@pytest.mark.asyncio
async def test_issued_tokens_are_unique_across_calls(pool):
    """secrets.token_urlsafe is RNG-backed; collisions are vanishingly
    unlikely but we want to surface a regression if someone swaps in a
    deterministic source."""
    seen: set[str] = set()
    for _ in range(50):
        plaintext, _ = await issue_token(pool, user_id="u-1", caller_mode="agent")
        assert plaintext not in seen
        seen.add(plaintext)


@pytest.mark.asyncio
async def test_lookup_round_trip(pool):
    plaintext, row = await issue_token(
        pool, user_id="u-2", caller_mode="agent", label="warp-runtime",
    )

    found = await lookup_token(pool, plaintext)

    assert found is not None
    assert found.token_hash == row.token_hash
    assert found.user_id == "u-2"
    assert found.caller_mode == "agent"
    assert found.label == "warp-runtime"


@pytest.mark.asyncio
async def test_lookup_bumps_last_used_at(pool):
    plaintext, _ = await issue_token(pool, user_id="u-3", caller_mode="supervisor")

    first = await lookup_token(pool, plaintext)
    assert first is not None
    assert first.last_used_at is not None

    # Sleep enough that NOW() advances on Postgres' microsecond clock.
    await asyncio.sleep(0.01)
    second = await lookup_token(pool, plaintext)
    assert second is not None
    assert second.last_used_at is not None
    assert second.last_used_at > first.last_used_at


@pytest.mark.asyncio
async def test_lookup_unknown_returns_none(pool):
    assert await lookup_token(pool, "weft-does-not-exist") is None


@pytest.mark.asyncio
async def test_lookup_empty_returns_none(pool):
    assert await lookup_token(pool, "") is None


@pytest.mark.asyncio
async def test_lookup_expired_returns_none(pool):
    plaintext, _ = await issue_token(
        pool,
        user_id="u-4",
        caller_mode="supervisor",
        # Already in the past — counts as expired even though the row
        # was just created.
        expires_in=timedelta(seconds=-1),
    )

    assert await lookup_token(pool, plaintext) is None


@pytest.mark.asyncio
async def test_lookup_revoked_returns_none(pool):
    plaintext, row = await issue_token(
        pool, user_id="u-5", caller_mode="agent",
    )
    assert await revoke_token(pool, row.token_hash) is True
    assert await lookup_token(pool, plaintext) is None


@pytest.mark.asyncio
async def test_revoke_is_idempotent(pool):
    _, row = await issue_token(pool, user_id="u-6", caller_mode="supervisor")

    assert await revoke_token(pool, row.token_hash) is True
    # Second call returns False because revoked_at is already set.
    assert await revoke_token(pool, row.token_hash) is False


@pytest.mark.asyncio
async def test_revoke_unknown_returns_false(pool):
    assert await revoke_token(pool, "deadbeef" * 8) is False


@pytest.mark.asyncio
async def test_revoke_empty_returns_false(pool):
    assert await revoke_token(pool, "") is False


@pytest.mark.asyncio
async def test_issue_token_rejects_invalid_caller_mode(pool):
    with pytest.raises(ValueError, match="caller_mode"):
        await issue_token(pool, user_id="u-7", caller_mode="root")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_issue_token_rejects_empty_user_id(pool):
    with pytest.raises(ValueError, match="user_id"):
        await issue_token(pool, user_id="", caller_mode="supervisor")


@pytest.mark.asyncio
async def test_list_tokens_default_excludes_revoked(pool):
    _, alive = await issue_token(pool, user_id="u-8", caller_mode="supervisor", label="a")
    _, dead = await issue_token(pool, user_id="u-8", caller_mode="agent", label="b")
    await revoke_token(pool, dead.token_hash)

    rows = await list_tokens(pool, "u-8")

    assert len(rows) == 1
    assert rows[0].token_hash == alive.token_hash


@pytest.mark.asyncio
async def test_list_tokens_include_revoked(pool):
    _, alive = await issue_token(pool, user_id="u-9", caller_mode="supervisor")
    _, dead = await issue_token(pool, user_id="u-9", caller_mode="agent")
    await revoke_token(pool, dead.token_hash)

    rows = await list_tokens(pool, "u-9", include_revoked=True)

    assert {r.token_hash for r in rows} == {alive.token_hash, dead.token_hash}


@pytest.mark.asyncio
async def test_list_tokens_scoped_to_user(pool):
    """A user's listing must NOT include another user's tokens — the
    middleware will lean on this for the per-user CLI listing in L5."""
    _, mine = await issue_token(pool, user_id="u-10", caller_mode="supervisor")
    await issue_token(pool, user_id="u-11", caller_mode="agent")

    rows = await list_tokens(pool, "u-10")

    assert [r.token_hash for r in rows] == [mine.token_hash]


@pytest.mark.asyncio
async def test_list_tokens_includes_expired_unrevoked(pool):
    """Expired-but-not-revoked rows still show up in the default
    listing — operators need to see what aged out without rummaging
    through include_revoked output."""
    _, expired = await issue_token(
        pool,
        user_id="u-12",
        caller_mode="supervisor",
        expires_in=timedelta(seconds=-1),
    )

    rows = await list_tokens(pool, "u-12")

    assert [r.token_hash for r in rows] == [expired.token_hash]


@pytest.mark.asyncio
async def test_list_tokens_orders_newest_first(pool):
    _, first = await issue_token(pool, user_id="u-13", caller_mode="supervisor")
    await asyncio.sleep(0.01)
    _, second = await issue_token(pool, user_id="u-13", caller_mode="agent")

    rows = await list_tokens(pool, "u-13")

    assert [r.token_hash for r in rows] == [second.token_hash, first.token_hash]


@pytest.mark.asyncio
async def test_token_row_is_active_predicate():
    """is_active mirrors the SQL filter; pure-Python check so the CLI
    can render status without re-querying."""
    now = datetime(2026, 4, 29, 12, 0, tzinfo=timezone.utc)

    live = TokenRow(
        token_hash="h", user_id="u", caller_mode="supervisor",
        created_at=now,
    )
    expired = TokenRow(
        token_hash="h", user_id="u", caller_mode="supervisor",
        created_at=now, expires_at=now - timedelta(seconds=1),
    )
    revoked = TokenRow(
        token_hash="h", user_id="u", caller_mode="supervisor",
        created_at=now, revoked_at=now,
    )

    assert live.is_active(now=now) is True
    assert expired.is_active(now=now) is False
    assert revoked.is_active(now=now) is False


@pytest.mark.asyncio
async def test_lookup_constant_after_first_use_does_not_change_other_columns(pool):
    plaintext, original = await issue_token(
        pool, user_id="u-14", caller_mode="agent", label="probe",
    )

    found = await lookup_token(pool, plaintext)
    assert found is not None

    # Hash, user, mode, label, created_at, expires_at, revoked_at all
    # come back identical — only last_used_at moves.
    assert found.token_hash == original.token_hash
    assert found.user_id == original.user_id
    assert found.caller_mode == original.caller_mode
    assert found.label == original.label
    assert found.created_at == original.created_at
    assert found.expires_at == original.expires_at
    assert found.revoked_at == original.revoked_at
