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

from weft import credentials as credentials_module
from weft.credentials import (
    LEGACY_ENV_KEY_LABEL,
    TOKEN_PREFIX,
    TokenRow,
    bootstrap_legacy_api_key,
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


# --- L3: bootstrap legacy WEFT_API_KEY ---------------------------------


@pytest.fixture(autouse=True)
def _reset_legacy_warn_state():
    """Each test starts with a clean rate-limiter so the WARNING fires
    deterministically when we expect it."""
    credentials_module._legacy_last_warned_at = None
    yield
    credentials_module._legacy_last_warned_at = None


@pytest.mark.asyncio
async def test_bootstrap_inserts_row_for_legacy_env_key(pool):
    inserted = await bootstrap_legacy_api_key(
        pool, api_key="weft-legacy-secret", default_user_id="u-bootstrap",
    )
    assert inserted is True

    rows = await list_tokens(pool, "u-bootstrap")
    assert len(rows) == 1
    assert rows[0].label == LEGACY_ENV_KEY_LABEL
    assert rows[0].caller_mode == "supervisor"

    # And the hash matches what lookup_token would compute against the
    # same plaintext — that's the contract L4 will rely on.
    found = await lookup_token(pool, "weft-legacy-secret")
    assert found is not None
    assert found.token_hash == rows[0].token_hash


@pytest.mark.asyncio
async def test_bootstrap_is_idempotent(pool):
    first = await bootstrap_legacy_api_key(
        pool, api_key="weft-legacy-secret", default_user_id="u-bootstrap",
    )
    second = await bootstrap_legacy_api_key(
        pool, api_key="weft-legacy-secret", default_user_id="u-bootstrap",
    )

    assert first is True
    assert second is False  # already present, no insert

    rows = await list_tokens(pool, "u-bootstrap", include_revoked=True)
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_bootstrap_no_op_when_api_key_missing(pool):
    inserted = await bootstrap_legacy_api_key(
        pool, api_key=None, default_user_id="u-bootstrap",
    )
    assert inserted is False
    rows = await pool.fetch("SELECT 1 FROM weft_tokens")
    assert rows == []


@pytest.mark.asyncio
async def test_bootstrap_no_op_when_default_user_id_missing(pool, caplog):
    """Bootstrap can't write a row without a user_id (NOT NULL); it
    must skip and warn rather than crash startup."""
    import logging

    with caplog.at_level(logging.WARNING, logger="weft.credentials"):
        inserted = await bootstrap_legacy_api_key(
            pool, api_key="weft-legacy-secret", default_user_id=None,
        )

    assert inserted is False
    assert any("WEFT_DEFAULT_USER_ID" in r.message for r in caplog.records)
    rows = await pool.fetch("SELECT 1 FROM weft_tokens")
    assert rows == []


@pytest.mark.asyncio
async def test_bootstrap_does_not_clobber_existing_row(pool):
    """If an operator already minted a token whose plaintext happens
    to equal WEFT_API_KEY (deeply unlikely but worth pinning), the
    existing row wins via ON CONFLICT DO NOTHING — no caller_mode or
    label drift."""
    plaintext, original = await issue_token(
        pool, user_id="u-real", caller_mode="agent", label="real-token",
    )

    inserted = await bootstrap_legacy_api_key(
        pool, api_key=plaintext, default_user_id="u-bootstrap",
    )
    assert inserted is False

    found = await lookup_token(pool, plaintext)
    assert found is not None
    assert found.user_id == "u-real"
    assert found.caller_mode == "agent"
    assert found.label == "real-token"
    assert found.token_hash == original.token_hash


@pytest.mark.asyncio
async def test_lookup_emits_deprecation_warning_on_legacy_row(pool, caplog):
    import logging

    await bootstrap_legacy_api_key(
        pool, api_key="weft-legacy-secret", default_user_id="u-bootstrap",
    )

    with caplog.at_level(logging.WARNING, logger="weft.credentials"):
        found = await lookup_token(pool, "weft-legacy-secret")

    assert found is not None
    assert found.label == LEGACY_ENV_KEY_LABEL
    assert any("Legacy WEFT_API_KEY" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_lookup_warning_is_rate_limited(pool, caplog):
    """Two lookups in quick succession produce exactly one WARNING —
    the rate limiter caps at once per process per hour."""
    import logging

    await bootstrap_legacy_api_key(
        pool, api_key="weft-legacy-secret", default_user_id="u-bootstrap",
    )

    with caplog.at_level(logging.WARNING, logger="weft.credentials"):
        await lookup_token(pool, "weft-legacy-secret")
        await lookup_token(pool, "weft-legacy-secret")
        await lookup_token(pool, "weft-legacy-secret")

    legacy_warnings = [
        r for r in caplog.records if "Legacy WEFT_API_KEY" in r.message
    ]
    assert len(legacy_warnings) == 1


@pytest.mark.asyncio
async def test_lookup_does_not_warn_for_non_legacy_rows(pool, caplog):
    import logging

    plaintext, _ = await issue_token(
        pool, user_id="u-real", caller_mode="supervisor", label="warp-runtime",
    )

    with caplog.at_level(logging.WARNING, logger="weft.credentials"):
        await lookup_token(pool, plaintext)

    legacy_warnings = [
        r for r in caplog.records if "Legacy WEFT_API_KEY" in r.message
    ]
    assert legacy_warnings == []
