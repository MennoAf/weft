"""Supabase paused-project auto-restore (weft/supabase.py + connection.create_pool).

Free-tier Supabase projects auto-pause after ~1 week idle and then refuse
connections at the socket layer. ``create_pool`` intercepts that failure for
Supabase DSNs and either auto-restores via the Management API (when an
account-scoped token is configured) or upgrades the raw socket error into an
actionable message (when it is not).

These tests mock the network boundary — ``asyncpg.create_pool`` and the
``weft.supabase`` Management-API helpers — so nothing hits Supabase. No DB
container needed.
"""

from __future__ import annotations

import pytest

import weft.db.connection as connection
import weft.supabase as supabase
from weft.config import DatabaseConfig, WeftConfig

# (async tests run under pytest-asyncio's auto mode — no per-test marker needed)

# 20-char lowercase ref, the Supabase project-ref shape.
_REF = "abcdefghijklmnopqrst"
_DIRECT_DSN = f"postgresql://postgres:pw@db.{_REF}.supabase.co:5432/postgres"
_POOLER_DSN = f"postgresql://postgres.{_REF}:pw@aws-0-us-east-1.pooler.supabase.com:6543/postgres"
_LOCAL_DSN = "postgresql://weft:weft_local@localhost:5433/weft"


def _config(dsn: str, *, token: str | None = None) -> WeftConfig:
    return WeftConfig(
        database=DatabaseConfig(url=dsn),
        supabase_access_token=token,
    )


def _fake_create_pool(*, failures: int):
    """asyncpg.create_pool stub: raise ConnectionRefusedError `failures` times, then succeed."""
    state = {"calls": 0}
    sentinel = object()

    async def _cp(dsn, **kwargs):
        state["calls"] += 1
        if state["calls"] <= failures:
            raise ConnectionRefusedError("connection refused (paused project)")
        return sentinel

    return _cp, state, sentinel


# ----------------------------------------------------------------------
# DSN parsing
# ----------------------------------------------------------------------


def test_extract_project_ref_direct():
    assert supabase.extract_project_ref(_DIRECT_DSN) == _REF


def test_extract_project_ref_pooler():
    assert supabase.extract_project_ref(_POOLER_DSN) == _REF


def test_extract_project_ref_non_supabase_is_none():
    assert supabase.extract_project_ref(_LOCAL_DSN) is None


def test_is_supabase_dsn():
    assert supabase.is_supabase_dsn(_DIRECT_DSN) is True
    assert supabase.is_supabase_dsn(_POOLER_DSN) is True
    assert supabase.is_supabase_dsn(_LOCAL_DSN) is False


# ----------------------------------------------------------------------
# create_pool — interception behavior
# ----------------------------------------------------------------------


async def test_non_supabase_failure_reraises_unchanged(monkeypatch):
    """A non-Supabase connection failure must propagate as-is (no restore path)."""
    cp, _, _ = _fake_create_pool(failures=99)
    monkeypatch.setattr(connection.asyncpg, "create_pool", cp)

    with pytest.raises(ConnectionRefusedError):
        await connection.create_pool(_config(_LOCAL_DSN))


async def test_paused_without_token_raises_actionable(monkeypatch):
    """Paused Supabase project + no token → clear error, never an auto-restore."""
    cp, _, _ = _fake_create_pool(failures=99)
    monkeypatch.setattr(connection.asyncpg, "create_pool", cp)

    called = {"restore": False}

    async def _no_restore(*a, **k):
        called["restore"] = True
        return True

    monkeypatch.setattr(supabase, "restore_project", _no_restore)

    with pytest.raises(ConnectionError) as ei:
        await connection.create_pool(_config(_DIRECT_DSN, token=None))

    assert "SUPABASE_ACCESS_TOKEN" in str(ei.value)
    assert _REF in str(ei.value)
    assert called["restore"] is False  # opt-in only: never restores without a token


async def test_restores_and_retries(monkeypatch):
    """Paused project + token → restore, wait, retry, return the pool."""
    cp, state, sentinel = _fake_create_pool(failures=1)
    monkeypatch.setattr(connection.asyncpg, "create_pool", cp)

    seen = {}

    async def _restore(ref, tok):
        seen["restore"] = (ref, tok)
        return True

    async def _wait(ref, tok, **k):
        seen["wait"] = (ref, tok)
        return True

    monkeypatch.setattr(supabase, "restore_project", _restore)
    monkeypatch.setattr(supabase, "wait_for_restore", _wait)

    pool = await connection.create_pool(_config(_DIRECT_DSN, token="sbp_tok"))

    assert pool is sentinel
    assert state["calls"] == 2  # failed once, retried once
    assert seen["restore"] == (_REF, "sbp_tok")
    assert seen["wait"][0] == _REF


async def test_restore_request_rejected_raises(monkeypatch):
    """If the Management API rejects the restore, surface a clear error."""
    cp, _, _ = _fake_create_pool(failures=99)
    monkeypatch.setattr(connection.asyncpg, "create_pool", cp)

    async def _restore(ref, tok):
        return False

    async def _wait(*a, **k):  # must not be reached
        raise AssertionError("wait_for_restore should not run after a rejected restore")

    monkeypatch.setattr(supabase, "restore_project", _restore)
    monkeypatch.setattr(supabase, "wait_for_restore", _wait)

    with pytest.raises(ConnectionError) as ei:
        await connection.create_pool(_config(_DIRECT_DSN, token="sbp_tok"))
    assert "Failed to restore" in str(ei.value)


async def test_restore_accepted_but_times_out_raises(monkeypatch):
    """Restore accepted but the project never comes healthy → timeout error."""
    cp, _, _ = _fake_create_pool(failures=99)
    monkeypatch.setattr(connection.asyncpg, "create_pool", cp)

    async def _restore(ref, tok):
        return True

    async def _wait(*a, **k):
        return False

    monkeypatch.setattr(supabase, "restore_project", _restore)
    monkeypatch.setattr(supabase, "wait_for_restore", _wait)

    with pytest.raises(ConnectionError) as ei:
        await connection.create_pool(_config(_DIRECT_DSN, token="sbp_tok"))
    assert "did not become available" in str(ei.value)
