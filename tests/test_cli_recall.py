"""Regression tests for the semantic-recall CLI command.

The recall command must use Weft's canonical pool factory, bind the local
installation identity for RLS, and release resources on every post-creation
failure path.

Author:  User
Version: 0.2.0
Python:  >= 3.9
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from click.testing import CliRunner

from weft.auth import current_user_id
from weft.cli import cli


def _config() -> SimpleNamespace:
    """Build the minimum configuration consumed by the recall command."""
    return SimpleNamespace(
        database=SimpleNamespace(url="postgresql://unused"),
        embedding=SimpleNamespace(provider="test", model="test-model", dimensions=3),
    )


def _install_recall_mocks(monkeypatch, *, pool, provider, search, user_id="owner-a"):
    """Install command dependencies and return captured acquire state."""
    captured: dict[str, object] = {}
    entered = object()

    async def fake_create_pool(received_config):
        captured["config"] = received_config
        return pool

    @asynccontextmanager
    async def fake_acquire(received_pool):
        captured["acquire_pool"] = received_pool
        captured["bound_user_id"] = current_user_id.get()
        captured["acquire_entered"] = True
        try:
            yield entered
        finally:
            captured["acquire_exited"] = True

    async def close_pool():
        captured["close_user_id"] = current_user_id.get()

    if pool.close.side_effect is None:
        pool.close.side_effect = close_pool

    monkeypatch.setattr("weft.cli.load_config", _config)
    monkeypatch.setattr("weft.db.connection.create_pool", fake_create_pool)
    monkeypatch.setattr("weft.db.connection.acquire", fake_acquire)
    monkeypatch.setattr("weft.embeddings.get_provider", lambda *args, **kwargs: provider)
    monkeypatch.setattr("weft.store.search_by_vector", search)
    monkeypatch.setattr("weft.config.user_identity.get_user_id", lambda: user_id)
    return captured


def test_recall_binds_owner_and_closes_pool(monkeypatch):
    """Recall passes the resolved owner through an RLS-scoped transaction."""
    captured: dict[str, object] = {}

    async def close_pool():
        captured["close_user_id"] = current_user_id.get()

    pool = SimpleNamespace(close=AsyncMock(side_effect=close_pool))
    provider = SimpleNamespace(embed=AsyncMock(return_value=[0.1, 0.2, 0.3]))

    async def fake_search(received_pool, embedding, **kwargs):
        captured["pool"] = received_pool
        captured["embedding"] = embedding
        captured["kwargs"] = kwargs
        captured["search_user_id"] = current_user_id.get()
        return []

    helper_captured = _install_recall_mocks(
        monkeypatch, pool=pool, provider=provider, search=fake_search
    )
    helper_captured.update(captured)
    captured = helper_captured
    previous_user_id = current_user_id.set("outer-owner")
    try:
        result = CliRunner().invoke(
            cli, ["recall", "codec regression", "--limit", "3", "--topic", "weft"]
        )
    finally:
        current_user_id.reset(previous_user_id)

    assert result.exit_code == 0, result.output
    assert captured["config"].database.url == "postgresql://unused"
    assert captured["acquire_pool"] is pool
    assert captured["pool"] is pool
    assert captured["embedding"] == [0.1, 0.2, 0.3]
    assert captured["kwargs"] == {"limit": 3, "topic": "weft", "user_id": "owner-a"}
    assert captured["bound_user_id"] == "owner-a"
    assert captured["search_user_id"] == "owner-a"
    assert captured["acquire_entered"] is True
    assert captured["acquire_exited"] is True
    assert captured["close_user_id"] == "outer-owner"
    pool.close.assert_awaited_once_with()


def test_recall_closes_pool_and_resets_scope_when_provider_creation_fails(monkeypatch):
    """Provider-construction failures do not leak the pool or identity scope."""
    pool = SimpleNamespace(close=AsyncMock())
    search = AsyncMock()
    captured = _install_recall_mocks(
        monkeypatch, pool=pool, provider=None, search=search
    )
    provider_error = RuntimeError("provider unavailable")
    monkeypatch.setattr(
        "weft.embeddings.get_provider", Mock(side_effect=provider_error)
    )
    previous_user_id = current_user_id.set("outer-owner")
    try:
        result = CliRunner().invoke(cli, ["recall", "provider failure"])
    finally:
        current_user_id.reset(previous_user_id)

    assert result.exit_code != 0
    assert result.exception is provider_error
    assert captured.get("acquire_entered") is None
    assert captured["close_user_id"] == "outer-owner"
    pool.close.assert_awaited_once_with()
    search.assert_not_awaited()


def test_recall_closes_pool_and_resets_scope_when_embedding_fails(monkeypatch):
    """Embedding failures do not leak the pool or identity scope."""
    pool = SimpleNamespace(close=AsyncMock())
    provider = SimpleNamespace(embed=AsyncMock(side_effect=RuntimeError("embed failed")))
    search = AsyncMock()
    captured = _install_recall_mocks(
        monkeypatch, pool=pool, provider=provider, search=search
    )
    previous_user_id = current_user_id.set("outer-owner")
    try:
        result = CliRunner().invoke(cli, ["recall", "embedding failure"])
    finally:
        current_user_id.reset(previous_user_id)

    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)
    assert captured.get("acquire_entered") is None
    assert captured["close_user_id"] == "outer-owner"
    pool.close.assert_awaited_once_with()
    search.assert_not_awaited()


def test_recall_closes_pool_and_resets_scope_when_search_fails(monkeypatch):
    """Search failures exit the transaction and still close the pool."""
    pool = SimpleNamespace(close=AsyncMock())
    provider = SimpleNamespace(embed=AsyncMock(return_value=[0.1, 0.2, 0.3]))
    search_error = RuntimeError("search failed")
    search = AsyncMock(side_effect=search_error)
    captured = _install_recall_mocks(
        monkeypatch, pool=pool, provider=provider, search=search
    )
    previous_user_id = current_user_id.set("outer-owner")
    try:
        result = CliRunner().invoke(cli, ["recall", "search failure"])
    finally:
        current_user_id.reset(previous_user_id)

    assert result.exit_code != 0
    assert result.exception is search_error
    assert captured["bound_user_id"] == "owner-a"
    assert captured["acquire_exited"] is True
    assert captured["close_user_id"] == "outer-owner"
    pool.close.assert_awaited_once_with()


def test_recall_does_not_close_pool_when_creation_fails(monkeypatch):
    """Pool-construction failures occur before a closeable pool exists."""
    pool_error = RuntimeError("pool unavailable")
    create_pool = AsyncMock(side_effect=pool_error)
    get_user_id = Mock(return_value="owner-a")
    monkeypatch.setattr("weft.cli.load_config", _config)
    monkeypatch.setattr("weft.db.connection.create_pool", create_pool)
    monkeypatch.setattr("weft.config.user_identity.get_user_id", get_user_id)

    previous_user_id = current_user_id.set("outer-owner")
    try:
        result = CliRunner().invoke(cli, ["recall", "pool failure"])
    finally:
        current_user_id.reset(previous_user_id)

    assert result.exit_code != 0
    assert result.exception is pool_error
    create_pool.assert_awaited_once()
    assert create_pool.await_args.args[0].database.url == "postgresql://unused"
    get_user_id.assert_not_called()


# RUN COMMANDS
#
# uv run pytest tests/test_cli_recall.py -q
# Expected output: five passing recall lifecycle regression tests.
