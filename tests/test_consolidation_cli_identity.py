"""Regression coverage for owner binding in CLI consolidation commands."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import asyncpg
import pytest
from click.testing import CliRunner

from weft.auth import current_user_id
from weft.db.connection import _pgvector_codec_init
from weft.cli import cli


@pytest.mark.parametrize(
    "args",
    [
        ["consolidate"],
        ["auto-consolidate", "--force"],
    ],
)
def test_consolidation_cli_binds_configured_owner(monkeypatch, args):
    """Both direct and scheduled CLI paths use the configured owner identity."""
    captured: dict[str, object] = {}
    pool = SimpleNamespace(close=AsyncMock())

    async def close_pool():
        captured["identity_during_close"] = current_user_id.get(None)

    pool.close.side_effect = close_pool

    async def create_pool(*args, **kwargs):
        return pool

    async def run_consolidation(*args, **kwargs):
        captured["identity_during_consolidation"] = current_user_id.get(None)
        return SimpleNamespace(
            decayed=[],
            duplicates_merged=[],
            contradictions_flagged=[],
            errors=["test report error"],
        )

    monkeypatch.setattr(
        "weft.cli.load_config",
        lambda: SimpleNamespace(
            database=SimpleNamespace(url="postgresql://unused"),
        ),
    )
    monkeypatch.setattr("asyncpg.create_pool", create_pool)
    monkeypatch.setattr("weft.config.user_identity.get_user_id", lambda: "cli-owner")
    monkeypatch.setattr("weft.consolidation.consolidate", run_consolidation)
    monkeypatch.setattr(
        "weft.consolidation.record_consolidation_run", AsyncMock(),
    )

    prior_token = current_user_id.set("caller-before-cli")
    try:
        result = CliRunner().invoke(cli, args)
    finally:
        current_user_id.reset(prior_token)

    assert result.exit_code == 0, result.output
    assert "test report error" in result.output
    assert captured["identity_during_consolidation"] == "cli-owner"
    assert captured["identity_during_close"] == "caller-before-cli"
    assert current_user_id.get(None) is None
    pool.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_consolidation_reports_missing_identity_relationship_skip(
    pool, pg_dsn, monkeypatch, caplog,
):
    """An unbound relationship write is refused and its pass skip is visible."""
    import logging

    import weft.consolidation as consolidation

    raw_pool = await asyncpg.create_pool(
        pg_dsn, min_size=1, max_size=2, init=_pgvector_codec_init,
    )

    async def duplicate_pass(*args, **kwargs):
        await consolidation._require_relationship_identity(raw_pool, "duplicate")
        return []

    monkeypatch.setattr(consolidation, "find_duplicates", duplicate_pass)
    caplog.set_level(logging.WARNING, logger="weft.consolidation")

    token = current_user_id.set(None)
    try:
        report = await consolidation.consolidate(raw_pool)
    finally:
        current_user_id.reset(token)
        await raw_pool.close()

    expected = (
        "Duplicate detection skipped: duplicate relationship write skipped: "
        "app.user_id is missing or invalid"
    )
    assert expected in report.errors
    assert report.duplicates_merged == []
    assert any(
        record.name == "weft.consolidation"
        and record.levelno == logging.WARNING
        and "consolidation.relationship_write_skipped" in record.getMessage()
        for record in caplog.records
    )
