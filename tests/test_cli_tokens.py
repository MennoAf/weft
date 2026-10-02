"""Tests for `weft tokens issue / list / revoke` (Phase 2.5 L5).

Drives the Click commands through CliRunner with WEFT_DATABASE_URL
pointed at the testcontainer Postgres so the commands talk to the
same DB the rest of the test suite uses.
"""

from __future__ import annotations

import json
import re

import pytest
from click.testing import CliRunner

from weft.cli import cli


@pytest.fixture
def runner(monkeypatch, pool, pg_container):
    """Click runner with WEFT_DATABASE_URL set to the testcontainer DSN.

    The pool fixture has already migrated the DB and TRUNCATEd it, so
    each test starts clean. The CLI commands open their own short-lived
    pool against the same URL — close-on-finish keeps that simple."""
    dsn = pg_container.get_connection_url().replace("+psycopg2", "")
    monkeypatch.setenv("WEFT_DATABASE_URL", dsn)
    return CliRunner()


def _extract_token(output: str) -> str:
    match = re.search(r"^Token:\s+(\S+)$", output, flags=re.MULTILINE)
    assert match, f"no Token: line in output:\n{output}"
    return match.group(1)


def _extract_hash(output: str) -> str:
    match = re.search(r"^Hash:\s+(\S+)$", output, flags=re.MULTILINE)
    assert match, f"no Hash: line in output:\n{output}"
    return match.group(1)


def test_export_defaults_to_caller_scope(runner, monkeypatch):
    """The packaged CLI must not perform a process-wide export by default."""
    captured = {}

    async def fake_export(pool, **kwargs):
        captured.update(kwargs)
        return json.dumps({"memories": [], "count": 0})

    monkeypatch.setenv("WEFT_USER_ID", "cli-owner")
    monkeypatch.setattr("weft.exporter.export_memories", fake_export)

    result = runner.invoke(cli, ["export", "--format", "json"])

    assert result.exit_code == 0, result.output
    assert captured["user_id"] == "cli-owner"


def test_export_all_requires_explicit_flag(runner, monkeypatch):
    """The all-user scope is only selected by the explicit --all flag."""
    captured = {}

    async def fake_export(pool, **kwargs):
        captured.update(kwargs)
        return json.dumps({"memories": [], "count": 0})

    monkeypatch.setattr("weft.exporter.export_memories", fake_export)
    result = runner.invoke(cli, ["export", "--format", "json", "--all"])

    assert result.exit_code == 0, result.output
    assert captured["user_id"] is None


def test_issue_supervisor_token_prints_plaintext_once(runner):
    result = runner.invoke(
        cli,
        ["tokens", "issue", "--user-id", "u-cli", "--mode", "supervisor", "--label", "face"],
    )
    assert result.exit_code == 0, result.output
    token = _extract_token(result.output)
    assert token.startswith("weft-")
    assert "Mode:  supervisor" in result.output
    assert "Label: face" in result.output
    assert "store this token now" in result.output.lower()


def test_issue_agent_token_with_expiry(runner):
    result = runner.invoke(
        cli,
        [
            "tokens", "issue",
            "--user-id", "u-cli",
            "--mode", "agent",
            "--label", "wick-runtime",
            "--expires-in", "7d",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Mode:  agent" in result.output
    assert "Expires:" in result.output


def test_issue_rejects_bad_mode(runner):
    result = runner.invoke(
        cli,
        ["tokens", "issue", "--user-id", "u-cli", "--mode", "root"],
    )
    assert result.exit_code != 0
    assert "Invalid value for '--mode'" in result.output


def test_issue_rejects_bad_expires_in(runner):
    result = runner.invoke(
        cli,
        [
            "tokens", "issue",
            "--user-id", "u-cli",
            "--mode", "supervisor",
            "--expires-in", "foo",
        ],
    )
    assert result.exit_code != 0
    assert "expires-in" in result.output.lower()


def test_list_empty_when_no_tokens(runner):
    result = runner.invoke(cli, ["tokens", "list", "--user-id", "u-cli"])
    assert result.exit_code == 0, result.output
    assert "No tokens found." in result.output


def test_list_shows_issued_tokens_in_table(runner):
    runner.invoke(
        cli,
        ["tokens", "issue", "--user-id", "u-cli", "--mode", "supervisor", "--label", "face"],
    )
    runner.invoke(
        cli,
        ["tokens", "issue", "--user-id", "u-cli", "--mode", "agent", "--label", "wick"],
    )

    result = runner.invoke(cli, ["tokens", "list", "--user-id", "u-cli"])
    assert result.exit_code == 0, result.output
    # Header present.
    assert "HASH" in result.output and "MODE" in result.output
    # Both labels visible.
    assert "face" in result.output
    assert "wick" in result.output
    # Both rows are active.
    active_lines = [l for l in result.output.splitlines() if "active" in l]
    assert len(active_lines) == 2


def test_list_default_excludes_revoked(runner):
    issue = runner.invoke(
        cli,
        ["tokens", "issue", "--user-id", "u-cli", "--mode", "agent", "--label", "ephemeral"],
    )
    token_hash = _extract_hash(issue.output)
    runner.invoke(cli, ["tokens", "revoke", token_hash])

    result = runner.invoke(cli, ["tokens", "list", "--user-id", "u-cli"])
    assert result.exit_code == 0, result.output
    assert "No tokens found." in result.output


def test_list_include_revoked_shows_revoked(runner):
    issue = runner.invoke(
        cli,
        ["tokens", "issue", "--user-id", "u-cli", "--mode", "agent", "--label", "ephemeral"],
    )
    token_hash = _extract_hash(issue.output)
    runner.invoke(cli, ["tokens", "revoke", token_hash])

    result = runner.invoke(
        cli, ["tokens", "list", "--user-id", "u-cli", "--include-revoked"],
    )
    assert result.exit_code == 0, result.output
    assert "ephemeral" in result.output
    assert "revoked" in result.output


def test_revoke_flips_status(runner):
    issue = runner.invoke(
        cli,
        ["tokens", "issue", "--user-id", "u-cli", "--mode", "supervisor"],
    )
    token_hash = _extract_hash(issue.output)

    revoke = runner.invoke(cli, ["tokens", "revoke", token_hash])
    assert revoke.exit_code == 0, revoke.output
    assert token_hash[:12] in revoke.output


def test_revoke_unknown_hash_reports_no_match(runner):
    bogus = "0" * 64
    result = runner.invoke(cli, ["tokens", "revoke", bogus])
    assert result.exit_code == 0, result.output
    assert "No live token matched" in result.output


def test_revoke_rejects_short_hash(runner):
    result = runner.invoke(cli, ["tokens", "revoke", "abc123"])
    assert result.exit_code != 0
    assert "64-char" in result.output
