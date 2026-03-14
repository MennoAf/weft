"""Tests for async git integration."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from weft.git_utils import get_recent_commits


@pytest.mark.asyncio
async def test_returns_commit_list():
    """Successful git log returns list of commit strings."""
    fake_stdout = b"abc1234 Add feature\ndef5678 Fix bug\n"

    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(fake_stdout, b""))
    proc.returncode = 0

    with patch("weft.git_utils.asyncio.create_subprocess_exec", return_value=proc):
        commits = await get_recent_commits()

    assert commits == ["abc1234 Add feature", "def5678 Fix bug"]


@pytest.mark.asyncio
async def test_empty_stdout_returns_empty_list():
    """No commits in range → empty list."""
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(b"", b""))
    proc.returncode = 0

    with patch("weft.git_utils.asyncio.create_subprocess_exec", return_value=proc):
        commits = await get_recent_commits()

    assert commits == []


@pytest.mark.asyncio
async def test_nonzero_returncode_returns_empty_list():
    """Non-zero exit (e.g., not a git repo) → empty list."""
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(b"", b"fatal: not a git repository"))
    proc.returncode = 128

    with patch("weft.git_utils.asyncio.create_subprocess_exec", return_value=proc):
        commits = await get_recent_commits()

    assert commits == []


@pytest.mark.asyncio
async def test_git_not_installed_returns_empty_list():
    """FileNotFoundError (git not installed) → empty list."""
    with patch(
        "weft.git_utils.asyncio.create_subprocess_exec",
        side_effect=FileNotFoundError("git"),
    ):
        commits = await get_recent_commits()

    assert commits == []


@pytest.mark.asyncio
async def test_timeout_returns_empty_list():
    """Subprocess timeout → empty list."""
    with patch(
        "weft.git_utils.asyncio.wait_for",
        side_effect=asyncio.TimeoutError(),
    ):
        commits = await get_recent_commits()

    assert commits == []


@pytest.mark.asyncio
async def test_since_parameter_passed_to_git():
    """Since timestamp is formatted as ISO 8601 for git."""
    since = datetime(2026, 3, 10, 12, 0, 0, tzinfo=timezone.utc)

    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(b"abc1234 commit\n", b""))
    proc.returncode = 0

    with patch("weft.git_utils.asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
        await get_recent_commits(since=since, max_count=5)

    # Check the args passed to create_subprocess_exec
    call_args = mock_exec.call_args[0]
    assert "--since=2026-03-10T12:00:00Z" in call_args
    assert "--max-count=5" in call_args


@pytest.mark.asyncio
async def test_repo_path_passed_to_git():
    """repo_path uses git -C for the right directory."""
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(b"", b""))
    proc.returncode = 0

    with patch("weft.git_utils.asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
        await get_recent_commits(repo_path="/tmp/myrepo")

    call_args = mock_exec.call_args[0]
    assert call_args[0] == "git"
    assert call_args[1] == "-C"
    assert call_args[2] == "/tmp/myrepo"
