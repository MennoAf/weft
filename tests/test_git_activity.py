"""Tests for weft.git_activity — runs against real git repos created in tmp_path."""

from __future__ import annotations

import asyncio
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from weft.git_activity import RepoActivity, scan_repo, scan_repos


def _git(*args: str, cwd: Path) -> None:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    subprocess.run(
        ["git", *args], cwd=cwd, env=env, check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _make_repo(path: Path, *, commits: int = 1) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git("init", "-q", "-b", "main", cwd=path)
    _git("config", "user.email", "test@example.com", cwd=path)
    _git("config", "user.name", "Test", cwd=path)
    for i in range(commits):
        f = path / f"file_{i}.txt"
        f.write_text(f"content {i}")
        _git("add", str(f), cwd=path)
        _git("commit", "-q", "-m", f"commit {i}: change file {i}", cwd=path)
    return path


@pytest.mark.asyncio
async def test_scan_repo_counts_recent_commits(tmp_path):
    repo = _make_repo(tmp_path / "alpha", commits=3)
    since = datetime.now(timezone.utc) - timedelta(hours=1)
    result = await scan_repo("alpha", str(repo), since=since)

    assert isinstance(result, RepoActivity)
    assert result.error is None
    assert result.commit_count == 3
    assert result.last_commit_subject is not None
    assert "commit 2" in result.last_commit_subject  # most recent
    assert len(result.subjects) == 3


@pytest.mark.asyncio
async def test_scan_repo_window_excludes_old_commits(tmp_path):
    repo = _make_repo(tmp_path / "beta", commits=2)
    # Future window — repo's commits are in the past relative to it
    since = datetime.now(timezone.utc) + timedelta(hours=1)
    result = await scan_repo("beta", str(repo), since=since)
    assert result.error is None
    assert result.commit_count == 0


@pytest.mark.asyncio
async def test_scan_repo_missing_path(tmp_path):
    result = await scan_repo("nope", str(tmp_path / "does_not_exist"), since=datetime.now(timezone.utc))
    assert result.commit_count == 0
    assert result.error == "repo path does not exist"


@pytest.mark.asyncio
async def test_scan_repo_not_a_git_repo(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    result = await scan_repo("plain", str(plain), since=datetime.now(timezone.utc))
    assert result.error == "not a git repository"


@pytest.mark.asyncio
async def test_scan_repos_runs_in_parallel(tmp_path):
    a = _make_repo(tmp_path / "a", commits=1)
    b = _make_repo(tmp_path / "b", commits=2)

    results = await scan_repos({"a": str(a), "b": str(b)}, window_hours=1)
    assert len(results) == 2
    by_id = {r.project_id: r for r in results}
    assert by_id["a"].commit_count == 1
    assert by_id["b"].commit_count == 2


@pytest.mark.asyncio
async def test_scan_repos_empty_config_returns_empty():
    assert await scan_repos({}) == []
