"""Git activity scanning for the daily brief.

Walks configured project repos and counts commits within a window.
Read-only — never mutates the repo. Uses ``git log --since`` so it
respects whatever date precision git applies.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RepoActivity:
    project_id: str
    repo_path: str
    commit_count: int = 0
    last_commit_subject: str | None = None
    last_commit_at: datetime | None = None
    error: str | None = None
    subjects: list[str] = field(default_factory=list)


async def _run_git(repo_path: Path, *args: str) -> tuple[int, str, str]:
    """Run a git subcommand against repo_path. Returns (rc, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        "git", "-C", str(repo_path), *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_b, stderr_b = await proc.communicate()
    return proc.returncode or 0, stdout_b.decode(errors="replace"), stderr_b.decode(errors="replace")


async def scan_repo(
    project_id: str,
    repo_path: str,
    *,
    since: datetime,
    max_subjects: int = 5,
) -> RepoActivity:
    """Count commits in repo_path since `since`. Captures up to N recent subjects.

    Returns a RepoActivity even on error so callers can decide how to surface
    repos that have moved or aren't git repos.
    """
    activity = RepoActivity(project_id=project_id, repo_path=repo_path)
    path = Path(repo_path).expanduser()
    if not path.exists():
        activity.error = "repo path does not exist"
        return activity
    if not (path / ".git").exists():
        activity.error = "not a git repository"
        return activity

    since_iso = since.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S +0000")

    try:
        rc, stdout, stderr = await _run_git(
            path,
            "log",
            f"--since={since_iso}",
            "--no-merges",
            "--pretty=format:%H%x09%cI%x09%s",
        )
        if rc != 0:
            activity.error = stderr.strip()[:200] or f"git log exit {rc}"
            return activity

        lines = [ln for ln in stdout.splitlines() if ln.strip()]
        activity.commit_count = len(lines)
        if lines:
            # Most recent commit is first
            top = lines[0].split("\t")
            if len(top) >= 3:
                try:
                    activity.last_commit_at = datetime.fromisoformat(top[1])
                except ValueError:
                    activity.last_commit_at = None
                activity.last_commit_subject = top[2]
            for ln in lines[:max_subjects]:
                parts = ln.split("\t", 2)
                if len(parts) >= 3:
                    activity.subjects.append(parts[2])
    except FileNotFoundError:
        activity.error = "git not found on PATH"
    except Exception as exc:  # pragma: no cover — last-resort safety
        logger.exception("git_activity.scan_repo unexpected_error project=%s", project_id)
        activity.error = f"unexpected: {exc.__class__.__name__}"

    return activity


async def scan_repos(
    project_repos: dict[str, str],
    *,
    window_hours: int = 24,
    as_of: datetime | None = None,
) -> list[RepoActivity]:
    """Scan all configured project repos in parallel. Returns activities (incl. errors)."""
    if not project_repos:
        return []
    if as_of is None:
        as_of = datetime.now(timezone.utc)
    since = as_of - timedelta(hours=window_hours)
    coros = [scan_repo(pid, path, since=since) for pid, path in project_repos.items()]
    return await asyncio.gather(*coros)
