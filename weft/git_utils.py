"""Async git integration for recent commit retrieval.

Stateless and pure — importable independently of the rest of weft.
Uses asyncio.create_subprocess_exec (not shell=True) to avoid injection.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# Default timeout for git subprocess calls
_GIT_TIMEOUT_SECONDS = 5


async def get_recent_commits(
    *,
    since: datetime | None = None,
    max_count: int = 20,
    repo_path: str | None = None,
) -> list[str]:
    """Get recent git commits as one-line strings.

    Returns empty list (never raises) if git is unavailable, not a repo,
    times out, or any other error occurs.

    Args:
        since: Only commits after this timestamp. Defaults to 24h ago.
        max_count: Maximum number of commits to return.
        repo_path: Path to the git repo. Defaults to cwd.
    """
    if since is None:
        since = datetime.now(timezone.utc) - timedelta(hours=24)

    since_iso = since.strftime("%Y-%m-%dT%H:%M:%SZ")

    args = ["git", "log", "--oneline", f"--since={since_iso}", f"--max-count={max_count}"]
    if repo_path:
        args = ["git", "-C", repo_path] + args[1:]

    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            ),
            timeout=_GIT_TIMEOUT_SECONDS,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=_GIT_TIMEOUT_SECONDS,
        )

        if proc.returncode != 0:
            logger.debug("git log returned %d: %s", proc.returncode, stderr.decode().strip())
            return []

        output = stdout.decode().strip()
        if not output:
            return []

        return output.split("\n")

    except FileNotFoundError:
        logger.warning("git binary not found")
        return []
    except asyncio.TimeoutError:
        logger.warning("git log timed out after %ds", _GIT_TIMEOUT_SECONDS)
        return []
    except OSError as e:
        logger.warning("git subprocess error: %s", e)
        return []
