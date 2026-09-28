"""Supabase Management API helpers — detect and restore paused projects."""

from __future__ import annotations

import asyncio
import logging
import re

logger = logging.getLogger(__name__)

# Matches Supabase direct connection or pooler hostnames
_SUPABASE_HOST_RE = re.compile(
    r"(?:db\.)?([a-z]{20,})\.supabase\.co"
    r"|"
    r"aws-0-[a-z0-9-]+\.pooler\.supabase\.com"
)

# Project ref is always 20 lowercase alpha chars in the hostname
_PROJECT_REF_RE = re.compile(r"([a-z]{20,})")


def extract_project_ref(dsn: str) -> str | None:
    """Extract the Supabase project ref from a database DSN.

    Works for both direct (db.<ref>.supabase.co) and pooler
    (aws-0-<region>.pooler.supabase.com) connection strings.
    For pooler URLs the ref is usually in the username or database name.
    """
    # Direct connection: db.<ref>.supabase.co
    m = re.search(r"db\.([a-z]{20,})\.supabase\.co", dsn)
    if m:
        return m.group(1)

    # Pooler connection: ref appears as the database name or in the URL path
    # e.g. REDACTED
    # or the user is postgres.REFHERE
    m = re.search(r"postgres\.([a-z]{20,})", dsn)
    if m:
        return m.group(1)

    return None


def is_supabase_dsn(dsn: str) -> bool:
    """Return True if the DSN points to a Supabase-hosted database."""
    return "supabase.co" in dsn or "supabase.com" in dsn


async def restore_project(
    project_ref: str,
    access_token: str,
    *,
    timeout: float = 30.0,
) -> bool:
    """Attempt to restore (unpause) a Supabase project via the Management API.

    Returns True if the restore request was accepted, False on failure.
    Uses only stdlib (urllib) to avoid adding a dependency.
    """
    import urllib.request
    import urllib.error

    url = f"https://api.supabase.com/v1/projects/{project_ref}/restore"
    req = urllib.request.Request(
        url,
        method="POST",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        data=b"{}",
    )
    def _request() -> bool:
        try:
            with urllib.request.urlopen(req, timeout=max(0.001, timeout)) as resp:
                status = resp.status
                logger.info(
                    "Supabase restore API responded %d for project %s",
                    status, project_ref,
                )
                return 200 <= status < 300
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:500]
            logger.warning(
                "Supabase restore API returned %d for project %s: %s",
                e.code, project_ref, body,
            )
            return False
        except OSError as e:
            logger.warning("Supabase restore API request failed: %s", e)
            return False

    return await asyncio.to_thread(_request)


async def wait_for_restore(
    project_ref: str,
    access_token: str,
    *,
    timeout: float = 120,
    poll_interval: float = 5,
) -> bool:
    """Poll Supabase until the project status is ACTIVE_HEALTHY.

    Returns True when the project is ready, False on timeout.
    """
    import json
    import urllib.request
    import urllib.error

    url = f"https://api.supabase.com/v1/projects/{project_ref}"
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(0.0, timeout)

    def _request(timeout: float) -> str:
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        with urllib.request.urlopen(req, timeout=max(0.001, timeout)) as resp:
            return json.loads(resp.read()).get("status", "")

    while (remaining := deadline - loop.time()) > 0:
        try:
            status = await asyncio.wait_for(
                asyncio.to_thread(_request, min(15.0, remaining)),
                timeout=remaining,
            )
            logger.info(
                "Supabase project %s status: %s (%.0fs remaining)",
                project_ref, status, remaining,
            )
            if status == "ACTIVE_HEALTHY":
                return True
        except (OSError, asyncio.TimeoutError) as e:
            logger.debug("Status poll failed: %s", e)

        remaining = deadline - loop.time()
        if remaining > 0:
            await asyncio.sleep(min(poll_interval, remaining))

    logger.warning(
        "Timed out waiting for Supabase project %s to restore (%.0fs)",
        project_ref, timeout,
    )
    return False
