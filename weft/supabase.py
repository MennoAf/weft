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
    # e.g. postgresql://postgres.REFHERE:pw@aws-0-us-east-1.pooler.supabase.com:6543/postgres
    # or the user is postgres.REFHERE
    m = re.search(r"postgres\.([a-z]{20,})", dsn)
    if m:
        return m.group(1)

    return None


def is_supabase_dsn(dsn: str) -> bool:
    """Return True if the DSN points to a Supabase-hosted database."""
    return "supabase.co" in dsn or "supabase.com" in dsn


async def restore_project(project_ref: str, access_token: str) -> bool:
    """Attempt to restore (unpause) a Supabase project via the Management API.

    Returns True if the restore request was accepted, False on failure.
    Uses only stdlib (urllib) to avoid adding a dependency.
    """
    import json
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
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
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
    except Exception as e:
        logger.warning("Supabase restore API request failed: %s", e)
        return False


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
    elapsed = 0.0

    while elapsed < timeout:
        try:
            req = urllib.request.Request(
                url,
                headers={"Authorization": f"Bearer {access_token}"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
                status = data.get("status", "")
                logger.info(
                    "Supabase project %s status: %s (%.0fs elapsed)",
                    project_ref, status, elapsed,
                )
                if status == "ACTIVE_HEALTHY":
                    return True
        except Exception as e:
            logger.debug("Status poll failed: %s", e)

        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

    logger.warning(
        "Timed out waiting for Supabase project %s to restore (%.0fs)",
        project_ref, timeout,
    )
    return False
