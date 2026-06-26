"""Digest cache read/write + write-invalidation for the topic-digest recall feature.

Implements the persistence layer for the topic_digests table (migration v57).
Three public functions:

    read_digest(pool, user_id, topic, scope) -> row dict | None
        Returns the fresh digest row, or None if missing or stale.

    write_digest(pool, user_id, topic, scope, content, provenance,
                 detector_version, digest_id=None) -> str
        Upserts a digest row (INSERT ... ON CONFLICT UPDATE). Returns the
        digest_id of the written row.

    mark_stale_for_tags(pool, user_id, tags) -> None
        Flips stale=true on any topic_digests row whose topic is in ``tags``
        for the given user_id.  Used by the store_memory write-invalidation hook.

Connection pattern: uses get_db(pool) to reuse the RLS-scoped connection when
inside an acquire() context — same idiom as topic_gather.py and store.py.

Spec: documents/prds/topic-digest-recall.md §The digest lifecycle, §Validation V3.
Loom task: loom-78432e69.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import asyncpg

from weft.db.connection import get_db

logger = logging.getLogger(__name__)


def _td_id() -> str:
    """Generate a prefixed digest id: td-{8 hex chars}."""
    return f"td-{uuid.uuid4().hex[:8]}"


async def read_digest(
    pool: asyncpg.Pool,
    user_id: str,
    topic: str,
    scope: str = "global",
) -> dict[str, Any] | None:
    """Read a fresh digest for (user_id, topic, scope).

    Returns the row as a dict when a non-stale digest exists.
    Returns None when:
      - No row exists for (user_id, topic, scope), OR
      - The row has stale=True.

    Callers that need to distinguish "missing" from "stale" should
    call the DB directly; this function intentionally collapses both
    to None so the caller always synthesizes a fresh digest on None.

    Uses get_db(pool) to reuse the RLS-scoped connection when inside
    an acquire() context, so RLS policies apply automatically.
    """
    row = await get_db(pool).fetchrow(
        """
        SELECT digest_id, user_id, topic, scope, content, provenance,
               generated_at, stale, detector_version
        FROM topic_digests
        WHERE user_id = $1
          AND topic   = $2
          AND scope   = $3
        """,
        user_id,
        topic,
        scope,
    )
    if row is None:
        return None
    # stale=True → not fresh; return None so caller regenerates
    if row["stale"]:
        return None
    return dict(row)


async def write_digest(
    pool: asyncpg.Pool,
    user_id: str,
    topic: str,
    content: str,
    detector_version: str,
    *,
    scope: str = "global",
    provenance: dict | None = None,
    digest_id: str | None = None,
) -> str:
    """Upsert a digest row for (user_id, topic, scope).

    On conflict (user_id, topic, scope) — the UNIQUE index from v57 —
    updates content, provenance, generated_at, detector_version, and
    resets stale=false (the new digest is fresh).

    Args:
        pool: asyncpg connection pool.
        user_id: Owner identity; must match app.user_id for RLS INSERT.
        topic: The topic tag string (e.g. 'weft', 'entity:Weft').
        content: The synthesized digest text.
        detector_version: Version string of the Haiku prompt used.
        scope: Scope qualifier; defaults to 'global'.
        provenance: Optional JSONB provenance map {memory_id: [spans]}.
        digest_id: Caller-supplied id; generated (td-{shortid}) if None.

    Returns:
        The digest_id of the written row.
    """
    import json

    row_id = digest_id or _td_id()
    provenance_json = json.dumps(provenance) if provenance is not None else None

    await get_db(pool).execute(
        """
        INSERT INTO topic_digests
            (digest_id, user_id, topic, scope, content, provenance,
             generated_at, stale, detector_version)
        VALUES
            ($1, $2, $3, $4, $5, $6::jsonb, now(), false, $7)
        ON CONFLICT (user_id, topic, scope) DO UPDATE
            SET content          = EXCLUDED.content,
                provenance       = EXCLUDED.provenance,
                generated_at     = now(),
                stale            = false,
                detector_version = EXCLUDED.detector_version
        """,
        row_id,
        user_id,
        topic,
        scope,
        content,
        provenance_json,
        detector_version,
    )
    return row_id


async def mark_stale_for_tags(
    pool: asyncpg.Pool,
    user_id: str,
    tags: list[str],
) -> None:
    """Flip stale=true on any digest whose topic is in ``tags`` for user_id.

    Called by the store_memory write-invalidation hook (V3) after a new
    memory is written.  Matches digests by the memory's topic tags directly
    (no alias resolution needed — the digest is keyed by the same topic
    string the memory uses).

    Uses get_db(pool) so when called inside an acquire() context the flip
    runs on the SAME connection/transaction as the memory write — satisfying
    the "same transaction boundary" requirement from done_when (3).

    Errors are swallowed with a warning log.  This is intentional:
      - "within same transaction boundary" is the *happy path*.
      - "best-effort non-raising" is the *safety guarantee*.
    The design resolves the tension by doing the stale flip on the same
    connection (atomic with the memory write when inside acquire()), but
    wrapping in try/except so a DB error here never aborts the memory write.
    """
    if not tags:
        return
    try:
        await get_db(pool).execute(
            """
            UPDATE topic_digests
               SET stale = true
             WHERE user_id = $1
               AND topic = ANY($2::text[])
            """,
            user_id,
            tags,
        )
    except Exception as exc:
        logger.warning(
            "mark_stale_for_tags failed (user=%s tags=%r): %s",
            user_id,
            tags,
            exc,
        )
