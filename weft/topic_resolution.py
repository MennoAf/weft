"""Topic resolver — L1 Resolution Ratchet (alias-first normalization).

resolve_topic(topic_string, user_id, pool) -> list[str]:
    FIRST looks up topic_resolution_aliases by normalized token under RLS.
    If a row exists, bumps hit_count, increments the topic_resolution.alias_hits
    counter, and returns resolved_tags.
    ELSE naive-normalizes to [lower(topic_string), 'entity:' + Title(topic_string)].

record_alias(token, resolved_tags, source, user_id, pool) upserts a row.

This is the FEEDBACK half of Compounding Loop L1 in the PRD
(documents/prds/topic-digest-recall.md §Compounding Loops): the alias map
must be consulted BEFORE naive normalization or the loop is dead.

Normalization idiom follows weft/views/belief_query.py _tokenize:
    re.findall(r'[a-z0-9]+', topic_string.lower())
    but we take only the first non-empty token for table lookups.

Spec: Loom task loom-04e8c1af.
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

import asyncpg

from weft.auth import current_user_id
from weft.counters import increment_counter
from weft.db.connection import acquire

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Counter name — must be exactly this string per task spec
# ---------------------------------------------------------------------------

COUNTER_TOPIC_RESOLUTION_ALIAS_HITS = "topic_resolution.alias_hits"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_token(topic_string: str) -> str:
    """Lowercase and extract the canonical token from a topic string.

    Mirrors the _tokenize() idiom in weft/views/belief_query.py:
    re.findall(r'[a-z0-9]+', s.lower()) — join all alphanumeric segments
    so 'entity:Weft' → 'entityweft' is not what we want.  Instead we
    lower() and strip, keeping the original structure with lowercase only,
    since topic tokens are typically simple words or kebab/colon strings.

    For alias key lookup we use the lowercased raw string (spaces stripped)
    to match what callers supply when saving aliases.
    """
    return topic_string.strip().lower()


def _naive_normalize(topic_string: str) -> list[str]:
    """Naive normalization: [lower(topic_string), 'entity:' + Title(topic_string)].

    Title-cases the first letter of each word segment for the entity tag,
    mirroring the entity-tagging convention (e.g. 'weft' → 'entity:Weft').
    """
    lower = topic_string.strip().lower()
    # Title case: capitalize first letter of each alphanumeric word
    titled = re.sub(r"[a-z0-9]+", lambda m: m.group(0).capitalize(), lower)
    return [lower, f"entity:{titled}"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def resolve_topic(
    topic_string: str,
    user_id: str,
    pool: asyncpg.Pool,
) -> list[str]:
    """Resolve a topic string to a list of canonical tags.

    Resolution order:
    1. Look up topic_resolution_aliases by normalized token under RLS
       (SET LOCAL app.user_id so RLS INSERT/UPDATE policies are satisfied).
       If a row exists → bump hit_count + increment alias_hits counter + return
       resolved_tags.
    2. ELSE → naive normalization: [lower(topic_string), 'entity:' + Title(topic_string)].

    Args:
        topic_string: The raw topic input (e.g. 'weft', 'Weft', 'entity:Weft').
        user_id: The owner identity; used to scope the RLS lookup and write.
        pool: asyncpg connection pool.

    Returns:
        List of resolved tag strings (always non-empty).
    """
    token = _normalize_token(topic_string)

    # Step 1: alias lookup under RLS. acquire() sets app.user_id (with the same
    # alphanumeric guard this code used to inline) and runs inside a transaction,
    # so the SELECT + hit_count UPDATE are atomic.
    row = None
    tok = current_user_id.set(user_id)
    try:
        async with acquire(pool) as conn:
            row = await conn.fetchrow(
                """
                SELECT resolved_tags
                FROM topic_resolution_aliases
                WHERE user_id = $1
                  AND topic_token = $2
                """,
                user_id,
                token,
            )

            if row is not None:
                resolved_tags: list[str] = list(row["resolved_tags"])

                # Bump hit_count inside the same transaction so the update
                # is atomic with the SELECT.
                await conn.execute(
                    """
                    UPDATE topic_resolution_aliases
                       SET hit_count  = hit_count + 1,
                           updated_at = now()
                     WHERE user_id    = $1
                       AND topic_token = $2
                    """,
                    user_id,
                    token,
                )
    finally:
        current_user_id.reset(tok)

    if row is not None:
        # Increment the named counter (best-effort, outside transaction).
        await increment_counter(pool, COUNTER_TOPIC_RESOLUTION_ALIAS_HITS)

        logger.debug(
            "resolve_topic: alias hit for token=%r user=%s → %r",
            token,
            user_id,
            resolved_tags,
        )
        return resolved_tags

    # Step 2: naive normalization
    result = _naive_normalize(topic_string)
    logger.debug(
        "resolve_topic: naive normalization for token=%r → %r", token, result
    )
    return result


async def record_alias(
    token: str,
    resolved_tags: Sequence[str],
    source: str,
    user_id: str,
    pool: asyncpg.Pool,
) -> None:
    """Upsert a topic_resolution_aliases row.

    ON CONFLICT (user_id, topic_token) updates resolved_tags, source,
    and updated_at.  hit_count is NOT reset on upsert.

    Args:
        token: The raw topic token to alias (will be normalized to lowercase).
        resolved_tags: The canonical tags this token should resolve to.
        source: One of 'manual' or 'learned' (enforced by DB CHECK).
        user_id: The owner identity; must match app.user_id for RLS.
        pool: asyncpg connection pool.
    """
    normalized_token = _normalize_token(token)
    tags_list = list(resolved_tags)

    # acquire() sets app.user_id (with the same alphanumeric guard this code
    # used to inline) so the RLS INSERT/UPDATE policy is satisfied, and runs
    # inside a transaction.
    tok = current_user_id.set(user_id)
    try:
        async with acquire(pool) as conn:
            await conn.execute(
                """
                INSERT INTO topic_resolution_aliases
                    (user_id, topic_token, resolved_tags, source, updated_at)
                VALUES ($1, $2, $3, $4, now())
                ON CONFLICT (user_id, topic_token) DO UPDATE
                    SET resolved_tags = EXCLUDED.resolved_tags,
                        source        = EXCLUDED.source,
                        updated_at    = now()
                """,
                user_id,
                normalized_token,
                tags_list,
                source,
            )
    finally:
        current_user_id.reset(tok)

    logger.debug(
        "record_alias: upserted token=%r user=%s source=%s tags=%r",
        normalized_token,
        user_id,
        source,
        tags_list,
    )
