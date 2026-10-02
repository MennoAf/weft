"""Layer 3 — instruction-shape quarantine for agent-provenance writes.

Threat model: agent-mode Face reads an indirect-injection payload (web
content, Slack message, email body) and writes a poisoned memory whose
content is shaped like an instruction. Future agent retrieval picks it
up as "relevant context" and acts on it.

Layer 3 is a heuristic guard at write-time: agent-provenance memories
whose content reads as instruction (imperative verbs, URLs, git remotes,
absolute paths outside the project, "when X do Y" structure) are written
with ``review_status='pending_review'`` and excluded from retrieval until
the supervisor reviews them via :func:`approve_pending` or
:func:`reject_pending`. False positives are acceptable — they only add
review friction, not data loss; false negatives are mitigated by Layer 2
(retrieval-time exclusion) and Layer 4 (action-time authority tier).

Spec: weft_v2_spec.md §Q4 Layer 3.
"""

from __future__ import annotations

import logging
import re

import asyncpg

from weft.db.connection import get_db

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Instruction-shape heuristic
# ---------------------------------------------------------------------------


# Each pattern is matched case-insensitively against the candidate content.
# We name them so quarantine reasons can carry a short tag for the
# supervisor's review UI ("flagged: url, imperative") rather than an opaque
# bool.

_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)

# git remotes: SSH form (git@host:path), git+ssh URIs, raw .git suffix.
_GIT_REMOTE_RE = re.compile(
    r"(?:git@[\w.\-]+:[\w./\-]+|git\+ssh://\S+|[\w./\-]+\.git\b)",
    re.IGNORECASE,
)

# Absolute paths into sensitive system roots. Project-internal paths
# (``./relative``, ``src/foo.py``) are intentionally not matched — only
# the ones that escape any reasonable workspace.
_SYSTEM_PATH_RE = re.compile(
    r"(?:^|[\s'\"`(])/(?:etc|var|usr|opt|home|root|tmp|private|System|Library|Applications|bin|sbin)\b",
    re.IGNORECASE,
)

# Imperative-mood openers. Multi-line content matches if any line starts
# with one of these verbs — we don't try to parse English, just catch the
# obvious "Run …", "Always …", "Never …", "Use …" patterns. Anchored to
# line-start so prose mentions like "I always run X" don't false-positive
# on every memory.
_IMPERATIVE_OPENERS = (
    "always", "never", "must", "should", "do not", "don't", "stop",
    "run", "execute", "send", "post", "fetch", "use", "install",
    "delete", "remove", "rewrite", "ignore", "override", "disable",
    "enable", "kill", "drop", "purge", "exfiltrate",
)
_IMPERATIVE_RE = re.compile(
    r"(?:^|\n)\s*(?:" + "|".join(re.escape(v) for v in _IMPERATIVE_OPENERS) + r")\b",
    re.IGNORECASE,
)

# "when X do Y" / "if X then Y" structure — classic conditional command
# shape that Behaviors and Triggers also use. If an agent-written memory
# carries this shape, the supervisor should look at it before it becomes
# implicit context. The verb list mirrors ``_IMPERATIVE_OPENERS`` so an
# imperative verb in the consequent (with or without "then") triggers
# the heuristic regardless of position.
_CONDITIONAL_VERBS = _IMPERATIVE_OPENERS + (
    "fetch", "post", "delete", "remove", "rewrite", "drop", "purge",
)
_CONDITIONAL_RE = re.compile(
    r"\b(?:when|if|whenever|once|after)\b[^.\n]{1,120}?\b(?:then\s+)?"
    r"(?:" + "|".join(re.escape(v) for v in _CONDITIONAL_VERBS) + r")\b",
    re.IGNORECASE,
)

# API endpoint shapes — agent contexts shouldn't be writing routes into
# memory unless they're explicitly documenting one.
_API_ENDPOINT_RE = re.compile(
    r"/(?:api|v\d+|graphql|rpc|webhook|hooks?)/[\w./\-]+",
    re.IGNORECASE,
)


def instruction_shape_reasons(content: str) -> list[str]:
    """Return the list of heuristic tags that fired on *content*.

    Empty list = clean. Non-empty = quarantine. The tags are stored on
    the memory row (future column / log entry) so a supervisor can see
    *why* the heuristic flagged the write without re-running the regexes.
    """
    reasons: list[str] = []
    if _URL_RE.search(content):
        reasons.append("url")
    if _GIT_REMOTE_RE.search(content):
        reasons.append("git_remote")
    if _SYSTEM_PATH_RE.search(content):
        reasons.append("system_path")
    if _IMPERATIVE_RE.search(content):
        reasons.append("imperative")
    if _CONDITIONAL_RE.search(content):
        reasons.append("conditional")
    if _API_ENDPOINT_RE.search(content):
        reasons.append("api_endpoint")
    return reasons


def looks_like_instruction(content: str) -> bool:
    """Convenience predicate: True if any heuristic fires."""
    return bool(instruction_shape_reasons(content))


# ---------------------------------------------------------------------------
# Quarantine review (supervisor-only at the tool layer)
# ---------------------------------------------------------------------------


async def list_pending(
    pool: asyncpg.Pool,
    *,
    limit: int = 50,
) -> list[dict]:
    """Return memories whose ``review_status='pending_review'``.

    Surfaces enough context for the supervisor to decide approve / reject
    without a follow-up read: id, content, who/what wrote it, when, and
    the original source / topic.
    """
    rows = await get_db(pool).fetch(
        """
        SELECT m.id, m.type, m.content, m.source, m.topic,
               m.write_provenance, m.agent_id, m.project_id, m.created_at,
               mc.target_id AS merge_target_id
        FROM memories m
        LEFT JOIN memory_relationships mc
               ON mc.source_id = m.id AND mc.relation = 'merge_candidate'
        WHERE m.review_status = 'pending_review'
        ORDER BY m.created_at DESC
        LIMIT $1
        """,
        limit,
    )
    # merge_target_id is non-null only for cross-project merge candidates (L2);
    # for those the supervisor can call the 'merge' action to append the facet
    # to the target. Plain injection-quarantine rows carry merge_target_id=None.
    return [dict(r) for r in rows]


async def mark_merge_candidate(
    pool: asyncpg.Pool, candidate_id: str, target_id: str
) -> None:
    """Flag a freshly-stored belief as a cross-project merge candidate.

    Sets ``review_status='pending_review'`` on the candidate and links it to
    the existing belief it would merge into via a ``merge_candidate`` edge
    (candidate -> target). The edge is what lets :func:`merge_pending` later
    append the facet to the right target; without it the candidate would sit
    reviewable-but-orphaned. Called by weft_remember's L2 post-store path
    (loom-c82bd8d8). Raw INSERT rather than store.add_relationship because
    store imports this module (avoids a cycle).
    """
    db = get_db(pool)
    await db.execute(
        "UPDATE memories SET review_status = 'pending_review' WHERE id = $1",
        candidate_id,
    )
    await db.execute(
        """
        INSERT INTO memory_relationships
            (source_id, target_id, relation, created_at, user_id)
        VALUES ($1, $2, 'merge_candidate', now(),
                nullif(current_setting('app.user_id', true), ''))
        ON CONFLICT (source_id, target_id, relation) DO NOTHING
        """,
        candidate_id,
        target_id,
    )


async def merge_pending(pool: asyncpg.Pool, candidate_id: str) -> dict | None:
    """Resolve a cross-project merge candidate by MERGING it into its target.

    A candidate (review_status='pending_review') linked to an existing belief
    via a ``merge_candidate`` edge is resolved by:
      1. appending the candidate's project_facets to the target belief
         (union, lowercase-preserving) and strengthening the target's
         confidence (GREATEST);
      2. archiving the candidate (status='archived') and clearing its
         pending_review flag so it leaves the review queue;
      3. recording a ``supersedes`` edge target -> candidate for lineage.

    Returns a small result dict on success, or ``None`` when *candidate_id*
    is not a pending merge candidate (no merge_candidate edge / not pending).
    This is the merge-aware counterpart to :func:`approve_pending` (which
    instead promotes a candidate to its own active belief — the keep-separate
    outcome).
    """
    db = get_db(pool)
    target_id = await db.fetchval(
        """
        SELECT mc.target_id
        FROM memory_relationships mc
        JOIN memories c ON c.id = mc.source_id
        WHERE mc.source_id = $1
          AND mc.relation = 'merge_candidate'
          AND c.review_status = 'pending_review'
        """,
        candidate_id,
    )
    if target_id is None:
        return None

    async with pool.acquire() as conn:
        async with conn.transaction():
            # 1. Append candidate facets to the target belief (union, distinct).
            merged = await conn.fetchval(
                """
                UPDATE memories t
                SET project_facets = (
                        SELECT array_agg(DISTINCT f ORDER BY f)
                        FROM unnest(t.project_facets || c.project_facets) AS f
                    ),
                    confidence = GREATEST(t.confidence, c.confidence),
                    updated_at = now()
                FROM memories c
                WHERE t.id = $1 AND c.id = $2
                  AND (t.user_id IS NULL
                       OR t.user_id = current_setting('app.user_id', true))
                RETURNING t.project_facets
                """,
                target_id,
                candidate_id,
            )
            if merged is None:
                # Target not visible to this user (RLS / ownership) — abort the
                # whole merge so we never archive a candidate without merging.
                return None
            # 2. Archive the candidate and drop it from the review queue.
            await conn.execute(
                """
                UPDATE memories
                SET status = 'archived',
                    review_status = 'active',
                    updated_at = now()
                WHERE id = $1
                  AND (user_id IS NULL
                       OR user_id = current_setting('app.user_id', true))
                """,
                candidate_id,
            )
            # 3. Lineage edge: target supersedes the merged-away candidate.
            await conn.execute(
                """
                INSERT INTO memory_relationships
                    (source_id, target_id, relation, created_at, user_id)
                VALUES ($1, $2, 'supersedes', now(),
                        nullif(current_setting('app.user_id', true), ''))
                ON CONFLICT (source_id, target_id, relation) DO NOTHING
                """,
                target_id,
                candidate_id,
            )
    logger.info(
        "quarantine: merged candidate %s into %s (facets now %s)",
        candidate_id, target_id, merged,
    )
    return {
        "candidate_id": candidate_id,
        "target_id": target_id,
        "target_project_facets": list(merged),
    }


async def approve_pending(pool: asyncpg.Pool, memory_id: str) -> bool:
    """Promote a quarantined write to supervisor-trusted active.

    Re-provenances to ``supervisor`` AND flips ``review_status`` to
    ``active`` so future retrievals surface it normally and Face mode
    no longer wraps it with the untrusted prefix. Returns True on a
    real promotion, False if the row wasn't pending.
    """
    result = await get_db(pool).execute(
        """
        UPDATE memories
        SET write_provenance = 'supervisor',
            review_status    = 'active',
            updated_at       = now()
        WHERE id = $1 AND review_status = 'pending_review'
        """,
        memory_id,
    )
    # asyncpg returns 'UPDATE n' — extract the count.
    affected = int(result.rsplit(" ", 1)[-1]) if result else 0
    if affected:
        logger.info("quarantine: approved pending memory %s", memory_id)
    return affected > 0


async def reject_pending(pool: asyncpg.Pool, memory_id: str) -> bool:
    """Hard-delete a quarantined write.

    Reject is destructive on purpose — the supervisor confirmed this was
    not memory worth keeping, so we don't soft-archive (which would still
    be reachable via ``include_pending_review`` / archived-only queries).
    Returns True on a real delete, False if the row wasn't pending.
    """
    result = await get_db(pool).execute(
        """
        DELETE FROM memories
        WHERE id = $1 AND review_status = 'pending_review'
        """,
        memory_id,
    )
    affected = int(result.rsplit(" ", 1)[-1]) if result else 0
    if affected:
        logger.info("quarantine: rejected pending memory %s", memory_id)
    return affected > 0
