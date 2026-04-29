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
        SELECT id, type, content, source, topic,
               write_provenance, agent_id, project_id, created_at
        FROM memories
        WHERE review_status = 'pending_review'
        ORDER BY created_at DESC
        LIMIT $1
        """,
        limit,
    )
    return [dict(r) for r in rows]


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
