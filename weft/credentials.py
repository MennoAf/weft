"""First-class bearer-token credentials backed by the ``weft_tokens`` table.

Phase 2.5 L2 — issuance, lookup, listing, and revocation for credentials
that bind ``caller_mode`` to the bearer token at mint time. Phase 2's
poisoning defense trusts the ``X-Weft-Caller-Mode`` request header at
the middleware boundary, which means an agent holding a valid token can
also claim ``caller_mode=supervisor`` and we believe it. L4 closes that
escalation by resolving caller mode from the token's row instead of the
header — this module is the authoritative source for those rows.

Token format: ``weft-<43 base64url chars>``. The prefix lets operators
grep logs and run leak-detection scanners; the 32-byte random suffix is
chosen with ``secrets.token_urlsafe(32)``. The plaintext is shown to the
caller exactly once at issuance and is never stored — only the SHA-256
hex digest goes to the database, so a database compromise can't be
replayed against the auth path.

Storage decisions in ``weft/db/migrations.py`` migration 40:

* PK on ``token_hash`` makes lookup an O(1) index hit.
* ``caller_mode`` is enum-checked at the DB layer (``supervisor`` /
  ``agent``). A buggy caller can't sneak a third value in.
* ``revoked_at`` and ``expires_at`` are nullable timestamps. Lookup
  filters both — a revoked or expired token is indistinguishable from
  ``None`` to the rest of the system.

Out of scope here: the legacy ``WEFT_API_KEY`` bootstrap (L3), the
middleware rewire that calls :func:`lookup_token` (L4), and the CLI /
MCP surfaces (L5 / L6).
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Literal

import asyncpg
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


CallerMode = Literal["supervisor", "agent"]
_VALID_CALLER_MODES: frozenset[str] = frozenset({"supervisor", "agent"})

TOKEN_PREFIX = "weft-"
_TOKEN_RANDOM_BYTES = 32  # secrets.token_urlsafe(32) → 43 base64url chars


class TokenRow(BaseModel):
    """A row in ``weft_tokens``. Never carries the plaintext token —
    that's returned separately by :func:`issue_token` and discarded
    after the caller persists or hands it off."""

    token_hash: str
    user_id: str
    caller_mode: CallerMode
    label: str | None = None
    created_at: datetime
    last_used_at: datetime | None = None
    expires_at: datetime | None = None
    revoked_at: datetime | None = None

    model_config = {"frozen": True}

    def is_active(self, *, now: datetime | None = None) -> bool:
        """True iff the row would resolve via :func:`lookup_token`.

        Mirrors the SQL filter so callers (CLI listing, audits) can
        check status without re-querying. ``now`` is injectable for
        tests; production callers should leave it as None."""
        ts = now or datetime.now(timezone.utc)
        if self.revoked_at is not None:
            return False
        if self.expires_at is not None and self.expires_at <= ts:
            return False
        return True


def _hash_plaintext(plaintext: str) -> str:
    """SHA-256 hex digest of the plaintext token.

    Hex digest (not raw bytes) so the value round-trips through ``TEXT``
    columns without base64 encoding. Constant-time compare isn't needed
    here — we hash before lookup, so the DB does an exact-match index
    probe rather than a string compare against the plaintext."""
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def _mint_plaintext() -> str:
    """Generate a fresh ``weft-<random>`` token. 32 random bytes →
    256 bits of entropy → ~43 base64url chars after the prefix."""
    return f"{TOKEN_PREFIX}{secrets.token_urlsafe(_TOKEN_RANDOM_BYTES)}"


def _row_to_model(row: asyncpg.Record) -> TokenRow:
    return TokenRow(
        token_hash=row["token_hash"],
        user_id=row["user_id"],
        caller_mode=row["caller_mode"],
        label=row["label"],
        created_at=row["created_at"],
        last_used_at=row["last_used_at"],
        expires_at=row["expires_at"],
        revoked_at=row["revoked_at"],
    )


async def issue_token(
    pool: asyncpg.Pool,
    *,
    user_id: str,
    caller_mode: CallerMode,
    label: str | None = None,
    expires_in: timedelta | None = None,
) -> tuple[str, TokenRow]:
    """Mint a new bearer credential.

    Returns ``(plaintext, row)``. Plaintext is the **only** time the
    caller can see the unhashed token — hand it to the operator (or the
    container being provisioned) and forget it. ``row`` is what gets
    persisted; it's safe to log.

    Caller is responsible for storing the plaintext somewhere durable
    (1Password, env var, secret manager) before this function returns
    — there is no recovery path if it's lost."""
    if caller_mode not in _VALID_CALLER_MODES:
        # The DB CHECK would reject this anyway, but failing here gives a
        # clean Python-level error before we burn a round-trip.
        raise ValueError(
            f"caller_mode must be 'supervisor' or 'agent', got {caller_mode!r}"
        )
    if not user_id:
        raise ValueError("user_id is required")

    plaintext = _mint_plaintext()
    token_hash = _hash_plaintext(plaintext)
    expires_at = (
        datetime.now(timezone.utc) + expires_in if expires_in is not None else None
    )

    row = await pool.fetchrow(
        """
        INSERT INTO weft_tokens (
            token_hash, user_id, caller_mode, label, expires_at
        ) VALUES ($1, $2, $3, $4, $5)
        RETURNING token_hash, user_id, caller_mode, label,
                  created_at, last_used_at, expires_at, revoked_at
        """,
        token_hash,
        user_id,
        caller_mode,
        label,
        expires_at,
    )
    logger.info(
        "issued token user=%s caller_mode=%s label=%s expires_at=%s",
        user_id, caller_mode, label, expires_at,
    )
    return plaintext, _row_to_model(row)


async def lookup_token(
    pool: asyncpg.Pool, plaintext_token: str
) -> TokenRow | None:
    """Resolve a bearer token to its row, or None.

    Returns None for: unknown hashes, revoked rows, and expired rows.
    Callers (the middleware) cannot tell the three cases apart — that's
    intentional, otherwise a probe could distinguish "this token never
    existed" from "this token existed but was revoked."

    On a successful match this also bumps ``last_used_at`` to now. The
    update is fire-and-forget from the caller's perspective; if the
    write fails it doesn't fail the lookup."""
    if not plaintext_token:
        return None

    token_hash = _hash_plaintext(plaintext_token)
    row = await pool.fetchrow(
        """
        UPDATE weft_tokens
        SET last_used_at = now()
        WHERE token_hash = $1
          AND revoked_at IS NULL
          AND (expires_at IS NULL OR expires_at > now())
        RETURNING token_hash, user_id, caller_mode, label,
                  created_at, last_used_at, expires_at, revoked_at
        """,
        token_hash,
    )
    if row is None:
        return None
    return _row_to_model(row)


async def revoke_token(pool: asyncpg.Pool, token_hash: str) -> bool:
    """Revoke by token_hash. Idempotent: returns True the first time
    the row flips revoked_at, False on subsequent calls (already
    revoked) and on unknown hashes.

    The argument is the **hash**, not plaintext. CLI / MCP surfaces
    derive the hash from a label / partial match before calling here —
    we never want a revocation path that takes plaintext, because that
    would re-introduce the leak vector L1 was designed to close."""
    if not token_hash:
        return False

    result = await pool.execute(
        """
        UPDATE weft_tokens
        SET revoked_at = now()
        WHERE token_hash = $1 AND revoked_at IS NULL
        """,
        token_hash,
    )
    # asyncpg returns "UPDATE <rowcount>" — parse the count.
    _, _, count_str = result.partition(" ")
    try:
        flipped = int(count_str.strip()) > 0
    except ValueError:
        flipped = False
    if flipped:
        logger.info("revoked token hash=%s…", token_hash[:8])
    return flipped


async def list_tokens(
    pool: asyncpg.Pool,
    user_id: str,
    *,
    include_revoked: bool = False,
) -> list[TokenRow]:
    """List a user's tokens, newest first.

    Default omits revoked rows so the common path (``weft tokens list``)
    only shows live credentials. Pass ``include_revoked=True`` for
    audits / forensics — expired-but-not-revoked rows are always
    included so operators see what aged out."""
    if include_revoked:
        rows = await pool.fetch(
            """
            SELECT token_hash, user_id, caller_mode, label,
                   created_at, last_used_at, expires_at, revoked_at
            FROM weft_tokens
            WHERE user_id = $1
            ORDER BY created_at DESC
            """,
            user_id,
        )
    else:
        rows = await pool.fetch(
            """
            SELECT token_hash, user_id, caller_mode, label,
                   created_at, last_used_at, expires_at, revoked_at
            FROM weft_tokens
            WHERE user_id = $1 AND revoked_at IS NULL
            ORDER BY created_at DESC
            """,
            user_id,
        )
    return [_row_to_model(r) for r in rows]
