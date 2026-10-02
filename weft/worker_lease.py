"""Durable single-worker ownership with database-backed fencing.

A lease is identified by a stable ``lease_key``.  Ownership is the pair of
opaque ``owner_token`` and monotonic ``generation`` returned by acquisition.
All state transitions are one SQL statement and use PostgreSQL's clock so
application clock skew cannot extend an expired owner.  Callers must use
``run_fenced`` for side effects; it verifies ownership immediately before the
callback and refuses to run after a takeover or expiry.
"""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass
from datetime import datetime
from typing import Awaitable, Callable

import asyncpg

DEFAULT_RENEWAL_INTERVAL_SECONDS = 10.0
DEFAULT_LEASE_DURATION_SECONDS = 30.0


class LeaseLostError(RuntimeError):
    """Raised when a worker attempts work after its lease was fenced."""


@dataclass(frozen=True, slots=True)
class LeaseState:
    lease_key: str
    owner_token: str
    generation: int
    acquired_at: datetime
    renewed_at: datetime
    expires_at: datetime


class WorkerLease:
    """Atomic lease-row operations for one worker and one named lease."""

    def __init__(
        self,
        db: asyncpg.Pool | asyncpg.Connection,
        lease_key: str,
        *,
        owner_token: str | None = None,
        lease_duration: float = DEFAULT_LEASE_DURATION_SECONDS,
        renewal_interval: float = DEFAULT_RENEWAL_INTERVAL_SECONDS,
    ) -> None:
        if not lease_key or len(lease_key) > 256:
            raise ValueError("lease_key must contain 1..256 characters")
        if lease_duration <= 0:
            raise ValueError("lease_duration must be positive")
        if renewal_interval <= 0 or renewal_interval >= lease_duration:
            raise ValueError("renewal_interval must be positive and less than lease_duration")
        self.db = db
        self.lease_key = lease_key
        self.owner_token = owner_token or secrets.token_urlsafe(32)
        self.lease_duration = float(lease_duration)
        self.renewal_interval = float(renewal_interval)
        self.generation: int | None = None

    @staticmethod
    def _state(row: asyncpg.Record | None) -> LeaseState | None:
        if row is None:
            return None
        return LeaseState(
            lease_key=row["lease_key"],
            owner_token=row["owner_token"],
            generation=row["generation"],
            acquired_at=row["acquired_at"],
            renewed_at=row["renewed_at"],
            expires_at=row["expires_at"],
        )

    async def acquire(self) -> LeaseState | None:
        """Acquire an unowned/expired row; return None for a live owner.

        The conflict update's predicate is evaluated while holding the unique
        index row lock, making simultaneous workers race-safe.  ``generation``
        advances on every successful ownership transition.
        """
        row = await self.db.fetchrow(
            """
            INSERT INTO worker_leases
                (lease_key, owner_token, generation, acquired_at, renewed_at, expires_at)
            VALUES ($1, $2, 1, clock_timestamp(), clock_timestamp(),
                    clock_timestamp() + ($3 * interval '1 second'))
            ON CONFLICT (lease_key) DO UPDATE
            SET owner_token = EXCLUDED.owner_token,
                generation = worker_leases.generation + 1,
                acquired_at = clock_timestamp(),
                renewed_at = clock_timestamp(),
                expires_at = clock_timestamp() + ($3 * interval '1 second')
            WHERE worker_leases.expires_at IS NULL
               OR worker_leases.expires_at <= clock_timestamp()
            RETURNING lease_key, owner_token, generation, acquired_at, renewed_at, expires_at
            """,
            self.lease_key,
            self.owner_token,
            self.lease_duration,
        )
        state = self._state(row)
        if state is not None:
            self.generation = state.generation
        return state

    async def renew(self) -> LeaseState | None:
        """Extend this exact owner/generation, or return None if fenced."""
        if self.generation is None:
            return None
        row = await self.db.fetchrow(
            """
            UPDATE worker_leases
            SET renewed_at = clock_timestamp(),
                expires_at = clock_timestamp() + ($4 * interval '1 second')
            WHERE lease_key = $1 AND owner_token = $2 AND generation = $3
              AND expires_at > clock_timestamp()
            RETURNING lease_key, owner_token, generation, acquired_at, renewed_at, expires_at
            """,
            self.lease_key,
            self.owner_token,
            self.generation,
            self.lease_duration,
        )
        return self._state(row)

    async def is_owner(self) -> bool:
        """Check current ownership against the database expiry clock."""
        if self.generation is None:
            return False
        return bool(
            await self.db.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1 FROM worker_leases
                    WHERE lease_key = $1 AND owner_token = $2 AND generation = $3
                      AND expires_at > clock_timestamp()
                )
                """,
                self.lease_key,
                self.owner_token,
                self.generation,
            )
        )

    async def release(self) -> bool:
        """Clear this owner only; never release a successor's lease."""
        if self.generation is None:
            return False
        result = await self.db.execute(
            """
            UPDATE worker_leases
            SET owner_token = NULL, acquired_at = NULL, renewed_at = NULL, expires_at = NULL
            WHERE lease_key = $1 AND owner_token = $2 AND generation = $3
            """,
            self.lease_key,
            self.owner_token,
            self.generation,
        )
        return result == "UPDATE 1"

    async def assert_owner(self) -> None:
        """Fail closed unless this worker still owns a non-expired row."""
        if not await self.is_owner():
            raise LeaseLostError(
                f"worker lease {self.lease_key!r} is no longer owned by this worker"
            )

    async def run_fenced(self, callback: Callable[[], Awaitable[object]]) -> object:
        """Guard a side-effect callback with an ownership check.

        The database check and callback cannot form a distributed transaction;
        a callback must therefore be idempotent or use its own fencing token.
        This guard ensures work never starts after a detected loss.
        """
        await self.assert_owner()
        return await callback()

    async def renew_loop(self, stop_event: asyncio.Event) -> None:
        """Renew periodically until stopped; loss terminates the loop."""
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.renewal_interval)
            except asyncio.TimeoutError:
                if await self.renew() is None:
                    raise LeaseLostError(f"worker lease {self.lease_key!r} renewal was fenced")
