"""Cost tracking — per-session and per-task token/cost recording.

Tracks API token usage and estimated costs for budgeting and governance.
Budget thresholds are stored as config and checked via check_budget().
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import asyncpg
from pydantic import BaseModel, Field

from weft.db.connection import get_db
from weft.models import _now, _weft_id

logger = logging.getLogger(__name__)


class CostEntryType(str, Enum):
    session = "session"
    task = "task"
    tool_call = "tool_call"


class CostEntry(BaseModel):
    """A single cost record — tokens used and estimated cost for an operation."""

    id: str = Field(default_factory=_weft_id)
    entry_type: CostEntryType = CostEntryType.session
    reference_id: str | None = None  # session_id, task_id, etc.
    model: str | None = None  # e.g. "claude-sonnet-4-20250514"
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    project_id: str | None = None
    agent_id: str | None = None
    user_id: str | None = None
    created_at: datetime = Field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for MCP tool responses."""
        d = self.model_dump(mode="json")
        d["entry_type"] = self.entry_type.value
        return d


class CostEntryCreate(BaseModel):
    """Input model for recording a cost entry."""

    entry_type: CostEntryType = CostEntryType.session
    reference_id: str | None = None
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    project_id: str | None = None
    agent_id: str | None = None


class CostSummary(BaseModel):
    """Aggregated cost summary over a time window."""

    total_entries: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    window_start: datetime | None = None
    window_end: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class BudgetStatus(BaseModel):
    """Result of a budget check."""

    daily_limit_usd: float
    daily_spent_usd: float
    within_budget: bool
    pct_used: float  # 0.0 to 100.0+
    remaining_usd: float

    def to_dict(self) -> dict[str, Any]:
        d = self.model_dump(mode="json")
        d["pct_used"] = round(self.pct_used, 2)
        d["daily_spent_usd"] = round(self.daily_spent_usd, 4)
        d["remaining_usd"] = round(self.remaining_usd, 4)
        return d


# ---------------------------------------------------------------------------
# Store layer
# ---------------------------------------------------------------------------


def _row_to_cost_entry(row: asyncpg.Record) -> CostEntry:
    """Convert a database row to a CostEntry model."""
    metadata = row["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return CostEntry(
        id=row["id"],
        entry_type=row["entry_type"],
        reference_id=row["reference_id"],
        model=row["model"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        total_tokens=row["total_tokens"],
        estimated_cost_usd=float(row["estimated_cost_usd"]),
        metadata=metadata or {},
        project_id=row["project_id"],
        agent_id=row["agent_id"],
        user_id=row["user_id"],
        created_at=row["created_at"],
    )


async def record_cost(
    pool: asyncpg.Pool, create: CostEntryCreate,
) -> CostEntry:
    """Insert a cost entry. Returns the created CostEntry."""
    entry_id = _weft_id()
    metadata_json = json.dumps(create.metadata)

    db = get_db(pool)
    row = await db.fetchrow(
        """
        INSERT INTO cost_entries (
            id, entry_type, reference_id, model,
            input_tokens, output_tokens, total_tokens, estimated_cost_usd,
            metadata, project_id, agent_id, user_id
        )
        VALUES (
            $1, $2, $3, $4,
            $5, $6, $7, $8,
            $9::jsonb, $10, $11,
            nullif(current_setting('app.user_id', true), '')
        )
        RETURNING *
        """,
        entry_id,
        create.entry_type.value,
        create.reference_id,
        create.model,
        create.input_tokens,
        create.output_tokens,
        create.total_tokens,
        create.estimated_cost_usd,
        metadata_json,
        create.project_id,
        create.agent_id,
    )
    return _row_to_cost_entry(row)


async def get_cost_summary(
    pool: asyncpg.Pool,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    entry_type: CostEntryType | None = None,
    project_id: str | None = None,
) -> CostSummary:
    """Aggregate cost entries over a time window."""
    clauses = []
    params: list[Any] = []
    idx = 1

    if since is not None:
        clauses.append(f"created_at >= ${idx}")
        params.append(since)
        idx += 1

    if until is not None:
        clauses.append(f"created_at <= ${idx}")
        params.append(until)
        idx += 1

    if entry_type is not None:
        clauses.append(f"entry_type = ${idx}")
        params.append(entry_type.value)
        idx += 1

    if project_id is not None:
        clauses.append(f"project_id = ${idx}")
        params.append(project_id)
        idx += 1

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    db = get_db(pool)
    row = await db.fetchrow(
        f"""
        SELECT
            count(*) as total_entries,
            coalesce(sum(input_tokens), 0) as total_input_tokens,
            coalesce(sum(output_tokens), 0) as total_output_tokens,
            coalesce(sum(total_tokens), 0) as total_tokens,
            coalesce(sum(estimated_cost_usd), 0) as total_cost_usd,
            min(created_at) as window_start,
            max(created_at) as window_end
        FROM cost_entries
        {where}
        """,
        *params,
    )
    return CostSummary(
        total_entries=row["total_entries"],
        total_input_tokens=row["total_input_tokens"],
        total_output_tokens=row["total_output_tokens"],
        total_tokens=row["total_tokens"],
        total_cost_usd=float(row["total_cost_usd"]),
        window_start=row["window_start"],
        window_end=row["window_end"],
    )


async def check_budget(
    pool: asyncpg.Pool,
    daily_limit_usd: float,
) -> BudgetStatus:
    """Check today's spending against a daily budget limit."""
    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )
    summary = await get_cost_summary(pool, since=today_start)
    spent = summary.total_cost_usd
    pct = (spent / daily_limit_usd * 100) if daily_limit_usd > 0 else 0.0

    return BudgetStatus(
        daily_limit_usd=daily_limit_usd,
        daily_spent_usd=spent,
        within_budget=spent <= daily_limit_usd,
        pct_used=pct,
        remaining_usd=max(0.0, daily_limit_usd - spent),
    )


async def list_cost_entries(
    pool: asyncpg.Pool,
    *,
    entry_type: CostEntryType | None = None,
    reference_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[CostEntry]:
    """List cost entries, newest first."""
    clauses = []
    params: list[Any] = []
    idx = 1

    if entry_type is not None:
        clauses.append(f"entry_type = ${idx}")
        params.append(entry_type.value)
        idx += 1

    if reference_id is not None:
        clauses.append(f"reference_id = ${idx}")
        params.append(reference_id)
        idx += 1

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    rows = await get_db(pool).fetch(
        f"""
        SELECT * FROM cost_entries
        {where}
        ORDER BY created_at DESC
        LIMIT ${idx} OFFSET ${idx + 1}
        """,
        *params,
        limit,
        offset,
    )
    return [_row_to_cost_entry(r) for r in rows]
