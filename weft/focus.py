"""weft_focus — post-intent re-prime that surfaces what the generic primer missed.

Focus is a supplement, not a replacement. The caller is responsible for having
already called weft_prime. Focus uses the intent + last session context to build
a compound query, excludes already-primed memories, and returns a tight-budget
result with focused memories, git changes, and last session summary.

Key design choices:
- Default budget 1200 tokens (supplemental, not full primer)
- Differential: excludes memory IDs already surfaced by prime
- No pagination: single shot, no "load more"
- Graceful degradation: works even without git or prior sessions
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import asyncpg

from weft.git_utils import get_recent_commits
from weft.models import MemoryStatus, MemoryType
from weft.session_tracking import get_session_memory_ids
from weft.store import list_memories, search_by_vector
from weft.tokens import estimate_tokens, truncate_to_token_budget

logger = logging.getLogger(__name__)

# Budget allocation — proportional caps within total budget
_BUDGET_SUMMARY = 0.15     # 15% for last session summary
_BUDGET_MEMORIES = 0.65    # 65% for focused memories
_BUDGET_GIT = 0.20         # 20% for git changes
_DEFAULT_BUDGET = 1200


@dataclass
class FocusResult:
    """Structured result from a focus operation."""

    intent: str
    focused_memories: list[dict] = field(default_factory=list)
    last_session_summary: str | None = None
    git_changes: list[str] = field(default_factory=list)
    total_tokens: int = 0
    budget_tokens: int = _DEFAULT_BUDGET
    excluded_count: int = 0

    def format(self) -> str:
        """Format as markdown for agent consumption."""
        parts: list[str] = []

        if self.last_session_summary:
            parts.append(f"## Last Session\n{self.last_session_summary}")

        if self.focused_memories:
            lines = ["## Focused Memories"]
            for mem in self.focused_memories:
                mid = mem.get("id", "?")[:12]
                mtype = mem.get("type", "?")
                content = mem.get("content", "")
                sim = mem.get("similarity", 0)
                lines.append(f"- **[{mtype}]** ({mid}, {sim:.2f}) {content}")
            parts.append("\n".join(lines))

        if self.git_changes:
            lines = ["## Recent Changes"]
            for commit in self.git_changes:
                lines.append(f"- {commit}")
            parts.append("\n".join(lines))

        if not parts:
            return "<!-- weft_focus: no additional context found -->"

        return "\n\n".join(parts)

    def to_dict(self) -> dict:
        """Convert to MCP response dict."""
        return {
            "intent": self.intent,
            "focused_memories": self.focused_memories,
            "last_session_summary": self.last_session_summary,
            "git_changes": self.git_changes,
            "total_tokens": self.total_tokens,
            "budget_tokens": self.budget_tokens,
            "excluded_count": self.excluded_count,
            "formatted": self.format(),
        }


async def build_focus(
    pool: asyncpg.Pool,
    *,
    intent: str,
    embedding_fn,
    project_id: str | None = None,
    agent_id: str | None = None,
    exclude_memory_ids: list[str] | None = None,
    budget_tokens: int = _DEFAULT_BUDGET,
    repo_path: str | None = None,
    limit: int = 10,
    threshold: float = 0.3,
) -> FocusResult:
    """Build a focused context supplement.

    Args:
        pool: Database connection pool.
        intent: What the agent is focusing on (required, non-empty).
        embedding_fn: Async callable that embeds text → vector.
        project_id: Scope to project (or None for global).
        agent_id: Scope to agent (or None for all).
        exclude_memory_ids: Memory IDs already surfaced (e.g., by primer).
            If None, auto-detects from session tracking.
        budget_tokens: Total token budget for the result.
        repo_path: Path to git repo (defaults to cwd).
        limit: Max memories to return.
        threshold: Minimum similarity score.
    """
    if not intent or not intent.strip():
        raise ValueError("intent is required for weft_focus")

    intent = intent.strip()

    # Auto-detect excluded IDs from session tracking if not provided
    if exclude_memory_ids is None:
        exclude_memory_ids = await get_session_memory_ids(pool)

    result = FocusResult(
        intent=intent,
        budget_tokens=budget_tokens,
        excluded_count=len(exclude_memory_ids),
    )

    # Budget allocation
    summary_budget = int(budget_tokens * _BUDGET_SUMMARY)
    memory_budget = int(budget_tokens * _BUDGET_MEMORIES)
    git_budget = int(budget_tokens * _BUDGET_GIT)

    # --- Step 1: Get last session context (handoff memory) ---
    handoff_content: str | None = None
    handoff_at = None
    try:
        handoffs = await list_memories(
            pool,
            memory_type=MemoryType.handoff,
            status=MemoryStatus.active,
            project_id=project_id,
            limit=1,
        )
        if handoffs:
            handoff = handoffs[0]
            handoff_at = handoff.created_at
            handoff_content = handoff.content
            # Truncate for summary display
            summary_text, _ = truncate_to_token_budget(
                handoff_content, summary_budget,
            )
            result.last_session_summary = summary_text
    except Exception as exc:
        logger.warning("Failed to retrieve handoff for focus: %s", exc)

    # --- Step 2: Build compound query and search ---
    # Weight intent higher by repeating it; append session context for breadth
    compound_query = intent
    if handoff_content:
        # Truncate handoff content for embedding (keep it short)
        handoff_for_query = handoff_content[:800]
        compound_query = f"{intent} {intent} {handoff_for_query}"

    try:
        query_embedding = await embedding_fn(compound_query)

        results = await search_by_vector(
            pool,
            query_embedding,
            limit=limit,
            threshold=threshold,
            status=MemoryStatus.active,
            project_id=project_id,
            agent_id=agent_id,
            exclude_ids=exclude_memory_ids if exclude_memory_ids else None,
        )

        # Pack memories within budget
        used_tokens = 0
        for r in results:
            mem_dict = {
                "id": r.memory.id,
                "type": r.memory.type.value,
                "content": r.memory.content,
                "similarity": round(r.similarity, 3),
                "topic": r.memory.topic,
            }
            entry_tokens = estimate_tokens(r.memory.content) + 20  # overhead
            if used_tokens + entry_tokens > memory_budget:
                break
            result.focused_memories.append(mem_dict)
            used_tokens += entry_tokens

    except Exception as exc:
        logger.warning("Focus semantic search failed: %s", exc)

    # --- Step 3: Get recent git changes ---
    try:
        commits = await get_recent_commits(
            since=handoff_at,
            repo_path=repo_path,
        )
        if commits:
            # Truncate to git budget
            used = 0
            for commit in commits:
                cost = estimate_tokens(commit) + 3  # "- " prefix overhead
                if used + cost > git_budget:
                    break
                result.git_changes.append(commit)
                used += cost
    except Exception as exc:
        logger.warning("Focus git integration failed: %s", exc)

    # --- Compute total tokens ---
    result.total_tokens = estimate_tokens(result.format())

    return result
