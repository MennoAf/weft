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

import logging
from dataclasses import dataclass, field

import asyncpg

from weft.git_utils import get_recent_commits
from weft.models import MemoryStatus, MemoryType
from weft.session_tracking import get_session_memory_ids
from weft.store import get_last_handoff_timestamp, get_memory_changes_since, list_memories, search_by_vector, search_cross_project
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
    cross_project_memories: list[dict] = field(default_factory=list)
    last_session_summary: str | None = None
    git_changes: list[str] = field(default_factory=list)
    changes_since: dict | None = None
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

        if self.cross_project_memories:
            lines = ["## Cross-Project Insights"]
            for mem in self.cross_project_memories:
                proj = mem.get("source_project", "?")
                content = mem.get("content", "")
                sim = mem.get("similarity", 0)
                lines.append(f"- [{proj}] ({sim:.2f}) {content[:120]}")
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
            "cross_project_memories": self.cross_project_memories,
            "last_session_summary": self.last_session_summary,
            "git_changes": self.git_changes,
            "changes_since": self.changes_since,
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
    changes_since: dict | None = None,
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
    if project_id is not None:
        try:
            handoffs = await list_memories(
                pool,
                memory_type=MemoryType.handoff,
                status=MemoryStatus.active,
                project_id=project_id,
                exact_scope=True,
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

    # --- Step 2b: Cross-project search ---
    if project_id is not None:
        try:
            from weft.config import load_config
            cfg = load_config()
            if cfg.retrieval.cross_project_search:
                main_ids = {m["id"] for m in result.focused_memories}
                cross_results = await search_cross_project(
                    pool, query_embedding,
                    exclude_project_id=project_id,
                    limit=cfg.retrieval.cross_project_limit,
                    threshold=threshold,
                    exclude_ids=list(main_ids),
                )
                for r in cross_results:
                    result.cross_project_memories.append({
                        "id": r.memory.id,
                        "type": r.memory.type.value,
                        "content": r.memory.content,
                        "similarity": round(r.similarity, 3),
                        "source_project": r.memory.project_id,
                    })
        except Exception as exc:
            logger.warning("Focus cross-project search failed: %s", exc)

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

    # --- Step 4: Changes since last session ---
    if changes_since is not None:
        result.changes_since = changes_since
    else:
        if project_id is not None:
            try:
                handoff_ts = await get_last_handoff_timestamp(
                    pool, project_id=project_id,
                )
                if handoff_ts is not None:
                    changes = await get_memory_changes_since(
                        pool, since=handoff_ts, project_id=project_id,
                    )
                    changes["recent_commits"] = result.git_changes
                    result.changes_since = changes
            except Exception as exc:
                logger.warning("Focus changes_since failed: %s", exc)

    # --- Compute total tokens ---
    result.total_tokens = estimate_tokens(result.format())

    return result
