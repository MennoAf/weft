"""Weft MCP tools — thin coordinators, ≤15 lines each."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Literal

import asyncpg
from fastmcp import Context

from weft.auth import resolve_caller_user_id
from weft.correlation import set_correlation_id
from weft.db.connection import acquire
from weft.fsck import list_orphan_memories
from weft.mcp.server import AppContext, mcp
from weft.behaviors import (
    delete_behavior,
    list_behaviors as list_behaviors_store,
    match_behaviors,
    store_behavior,
    touch_behavior,
)
from weft.entities import (
    get_entity,
    get_entity_memories,
    link_mention,
    search_entities,
    store_entity,
)
from weft.episodes import (
    add_memory_to_episode,
    close_episode,
    create_episode,
    get_episode,
    get_episode_memories,
    graduate_episode,
    list_episodes,
    timeline_query,
)
from weft.models import (
    BehaviorCreate,
    BehaviorScope,
    EntityCreate,
    EntityType,
    EpisodeCreate,
    EpisodeStatus,
    EpisodeWithMemories,
    MemoryCreate,
    MemorySource,
    MemorySourceLiteral,
    MemoryStatus,
    MemoryType,
    MemoryTypeLiteral,
    ModeCreate,
    ModeWeights,
    NudgeMode,
    NudgeModeLiteral,
    RelationType,
    TrackerCreate,
    TrackerKind,
    TrackerKindLiteral,
    TrackerState,
    TrackerStateLiteral,
)
from weft.quarantine import (
    approve_pending as approve_pending_quarantine,
    list_pending as list_pending_quarantine,
    mark_merge_candidate as mark_merge_candidate_quarantine,
    merge_pending as merge_pending_quarantine,
    reject_pending as reject_pending_quarantine,
)
from weft.tokens import estimate_tokens
from weft.session_tracking import (
    boost_session_memories,
    boost_session_turns,
    log_memory_access,
    log_turn_access,
)
from weft.store import (
    add_relationship,
    count_by_vector,
    delete_memory,
    embed_text_for_memory,
    get_recent_writes,
    get_relationships,
    get_stats,
    list_memories,
    record_feedback,
    remove_relationship,
    search_by_keyword,
    search_by_vector,
    search_cross_project,
    search_hybrid,
    store_memory,
    touch_memory,
    update_memory,
    bump_retrieval_telemetry,
    log_recall_query,
)

logger = logging.getLogger(__name__)

# Exceptions that indicate the database is unreachable (not application logic errors).
_DB_ERRORS = (OSError, asyncpg.PostgresError, asyncpg.InterfaceError, ConnectionRefusedError)

# Exceptions from invalid input (bad enum values, wrong types, etc.)
_INPUT_ERRORS = (ValueError, TypeError)


def _coerce_list(value: str | list | None) -> list | None:
    """Coerce a JSON-string list to a native list.

    Some MCP clients serialize list params as JSON strings instead of arrays.
    E.g. topic arrives as '["a","b"]' instead of ["a","b"]. This helper
    transparently handles that so Pydantic validation succeeds.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return value  # let Pydantic raise if still wrong


_RELATIVE_DATE_RE = re.compile(r"^(\d+)\s*(d|days?|w|weeks?|m|months?)$", re.IGNORECASE)

_UNIT_DAYS = {"d": 1, "day": 1, "days": 1, "w": 7, "week": 7, "weeks": 7, "m": 30, "month": 30, "months": 30}


def _parse_review_after(value: str | None) -> datetime | None:
    """Parse a review_after value: ISO timestamp or relative like '30d', '2w', '3m'."""
    if value is None:
        return None
    match = _RELATIVE_DATE_RE.match(value.strip())
    if match:
        amount, unit = int(match.group(1)), match.group(2).lower()
        days = amount * _UNIT_DAYS[unit]
        return datetime.now(timezone.utc) + timedelta(days=days)
    # Try ISO format
    return datetime.fromisoformat(value)


async def _detect_project_id(ctx: Context) -> str | None:
    """Auto-detect project_id from MCP client roots.

    Uses the directory name of the first root URI as the project identifier.
    E.g. file:///Users/jason/Projects/Weft → "weft"
    Returns None if roots are unavailable or empty.
    """
    try:
        roots = await ctx.list_roots()
        if roots:
            uri = str(roots[0].uri)
            # file:///path/to/ProjectName → "projectname"
            path = uri.replace("file://", "").rstrip("/")
            name = path.rsplit("/", 1)[-1] if "/" in path else path
            return name.lower() or None
    except Exception as e:
        logger.debug("detect_project_id failed: %s", e, exc_info=True)
    return None


_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)


async def _resolve_project_id(ctx: Context, explicit: str | None) -> str | None:
    """Return the explicit project_id if provided, otherwise auto-detect.

    Raises ValueError if the explicit value looks like a UUID — project_ids
    should be human-readable directory names (e.g. 'delphi', not a Loom
    project UUID).
    """
    if explicit is not None:
        if _UUID_RE.match(explicit):
            raise ValueError(
                f"project_id looks like a UUID ({explicit}). "
                "Use the directory/folder name instead (e.g. 'delphi', 'loom', 'weft'). "
                "UUIDs are Loom project IDs — Weft project_ids should be "
                "human-readable names that match the working directory."
            )
        return explicit
    return await _detect_project_id(ctx)


def _input_error_response(tool_name: str, error: Exception) -> dict:
    """Standard error response for invalid input parameters."""
    logger.info("Invalid input in %s: %s", tool_name, error)
    return {"error": "Invalid input", "detail": str(error), "tool": tool_name}


def _db_error_response(tool_name: str, error: Exception) -> dict:
    """Standard error response when database is unavailable."""
    detail = type(error).__name__
    if isinstance(error, asyncpg.InterfaceError):
        detail = "stale connection pool (will auto-recover)"
    elif isinstance(error, ConnectionRefusedError):
        detail = "database unreachable"
    logger.warning("Database unavailable in %s: %s — %s", tool_name, detail, error)
    return {"error": "Database unavailable", "detail": detail, "degraded": True, "tool": tool_name}


def _extract_primer_memory_ids(result: dict) -> list[str]:
    """Extract memory IDs from all sections of a primer result."""
    ids: list[str] = []
    # Sections that contain memory dicts with "id" keys
    for section in ("rules", "handoff", "recent_work", "decisions"):
        for item in result.get(section, []):
            if isinstance(item, dict) and "id" in item:
                ids.append(item["id"])
    # Issues have a nested structure
    issues = result.get("issues", {})
    for item in issues.get("items", []):
        if isinstance(item, dict) and "id" in item:
            ids.append(item["id"])
    return ids


@mcp.tool()
async def weft_remember(
    ctx: Context,
    content: str,
    type: MemoryTypeLiteral = "fact",
    topic: list[str] | None = None,
    source: MemorySourceLiteral = "conversation",
    confidence: float = 0.7,
    project_id: str | None = None,
    agent_id: str | None = None,
    workspace_id: str | None = None,
    check_contradictions: bool = True,
    pinned: bool = False,
    review_after: str | None = None,
    project_facets: list[str] | None = None,
) -> dict:
    """Store a new memory with type, topics, content, confidence, and source.
    If project_id is omitted, auto-detects from the client's working directory.

    workspace_id: optional shared-brain scope. When set, the caller must be
    a member of that workspace; the resulting memory is readable by every
    workspace member. Use ``weft_workspace_create`` first.

    review_after: optional lifecycle date. Accepts ISO timestamp or relative
    durations like '30d', '2w', '3m'. Memories past their review_after date
    are flagged in the primer so the agent can confirm, revise, or archive them.

    project_facets: optional list of project names to pre-seed on the new memory's
    project_facets column. Values are normalized to lowercase. When omitted, the
    column is initialized to [detected project] via init_project_facets (current
    default behaviour). Explicit facets let a caller pre-declare cross-project
    membership (e.g. a belief known to span 'weft' and 'loom') on initial store."""
    try:
        cid = set_correlation_id()
        logger.debug("weft_remember start [%s]", cid)
        from weft.extract import validate_memory_content
        ok, reason = validate_memory_content(content)
        if not ok:
            return _input_error_response(
                "weft_remember",
                ValueError(
                    f"content rejected: {reason}. "
                    "weft_remember requires a meaningful memory body — "
                    "bare headings and trailing-colon fragments are rejected "
                    "to prevent recall pollution. Include the body in the "
                    "content field, or call weft_behavior_add for short rules."
                ),
            )
        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)
        if workspace_id is not None:
            from weft.workspaces import is_member as _ws_is_member
            caller_uid = resolve_caller_user_id()
            async with acquire(app.pool):
                if not await _ws_is_member(app.pool, workspace_id, caller_uid):
                    raise ValueError(
                        f"caller {caller_uid} is not a member of workspace "
                        f"{workspace_id}"
                    )
        create = MemoryCreate(
            type=MemoryType(type),
            content=content,
            topic=_coerce_list(topic) or [],
            source=MemorySource(source),
            confidence=confidence,
            project_id=resolved_project,
            agent_id=agent_id,
            workspace_id=workspace_id,
            pinned=pinned,
            review_after=_parse_review_after(review_after),
        )
        embedding = None
        embedding_failed = False
        try:
            embedding = await app.embedding.embed(
                embed_text_for_memory(content, create.topic)
            )
        except Exception as embed_err:
            logger.warning(
                "Embedding failed for weft_remember, storing without vector: %s",
                embed_err,
            )
            embedding_failed = True
        async with acquire(app.pool):
            # Pre-insert dedup check (requires embedding)
            dedup = None  # track for post-store L2 handling
            if embedding and not pinned:
                from weft.consolidation import check_dedup_on_store
                dedup = await check_dedup_on_store(
                    app.pool, content, embedding,
                    new_confidence=confidence,
                    memory_type=create.type,
                    project_id=resolved_project,
                )
                if dedup.is_duplicate:
                    # facet_appended (cross-project auto-merge) or within-project dedup
                    result = dedup.existing_memory.to_dict()
                    result["dedup"] = dedup.to_dict()
                    if dedup.existing_memory:
                        await app.cache.set_memory(dedup.existing_memory)
                        await app.cache.invalidate_stats()
                    return result

            memory = await store_memory(app.pool, create, embedding=embedding)
            await app.cache.set_memory(memory)
            await app.cache.invalidate_stats()

            # L2 post-store handling ─────────────────────────────────────────
            # (a) Cross-project merge candidate: route through the quarantine review
            #     surface (weft_quarantine_review) so a supervisor can approve/reject.
            if (
                dedup is not None
                and dedup.action == "merge_candidate"
                and dedup.existing_memory is not None
            ):
                # Mark pending_review + link candidate -> existing belief so the
                # quarantine merge action knows what to append the facet to.
                # Without the edge the candidate sits reviewable-but-orphaned
                # (loom-c82bd8d8).
                await mark_merge_candidate_quarantine(
                    app.pool, memory.id, dedup.existing_memory.id,
                )
                memory = memory.model_copy(update={"review_status": "pending_review"})
                logger.info(
                    "weft_remember: merge candidate %s marked pending_review "
                    "(existing=%s, sim=%.3f)",
                    memory.id,
                    dedup.existing_memory.id if dedup.existing_memory else "?",
                    dedup.similarity,
                )

            # (b) Initialize project_facets for project-scoped stores so future
            #     cross-project dedup can find and facet-merge this belief.
            #     Explicit project_facets param (normalized to lowercase) seeds
            #     directly; omitted falls back to auto-detection from project_id.
            if project_facets is not None:
                normalized_facets = [f.lower() for f in _coerce_list(project_facets) or []]
                # UNION: always include the resolved/detected current project (lowercased)
                # so the belief is still boosted in the project it was stored under.
                if resolved_project:
                    rp = resolved_project.lower()
                    if rp not in normalized_facets:
                        normalized_facets = sorted(set(normalized_facets) | {rp})
                from weft.db.connection import get_db as _get_db
                await _get_db(app.pool).execute(
                    "UPDATE memories SET project_facets = $1::text[] WHERE id = $2",
                    normalized_facets,
                    memory.id,
                )
                memory = memory.model_copy(update={"project_facets": normalized_facets})
            elif resolved_project:
                from weft.consolidation import init_project_facets
                await init_project_facets(app.pool, memory.id, resolved_project)
                memory = memory.model_copy(update={"project_facets": [resolved_project.lower()]})
            # ────────────────────────────────────────────────────────────────

            # Canary enrollment: O(1) INSERT, no LLM call (PRD §V5, loom-c27ab1d2).
            # Runs inside the acquire() context so app.user_id GUC is active.
            try:
                from weft.canary import enroll_canary
                await enroll_canary(app.pool, memory.id, content)
            except Exception as _canary_err:
                logger.warning(
                    "weft_remember: canary enrollment failed (non-fatal, id=%s): %s",
                    memory.id, _canary_err,
                )
            result = memory.to_dict()
            if embedding_failed:
                result["warning"] = (
                    "Memory saved but embedding failed — not searchable by "
                    "semantic similarity until next re-embed cycle."
                )
            if check_contradictions and embedding:
                from weft.consolidation import check_contradictions_on_store
                warnings = await check_contradictions_on_store(
                    app.pool, memory.id, embedding,
                    memory_type=create.type,
                    project_id=resolved_project,
                )
                if warnings:
                    result["contradiction_warnings"] = warnings
                    result["contradiction_warnings_text"] = [
                        f'Warning: This may contradict memory {w["memory_id"]}: "{w["content_preview"]}"'
                        for w in warnings
                    ]
                    # Auto-create a contradiction alert so it's visible next session
                    try:
                        from weft.memory_hygiene_alerts import create_contradiction_alert
                        await create_contradiction_alert(
                            app.pool,
                            new_memory_id=memory.id,
                            contradictions=warnings,
                        )
                    except Exception as e:
                        logger.warning("contradiction_alert_dispatch failed: %s", e, exc_info=True)
            return result
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_remember", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_remember", e)


async def _weft_recall_turns(
    ctx: Context,
    *,
    query: str,
    project_id: str | None,
    limit: int,
) -> dict:
    """Turn-tier dispatch for ``weft_recall(tier='turns'|'auto'→turns)``.

    Multi-anchor queries get the per-anchor split (``anchors`` map);
    everything else is a single hybrid recall keyed under the query.
    Resolves the project_id the same way the belief path does so the
    same default-detection rules apply.

    When ``WEFT_HIERARCHICAL=1`` is set in the env, the flat recall path
    is replaced with the hierarchical descent
    (:func:`weft.episode_turns.recall_turns_hierarchical`): rank episodes
    first, then descend to turns scoped to that episode set. Multi-anchor
    splits are skipped under the flag — the descent is a single coarse-
    to-fine pipeline keyed under the original query, which trades the
    per-anchor split for a tighter candidate set. The flag check happens
    here, not inside ``recall_turns_hierarchical`` itself, so the new
    function stays directly testable without env-var dance.
    """
    try:
        cid = set_correlation_id()
        logger.debug("weft_recall.turns start [%s] query=%r", cid, query[:50])
        app: AppContext = ctx.request_context.lifespan_context
        from weft.turn_recall import temporal_anchor

        hierarchical = os.environ.get("WEFT_HIERARCHICAL") == "1"
        resolved_project = await _resolve_project_id(ctx, project_id)
        async with acquire(app.pool):
            if hierarchical:
                from weft.episode_turns import recall_turns_hierarchical
                turns = await recall_turns_hierarchical(
                    app.pool, query,
                    project_id=resolved_project,
                    top_k_episodes=10,
                    top_k_turns=limit,
                    embedder=app.embedding,
                )
                # Keep the same {anchor: [turns]} shape downstream code
                # downstream expects so the dedup / logging blocks work
                # without additional branching. The original-query key
                # mirrors temporal_anchor's no-anchor fallback path.
                anchored = {query: turns}
            else:
                anchored = await temporal_anchor(
                    app.pool, query,
                    project_id=resolved_project,
                    top_k_per_anchor=max(1, limit // 2),
                    embedder=app.embedding,
                )

        # Flatten dedup'd turns for a single ``turns`` array (the most
        # common consumer shape), and surface the per-anchor mapping for
        # callers that want to do anchored arithmetic.
        seen: set[str] = set()
        flat: list[dict] = []
        for turns in anchored.values():
            for t in turns:
                if t.id in seen:
                    continue
                seen.add(t.id)
                flat.append(t.to_dict())
                if len(flat) >= limit:
                    break
            if len(flat) >= limit:
                break

        response: dict = {
            "query": query,
            "tier": "turns",
            "count": len(flat),
            "turns": flat,
        }
        # Only include the anchors map when the planner actually split the
        # query — for single-recall fallback the map is just {query: [...]}
        # which is redundant with `turns` and just costs tokens.
        if len(anchored) > 1 or (
            len(anchored) == 1 and next(iter(anchored)) != query
        ):
            response["anchors"] = {
                anchor: [t.to_dict() for t in turns]
                for anchor, turns in anchored.items()
            }

        # Fire-and-forget: log session access (outside acquire — system-level op).
        # Mirrors the belief-tier wiring at weft_recall above. Dedup across the
        # anchor map so a turn surfaced under multiple anchors is logged once.
        accessed_ids: list[str] = []
        seen_log: set[str] = set()
        for turns in anchored.values():
            for t in turns:
                if t.id not in seen_log:
                    seen_log.add(t.id)
                    accessed_ids.append(t.id)
        if accessed_ids:
            asyncio.create_task(
                log_turn_access(
                    app.pool, accessed_ids, tool_name="recall",
                ),
                name="weft-session-log-recall-turns",
            )
        return response
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_recall", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_recall", e)


async def _weft_recall_both(
    ctx: Context,
    *,
    query: str,
    project_id: str | None,
    agent_id: str | None,
    user_id: str | None,
    limit: int,
    retrieval_mode: str,
    memory_status: MemoryStatus,
    memory_type: MemoryType | None,
    topic: str | None,
) -> dict:
    """Both-tier dispatch: RRF fuse belief-tier memories + turn-tier dialogue.

    Mirrors the scoping discipline of the belief and turn paths so the
    same project_id/agent_id/user_id rules apply consistently — the spec
    is "no cross-project bleed" so all three flow into the belief half,
    project_id flows into both halves.
    """
    try:
        cid = set_correlation_id()
        logger.debug("weft_recall.both start [%s] query=%r", cid, query[:50])
        app: AppContext = ctx.request_context.lifespan_context
        from weft.retrieval_modes import (
            include_agent_provenance,
            sources_for_mode,
            wrap_untrusted_for_face,
        )
        from weft.turn_recall import recall_both

        sources = sources_for_mode(retrieval_mode)
        agent_provenance_ok = include_agent_provenance(retrieval_mode)
        resolved_project = await _resolve_project_id(ctx, project_id)

        async with acquire(app.pool):
            fused = await recall_both(
                app.pool, query,
                project_id=resolved_project,
                top_k=limit,
                embedder=app.embedding,
                agent_id=agent_id,
                user_id=user_id,
                status=memory_status,
                memory_type=memory_type,
                topic=topic,
                sources=sources,
                include_agent_provenance=agent_provenance_ok,
            )

            # Face-mode wrapping for belief-tier payloads — same rule as
            # the belief path uses below. Turns are dialogue traces, not
            # writeable rows, so they don't carry write_provenance.
            if retrieval_mode == "face":
                for entry in fused:
                    if entry["kind"] == "memory":
                        payload = entry["payload"]
                        # MemoryRecall.to_dict() embeds the memory row
                        # plus similarity/relevance_score. write_provenance
                        # comes from the underlying memory dict.
                        wp = payload.get("write_provenance")
                        payload["content"] = wrap_untrusted_for_face(
                            payload["content"], wp,
                        )

        # Fire-and-forget: log access to BOTH memory + turn IDs so the
        # session-tracking layer keeps both surfaces in its working set.
        memory_ids = [
            e["payload"]["id"] for e in fused if e["kind"] == "memory"
        ]
        turn_ids = [
            e["payload"]["id"] for e in fused if e["kind"] == "turn"
        ]
        # Compounding-loop Step 1 (v49): bump per-memory retrieval telemetry
        # for memory-kind entries only. Turns are tracked separately via
        # turn_access_log (v46) and have no analogous column pair.
        if memory_ids:
            await bump_retrieval_telemetry(app.pool, memory_ids)
        if memory_ids:
            asyncio.create_task(
                log_memory_access(
                    app.pool, memory_ids, "recall",
                    retrieval_mode=retrieval_mode,
                ),
                name="weft-session-log-recall-both-memory",
            )
        if turn_ids:
            asyncio.create_task(
                log_turn_access(
                    app.pool, turn_ids, tool_name="recall",
                ),
                name="weft-session-log-recall-both-turn",
            )

        return {
            "query": query,
            "tier": "both",
            "count": len(fused),
            "results": fused,
        }
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_recall", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_recall", e)


@mcp.tool()
async def weft_recall(
    ctx: Context,
    query: str,
    topic: str | None = None,
    type: MemoryTypeLiteral | None = None,
    status: str | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
    limit: int = 10,
    threshold: float = 0.3,
    mode: str = "hybrid",
    retrieval_mode: str = "face",
    user_id: str | None = None,
    tier: str = "auto",
) -> dict:
    """Retrieve memories by semantic query, keyword search, or hybrid (default).

    mode: 'semantic' (vector only), 'keyword' (BM25 full-text only), or 'hybrid' (RRF fusion of both).
    Hybrid mode combines vector similarity and BM25 keyword matching using Reciprocal Rank Fusion.
    Keyword mode does not require embeddings and works on exact/stemmed word matches.

    retrieval_mode: 'face' (default — excludes codebase ingest noise), 'code' (includes ingest
    and code-context sources for agent-in-repo queries), or 'all' (no source filter).

    user_id: Caller identity override. Defaults to the authenticated caller
    (resolve_caller_user_id): the request credential's user_id when present,
    else this installation's UUID.
    Filters to user-owned rows OR truly-global rows (user_id IS NULL).
    retrieval_mode and scope are orthogonal — user_id composes independently with both.
    These are three independent knobs: retrieval_mode (face/code/all), scope (user/project/agent),
    and user_id — none implies the other.

    tier: 'belief' (default belief-tier semantic recall over `memories`), 'turns' (turn-tier
    raw dialogue trace over `episode_turns`, with multi-anchor splitting on temporal queries),
    'both' (RRF-fused belief + turn results for episodic-recall queries), or 'auto'
    (regex-based query planner routes episodic markers to 'both', temporal markers to 'turns',
    everything else to 'belief'). When the chosen tier is 'turns', the response carries
    `tier: "turns"` and a `turns` array instead of `results`; multi-anchor temporal queries
    also include an `anchors` mapping from anchor phrase → returned turns so a Reader can do
    anchored arithmetic. When the chosen tier is 'both', the response carries `tier: "both"`
    and a `results` array of unified entries `{kind, payload, rank, rrf_score}` where `kind`
    is 'memory' or 'turn'.
    """
    if user_id is None:
        user_id = resolve_caller_user_id()

    # Tier dispatch happens FIRST. The turns and both paths don't reuse
    # the belief-tier helpers below — they have their own scoring,
    # response shapes, and (for multi-anchor queries) sub-recall splits.
    if tier not in ("auto", "belief", "turns", "both"):
        return _input_error_response(
            "weft_recall",
            ValueError(
                f"tier must be 'auto', 'belief', 'turns', or 'both'; got {tier!r}"
            ),
        )
    if tier == "auto":
        from weft.turn_recall import route_query_to_tier
        tier = route_query_to_tier(query)

    # Step 1.5 (v50) — fire-and-forget query log. Captures EVERY weft_recall
    # invocation regardless of which downstream tier path runs (belief / turns /
    # both / belief-view fallback). Logged after tier resolution so the row
    # carries the concrete tier the call actually executed against, not 'auto'.
    app_for_log: AppContext = ctx.request_context.lifespan_context
    asyncio.create_task(
        log_recall_query(
            app_for_log.pool,
            tool_name="recall",
            query_text=query,
            project_id=project_id,
            tier=tier,
            mode=mode,
            retrieval_mode=retrieval_mode,
        ),
        name="weft-recall-query-log",
    )

    # Never-miss fallback annotation. When the router sends a query to the
    # turns tier but that tier has nothing, we recover with belief recall
    # instead of returning an empty hand (see the turns dispatch below). This
    # carries a transparency marker onto the belief response so a consumer can
    # tell a fallback happened — structured over inferred.
    tier_fallback: dict | None = None

    if tier == "turns":
        turns_response = await _weft_recall_turns(
            ctx,
            query=query,
            project_id=project_id,
            limit=limit,
        )
        # Resilience over routing precision: the tier router is a single hard
        # regex guess, and when it guesses wrong (e.g. an entity/belief query
        # that merely contains a temporal word like "before our meeting") the
        # turns tier answers empty and the caller gets nothing. Rather than
        # tune the router to the exact phrasings that trip it, recover here:
        # an empty turns result falls through to belief recall so the answer
        # still surfaces if belief holds it. Fires ONLY on a genuinely empty
        # turns result, so temporal queries that DO have turn answers are
        # untouched — this protects the turn tier's purpose instead of narrowing
        # it. Cost: a second recall pass on the (rare) empty-turns path.
        if turns_response.get("count", 0) > 0:
            return turns_response
        logger.debug(
            "weft_recall: turns tier empty for %r — falling back to belief",
            query[:50],
        )
        tier = "belief"
        tier_fallback = {"from": "turns", "reason": "empty_turns_result"}
    if tier == "both":
        return await _weft_recall_both(
            ctx,
            query=query,
            project_id=project_id,
            agent_id=agent_id,
            user_id=user_id,
            limit=limit,
            retrieval_mode=retrieval_mode,
            memory_status=MemoryStatus(status) if status else MemoryStatus.active,
            memory_type=MemoryType(type) if type else None,
            topic=topic,
        )

    # === Belief-view primary lookup (loom-719e74b0) ===
    # When tier='belief', first check the new belief_claims table for active
    # claims that match the query.  Returns them in the standard response shape
    # if any are found, skipping the legacy memories search entirely.
    # If the primary lookup returns nothing (or errors), falls through to the
    # existing memories search below — the legacy path is NEVER broken.
    if tier == "belief":
        try:
            app: AppContext = ctx.request_context.lifespan_context
            from weft.views.belief_query import search_belief_claims
            async with acquire(app.pool):
                claim_results = await search_belief_claims(
                    app.pool,
                    query=query,
                    user_id=user_id,
                    scope="global",
                    limit=limit,
                )
            if claim_results:
                claim_response = {
                    "query": query,
                    "mode": mode,
                    "tier": "belief-view",
                    "count": len(claim_results),
                    "results": [r.to_recall_dict() for r in claim_results],
                }
                if tier_fallback:
                    claim_response["tier_fallback"] = tier_fallback
                return claim_response
        except (
            asyncpg.PostgresConnectionError,
            asyncpg.InterfaceError,
            ConnectionError,
            asyncio.TimeoutError,
            ImportError,
        ) as exc:
            # Hard rule: belief-view failures must NOT break the legacy
            # belief tier.  Log and fall through to the existing memories search.
            # Only transient / infrastructure errors are caught here — programming
            # errors (ValueError, KeyError) and integrity violations propagate so
            # the caller learns about bugs.
            logger.warning(
                "belief_view_query failed; falling back to memories search: %s",
                exc,
                exc_info=True,
            )

    # --- Enumeration-intent router (Phase 1, V7) ---
    # Detect "list all / every / how many / enumerate" and fire the deterministic
    # complete gather CONCURRENTLY with the top-k search below. The gap between
    # what similarity surfaces (top-k) and what membership knows (the gather) is
    # then returned as a reconciliation header instead of silently dropping
    # sub-cutoff members. Best-effort: gather_enumeration swallows its own errors,
    # and the reconcile step is wrapped, so this never breaks recall.
    enum_task: asyncio.Task | None = None
    if tier == "belief":
        try:
            from weft.enumeration_router import detect_enumeration_intent, gather_enumeration
            is_enum, enum_noun = detect_enumeration_intent(query)
            enum_target = topic or enum_noun
            if is_enum and enum_target:
                app_enum: AppContext = ctx.request_context.lifespan_context
                enum_task = asyncio.create_task(
                    gather_enumeration(app_enum.pool, enum_target, user_id),
                    name="weft-recall-enumeration-gather",
                )
        except Exception as exc:  # noqa: BLE001 - never break recall on router setup
            logger.debug("enumeration router setup failed: %s", exc, exc_info=True)
            enum_task = None

    try:
        cid = set_correlation_id()
        logger.debug("weft_recall start [%s] query=%r mode=%s", cid, query[:50], mode)
        app: AppContext = ctx.request_context.lifespan_context
        memory_type = MemoryType(type) if type else None
        memory_status = MemoryStatus(status) if status else MemoryStatus.active

        if mode not in ("semantic", "keyword", "hybrid"):
            mode = "hybrid"

        from weft.retrieval_modes import (
            include_agent_provenance,
            sources_for_mode,
            wrap_untrusted_for_face,
        )
        sources = sources_for_mode(retrieval_mode)
        # Phase 2 Layer 2: agent-context retrieval (mode='code') default-
        # excludes agent-provenance rows; Face/all retrieval keeps them
        # and the projection below wraps them with the untrusted-write
        # prefix so Jason can tell what came from an agent.
        agent_provenance_ok = include_agent_provenance(retrieval_mode)

        # Keyword mode doesn't need an embedding
        embedding = None
        if mode in ("semantic", "hybrid"):
            embedding = await app.embedding.embed(query)

        # Facet-boost recall (associative / face path):
        # Drop the hard project wall and instead rank by facet overlap.
        # The single query surfaces beliefs across all their registered
        # projects, with a _FACET_BOOST multiplier when the current project
        # is in project_facets.  The separate cross-project second-pass is
        # retired for this path — a faceted belief surfaces through the
        # primary ranked results, not a penalised second list.
        #
        # Catalog path (retrieval_mode='code'): the hard project_id wall is
        # preserved unchanged — ingest scoping must not leak across repos.
        facet_boost_project_id: str | None = None
        if retrieval_mode == "face":
            _raw_boost_id = await _resolve_project_id(ctx, project_id)
            facet_boost_project_id = _raw_boost_id.lower() if _raw_boost_id else None

        # When facet boost is active, suppress the project wall in search calls.
        _search_project_id = None if retrieval_mode == "face" else project_id

        async with acquire(app.pool):
            if mode == "keyword":
                results = await search_by_keyword(
                    app.pool,
                    query,
                    limit=limit,
                    status=memory_status,
                    memory_type=memory_type,
                    topic=topic,
                    project_id=_search_project_id,
                    agent_id=agent_id,
                    sources=sources,
                    user_id=user_id,
                    include_agent_provenance=agent_provenance_ok,
                    facet_boost_project_id=facet_boost_project_id,
                )
            elif mode == "hybrid":
                results = await search_hybrid(
                    app.pool,
                    query,
                    embedding,
                    limit=limit,
                    threshold=threshold,
                    status=memory_status,
                    memory_type=memory_type,
                    topic=topic,
                    project_id=_search_project_id,
                    agent_id=agent_id,
                    sources=sources,
                    user_id=user_id,
                    include_agent_provenance=agent_provenance_ok,
                    facet_boost_project_id=facet_boost_project_id,
                )
            else:  # semantic
                results = await search_by_vector(
                    app.pool,
                    embedding,
                    limit=limit,
                    threshold=threshold,
                    status=memory_status,
                    memory_type=memory_type,
                    topic=topic,
                    project_id=_search_project_id,
                    agent_id=agent_id,
                    sources=sources,
                    user_id=user_id,
                    include_agent_provenance=agent_provenance_ok,
                    facet_boost_project_id=facet_boost_project_id,
                )

            # Touch accessed memories and enrich with entities. When the
            # caller is reading in Face mode, agent-provenance rows are
            # included but their content is wrapped with the untrusted-
            # write prefix so an injected memory cannot impersonate a
            # self-authored fact (Phase 2 / Layer 2).
            enriched = []
            for r in results:
                await touch_memory(app.pool, r.memory.id)
                d = r.to_dict()
                if retrieval_mode == "face":
                    d["content"] = wrap_untrusted_for_face(
                        d["content"], r.memory.write_provenance,
                    )
                try:
                    from weft.entities import get_memory_entities
                    ents = await get_memory_entities(app.pool, r.memory.id)
                    if ents:
                        d["entities"] = [{"name": e.name, "type": e.entity_type.value} for e in ents]
                except Exception as e:
                    logger.debug("entity_enrichment failed for %s: %s", r.memory.id, e, exc_info=True)
                enriched.append(d)

            # Count total matches above threshold (semantic/hybrid only)
            total_matches = None
            if mode in ("semantic", "hybrid") and embedding is not None:
                total_matches = await count_by_vector(
                    app.pool,
                    embedding,
                    threshold=threshold,
                    status=memory_status,
                    memory_type=memory_type,
                    topic=topic,
                    project_id=project_id,
                    agent_id=agent_id,
                    sources=sources,
                    include_agent_provenance=agent_provenance_ok,
                )

            response: dict = {"query": query, "mode": mode, "count": len(results), "results": enriched}
            if tier_fallback:
                response["tier_fallback"] = tier_fallback
            if total_matches is not None and total_matches > len(results):
                response["total_matches"] = total_matches
                response["showing"] = f"Showing {len(results)} of {total_matches} matches"

            # Cross-project search: secondary pass for non-face modes only.
            # Face mode uses facet-boost in the primary query (above), so
            # cross-project beliefs already surface there — no separate pass.
            if embedding is not None and retrieval_mode != "face":
                resolved_project = await _resolve_project_id(ctx, project_id)
                if resolved_project is not None:
                    try:
                        from weft.config import load_config
                        cfg = load_config()
                        if cfg.retrieval.cross_project_search:
                            main_ids = {r.memory.id for r in results}
                            cross_results = await search_cross_project(
                                app.pool, embedding,
                                exclude_project_id=resolved_project,
                                limit=cfg.retrieval.cross_project_limit,
                                threshold=threshold,
                                status=memory_status,
                                memory_type=memory_type,
                                exclude_ids=list(main_ids),
                                sources=sources,
                                include_agent_provenance=agent_provenance_ok,
                            )
                            if cross_results:
                                response["cross_project"] = [
                                    {
                                        **r.to_dict(),
                                        "source_project": r.memory.project_id,
                                    }
                                    for r in cross_results
                                ]
                    except Exception as exc:
                        logger.warning("Cross-project search failed: %s", exc)

        # Compounding-loop Step 1 (v49): bump per-memory retrieval telemetry
        # for every id surfaced in this response — primary AND cross-project.
        # Awaited inline so the writes commit before the session-access log
        # task scheduled below races on the same rows.
        bump_ids = [r.memory.id for r in results]
        if "cross_project" in response:
            bump_ids.extend(r["id"] for r in response["cross_project"])
        if bump_ids:
            await bump_retrieval_telemetry(app.pool, bump_ids)

        # Fire-and-forget: log session access (outside acquire — system-level op)
        if results:
            asyncio.create_task(
                log_memory_access(
                    app.pool,
                    [r.memory.id for r in results],
                    "recall",
                    retrieval_mode=retrieval_mode,
                ),
                name="weft-session-log-recall",
            )

        # --- Enumeration answer (Phase 1, V7 → V8: unambiguous) ---
        # For an enumeration ask ("how many plants", "list all my meds") the
        # limit-bounded top-k `results` is NOT the answer — it is a relevance
        # slice of the whole corpus, so its length over- or under-counts. Await
        # the concurrent complete gather and hand the agent the answer as
        # explicit structured fields it can KNOW rather than infer (design
        # principle: prefer structured data over forcing the model to guess):
        #   * response["enumeration"]["count"]   — THE count ("how many")
        #   * response["enumeration"]["members"] — THE complete list ("list
        #     all"), already assembled; no union of results + remainder needed
        # When the gather is COMPLETE we also correct the top-level `count` so
        # the most-obvious field is the right one — don't leave a misleading
        # count next to the real answer. `results` stays the ranked top-k so
        # relevance ordering is still available (the answer, if not in the top
        # k, is in enumeration.members).
        if enum_task is not None:
            try:
                resolved_tags, gather_result = await enum_task
                if gather_result is not None and gather_result["memories"]:
                    members = gather_result["memories"]
                    member_count = len(members)
                    complete = gather_result["complete"]
                    shown_ids = {r.memory.id for r in results}
                    extra_in_members = sum(
                        1 for m in members if m.id not in shown_ids
                    )
                    response["enumeration"] = {
                        "target": enum_target,
                        "resolved_tags": resolved_tags,
                        "count": member_count,
                        "complete": complete,
                        "truncated": gather_result["truncated"],
                        "similarity_count": len(results),
                        "members": [
                            {
                                "id": m.id,
                                "type": m.type.value,
                                "content": m.content,
                                "topic": m.topic,
                                "created_at": m.created_at.isoformat(),
                            }
                            for m in members
                        ],
                        "summary": (
                            f"{member_count} {enum_target}: the complete set is in "
                            f"enumeration.members; results shows the top "
                            f"{len(results)} by relevance "
                            f"({extra_in_members} more only in enumeration.members)"
                        ),
                    }
                    # Make the most-obvious field correct — but only when the
                    # gather is COMPLETE (a truncated gather must not assert an
                    # exact count). The corpus-wide match tally is misleading as
                    # an answer to "how many X", so drop it here.
                    if complete:
                        response["count"] = member_count
                        response.pop("total_matches", None)
                        response.pop("showing", None)
            except Exception as exc:  # noqa: BLE001 - enumeration augmentation never breaks recall
                logger.warning("enumeration answer assembly failed: %s", exc, exc_info=True)

        return response
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_recall", e)
    except _DB_ERRORS as e:
        logger.warning("Database unavailable in weft_recall: %s", e)
        from weft.fallback import search_fallback
        results = search_fallback(query, limit=limit)
        return {"query": query, "mode": mode, "count": len(results), "results": results, "degraded": True}


@mcp.tool()
async def weft_forget(
    ctx: Context,
    memory_id: str,
    hard: bool = False,
) -> dict:
    """Archive a memory (soft-delete) or hard-delete it."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        async with acquire(app.pool):
            deleted = await delete_memory(app.pool, memory_id, hard=hard)
            await app.cache.invalidate_memory(memory_id)
            await app.cache.invalidate_stats()
            return {"memory_id": memory_id, "deleted": deleted, "hard": hard}
    except _DB_ERRORS as e:
        return _db_error_response("weft_forget", e)


@mcp.tool()
async def weft_quarantine_review(
    ctx: Context,
    action: Literal["list", "approve", "reject", "merge"] = "list",
    memory_id: str | None = None,
    limit: int = 50,
) -> dict:
    """Phase 2 / Layer 3 — review agent-provenance writes flagged as
    instruction-shaped at write-time, plus cross-project merge candidates.

    Actions:
    * ``list`` (default): return pending memories with their content,
      provenance, and origin so the supervisor can decide. Rows that are
      cross-project merge candidates (L2) carry a non-null
      ``merge_target_id`` — the existing belief they would merge into.
    * ``approve``: re-provenance the row to 'supervisor' and flip
      ``review_status`` back to 'active'. Surfaces normally afterwards.
      For a merge candidate this is the *keep-separate* outcome (promote
      it to its own active belief).
    * ``reject``: hard-delete the row. Use when the heuristic correctly
      caught injected / poisoned content.
    * ``merge``: cross-project merge-candidate only — append the
      candidate's project facet to its linked target belief and archive
      the candidate. Errors if the row is not a pending merge candidate.

    This tool is supervisor-only at the trust-tier level — calling it
    from agent-mode context defeats the whole layer. The Phase 2 gate
    is enforced here in the tool body, not at the MCP boundary.
    """
    try:
        if action == "list":
            app: AppContext = ctx.request_context.lifespan_context
            async with acquire(app.pool):
                items = await list_pending_quarantine(app.pool, limit=limit)
            return {"count": len(items), "pending": items}

        if memory_id is None:
            return {"error": f"memory_id required for action='{action}'"}

        from weft.auth import is_agent_caller
        if is_agent_caller():
            # An agent-mode caller approving its own quarantined writes
            # would round-trip the entire defense to zero. Refuse loudly.
            return {
                "error": (
                    "weft_quarantine_review approve/reject are supervisor-only "
                    "(Phase 2 / Layer 3)"
                ),
            }

        app: AppContext = ctx.request_context.lifespan_context
        async with acquire(app.pool):
            if action == "approve":
                ok = await approve_pending_quarantine(app.pool, memory_id)
                if ok:
                    await app.cache.invalidate_memory(memory_id)
                return {"memory_id": memory_id, "approved": ok}
            if action == "reject":
                ok = await reject_pending_quarantine(app.pool, memory_id)
                if ok:
                    await app.cache.invalidate_memory(memory_id)
                return {"memory_id": memory_id, "rejected": ok}
            if action == "merge":
                result = await merge_pending_quarantine(app.pool, memory_id)
                if result is None:
                    return {
                        "memory_id": memory_id,
                        "merged": False,
                        "error": (
                            "not a pending cross-project merge candidate "
                            "(no merge_candidate link)"
                        ),
                    }
                # Invalidate both the archived candidate and the updated target.
                await app.cache.invalidate_memory(memory_id)
                await app.cache.invalidate_memory(result["target_id"])
                await app.cache.invalidate_stats()
                return {"memory_id": memory_id, "merged": True, **result}

        return {"error": f"unknown action: {action}"}
    except _DB_ERRORS as e:
        return _db_error_response("weft_quarantine_review", e)


@mcp.tool()
async def weft_context(
    ctx: Context,
    query: str,
    budget_tokens: int = 4000,
    topic: str | None = None,
    type: MemoryTypeLiteral | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
    max_per_topic: int = 3,
) -> dict:
    """Budget-aware context loading: best memories for a situation within N tokens."""
    try:
        cid = set_correlation_id()
        logger.debug("weft_context start [%s] budget=%d", cid, budget_tokens)
        from weft.context import build_context

        app: AppContext = ctx.request_context.lifespan_context
        memory_type = MemoryType(type) if type else None
        embedding = await app.embedding.embed(query)
        async with acquire(app.pool):
            result = await build_context(
                app.pool, embedding,
                budget_tokens=budget_tokens, max_per_topic=max_per_topic,
                memory_type=memory_type, topic=topic, project_id=project_id,
                agent_id=agent_id,
            )
            # Touch the memories that made it into context
            for mem_dict in result["memories"]:
                await touch_memory(app.pool, mem_dict["id"])

        # Fire-and-forget: log session access (outside acquire — system-level op)
        import asyncio
        mem_ids = [m["id"] for m in result["memories"]]
        if mem_ids:
            asyncio.create_task(
                log_memory_access(app.pool, mem_ids, "context"),
                name="weft-session-log-context",
            )

        return result
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_context", e)
    except _DB_ERRORS as e:
        logger.warning("Database unavailable in weft_context: %s", e)
        from weft.fallback import search_fallback
        results = search_fallback(query)
        return {
            "memories": results,
            "count": len(results),
            "tokens_used": 0,
            "budget_tokens": budget_tokens,
            "degraded": True,
        }


@mcp.tool()
async def weft_revise(
    ctx: Context,
    memory_id: str,
    new_content: str,
    new_confidence: float | None = None,
    new_topic: list[str] | None = None,
    new_type: MemoryTypeLiteral | None = None,
    new_project_id: str | None = None,
    new_pinned: bool | None = None,
    review_after: str | None = None,
) -> dict:
    """Update a memory's content, creating a new version that supersedes the old one.

    Preserves the predecessor's pinned state and project_id by default — pass
    new_pinned/new_project_id only when explicitly changing them.

    new_project_id: optional project_id for the new version. Use this to fix
    memories saved under the wrong project (e.g. a UUID instead of the
    directory name). Must be a human-readable name, not a UUID.

    new_pinned: optional explicit pin override for the new version. Omit to
    inherit the predecessor's pin state.

    review_after: optional lifecycle date for the new version. Accepts ISO
    timestamp or relative durations like '30d', '2w', '3m'."""
    try:
        from weft.models import MemoryType as _MT
        from weft.revise import revise_memory

        resolved_type = _MT(new_type) if new_type else None
        if new_project_id is not None and _UUID_RE.match(new_project_id):
            return _input_error_response(
                "weft_revise",
                ValueError(
                    f"new_project_id looks like a UUID ({new_project_id}). "
                    "Use the directory/folder name instead."
                ),
            )
        # Build kwargs conditionally so omitted MCP args preserve predecessor
        # state via revise_memory's _UNSET sentinels. Passing None explicitly
        # would overwrite the inherited value with NULL.
        revise_kwargs: dict = {}
        if new_project_id is not None:
            revise_kwargs["new_project_id"] = new_project_id
        if new_pinned is not None:
            revise_kwargs["new_pinned"] = new_pinned
        if review_after is not None:
            parsed = _parse_review_after(review_after)
            if parsed is not None:
                revise_kwargs["review_after"] = parsed
        app: AppContext = ctx.request_context.lifespan_context
        embedding = await app.embedding.embed(
            embed_text_for_memory(new_content, _coerce_list(new_topic))
        )
        async with acquire(app.pool):
            new, old = await revise_memory(
                app.pool, memory_id, new_content,
                embedding=embedding, new_confidence=new_confidence,
                new_topic=_coerce_list(new_topic), new_type=resolved_type,
                **revise_kwargs,
            )
            await app.cache.set_memory(new)
            await app.cache.invalidate_memory(old.id)
            await app.cache.invalidate_stats()
            return {"new": new.to_dict(), "superseded": old.to_dict()}
    except _DB_ERRORS as e:
        return _db_error_response("weft_revise", e)


@mcp.tool()
async def weft_relate(
    ctx: Context,
    action: str,
    memory_id: str,
    target_id: str | None = None,
    relation: str | None = None,
) -> dict:
    """Manage relationships between memories: add, get, or remove."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        async with acquire(app.pool):
            if action == "add":
                rel = await add_relationship(app.pool, source_id=memory_id, target_id=target_id, relation=RelationType(relation))
                return {"source_id": rel.source_id, "target_id": rel.target_id, "relation": rel.relation.value, "created_at": rel.created_at.isoformat()}
            elif action == "get":
                rels = await get_relationships(app.pool, memory_id, relation=RelationType(relation) if relation else None)
                return {"memory_id": memory_id, "count": len(rels), "relationships": [
                    {"source_id": r.source_id, "target_id": r.target_id, "relation": r.relation.value, "created_at": r.created_at.isoformat()} for r in rels
                ]}
            elif action == "remove":
                removed = await remove_relationship(app.pool, source_id=memory_id, target_id=target_id, relation=RelationType(relation))
                return {"memory_id": memory_id, "target_id": target_id, "relation": relation, "removed": removed}
            else:
                return {"error": f"Unknown action: {action}. Use 'add', 'get', or 'remove'."}
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_relate", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_relate", e)


@mcp.tool()
async def weft_consolidate(ctx: Context, dry_run: bool = False) -> dict:
    """Run consolidation: decay stale memories, merge duplicates, flag contradictions."""
    try:
        cid = set_correlation_id()
        logger.debug("weft_consolidate start [%s] dry_run=%s", cid, dry_run)
        from weft.consolidation import consolidate
        app: AppContext = ctx.request_context.lifespan_context
        report = await consolidate(app.pool, dry_run=dry_run)
        await app.cache.invalidate_stats()
        return report.to_dict()
    except _DB_ERRORS as e:
        return _db_error_response("weft_consolidate", e)


@mcp.tool()
async def weft_feedback(
    ctx: Context,
    memory_id: str,
    helpful: bool,
) -> dict:
    """Record whether a memory was helpful. Adjusts usefulness score for future ranking."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        async with acquire(app.pool):
            result = await record_feedback(app.pool, memory_id, helpful)
            await app.cache.invalidate_memory(memory_id)
            return result
    except _DB_ERRORS as e:
        return _db_error_response("weft_feedback", e)


@mcp.tool()
async def weft_pin(
    ctx: Context,
    memory_id: str,
    pinned: bool = True,
) -> dict:
    """Pin or unpin a memory. Pinned memories are always included in prime and context calls."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        async with acquire(app.pool):
            updated = await update_memory(app.pool, memory_id, pinned=pinned)
            if not updated:
                return {"error": f"Memory {memory_id} not found"}
            await app.cache.invalidate_memory(memory_id)
            return {"memory_id": memory_id, "pinned": updated.pinned}
    except _DB_ERRORS as e:
        return _db_error_response("weft_pin", e)


@mcp.tool()
async def weft_prime(
    ctx: Context,
    project_id: str | None = None,
    agent_id: str | None = None,
    budget_tokens: int = 2400,
    query: str | None = None,
    disclosure: Literal["full", "progressive"] = "progressive",
    mode: str | None = None,
) -> dict:
    """Session primer: assemble structured context for session startup.
    If project_id is omitted, auto-detects from the client's working directory.

    query: optional intent/topic string to bias which behaviors, decisions,
    issues, and recent work are surfaced. When provided, those sections use
    semantic similarity to rank more relevant items higher.

    disclosure: 'progressive' (default) returns tier-1 sections (rules,
    handoff, issues, anti-patterns) with full content, and tier-2
    sections (decisions, recent_work, behaviors, entities) as counts
    only. Use weft_focus to load deferred sections when relevant.
    'full' returns all sections with content.

    mode: optional name of a retrieval mode/persona (e.g., 'research',
    'coding'). Adjusts section weights via ModeWeights — behavior_boost
    and entity_boost scale section token caps, recency_bias shifts
    milestone ranking toward recency. Falls back to defaults if not found."""
    try:
        cid = set_correlation_id()
        logger.debug("weft_prime start [%s] project=%s", cid, project_id)
        from weft.primer import build_primer

        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)

        # Compute query embedding if provided (best-effort).
        query_vec: list[float] | None = None
        if query and query.strip():
            try:
                query_vec = await app.embedding.embed(query.strip())
            except Exception as exc:
                logger.warning("Failed to embed primer query (non-fatal): %s", exc)

        # NOTE: build_primer uses asyncio.gather for parallel section fetches,
        # so we do NOT wrap it in acquire() — concurrent queries on a shared
        # connection would crash.  RLS SELECT policies handle NULL user_id
        # gracefully (showing global rows).
        result = await build_primer(
            app.pool,
            project_id=resolved_project,
            agent_id=agent_id,
            budget_tokens=budget_tokens,
            query_vec=query_vec,
            disclosure=disclosure,
            mode=mode,
        )

        # Reconciliation-meter health (recall canary). Attached as a tier-1
        # field so a DARK meter screams on every prime — the load-bearing
        # surface that makes "the meter went dark" impossible to miss. Scoped
        # to the authenticated caller explicitly (not the GUC). Best-effort:
        # never let the meter's own health break the primer.
        try:
            from weft.canary import canary_health
            _ch = await canary_health(app.pool, resolve_caller_user_id())
            if _ch:
                result["recall_canary"] = _ch
        except Exception as exc:
            logger.debug("recall_canary health skipped (non-fatal): %s", exc)

        # Fire-and-forget tasks run without acquire — they're system-level ops
        # that don't need user scoping.
        try:
            import asyncio
            from weft.consolidation import consolidate_if_due
            asyncio.create_task(
                consolidate_if_due(app.pool),
                name="weft-auto-consolidation",
            )
        except Exception as exc:
            logger.debug("Auto-consolidation scheduling skipped: %s", exc)

        try:
            primer_mem_ids = _extract_primer_memory_ids(result)
            if primer_mem_ids:
                asyncio.create_task(
                    log_memory_access(app.pool, primer_mem_ids, "prime"),
                    name="weft-session-log-prime",
                )
        except Exception as exc:
            logger.debug("Prime session logging skipped: %s", exc)

        return result
    except _DB_ERRORS as e:
        logger.warning("Database unavailable in weft_prime: %s", e)
        from weft.fallback import read_fallback
        content = read_fallback()
        from weft.primer import _ONBOARDING_TEXT, _SECTION_HINTS
        from weft.tokens import estimate_tokens, truncate_to_token_budget

        # Truncate fallback content to budget — raw exports can be huge.
        handoff_section: list[dict] = []
        if content:
            cost = estimate_tokens(content)
            if cost > budget_tokens:
                content, cost = truncate_to_token_budget(content, budget_tokens)
            handoff_section = [{"content": content, "type": "fallback"}]

        return {
            "grounding": None,
            "rules": [],
            "behaviors": [],
            "handoff": handoff_section,
            "recent_work": [],
            "issues": {"count": 0, "items": []},
            "decisions": [],
            "entities": [],
            "total_tokens": 0,
            "budget_tokens": budget_tokens,
            "budget_remaining": budget_tokens,
            "excluded": 0,
            "degraded": True,
            "section_tokens": {},
            "hints": dict(_SECTION_HINTS),
            "onboarding": _ONBOARDING_TEXT,
        }


@mcp.tool()
async def weft_focus(
    ctx: Context,
    intent: str,
    project_id: str | None = None,
    agent_id: str | None = None,
    budget_tokens: int = 1200,
) -> dict:
    """Post-intent re-prime: surface memories the generic primer missed.

    Call after weft_prime when you know what you're working on. Focus builds
    a compound query from your intent + last session context, excludes memories
    already surfaced by prime, and returns a tight-budget supplement.

    Unlike prime (broad session startup), focus is narrow and intent-driven.
    Can be called multiple times as intent shifts mid-session.

    intent: what you're focusing on (e.g., "implement session tracking")
    budget_tokens: max tokens in result (default 1200, supplemental to prime)"""
    try:
        cid = set_correlation_id()
        logger.debug("weft_focus start [%s] intent=%r", cid, intent[:50])
        from weft.focus import build_focus

        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)

        async with acquire(app.pool):
            result = await build_focus(
                app.pool,
                intent=intent,
                embedding_fn=app.embedding.embed,
                project_id=resolved_project,
                agent_id=agent_id,
                budget_tokens=budget_tokens,
            )

        # Fire-and-forget: log focused memories (outside acquire — system-level op)
        import asyncio
        focused_ids = [m["id"] for m in result.focused_memories]
        if focused_ids:
            asyncio.create_task(
                log_memory_access(app.pool, focused_ids, "focus"),
                name="weft-session-log-focus",
            )

        return result.to_dict()
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_focus", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_focus", e)


@mcp.tool()
async def weft_status(
    ctx: Context,
    topic: str,
    synthesize: bool = False,
    budget_tokens: int = 2000,
) -> dict:
    """Topic-anchored memory status: gather every active memory for a topic, optionally
    synthesize into a narrative digest.

    topic: The topic string to query (e.g. 'weft', 'loom', 'entity:Windward').
    synthesize: False (default) returns Tier-1 only — complete memory set, zero LLM calls.
                True requests Tier-2 — a cached or freshly synthesized narrative digest.
    budget_tokens: Maximum output tokens for Tier-2 synthesis (default 2000).

    Returns:
      {
        topic, resolved_tags: [str, ...],
        memories: [ {id, type, content, topic, created_at}, ... ],  # complete, ordered by created_at
        complete: bool, truncated: bool,
        digest: { content, provenance: {memory_id: [...spans]}, generated_at, stale } | null,
      }
    """
    try:
        from weft.auth import current_user_id
        from weft.cost_tracking import CostEntryCreate, CostEntryType, record_cost
        from weft.topic_digest_cache import read_digest, write_digest
        from weft.topic_gather import gather_topic_memories
        from weft.topic_resolution import resolve_topic
        from weft.views.topic_synthesis import SYNTHESIZER_VERSION, _MODEL, synthesize_digest

        app: AppContext = ctx.request_context.lifespan_context
        user_id = resolve_caller_user_id()

        # --- Tier-1: resolve + gather ---
        resolved_tags = await resolve_topic(topic, user_id, app.pool)
        gather_result = await gather_topic_memories(
            app.pool,
            tags=resolved_tags,
            user_id=user_id,
            budget_tokens=budget_tokens,
        )
        memory_count = len(gather_result["memories"])
        was_empty = memory_count == 0
        logger.debug(
            "weft_status: topic=%r resolved_tags=%r memory_count=%d was_empty=%s",
            topic,
            resolved_tags,
            memory_count,
            was_empty,
        )

        # L1 Resolution Ratchet feed: log this topic ask as a recall_query row so
        # the compounding loop can read was_empty (result_count == 0) signals from
        # weft_status, the same way weft_recall logs its calls. result_count
        # carries was_empty: 0 means the topic resolved to nothing.
        #
        # log_recall_query relies on the app.user_id GUC default to fill the
        # NOT-NULL user_id column, and acquire() issues SET LOCAL app.user_id
        # from the current_user_id contextvar AT ENTRY — so the contextvar must
        # be set BEFORE entering acquire(), not inside it. (In prod the auth
        # middleware has already set it; setting it here also covers callers
        # that haven't, e.g. tests.) log_recall_query swallows its own DB errors
        # and must never break the user path.
        tok = current_user_id.set(user_id)
        try:
            async with acquire(app.pool):
                await log_recall_query(
                    app.pool,
                    tool_name="status",
                    query_text=topic,
                    result_count=memory_count,
                )
        finally:
            current_user_id.reset(tok)

        memories_payload = [
            {
                "id": m.id,
                "type": m.type.value,
                "content": m.content,
                "topic": m.topic,
                "created_at": m.created_at.isoformat(),
            }
            for m in gather_result["memories"]
        ]

        response: dict = {
            "topic": topic,
            "resolved_tags": resolved_tags,
            "memories": memories_payload,
            "complete": gather_result["complete"],
            "truncated": gather_result["truncated"],
            "digest": None,
        }

        if not synthesize:
            return response

        # --- Tier-2: cache read → synthesize → cache write ---
        async with acquire(app.pool):
            # Set user context so RLS-scoped cache reads/writes work
            tok = current_user_id.set(user_id)
            try:
                cached_digest = await read_digest(
                    app.pool, user_id=user_id, topic=topic, scope="global"
                )
            finally:
                current_user_id.reset(tok)

        if cached_digest is not None:
            # Cache hit — zero model calls, zero cost_entries row (V6)
            response["digest"] = {
                "content": cached_digest["content"],
                "provenance": cached_digest["provenance"],
                "generated_at": cached_digest["generated_at"].isoformat()
                if hasattr(cached_digest["generated_at"], "isoformat")
                else str(cached_digest["generated_at"]),
                "stale": cached_digest["stale"],
            }
            return response

        # Cache miss — call synthesizer
        result = await synthesize_digest(
            gather_result["memories"], budget_tokens=budget_tokens
        )

        if result.status == "synthesized":
            # Write cache entry + record cost
            async with acquire(app.pool):
                tok = current_user_id.set(user_id)
                try:
                    await write_digest(
                        app.pool,
                        user_id=user_id,
                        topic=topic,
                        content=result.content,
                        detector_version=SYNTHESIZER_VERSION,
                        scope="global",
                        provenance=result.provenance,
                    )
                finally:
                    current_user_id.reset(tok)

            # Set user context BEFORE acquire() so its SET LOCAL app.user_id
            # fires and the cost_entries.user_id GUC default resolves to the
            # caller (not NULL). See weft-49bd0550 / weft-93733760.
            tok = current_user_id.set(user_id)
            try:
                async with acquire(app.pool):
                    await record_cost(
                        app.pool,
                        CostEntryCreate(
                            entry_type=CostEntryType.topic_synthesis,
                            reference_id=topic,
                            model=_MODEL,
                            input_tokens=result.input_tokens,
                            output_tokens=result.output_tokens,
                            total_tokens=result.input_tokens + result.output_tokens,
                            estimated_cost_usd=result.cost_usd,
                            metadata={},
                        ),
                    )
            finally:
                current_user_id.reset(tok)

            response["digest"] = {
                "content": result.content,
                "provenance": result.provenance,
                "generated_at": None,
                "stale": False,
            }

        elif result.status == "abstained":
            # Record abstention in cost_entries (V telemetry) — no cache write.
            # Set user context BEFORE acquire() so the row is attributed to the
            # caller, not NULL (weft-49bd0550 / weft-93733760).
            tok = current_user_id.set(user_id)
            try:
                async with acquire(app.pool):
                    await record_cost(
                        app.pool,
                        CostEntryCreate(
                            entry_type=CostEntryType.topic_synthesis,
                            reference_id=topic,
                            model=_MODEL,
                            input_tokens=0,
                            output_tokens=0,
                            total_tokens=0,
                            estimated_cost_usd=0.0,
                            metadata={
                                "abstained": True,
                                "projected": result.projected_cost_usd,
                                "memory_count": result.memory_count,
                            },
                        ),
                    )
            finally:
                current_user_id.reset(tok)
            # Graceful non-synthesized response — Tier-1 memories remain intact
            response["digest"] = None
            response["synthesis_status"] = "abstained"

        else:
            # error or empty — graceful fallback, no cache write, no cost record
            response["digest"] = None
            response["synthesis_status"] = result.status

        return response

    except _INPUT_ERRORS as e:
        return _input_error_response("weft_status", e)
    except _DB_ERRORS as e:
        logger.warning("Database unavailable in weft_status: %s", e)
        return {"degraded": True, "error": "Database unavailable", "topic": topic}


@mcp.tool()
async def weft_extract(
    ctx: Context,
    text: str,
    min_confidence: float = 0.5,
) -> dict:
    """Extract memory candidates from text. Returns proposals for review — does NOT auto-store."""
    from weft.extract import extract_candidates
    candidates = extract_candidates(text, min_confidence=min_confidence)
    return {"count": len(candidates), "candidates": candidates}


@mcp.tool()
async def weft_learn(
    ctx: Context,
    content: str,
    task_id: str | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
    min_confidence: float = 0.7,
) -> dict:
    """Store lessons learned from completed work. Extracts memories from free-text
    notes about what was learned (gotchas, patterns, fixes) and auto-stores
    candidates above the confidence threshold. Designed for post-task capture —
    call after loom_done with what the agent (or you) learned during the task.
    If project_id is omitted, auto-detects from the client's working directory."""
    try:
        from weft.extract import extract_behaviors, extract_candidates

        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)
        candidates = extract_candidates(content, min_confidence=min_confidence)
        behavior_candidates = extract_behaviors(content, min_confidence=min_confidence)

        # Also store the raw content as a solution memory if no patterns matched
        # but the content is substantial enough to be useful
        if not candidates and len(content.split()) >= 10:
            candidates = [{
                "content": content.strip(),
                "type": "solution",
                "confidence": min_confidence,
                "topic": [],
            }]

        async with acquire(app.pool):
            stored: list[dict] = []
            for c in candidates:
                create = MemoryCreate(
                    type=MemoryType(c["type"]),
                    content=c["content"],
                    topic=c.get("topic", []) + ([f"task:{task_id}"] if task_id else []),
                    source=MemorySource.conversation,
                    confidence=c["confidence"],
                    project_id=resolved_project,
                    agent_id=agent_id,
                )
                embedding = await app.embedding.embed(
                    embed_text_for_memory(c["content"], create.topic)
                )
                memory = await store_memory(app.pool, create, embedding=embedding)
                await app.cache.set_memory(memory)
                stored.append(memory.to_dict())

            # Auto-create a milestone when task_id is provided so primer's
            # recent_work section shows completed task breadcrumbs.
            milestone_dict: dict | None = None
            if task_id:
                # Build a concise milestone from the first ~100 words of content
                words = content.split()
                summary = " ".join(words[:100]) + ("..." if len(words) > 100 else "")
                milestone_create = MemoryCreate(
                    type=MemoryType.milestone,
                    content=summary,
                    topic=[f"task:{task_id}"],
                    source=MemorySource.conversation,
                    confidence=0.9,
                    project_id=resolved_project,
                    agent_id=agent_id,
                )
                ms_embedding = await app.embedding.embed(
                    embed_text_for_memory(summary, milestone_create.topic)
                )
                milestone = await store_memory(
                    app.pool, milestone_create, embedding=ms_embedding,
                )
                await app.cache.set_memory(milestone)
                milestone_dict = milestone.to_dict()

            # Auto-extract and store behavior candidates (trigger → action rules)
            stored_behaviors: list[dict] = []
            for bc in behavior_candidates:
                scope = BehaviorScope.project if resolved_project else BehaviorScope.global_
                create_b = BehaviorCreate(
                    trigger_pattern=bc["trigger_pattern"],
                    action=bc["action"],
                    confidence=bc["confidence"],
                    scope=scope,
                    project_id=resolved_project,
                    agent_id=agent_id,
                )
                b_embedding = await app.embedding.embed(bc["trigger_pattern"])
                behavior = await store_behavior(app.pool, create_b, embedding=b_embedding)
                stored_behaviors.append(behavior.to_dict())

            await app.cache.invalidate_stats()

        # Boost usefulness (outside acquire — system-level op)
        session_boost: dict = {}
        try:
            session_boost = await boost_session_memories(app.pool)
        except Exception as exc:
            logger.warning("Session boost failed during learn: %s", exc)

        # Turn-tier boost mirrors the belief-tier boost at session end (P1.A2).
        turn_boost: dict = {}
        try:
            turn_boost = await boost_session_turns(app.pool)
        except Exception as exc:
            logger.warning("Turn-tier session boost failed during learn: %s", exc)

        return {
            "candidates_found": len(candidates),
            "stored": len(stored),
            "memories": stored,
            "behaviors_extracted": len(stored_behaviors),
            "behaviors": stored_behaviors,
            "task_id": task_id,
            "milestone": milestone_dict,
            "session_boost": session_boost,
            "turn_boost": turn_boost,
        }
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_learn", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_learn", e)


@mcp.tool()
async def weft_feedback_general(
    ctx: Context,
    feedback: str,
    category: str = "suggestion",
    agent_id: str | None = None,
) -> dict:
    """Submit general feedback about Weft itself — friction points, feature requests,
    or praise. Unlike weft_feedback (per-memory ratings), this captures product-level
    observations from agents using Weft in the field."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        valid_categories = ("suggestion", "friction", "praise", "bug")
        if category not in valid_categories:
            return _input_error_response(
                "weft_feedback_general",
                ValueError(f"category must be one of {valid_categories}, got '{category}'"),
            )
        create = MemoryCreate(
            type=MemoryType.fact,
            content=f"[{category.upper()}] {feedback}",
            topic=["weft-feedback", category],
            source=MemorySource.conversation,
            confidence=0.8,
            project_id="weft",
            agent_id=agent_id,
        )
        embedding = await app.embedding.embed(
            embed_text_for_memory(feedback, create.topic)
        )
        async with acquire(app.pool):
            memory = await store_memory(app.pool, create, embedding=embedding)
            await app.cache.set_memory(memory)
            await app.cache.invalidate_stats()
            return {"id": memory.id, "category": category, "stored": True}
    except _DB_ERRORS as e:
        return _db_error_response("weft_feedback_general", e)


@mcp.tool()
async def weft_handoff(
    ctx: Context,
    summary: str,
    in_progress: str | None = None,
    next_steps: str | None = None,
    open_questions: str | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
) -> dict:
    """Session handoff: capture context for the next session before clearing.

    Call this before ending a session to preserve continuity. The next session's
    weft_prime will surface the most recent handoff prominently so the incoming
    agent can pick up where you left off.

    Structure your handoff like a shift note:
    - summary: what was accomplished this session
    - in_progress: what's partially done or needs follow-up
    - next_steps: what you'd recommend doing next and why
    - open_questions: unresolved decisions or things to investigate"""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)

        # Build structured content
        parts = [f"## Session Handoff\n\n**Summary:** {summary}"]
        if in_progress:
            parts.append(f"\n**In Progress:** {in_progress}")
        if next_steps:
            parts.append(f"\n**Next Steps:** {next_steps}")
        if open_questions:
            parts.append(f"\n**Open Questions:** {open_questions}")
        content = "\n".join(parts)

        create = MemoryCreate(
            type=MemoryType.handoff,
            content=content,
            topic=["session-handoff"],
            source=MemorySource.conversation,
            confidence=1.0,
            project_id=resolved_project,
            agent_id=agent_id,
        )
        embedding = await app.embedding.embed(
            embed_text_for_memory(content, create.topic)
        )
        async with acquire(app.pool):
            memory = await store_memory(app.pool, create, embedding=embedding)
            await app.cache.set_memory(memory)
            await app.cache.invalidate_stats()

            # Auto-prune: archive previous handoffs for this project so they
            # don't accumulate.  Only the most recent handoff matters.
            pruned_count = 0
            try:
                prev = await list_memories(
                    app.pool,
                    memory_type=MemoryType.handoff,
                    status=MemoryStatus.active,
                    project_id=resolved_project,
                    exact_scope=True,
                    limit=100,
                )
                for old in prev:
                    if old.id != memory.id:
                        await update_memory(
                            app.pool, old.id, status=MemoryStatus.archived,
                        )
                        pruned_count += 1
            except Exception as exc:
                logger.warning("Handoff auto-prune failed: %s", exc)

            # Auto-episode: close open episodes and create a new one
            closed_ids = []
            new_episode_id = None
            try:
                open_eps = await list_episodes(
                    app.pool, project_id=resolved_project,
                    status=EpisodeStatus.open, limit=10,
                )
                # Each close_episode below rewrites `summary` to the
                # handoff summary, so we recompute the embedding to match.
                # Per-episode text uses the episode's own title with the
                # new summary so the vector reflects the post-close state.
                for ep in open_eps:
                    close_embedding = await _embed_episode_text(
                        app, _episode_embed_text(ep.title, summary),
                    )
                    closed = await close_episode(
                        app.pool, ep.id,
                        summary=summary,
                        embedding=close_embedding,
                    )
                    if closed:
                        await add_memory_to_episode(app.pool, ep.id, memory.id)
                        closed_ids.append(ep.id)

                new_title = f"Session after: {summary[:80]}"
                new_embedding = await _embed_episode_text(
                    app, _episode_embed_text(new_title, None),
                )
                new_ep = await create_episode(
                    app.pool,
                    EpisodeCreate(
                        title=new_title,
                        project_id=resolved_project,
                        agent_id=agent_id,
                    ),
                    embedding=new_embedding,
                )
                await add_memory_to_episode(app.pool, new_ep.id, memory.id)
                new_episode_id = new_ep.id
            except Exception as exc:
                logger.warning("Auto-episode failed during handoff: %s", exc)

        # Boost usefulness (outside acquire — system-level op)
        session_boost: dict = {}
        try:
            session_boost = await boost_session_memories(app.pool)
        except Exception as exc:
            logger.warning("Session boost failed during handoff: %s", exc)

        # Turn-tier boost mirrors the belief-tier boost at session end (P1.A2).
        turn_boost: dict = {}
        try:
            turn_boost = await boost_session_turns(app.pool)
        except Exception as exc:
            logger.warning("Turn-tier session boost failed during handoff: %s", exc)

        return {
            "id": memory.id,
            "project_id": resolved_project,
            "stored": True,
            "previous_handoffs_archived": pruned_count,
            "episodes_closed": closed_ids,
            "episode_opened": new_episode_id,
            "session_boost": session_boost,
            "turn_boost": turn_boost,
        }
    except _DB_ERRORS as e:
        return _db_error_response("weft_handoff", e)


# ── Skills ──────────────────────────────────────────────────────────


@mcp.tool()
async def weft_weekly_recap(
    ctx: Context,
    days: int = 7,
    project_id: str | None = None,
) -> dict:
    """Weekly recap: memories from the last N days grouped by type and topic.

    Returns decisions, issues, milestones, top topics, and recent activity.
    Use this to generate status updates, standup notes, or session summaries."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)
        from weft.skills import weekly_recap

        async with acquire(app.pool):
            return await weekly_recap(app.pool, days=days, project_id=resolved_project)
    except _DB_ERRORS as e:
        return _db_error_response("weft_weekly_recap", e)


@mcp.tool()
async def weft_search_all(
    ctx: Context,
    query: str | None = None,
    topic: str | None = None,
    memory_type: MemoryTypeLiteral | None = None,
    days: int | None = None,
    limit: int = 20,
    retrieval_mode: str = "face",
) -> dict:
    """Cross-project brain-wide search combining semantic and filter queries.

    At least one filter is required. Searches across ALL projects (not scoped).
    Use for finding information that spans projects or when you don't know
    which project something belongs to.

    retrieval_mode: 'face' (default — excludes codebase ingest noise),
    'code' (includes ingest + code-context for agent-in-repo queries),
    or 'all' (no source filter).
    """
    try:
        app: AppContext = ctx.request_context.lifespan_context
        from weft.skills import search_all

        # Step 1.5 (v50) — fire-and-forget query log. Same observation
        # window as weft_recall: every cross-project search counts toward
        # the calls/week + repeat-query metrics. Logged before the actual
        # search runs so a search failure still leaves a row.
        if query:
            asyncio.create_task(
                log_recall_query(
                    app.pool,
                    tool_name="search_all",
                    query_text=query,
                    project_id=None,
                    retrieval_mode=retrieval_mode,
                ),
                name="weft-search-all-query-log",
            )

        async with acquire(app.pool):
            result = await search_all(
                app.pool, app.embedding,
                query=query, topic=topic, memory_type=memory_type,
                days=days, limit=limit, retrieval_mode=retrieval_mode,
            )

        # Fire-and-forget read-side audit log (Phase 2 follow-on / mig 41).
        # Brain-wide search returning a poisoned memory must be traceable
        # the same way recall is — incident response shouldn't have to
        # care whether the read came from weft_recall or weft_search_all.
        # asyncio is imported at module level — no shadowing local import.
        result_ids = [r["id"] for r in result.get("results", []) if isinstance(r, dict) and "id" in r]
        # Cross-project results returned alongside the main hit set also
        # count as retrievals for telemetry purposes (handoff: "any memory
        # returned in results"). search_all flattens them, but the dict
        # form is conservative against future shape changes.
        cross_ids = [
            r["id"] for r in result.get("cross_project", [])
            if isinstance(r, dict) and "id" in r
        ]
        bump_ids = result_ids + cross_ids
        if bump_ids:
            await bump_retrieval_telemetry(app.pool, bump_ids)
        if result_ids:
            asyncio.create_task(
                log_memory_access(
                    app.pool, result_ids, "search_all",
                    retrieval_mode=retrieval_mode,
                ),
                name="weft-session-log-search-all",
            )
        return result
    except _DB_ERRORS as e:
        return _db_error_response("weft_search_all", e)


@mcp.tool()
async def weft_project_status(
    ctx: Context,
    project_id: str | None = None,
    days: int = 30,
) -> dict:
    """Project status: memories for a project weighted by importance.

    Prioritizes decisions, issues, and milestones. Includes recent activity.
    Use to get a quick overview of where a project stands."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)
        if not resolved_project:
            return {"error": "project_id required — pass explicitly or run from a project directory"}
        from weft.skills import project_status

        async with acquire(app.pool):
            return await project_status(app.pool, project_id=resolved_project, days=days)
    except _DB_ERRORS as e:
        return _db_error_response("weft_project_status", e)


@mcp.tool()
async def weft_projects(ctx: Context) -> dict:
    """Cross-project handoff index — discover what projects exist.

    Returns one entry per project_id that has at least one active memory,
    sorted so projects with the freshest handoff come first (projects with
    no handoff sort to the end). Each entry has last_handoff_at,
    last_handoff_summary (the **Summary:** line, truncated), last_activity_at,
    and memory_count.

    Use this when a session in project A needs to find work in project B —
    call weft_projects() to discover the canonical project_id, then
    weft_prime(project_id="<id>") to load that project's handoff and rules.
    """
    try:
        app: AppContext = ctx.request_context.lifespan_context
        from weft.skills import list_projects_with_handoffs

        async with acquire(app.pool):
            projects = await list_projects_with_handoffs(app.pool)
        return {"count": len(projects), "projects": projects}
    except _DB_ERRORS as e:
        return _db_error_response("weft_projects", e)


@mcp.tool()
async def weft_meal_plan(
    ctx: Context,
    lissy_approved: bool | None = None,
    cuisine: str | None = None,
    tag: str | None = None,
    limit: int = 20,
) -> dict:
    """Query recipe memories for meal planning.

    Filter by Lissy-approved, cuisine type, or tags. Returns full recipe
    content for planning meals."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        from weft.skills import meal_plan

        async with acquire(app.pool):
            return await meal_plan(
                app.pool, lissy_approved=lissy_approved,
                cuisine=cuisine, tag=tag, limit=limit,
            )
    except _DB_ERRORS as e:
        return _db_error_response("weft_meal_plan", e)


@mcp.tool()
async def weft_up_next(
    ctx: Context,
    days: int = 7,
    include_no_date: bool = False,
) -> dict:
    """Upcoming tasks: open tasks due in the next N days from Obsidian notes.

    Returns overdue tasks and tasks due soon, sorted by date and priority.
    Tasks come from Obsidian checkbox items with Tasks plugin emoji dates."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        from weft.skills import up_next

        async with acquire(app.pool):
            return await up_next(app.pool, days=days, include_no_date=include_no_date)
    except _DB_ERRORS as e:
        return _db_error_response("weft_up_next", e)


@mcp.tool()
async def weft_slack_sync(
    ctx: Context,
    channel_ids: list[str] | None = None,
    channel_names: list[str] | None = None,
    limit_per_channel: int = 50,
    bot_token: str | None = None,
) -> dict:
    """Sync Slack channel history into Weft memories.

    Two modes:
    1. With bot_token: uses slack_sdk directly (for CLI/cron).
    2. Without bot_token: caller must pre-fetch messages using Slack MCP
       tools and pass via channel_ids. This tool fetches what it can
       and stores the results.

    If bot_token is provided, syncs all non-excluded channels automatically.
    Otherwise, provide channel_ids or channel_names to target specific channels.
    """
    try:
        app: AppContext = ctx.request_context.lifespan_context

        if bot_token:
            # SDK mode — fully automated
            from weft.slack.sync import sync_slack_sdk

            async with acquire(app.pool):
                result = await sync_slack_sdk(
                    app.pool,
                    bot_token,
                    app.embedding,
                    limit_per_channel=limit_per_channel,
                )
                return {
                    "mode": "sdk",
                    "channels_synced": result.channels_synced,
                    "messages_found": result.messages_found,
                    "messages_synced": result.messages_synced,
                    "messages_skipped": result.messages_skipped,
                    "messages_updated": result.messages_updated,
                    "memories_created": result.memories_created,
                    "memories_archived": result.memories_archived,
                }
        else:
            # Return instructions for MCP-based sync workflow
            channel_ids = _coerce_list(channel_ids)
            channel_names = _coerce_list(channel_names)
            return {
                "mode": "mcp_assisted",
                "instructions": (
                    "To sync Slack channels without a bot token, use the Slack MCP "
                    "tools to read channel history, then call weft_slack_ingest with "
                    "the fetched messages. Steps:\n"
                    "1. slack_search_channels to find channel IDs\n"
                    "2. slack_read_channel for each channel\n"
                    "3. slack_read_thread for threaded messages\n"
                    "4. weft_slack_ingest with the collected data"
                ),
                "channel_ids": channel_ids,
                "channel_names": channel_names,
            }
    except _DB_ERRORS as e:
        return _db_error_response("weft_slack_sync", e)


@mcp.tool()
async def weft_slack_ingest(
    ctx: Context,
    channel_id: str,
    channel_name: str,
    messages: list[dict],
    threads: dict | None = None,
    user_names: dict | None = None,
) -> dict:
    """Ingest pre-fetched Slack messages into Weft memories.

    Use this after reading channel history via Slack MCP tools.
    Pass raw message objects from slack_read_channel and thread
    replies from slack_read_thread.

    Args:
        channel_id: The Slack channel ID (e.g. C0A8VD15M5X)
        channel_name: The channel name (e.g. 'general')
        messages: List of raw Slack message dicts from channel history
        threads: Optional {thread_ts: [reply_dicts]} for threaded messages
        user_names: Optional {user_id: display_name} mapping
    """
    try:
        app: AppContext = ctx.request_context.lifespan_context
        from weft.slack.sync import ChannelInfo, sync_slack_messages

        channel = ChannelInfo(id=channel_id, name=channel_name)
        messages = _coerce_list(messages) or []
        if isinstance(threads, str):
            threads = json.loads(threads)

        async with acquire(app.pool):
            result = await sync_slack_messages(
                app.pool,
                channels=[channel],
                messages_by_channel={channel_id: messages},
                embedding_provider=app.embedding,
                threads_by_channel={channel_id: threads} if threads else None,
                user_names=user_names,
            )
        return {
            "channel": channel_name,
            "messages_found": result.messages_found,
            "messages_synced": result.messages_synced,
            "messages_skipped": result.messages_skipped,
            "memories_created": result.memories_created,
        }
    except _DB_ERRORS as e:
        return _db_error_response("weft_slack_ingest", e)
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_slack_ingest", e)


# --- Behavior tools ---


@mcp.tool()
async def weft_behavior_add(
    ctx: Context,
    trigger_pattern: str,
    action: str,
    confidence: float = 0.7,
    scope: str = "global",
    project_id: str | None = None,
    agent_id: str | None = None,
    priority: int = 0,
) -> dict:
    """Store a persistent behavioral rule for agents.

    trigger_pattern: describes WHEN this behavior should activate (embedded for semantic matching).
    action: describes WHAT the agent should do when the trigger matches.
    scope: 'global' (all projects), 'project' (specific project), or 'agent' (specific agent).
    priority: higher values override lower-priority behaviors (default 0).

    project_id resolution: scope='global' forces project_id=NULL so the
    behavior fires across every project (auto-detect from CWD would
    silently bind it to the calling project, defeating global scope).
    scope='project' or 'agent' auto-detects from the client's working
    directory if project_id is omitted.
    """
    try:
        app: AppContext = ctx.request_context.lifespan_context
        behavior_scope = BehaviorScope(scope)
        if behavior_scope == BehaviorScope.global_:
            # Global scope must not bind to a project; honoring the
            # auto-detected CWD here would scope the row to that project
            # and silently break cross-project firing. Caller-supplied
            # project_id is also ignored — global means global.
            resolved_project = None
        else:
            resolved_project = await _resolve_project_id(ctx, project_id)
        create = BehaviorCreate(
            trigger_pattern=trigger_pattern,
            action=action,
            confidence=confidence,
            scope=behavior_scope,
            project_id=resolved_project,
            agent_id=agent_id,
            priority=priority,
        )
        embedding = await app.embedding.embed(trigger_pattern)
        async with acquire(app.pool):
            behavior = await store_behavior(app.pool, create, embedding=embedding)
            return behavior.to_dict()
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_behavior_add", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_behavior_add", e)


@mcp.tool()
async def weft_behavior_match(
    ctx: Context,
    situation: str,
    project_id: str | None = None,
    agent_id: str | None = None,
    limit: int = 5,
    threshold: float = 0.3,
    user_id: str | None = None,
) -> dict:
    """Find behavioral rules that match a described situation.
    If project_id is omitted, auto-detects from the client's working directory.

    situation: free-text description of the current context or task.
    Returns behaviors ranked by relevance (similarity * confidence * priority).

    user_id: Caller identity override. Defaults to the authenticated caller
    (resolve_caller_user_id): the request credential's user_id when present,
    else this installation's UUID.
    Filters to user-owned rows OR truly-global rows (user_id IS NULL).
    retrieval_mode and scope are orthogonal — user_id composes independently with both.
    """
    if user_id is None:
        user_id = resolve_caller_user_id()
    try:
        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)
        embedding = await app.embedding.embed(situation)
        async with acquire(app.pool):
            results = await match_behaviors(
                app.pool,
                embedding,
                limit=limit,
                threshold=threshold,
                project_id=resolved_project,
                agent_id=agent_id,
                user_id=user_id,
            )
            # Touch matched behaviors to track usage
            for r in results:
                await touch_behavior(app.pool, r.behavior.id)
            return {
                "situation": situation,
                "count": len(results),
                "behaviors": [r.to_dict() for r in results],
            }
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_behavior_match", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_behavior_match", e)


@mcp.tool()
async def weft_behavior_list(
    ctx: Context,
    scope: str | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
    enabled: bool | None = True,
    limit: int = 50,
    user_id: str | None = None,
) -> dict:
    """List stored behavioral rules with optional filters.
    If project_id is omitted, auto-detects from the client's working directory.

    Returns behaviors ordered by priority (highest first).

    user_id: Caller identity override. Defaults to the authenticated caller
    (resolve_caller_user_id): the request credential's user_id when present,
    else this installation's UUID.
    Filters to user-owned rows OR truly-global rows (user_id IS NULL).
    retrieval_mode and scope are orthogonal — user_id composes independently with both.
    """
    if user_id is None:
        user_id = resolve_caller_user_id()
    try:
        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)
        behavior_scope = BehaviorScope(scope) if scope else None
        async with acquire(app.pool):
            results = await list_behaviors_store(
                app.pool,
                scope=behavior_scope,
                project_id=resolved_project,
                agent_id=agent_id,
                enabled=enabled,
                limit=limit,
                user_id=user_id,
            )
            return {
                "count": len(results),
                "behaviors": [b.to_dict() for b in results],
            }
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_behavior_list", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_behavior_list", e)


@mcp.tool()
async def weft_behavior_delete(
    ctx: Context,
    behavior_id: str,
    hard: bool = False,
) -> dict:
    """Delete a behavioral rule by ID.

    hard=False (default): soft-delete (archives the behavior).
    hard=True: permanently removes the behavior from the database."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        async with acquire(app.pool):
            deleted = await delete_behavior(app.pool, behavior_id, hard=hard)
            return {
                "behavior_id": behavior_id,
                "deleted": deleted,
                "hard": hard,
            }
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_behavior_delete", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_behavior_delete", e)


# --- Episode tools ---


def _episode_embed_text(title: str, summary: str | None) -> str:
    """Compose the text fed to the episode embedder.

    Mirrors ``TABLE_TEXT_EXPRESSIONS["episodes"]`` in
    :mod:`weft.db.reembed` (``title || ' ' || COALESCE(summary, '')``)
    so write-time and backfill embeddings are derived from the same text.
    """
    return f"{title} {summary or ''}"


async def _embed_episode_text(app: AppContext, text: str) -> list[float] | None:
    """Embed an episode's text, mirroring store_memory's failure handling.

    Returns ``None`` on any embed failure and logs a warning. The episode
    write proceeds with a NULL embedding; the v47 startup backfill
    (``reembed_table('episodes', ...)``) catches it on the next boot.
    Uses ``app.episode_embedding`` (the per-tier provider) and falls back
    to ``app.embedding`` for safety in case startup wiring failed.
    """
    provider = app.episode_embedding or app.embedding
    try:
        return await provider.embed(text)
    except Exception as exc:
        logger.warning(
            "Episode embedding failed, storing without vector: %s", exc,
        )
        return None


@mcp.tool()
async def weft_episode_create(
    ctx: Context,
    title: str,
    summary: str | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
) -> dict:
    """Create a new open episode — a time-bounded grouping of memories.

    Episodes track causal sequences of decisions, actions, and learnings.
    If project_id is omitted, auto-detects from the client's working directory."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)
        # Embed BEFORE acquire — the network call shouldn't hold a tx open.
        embedding = await _embed_episode_text(
            app, _episode_embed_text(title, summary),
        )
        async with acquire(app.pool):
            ep = await create_episode(
                app.pool,
                EpisodeCreate(
                    title=title,
                    summary=summary,
                    project_id=resolved_project,
                    agent_id=agent_id,
                ),
                embedding=embedding,
            )
            return ep.to_dict()
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_episode_create", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_episode_create", e)


@mcp.tool()
async def weft_episode_add(
    ctx: Context,
    episode_id: str,
    memory_id: str,
    position: int | None = None,
) -> dict:
    """Link a memory to an episode. Idempotent — linking the same memory twice is a no-op.

    If position is omitted, auto-assigns the next sequential position."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        async with acquire(app.pool):
            created = await add_memory_to_episode(
                app.pool, episode_id, memory_id, position=position,
            )
            return {
                "episode_id": episode_id,
                "memory_id": memory_id,
                "created": created,
            }
    except _DB_ERRORS as e:
        return _db_error_response("weft_episode_add", e)


@mcp.tool()
async def weft_episode_timeline(
    ctx: Context,
    start: str | None = None,
    end: str | None = None,
    hours: int = 24,
    project_id: str | None = None,
    agent_id: str | None = None,
    limit: int = 20,
    user_id: str | None = None,
) -> dict:
    """Find episodes overlapping a time range.

    If start/end are omitted, defaults to the last N hours (default 24).
    start/end accept ISO timestamps or relative values like '2d', '1w'.
    If project_id is omitted, auto-detects from the client's working directory.
    Open episodes (no end time) match any range after their start.

    user_id: Caller identity override. Defaults to the authenticated caller
    (resolve_caller_user_id): the request credential's user_id when present,
    else this installation's UUID.
    Filters to user-owned rows OR truly-global rows (user_id IS NULL).
    retrieval_mode and scope are orthogonal — user_id composes independently with both.
    """
    if user_id is None:
        user_id = resolve_caller_user_id()
    try:
        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)

        now = datetime.now(timezone.utc)
        if end is not None:
            parsed_end = _parse_review_after(end) or now
        else:
            parsed_end = now
        if start is not None:
            parsed_start = _parse_review_after(start) or (now - timedelta(hours=hours))
        else:
            parsed_start = now - timedelta(hours=hours)

        async with acquire(app.pool):
            results = await timeline_query(
                app.pool,
                start=parsed_start,
                end=parsed_end,
                project_id=resolved_project,
                agent_id=agent_id,
                limit=limit,
                user_id=user_id,
            )
            return {
                "start": parsed_start.isoformat(),
                "end": parsed_end.isoformat(),
                "count": len(results),
                "episodes": [ep.to_dict() for ep in results],
            }
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_episode_timeline", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_episode_timeline", e)


@mcp.tool()
async def weft_episode_context(
    ctx: Context,
    episode_id: str,
    budget_tokens: int = 2000,
    include_episode: bool = True,
) -> dict:
    """Load an episode and its memories within a token budget.

    Returns the episode metadata and as many linked memories as fit
    within budget_tokens. Memories are returned in position order."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        async with acquire(app.pool):
            ep = await get_episode(app.pool, episode_id)
            if ep is None:
                return {"error": f"Episode {episode_id} not found"}

            memories = await get_episode_memories(app.pool, episode_id)

            # Pack memories within token budget
            budget = budget_tokens
            if include_episode:
                ep_text = f"{ep.title}: {ep.summary or ''}"
                budget -= estimate_tokens(ep_text)

            packed = []
            total_tokens = 0
            for mem in memories:
                cost = estimate_tokens(mem.content)
                if total_tokens + cost > budget:
                    break
                packed.append(mem)
                total_tokens += cost

            ewm = EpisodeWithMemories(episode=ep, memories=packed)
            result = ewm.to_dict()
            result["tokens_used"] = total_tokens
            result["tokens_budget"] = budget_tokens
            result["memories_truncated"] = len(memories) - len(packed)
            return result
    except _DB_ERRORS as e:
        return _db_error_response("weft_episode_context", e)


@mcp.tool()
async def weft_episode_graduate(
    ctx: Context,
    episode_id: str,
    memory_type: str = "fact",
    content: str | None = None,
    topic: list[str] | None = None,
    confidence: float = 0.7,
) -> dict:
    """Graduate an episode into a persistent memory.

    Converts an episode's accumulated context into a long-term memory record.
    The episode is marked as 'graduated' and linked to the new memory.
    Use this when an episode contains insights worth preserving beyond its TTL.

    memory_type: type for the new memory ('fact', 'pattern', 'solution',
        'architecture', 'decision', etc.)
    content: custom memory content. If omitted, uses episode title + summary.
    topic: optional tags for the new memory.
    confidence: confidence score for the new memory (0.0-1.0, default 0.7)."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        topic = _coerce_list(topic) or []

        # Generate embedding for the memory content
        embed_text = content
        if embed_text is None:
            ep = await get_episode(app.pool, episode_id)
            if ep is None:
                return {"error": f"Episode {episode_id} not found"}
            parts = [ep.title]
            if ep.summary:
                parts.append(ep.summary)
            embed_text = "\n\n".join(parts)

        embedding = await app.embedding.embed(embed_text)

        async with acquire(app.pool):
            updated_ep, memory = await graduate_episode(
                app.pool,
                episode_id,
                memory_type=MemoryType(memory_type),
                content=content,
                topic=topic,
                confidence=confidence,
                embedding=embedding,
            )
            await app.cache.invalidate_stats()
            result = updated_ep.to_dict()
            result["graduated_memory"] = memory.to_dict()
            return result
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_episode_graduate", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_episode_graduate", e)


# --- Entity tools ---


@mcp.tool()
async def weft_entity_create(
    ctx: Context,
    name: str,
    entity_type: str = "concept",
    aliases: list[str] | None = None,
    description: str | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
) -> dict:
    """Create a first-class entity (person, project, company, tool, concept).

    Entities group memories about a subject. Link memories via weft_entity_link.
    If project_id is omitted, auto-detects from the client's working directory.

    entity_type: 'person', 'project', 'company', 'tool', or 'concept'."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)
        aliases = _coerce_list(aliases) or []
        embed_text = name + (f": {description}" if description else "")
        embedding = await app.embedding.embed(embed_text)
        async with acquire(app.pool):
            ent = await store_entity(app.pool, EntityCreate(
                name=name,
                entity_type=EntityType(entity_type),
                aliases=aliases,
                description=description,
                project_id=resolved_project,
                agent_id=agent_id,
            ), embedding=embedding)
            return ent.to_dict()
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_entity_create", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_entity_create", e)


@mcp.tool()
async def weft_entity_link(
    ctx: Context,
    entity_id: str,
    memory_id: str,
) -> dict:
    """Link a memory to an entity. Idempotent — linking twice is a no-op.

    Increments the entity's mention count on first link."""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        async with acquire(app.pool):
            created = await link_mention(app.pool, entity_id, memory_id)
            return {
                "entity_id": entity_id,
                "memory_id": memory_id,
                "created": created,
            }
    except _DB_ERRORS as e:
        return _db_error_response("weft_entity_link", e)


@mcp.tool()
async def weft_entity_search(
    ctx: Context,
    query: str,
    entity_type: str | None = None,
    project_id: str | None = None,
    limit: int = 10,
    threshold: float = 0.3,
    user_id: str | None = None,
) -> dict:
    """Search for entities by semantic similarity.

    If project_id is omitted, auto-detects from the client's working directory.
    Returns entities ranked by relevance to the query.

    user_id: Caller identity override. Defaults to the authenticated caller
    (resolve_caller_user_id): the request credential's user_id when present,
    else this installation's UUID.
    Filters to user-owned rows OR truly-global rows (user_id IS NULL).
    retrieval_mode and scope are orthogonal — user_id composes independently with both.
    """
    if user_id is None:
        user_id = resolve_caller_user_id()
    try:
        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)
        embedding = await app.embedding.embed(query)
        etype = EntityType(entity_type) if entity_type else None
        async with acquire(app.pool):
            results = await search_entities(
                app.pool, embedding,
                entity_type=etype,
                project_id=resolved_project,
                limit=limit,
                threshold=threshold,
                user_id=user_id,
            )
            return {
                "query": query,
                "count": len(results),
                "entities": [
                    {**ent.to_dict(), "similarity": round(sim, 4)}
                    for ent, sim in results
                ],
            }
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_entity_search", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_entity_search", e)


@mcp.tool()
async def weft_entity_context(
    ctx: Context,
    entity_id: str,
    budget_tokens: int = 2000,
) -> dict:
    """Load an entity and its linked memories within a token budget.

    Returns the entity metadata and as many linked memories as fit
    within budget_tokens. Memories ordered by mention time (newest first).

    The response includes:
    - truncated: True if the entity's total memory set exceeds the 100-row cap
    - memory_count: Number of memories returned in this response (after token budget filtering)
    - memories_truncated: Number of memories dropped due to token budget (not including
      the database-level 100-row cap)"""
    try:
        app: AppContext = ctx.request_context.lifespan_context
        async with acquire(app.pool):
            ent = await get_entity(app.pool, entity_id)
            if ent is None:
                return {"error": f"Entity {entity_id} not found"}

            memories, db_truncated = await get_entity_memories(app.pool, entity_id)

            # Reserve tokens for entity metadata
            ent_text = f"{ent.name}: {ent.description or ''}"
            budget = budget_tokens - estimate_tokens(ent_text)

            packed = []
            total_tokens = 0
            for mem in memories:
                cost = estimate_tokens(mem.content)
                if total_tokens + cost > budget:
                    break
                packed.append(mem)
                total_tokens += cost

            result = ent.to_dict()
            result["memories"] = [m.to_dict() for m in packed]
            result["memory_count"] = len(packed)
            result["tokens_used"] = total_tokens
            result["tokens_budget"] = budget_tokens
            result["memories_truncated"] = len(memories) - len(packed)
            result["truncated"] = db_truncated
            return result
    except _DB_ERRORS as e:
        return _db_error_response("weft_entity_context", e)


# --- Mode tools ---


@mcp.tool()
async def weft_mode_set(
    ctx: Context,
    name: str,
    description: str | None = None,
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5,
    recency_bias: float = 0.0,
    entity_boost: float = 1.0,
    behavior_boost: float = 1.0,
    project_id: str | None = None,
    agent_id: str | None = None,
) -> dict:
    """Create or update a named retrieval mode with custom weights.

    Modes tune how weft_prime assembles context. Use different modes for
    different tasks — e.g., 'research' with high vector_weight for semantic
    depth, 'coding' with high behavior_boost for rules-heavy priming.

    vector_weight/bm25_weight: balance semantic vs keyword search [0.0–1.0].
    recency_bias: favor recent memories [0.0–1.0], 0 = no recency preference.
    entity_boost: scale entity section capacity [0.0–10.0], 1.0 = default.
    behavior_boost: scale behavior section capacity [0.0–10.0], 1.0 = default."""
    try:
        from weft.modes import upsert_mode

        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)
        weights = ModeWeights(
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            recency_bias=recency_bias,
            entity_boost=entity_boost,
            behavior_boost=behavior_boost,
        )
        create = ModeCreate(
            name=name,
            description=description,
            weights=weights,
            project_id=resolved_project,
            agent_id=agent_id,
        )
        async with acquire(app.pool):
            mode = await upsert_mode(app.pool, create)
            return {"success": True, "mode": mode.to_dict()}
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_mode_set", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_mode_set", e)


@mcp.tool()
async def weft_mode_list(ctx: Context, user_id: str | None = None) -> dict:
    """List all saved retrieval modes for the current user.

    Returns modes ordered by name with their weight configurations.

    user_id: Caller identity override. Defaults to the authenticated caller
    (resolve_caller_user_id): the request credential's user_id when present,
    else this installation's UUID.
    Filters to user-owned rows OR truly-global rows (user_id IS NULL).
    retrieval_mode and scope are orthogonal — user_id composes independently with both.
    """
    if user_id is None:
        user_id = resolve_caller_user_id()
    try:
        from weft.modes import list_modes

        app: AppContext = ctx.request_context.lifespan_context
        async with acquire(app.pool):
            modes = await list_modes(app.pool, user_id=user_id)
            return {
                "count": len(modes),
                "modes": [m.to_dict() for m in modes],
            }
    except _DB_ERRORS as e:
        return _db_error_response("weft_mode_list", e)


@mcp.tool()
async def weft_mode_delete(ctx: Context, name: str) -> dict:
    """Delete a saved retrieval mode by name.

    Returns success=True if deleted, or an error if the mode was not found."""
    try:
        from weft.modes import delete_mode

        app: AppContext = ctx.request_context.lifespan_context
        async with acquire(app.pool):
            deleted = await delete_mode(app.pool, name)
            if deleted:
                return {"success": True, "deleted": name}
            return {"success": False, "error": f"Mode '{name}' not found"}
    except _DB_ERRORS as e:
        return _db_error_response("weft_mode_delete", e)


# ── Alert tools ─────────────────────────────────────────────────────


@mcp.tool()
async def weft_alert_create(
    ctx: Context,
    alert_type: str,
    title: str,
    trigger_at: str,
    body: str | None = None,
    channel: str = "log",
    channel_target: str | None = None,
    payload: dict | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
) -> dict:
    """Schedule a proactive alert that fires at trigger_at.

    alert_type: due_task, stale_decision, follow_up, or custom.
    trigger_at: ISO8601 datetime string (must be timezone-aware, e.g.
        '2026-03-21T10:00:00+00:00'). Past times fire on next scheduler tick.
    channel: 'log' (default) or 'slack'.
    channel_target: required when channel='slack' (e.g. '#alerts').
    payload: optional JSON-serializable dict of extra data."""
    try:
        from datetime import datetime, timezone

        from weft.alerts import create_alert
        from weft.models import AlertChannel, AlertCreate, AlertType

        # Validate alert_type
        valid_types = [t.value for t in AlertType]
        if alert_type not in valid_types:
            return _input_error_response(
                "weft_alert_create",
                ValueError(f"Invalid alert_type '{alert_type}'. Valid: {valid_types}"),
            )

        # Validate channel
        valid_channels = [c.value for c in AlertChannel]
        if channel not in valid_channels:
            return _input_error_response(
                "weft_alert_create",
                ValueError(f"Invalid channel '{channel}'. Valid: {valid_channels}"),
            )

        # Validate slack requires channel_target
        if channel == "slack" and not channel_target:
            return _input_error_response(
                "weft_alert_create",
                ValueError("channel_target is required when channel is 'slack'"),
            )

        # Parse trigger_at
        try:
            trigger_dt = datetime.fromisoformat(trigger_at)
        except ValueError:
            return _input_error_response(
                "weft_alert_create",
                ValueError(
                    f"Invalid trigger_at '{trigger_at}'. "
                    "Use ISO8601 format, e.g. '2026-03-21T10:00:00+00:00'"
                ),
            )
        if trigger_dt.tzinfo is None:
            trigger_dt = trigger_dt.replace(tzinfo=timezone.utc)

        # Validate payload is JSON-serializable
        if payload is not None:
            import json
            try:
                json.dumps(payload)
            except (TypeError, ValueError) as e:
                return _input_error_response(
                    "weft_alert_create",
                    ValueError(f"payload must be JSON-serializable: {e}"),
                )

        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)

        create = AlertCreate(
            alert_type=AlertType(alert_type),
            title=title,
            body=body,
            trigger_at=trigger_dt,
            channel=AlertChannel(channel),
            channel_target=channel_target,
            payload=payload or {},
            project_id=resolved_project,
            agent_id=agent_id,
        )
        async with acquire(app.pool):
            alert = await create_alert(app.pool, create)
            return {"success": True, "alert": alert.to_dict()}
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_alert_create", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_alert_create", e)


@mcp.tool()
async def weft_alert_list(
    ctx: Context,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """List alerts for the current user, newest first.

    status: optional filter — 'pending', 'fired', or 'dismissed'.
    Returns all alerts if status is omitted."""
    try:
        from weft.alerts import list_alerts
        from weft.models import AlertStatus

        # Validate status if provided
        if status is not None:
            valid_statuses = [s.value for s in AlertStatus]
            if status not in valid_statuses:
                return _input_error_response(
                    "weft_alert_list",
                    ValueError(f"Invalid status '{status}'. Valid: {valid_statuses}"),
                )
            status_enum = AlertStatus(status)
        else:
            status_enum = None

        app: AppContext = ctx.request_context.lifespan_context
        async with acquire(app.pool):
            alerts = await list_alerts(
                app.pool, status=status_enum, limit=limit, offset=offset
            )
            return {
                "count": len(alerts),
                "alerts": [a.to_dict() for a in alerts],
            }
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_alert_list", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_alert_list", e)


@mcp.tool()
async def weft_alert_dismiss(ctx: Context, alert_id: str) -> dict:
    """Dismiss a pending alert by ID. Prevents it from firing.

    Returns success=True if dismissed, success=False if not found or
    already dismissed/fired."""
    try:
        from weft.alerts import dismiss_alert

        if not alert_id or not alert_id.startswith("weft-"):
            return _input_error_response(
                "weft_alert_dismiss",
                ValueError(f"Invalid alert_id '{alert_id}'. Expected format: 'weft-...'"),
            )

        app: AppContext = ctx.request_context.lifespan_context
        async with acquire(app.pool):
            dismissed = await dismiss_alert(app.pool, alert_id)
            if dismissed:
                return {"success": True, "dismissed": alert_id}
            return {"success": False, "error": f"Alert '{alert_id}' not found or not pending"}
    except _DB_ERRORS as e:
        return _db_error_response("weft_alert_dismiss", e)


@mcp.tool()
async def weft_alert_suppress(
    ctx: Context,
    alert_type: str,
    dedup_key: str,
    hours: float = 24.0,
    reason: str | None = None,
) -> dict:
    """Mute (alert_type, dedup_key) for *hours* hours. Survives cooldown.

    alert_type: one of the AlertType values (e.g. 'loom_stale_claim').
    dedup_key:  producer-specific scope key, typically 'task:<id>',
                'project:<id>', 'memory:<id>', or 'global' for singletons.
                Use weft_alert_list to inspect existing keys.
    hours:      how long to suppress; must be > 0.
    reason:     optional human-readable reason ('on vacation', 'flapping').
    """
    try:
        from datetime import datetime, timedelta, timezone

        from weft.alert_dedup import suppress
        from weft.models import AlertType

        valid_types = [t.value for t in AlertType]
        if alert_type not in valid_types:
            return _input_error_response(
                "weft_alert_suppress",
                ValueError(f"Invalid alert_type '{alert_type}'. Valid: {valid_types}"),
            )
        if hours <= 0:
            return _input_error_response(
                "weft_alert_suppress",
                ValueError("hours must be > 0"),
            )
        if not dedup_key:
            return _input_error_response(
                "weft_alert_suppress",
                ValueError("dedup_key must be a non-empty string"),
            )

        until = datetime.now(timezone.utc) + timedelta(hours=hours)
        app: AppContext = ctx.request_context.lifespan_context
        async with acquire(app.pool):
            state = await suppress(
                app.pool, AlertType(alert_type), dedup_key,
                until=until, reason=reason,
            )
            return {"success": True, "state": state.to_dict()}
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_alert_suppress", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_alert_suppress", e)


@mcp.tool()
async def weft_alert_unsuppress(
    ctx: Context,
    alert_type: str,
    dedup_key: str,
) -> dict:
    """Lift any active suppression for (alert_type, dedup_key).

    Returns success=True if a suppression was cleared (or no row existed
    with one), success=False if no state row found at all.
    """
    try:
        from weft.alert_dedup import clear_suppression
        from weft.models import AlertType

        valid_types = [t.value for t in AlertType]
        if alert_type not in valid_types:
            return _input_error_response(
                "weft_alert_unsuppress",
                ValueError(f"Invalid alert_type '{alert_type}'. Valid: {valid_types}"),
            )

        app: AppContext = ctx.request_context.lifespan_context
        async with acquire(app.pool):
            cleared = await clear_suppression(
                app.pool, AlertType(alert_type), dedup_key,
            )
            return {"success": cleared, "alert_type": alert_type, "dedup_key": dedup_key}
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_alert_unsuppress", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_alert_unsuppress", e)


# ── Check-in tools ──────────────────────────────────────────────────


@mcp.tool()
async def weft_check_in(
    ctx: Context,
    mood: int | None = None,
    sleep_hours: float | None = None,
    energy: int | None = None,
    notes: str | None = None,
    logged_at: str | None = None,
) -> dict:
    """Log a mood/sleep/energy check-in for personal tracking.

    All fields are optional — log whatever you have:
    mood: 1 (terrible) to 5 (great).
    sleep_hours: hours of sleep (e.g. 7.5).
    energy: 1 (exhausted) to 5 (wired).
    notes: free-text context.
    logged_at: ISO8601 timestamp (defaults to now)."""
    try:
        from datetime import datetime, timezone

        from weft.check_ins import create_check_in
        from weft.models import CheckInCreate

        if mood is None and sleep_hours is None and energy is None and notes is None:
            return _input_error_response(
                "weft_check_in",
                ValueError("Provide at least one of: mood, sleep_hours, energy, notes"),
            )

        logged_dt = None
        if logged_at is not None:
            try:
                logged_dt = datetime.fromisoformat(logged_at)
                if logged_dt.tzinfo is None:
                    logged_dt = logged_dt.replace(tzinfo=timezone.utc)
            except ValueError:
                return _input_error_response(
                    "weft_check_in",
                    ValueError(f"Invalid logged_at '{logged_at}'. Use ISO8601 format."),
                )

        create = CheckInCreate(
            mood=mood,
            sleep_hours=sleep_hours,
            energy=energy,
            notes=notes,
            logged_at=logged_dt,
        )
        app: AppContext = ctx.request_context.lifespan_context
        async with acquire(app.pool):
            check_in = await create_check_in(app.pool, create)
            return {"success": True, "check_in": check_in.to_dict()}
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_check_in", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_check_in", e)


@mcp.tool()
async def weft_check_in_history(
    ctx: Context,
    limit: int = 30,
    offset: int = 0,
) -> dict:
    """View recent check-in history with stats.

    Returns the last N check-ins plus 30-day aggregated stats
    (averages, min/max for mood, sleep, energy)."""
    try:
        from weft.check_ins import get_check_in_stats, list_check_ins

        app: AppContext = ctx.request_context.lifespan_context
        async with acquire(app.pool):
            check_ins = await list_check_ins(app.pool, limit=limit, offset=offset)
            stats = await get_check_in_stats(app.pool)
            return {
                "count": len(check_ins),
                "check_ins": [c.to_dict() for c in check_ins],
                "stats_30d": stats,
            }
    except _DB_ERRORS as e:
        return _db_error_response("weft_check_in_history", e)


@mcp.tool()
async def weft_daily_brief(
    ctx: Context,
    date: str | None = None,
) -> dict:
    """Assemble a daily brief — morning digest of what needs attention.

    Pulls from: memories due for review, recent handoffs, check-in trends,
    Loom ready tasks, and pending alerts. Returns markdown and Slack Block Kit.

    date: optional ISO date string (YYYY-MM-DD). Defaults to today."""
    try:
        from weft.config import DailyBriefConfig
        from weft.daily_brief import assemble_daily_brief

        app: AppContext = ctx.request_context.lifespan_context
        brief_config = DailyBriefConfig(
            timezone=app.config.daily_brief.timezone,
        )

        target_date = None
        if date:
            try:
                from datetime import datetime as dt_mod
                from datetime import timezone as tz_mod

                parsed = dt_mod.fromisoformat(date)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=tz_mod.utc)
                target_date = parsed
            except ValueError:
                return _input_error_response("weft_daily_brief", ValueError(f"Invalid date: {date!r}"))

        async with acquire(app.pool):
            result = await assemble_daily_brief(
                app.pool, brief_config, target_date=target_date
            )
            return {
                "markdown": result.markdown,
                "slack_blocks": result.slack_blocks,
                "generated_at": result.generated_at.isoformat(),
            }
    except _DB_ERRORS as e:
        return _db_error_response("weft_daily_brief", e)


@mcp.tool()
async def weft_ingest(
    ctx: Context,
    text: str,
    source: str = "conversation",
    author: str | None = None,
    metadata: dict | None = None,
    project_id: str | None = None,
) -> dict:
    """Run text through the smart ingestion pipeline.

    Classifies intent via LLM, extracts entities and dates, and routes to
    appropriate Weft subsystems (memories, entities, alerts).

    source: origin of the text (e.g. 'slack', 'email', 'cli').
    metadata: optional dict of extra context (channel, thread_ts, etc.).
    Returns structured IngestResult with counts of created objects."""
    try:
        from weft.ingest_pipeline import IngestItem, process

        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)

        item = IngestItem(
            text=text,
            source=source,
            author=author,
            metadata=metadata or {},
        )

        async with acquire(app.pool):
            result = await process(
                item,
                app.pool,
                app.embedding,
                project_id=resolved_project,
            )

        return {
            "memories_created": result.memories_created,
            "entities_created": result.entities_created,
            "entities_linked": result.entities_linked,
            "alerts_created": result.alerts_created,
            "intents": len(result.intents),
            "errors": result.errors,
        }
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_ingest", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_ingest", e)


@mcp.tool()
async def weft_check_health(
    ctx: Context,
) -> dict:
    """Run a read-only health check across all alert subsystems.

    Evaluates check-in patterns, Loom task awareness, and memory hygiene
    without creating alerts or writing to the database. Returns a unified
    summary of findings and any evaluator errors.

    Also surfaces PROOF metrics for the compounding recall loop:
    - reask_rate: fraction of recent recall queries that are near-duplicate
      re-asks (a rising rate signals the loop is not correcting misses).
    - auto_originated_tier_changes_30d: count of autonomy tier changes
      auto-originated by the calibration loop in the last 30 days (zero
      while calibration_records grow is the dead tell the loop has stalled).
    - replay_queue_depth: count of pending rows in replay_queue (episodes
      awaiting belief re-extraction; rises when the replay loop falls behind).
    - replay_queue_stale_pending: count of pending rows older than the retention
      staleness window (REPLAY_QUEUE_STALENESS_DAYS). Non-zero means the executor
      is not draining rows to a terminal status — those rows no longer pin their
      turns (L4 retention sheds them), but the executor still needs attention.
    - replay_claims_30d: count of belief_claims written by the replay loop in
      the last 30 days. Returns 0 until Epic 3 wires the replay writer.
      Epic 3 MUST set detector_version to a value starting with 'replay-'
      (e.g. 'replay-v1') when writing replay-origin claims.
    - failure_counters: aggregate counts for silently-swallowed failure sites
      (replay.enqueue.failed, calibration.auto_promote.failed,
      replay.executor.failed). A rising count while the corresponding success
      metric stays flat is the tell that a log-and-continue path is broken."""
    try:
        from weft.calibration import count_auto_originated_tier_changes
        from weft.counters import FAILURE_COUNTERS, get_counters
        from weft.db.connection import get_db
        from weft.health_check import run_all_evaluators, summary_to_dict
        from weft.reask import compute_reask_rate
        from weft.replay import (
            REPLAY_DETECTOR_VERSION_PREFIX,
            REPLAY_QUEUE_STATUS_PENDING,
            count_stale_pending_replays,
        )
        from weft.store import get_recent_recall_queries

        app: AppContext = ctx.request_context.lifespan_context
        async with acquire(app.pool):
            result = await run_all_evaluators(app.pool)
            # PROOF metric 1: re-ask rate (query-based, last 30 min window)
            reask_rows = await get_recent_recall_queries(app.pool, window_minutes=30)
            reask_rate = compute_reask_rate(reask_rows)
            # PROOF metric 2: auto-originated tier changes in the last 30 days
            since_30d = datetime.now(timezone.utc) - timedelta(days=30)
            auto_tier_count = await count_auto_originated_tier_changes(
                app.pool, since=since_30d
            )
            # PROOF metric 3: pending replay_queue depth (real query; 0 when
            # no replays are queued, rises as the loop enqueues missed episodes)
            db = get_db(app.pool)
            replay_queue_depth: int = await db.fetchval(
                f"SELECT count(*) FROM replay_queue WHERE status = '{REPLAY_QUEUE_STATUS_PENDING}'"
            )
            # PROOF metric 3b: STALE pending rows — pending longer than the
            # retention staleness window (weft-99cac4e5). The L4 retention guard
            # stops honoring these, so their turns can shed; a non-zero count is
            # the visible tell that the executor is NOT draining rows to a
            # terminal status (loop not running, unwritable user_id, poison row).
            # replay_queue_depth can look healthy while stale rows accumulate, so
            # this is surfaced as its own signal rather than folded into depth.
            replay_queue_stale_pending: int = await count_stale_pending_replays(
                app.pool
            )
            # PROOF metric 4: replay-origin belief_claims in last 30 days.
            # The replay executor (E2.L7) stamps detector_version with the
            # 'replay-' prefix (both the Haiku REPLAY_AGGREGATE_DETECTOR_VERSION
            # and the escalated Sonnet REPLAY_AGGREGATE_SONNET_DETECTOR_VERSION),
            # so this query returns real counts as the loop mints claims — no
            # code change here needed when the writer runs.
            # DEAD-TELL (rollback signal): replay_claims_30d climbing while
            # reask_rate stays flat on replayed topics means the loop is minting
            # claims that are NOT what the misses needed — i.e. the replay path
            # is producing volume without closing recall gaps. Treat a rising
            # replay_claims_30d with no corresponding reask_rate decline as a
            # signal to roll back / re-examine the aggregate detector, not as
            # healthy progress.
            # NOTE: source_provenance cannot carry 'replay' — the CHECK
            # constraint on belief_claims only allows ('user_stated',
            # 'agent_suggested', 'joint_decision'). detector_version is the
            # correct unconstrained field for the replay-origin marker.
            replay_claims_30d: int = await db.fetchval(
                f"""
                SELECT count(*)
                FROM belief_claims
                WHERE detector_version LIKE '{REPLAY_DETECTOR_VERSION_PREFIX}%'
                  AND occurred_at >= $1
                """,
                since_30d,
            )
            # Aggregate failure counters: silently-swallowed failure sites
            # log-and-continue, so a persistent break is invisible without an
            # aggregate signal. A rising replay.enqueue.failed while
            # replay_queue_depth stays pinned at 0 is the "enqueue is silently
            # broken" tell. replay.executor.failed reads 0 until E2.L7 lands.
            failure_counters = await get_counters(app.pool, FAILURE_COUNTERS)

        payload = summary_to_dict(result)
        payload["reask_rate"] = reask_rate
        payload["auto_originated_tier_changes_30d"] = auto_tier_count
        payload["replay_queue_depth"] = replay_queue_depth
        payload["replay_queue_stale_pending"] = replay_queue_stale_pending
        payload["replay_claims_30d"] = replay_claims_30d
        payload["failure_counters"] = failure_counters
        return payload
    except _DB_ERRORS as e:
        return _db_error_response("weft_check_health", e)
    except Exception as e:
        return {"error": f"Health check failed: {type(e).__name__}: {e}"}


# ── Autonomy policy tools ──────────────────────────────────────────


@mcp.tool()
async def weft_autonomy_check(
    ctx: Context,
    action: str,
) -> dict:
    """Check the autonomy tier for an action.

    Returns the effective tier (never/earned/always) and whether the action
    is permitted. Unknown actions default to 'earned' (conservative but not
    blocked).

    action: the action identifier (e.g. 'send_slack_message', 'deploy')."""
    try:
        from weft.autonomy import AutonomyTier, get_policy_by_action, get_tier_for_action

        app: AppContext = ctx.request_context.lifespan_context
        async with acquire(app.pool):
            tier = await get_tier_for_action(app.pool, action)
            policy = await get_policy_by_action(app.pool, action)

        result: dict = {
            "action": action,
            "tier": tier.value,
            "permitted": tier == AutonomyTier.always,
            "requires_approval": tier == AutonomyTier.earned,
            "blocked": tier == AutonomyTier.never,
        }
        if policy:
            result["policy_id"] = policy.id
            result["description"] = policy.description
            result["conditions"] = policy.conditions
        else:
            result["policy_id"] = None
            result["rationale"] = "No policy found — defaulting to 'earned' (requires approval)"
        return result
    except _DB_ERRORS as e:
        return _db_error_response("weft_autonomy_check", e)


@mcp.tool()
async def weft_autonomy_set(
    ctx: Context,
    action: str,
    tier: str = "never",
    description: str | None = None,
    conditions: dict | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
    enabled: bool = True,
) -> dict:
    """Create or register an autonomy policy for an action.

    tier: 'never' (hard stop), 'earned' (requires approval), or 'always' (safe zone).
    conditions: optional JSON dict of constraints (e.g. {"environment": "staging"}).
    Default tier is 'never' — conservative by design."""
    try:
        from weft.autonomy import ActionPolicyCreate, AutonomyTier, create_policy

        valid_tiers = [t.value for t in AutonomyTier]
        if tier not in valid_tiers:
            return _input_error_response(
                "weft_autonomy_set",
                ValueError(f"Invalid tier '{tier}'. Valid: {valid_tiers}"),
            )

        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)

        create = ActionPolicyCreate(
            action=action,
            tier=AutonomyTier(tier),
            description=description,
            conditions=conditions or {},
            project_id=resolved_project,
            agent_id=agent_id,
            enabled=enabled,
        )
        async with acquire(app.pool):
            policy = await create_policy(app.pool, create)
        return {"success": True, "policy": policy.to_dict()}
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_autonomy_set", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_autonomy_set", e)


@mcp.tool()
async def weft_autonomy_list(
    ctx: Context,
    tier: str | None = None,
    enabled_only: bool = True,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """List autonomy policies, optionally filtered by tier.

    tier: optional filter — 'never', 'earned', or 'always'.
    enabled_only: if true (default), only returns enabled policies."""
    try:
        from weft.autonomy import AutonomyTier, list_policies

        tier_enum = None
        if tier is not None:
            valid_tiers = [t.value for t in AutonomyTier]
            if tier not in valid_tiers:
                return _input_error_response(
                    "weft_autonomy_list",
                    ValueError(f"Invalid tier '{tier}'. Valid: {valid_tiers}"),
                )
            tier_enum = AutonomyTier(tier)

        app: AppContext = ctx.request_context.lifespan_context
        async with acquire(app.pool):
            policies = await list_policies(
                app.pool, tier=tier_enum, enabled_only=enabled_only,
                limit=limit, offset=offset,
            )
        return {
            "count": len(policies),
            "policies": [p.to_dict() for p in policies],
        }
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_autonomy_list", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_autonomy_list", e)


@mcp.tool()
async def weft_autonomy_calibrate(
    ctx: Context,
    policy_id: str,
    new_tier: str,
    reason: str | None = None,
    agent_id: str | None = None,
) -> dict:
    """Change an action policy's tier, recording a calibration event.

    Promotes or demotes a policy between tiers. NEVER-tier policies are
    immutable hard-stops and cannot be changed (returns an error).

    policy_id: the policy to calibrate (weft-... ID).
    new_tier: target tier — 'earned' or 'always'.
    reason: why the tier is changing (recorded for audit trail)."""
    try:
        from weft.autonomy import AutonomyTier, list_calibration_events, update_policy_tier

        valid_tiers = [t.value for t in AutonomyTier]
        if new_tier not in valid_tiers:
            return _input_error_response(
                "weft_autonomy_calibrate",
                ValueError(f"Invalid new_tier '{new_tier}'. Valid: {valid_tiers}"),
            )

        if not policy_id or not policy_id.startswith("weft-"):
            return _input_error_response(
                "weft_autonomy_calibrate",
                ValueError(f"Invalid policy_id '{policy_id}'. Expected format: 'weft-...'"),
            )

        app: AppContext = ctx.request_context.lifespan_context
        async with acquire(app.pool):
            updated = await update_policy_tier(
                app.pool, policy_id, AutonomyTier(new_tier),
                reason=reason, agent_id=agent_id,
            )
            events = await list_calibration_events(app.pool, policy_id, limit=5)

        return {
            "success": True,
            "policy": updated.to_dict(),
            "calibration_history": [e.to_dict() for e in events],
        }
    except (ValueError, LookupError) as e:
        return _input_error_response("weft_autonomy_calibrate", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_autonomy_calibrate", e)


# ── Cost tracking tools ─────────────────────────────────────────────


@mcp.tool()
async def weft_cost_record(
    ctx: Context,
    entry_type: str = "session",
    reference_id: str | None = None,
    model: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int = 0,
    estimated_cost_usd: float = 0.0,
    metadata: dict | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
) -> dict:
    """Record a cost entry for token usage and estimated spend.

    entry_type: 'session', 'task', or 'tool_call'.
    reference_id: optional identifier (session ID, task ID, etc.).
    model: the model used (e.g. 'claude-sonnet-4-20250514').
    input_tokens/output_tokens/total_tokens: token counts.
    estimated_cost_usd: estimated cost in USD."""
    try:
        from weft.cost_tracking import CostEntryCreate, CostEntryType, record_cost

        valid_types = [t.value for t in CostEntryType]
        if entry_type not in valid_types:
            return _input_error_response(
                "weft_cost_record",
                ValueError(f"Invalid entry_type '{entry_type}'. Valid: {valid_types}"),
            )

        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)

        create = CostEntryCreate(
            entry_type=CostEntryType(entry_type),
            reference_id=reference_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost_usd,
            metadata=metadata or {},
            project_id=resolved_project,
            agent_id=agent_id,
        )
        async with acquire(app.pool):
            entry = await record_cost(app.pool, create)
        return {"success": True, "entry": entry.to_dict()}
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_cost_record", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_cost_record", e)


@mcp.tool()
async def weft_cost_summary(
    ctx: Context,
    since: str | None = None,
    until: str | None = None,
    entry_type: str | None = None,
    project_id: str | None = None,
) -> dict:
    """Get aggregated cost summary over a time window.

    since/until: ISO8601 datetime strings (optional, defaults to all time).
    entry_type: optional filter — 'session', 'task', or 'tool_call'.
    project_id: optional filter by project."""
    try:
        from weft.cost_tracking import CostEntryType, get_cost_summary

        since_dt = None
        until_dt = None
        type_enum = None

        if since is not None:
            since_dt = datetime.fromisoformat(since)
            if since_dt.tzinfo is None:
                since_dt = since_dt.replace(tzinfo=timezone.utc)

        if until is not None:
            until_dt = datetime.fromisoformat(until)
            if until_dt.tzinfo is None:
                until_dt = until_dt.replace(tzinfo=timezone.utc)

        if entry_type is not None:
            valid_types = [t.value for t in CostEntryType]
            if entry_type not in valid_types:
                return _input_error_response(
                    "weft_cost_summary",
                    ValueError(f"Invalid entry_type '{entry_type}'. Valid: {valid_types}"),
                )
            type_enum = CostEntryType(entry_type)

        resolved_project = await _resolve_project_id(ctx, project_id)

        app: AppContext = ctx.request_context.lifespan_context
        async with acquire(app.pool):
            summary = await get_cost_summary(
                app.pool,
                since=since_dt,
                until=until_dt,
                entry_type=type_enum,
                project_id=resolved_project,
            )
        return summary.to_dict()
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_cost_summary", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_cost_summary", e)


@mcp.tool()
async def weft_calibrate(
    ctx: Context,
    action_category: str,
    action_description: str,
    outcome: str,
    agent_id: str | None = None,
    project_id: str | None = None,
    context: dict | None = None,
) -> dict:
    """Record a calibration event — whether an agent action was approved,
    rejected, or modified by the user.

    After recording, evaluates whether the action_category warrants a
    tier promotion or demotion based on recent calibration history.

    outcome: 'approved', 'rejected', or 'modified'.
    context: optional dict of extra metadata about the action."""
    try:
        from weft.calibration import evaluate_tier_change, record_calibration
        from weft.models import CalibrationCreate, CalibrationOutcome

        valid_outcomes = [o.value for o in CalibrationOutcome]
        if outcome not in valid_outcomes:
            return _input_error_response(
                "weft_calibrate",
                ValueError(f"Invalid outcome '{outcome}'. Valid: {valid_outcomes}"),
            )

        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)

        create = CalibrationCreate(
            action_category=action_category,
            action_description=action_description,
            outcome=CalibrationOutcome(outcome),
            agent_id=agent_id,
            project_id=resolved_project,
            context=context or {},
        )
        async with acquire(app.pool):
            record = await record_calibration(app.pool, create)
            evaluation = await evaluate_tier_change(
                app.pool,
                action_category,
                project_id=resolved_project,
            )
        return {
            "success": True,
            "record": record.to_dict(),
            "tier_evaluation": evaluation,
        }
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_calibrate", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_calibrate", e)


@mcp.tool()
async def weft_calibration_summary(
    ctx: Context,
    action_category: str | None = None,
    project_id: str | None = None,
    since_days: int | None = None,
) -> dict:
    """Get aggregate calibration statistics — approval rates overall and
    per action category.

    action_category: filter to a specific category (optional).
    since_days: only count records from the last N days (optional).

    Returns total, approved, rejected, modified counts, approval_rate,
    and per-category breakdown."""
    try:
        from weft.calibration import get_calibration_summary

        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)

        since = None
        if since_days is not None:
            if since_days <= 0:
                return _input_error_response(
                    "weft_calibration_summary",
                    ValueError("since_days must be positive"),
                )
            since = datetime.now(timezone.utc) - timedelta(days=since_days)

        async with acquire(app.pool):
            summary = await get_calibration_summary(
                app.pool,
                action_category=action_category,
                project_id=resolved_project,
                since=since,
            )
        return {"success": True, **summary}
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_calibration_summary", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_calibration_summary", e)


@mcp.tool()
async def weft_calibration_history(
    ctx: Context,
    action_category: str | None = None,
    outcome: str | None = None,
    project_id: str | None = None,
    limit: int = 20,
) -> dict:
    """List recent calibration records with optional filters.

    action_category: filter by category (optional).
    outcome: filter by outcome — 'approved', 'rejected', or 'modified' (optional).
    limit: max records to return (default 20)."""
    try:
        from weft.calibration import list_calibrations
        from weft.models import CalibrationOutcome

        if outcome is not None:
            valid_outcomes = [o.value for o in CalibrationOutcome]
            if outcome not in valid_outcomes:
                return _input_error_response(
                    "weft_calibration_history",
                    ValueError(f"Invalid outcome '{outcome}'. Valid: {valid_outcomes}"),
                )

        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)

        outcome_enum = CalibrationOutcome(outcome) if outcome else None
        async with acquire(app.pool):
            records = await list_calibrations(
                app.pool,
                action_category=action_category,
                outcome=outcome_enum,
                project_id=resolved_project,
                limit=limit,
            )
        return {
            "success": True,
            "records": [r.to_dict() for r in records],
            "count": len(records),
        }
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_calibration_history", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_calibration_history", e)


@mcp.tool()
async def weft_degradation_set(
    ctx: Context,
    name: str,
    trigger_type: str,
    action: str,
    condition: dict | None = None,
    description: str | None = None,
    cooldown_minutes: float | None = None,
    max_fires: int | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
) -> dict:
    """Create a degradation policy — a rule that fires a response action
    when system health degrades.

    trigger_type: 'low_confidence', 'api_error', 'context_decay', or 'budget_breach'.
    action: 'pause', 'escalate', 'restart', or 'restrict'.
    condition: dict with trigger-specific params (see below).

    Condition requirements per trigger_type:
      low_confidence: {"threshold": 0.3}  (float 0.0-1.0)
      api_error:      {"max_errors": 5, "window_minutes": 10}
      context_decay:  {"max_age_hours": 24}
      budget_breach:  {"max_tokens": 100000}"""
    try:
        from weft.degradation import create_policy
        from weft.models import (
            DegradationAction,
            DegradationPolicyCreate,
            DegradationTriggerType,
        )

        valid_triggers = [t.value for t in DegradationTriggerType]
        if trigger_type not in valid_triggers:
            return _input_error_response(
                "weft_degradation_set",
                ValueError(f"Invalid trigger_type '{trigger_type}'. Valid: {valid_triggers}"),
            )

        valid_actions = [a.value for a in DegradationAction]
        if action not in valid_actions:
            return _input_error_response(
                "weft_degradation_set",
                ValueError(f"Invalid action '{action}'. Valid: {valid_actions}"),
            )

        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)

        create = DegradationPolicyCreate(
            name=name,
            trigger_type=DegradationTriggerType(trigger_type),
            condition=condition or {},
            action=DegradationAction(action),
            description=description,
            cooldown_minutes=cooldown_minutes,
            max_fires=max_fires,
            project_id=resolved_project,
            agent_id=agent_id,
        )
        async with acquire(app.pool):
            policy = await create_policy(app.pool, create)
        return {"success": True, "policy": policy.to_dict()}
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_degradation_set", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_degradation_set", e)


@mcp.tool()
async def weft_degradation_check(
    ctx: Context,
    confidence: float | None = None,
    error_count: int | None = None,
    context_age_hours: float | None = None,
    tokens_used: int | None = None,
    project_id: str | None = None,
) -> dict:
    """Check current metrics against active degradation policies and fire
    any that match.

    Pass the metrics you have — policies only trigger on metrics they
    care about. Returns a list of triggered policies with the action
    to take.

    confidence: current confidence level (0.0-1.0).
    error_count: number of API errors in recent window.
    context_age_hours: hours since context was refreshed.
    tokens_used: total tokens consumed in session."""
    try:
        from weft.degradation import update_degradation_state

        metrics: dict[str, float | int] = {}
        if confidence is not None:
            metrics["confidence"] = confidence
        if error_count is not None:
            metrics["error_count"] = error_count
        if context_age_hours is not None:
            metrics["context_age_hours"] = context_age_hours
        if tokens_used is not None:
            metrics["tokens_used"] = tokens_used

        if not metrics:
            return _input_error_response(
                "weft_degradation_check",
                ValueError("At least one metric must be provided"),
            )

        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)

        async with acquire(app.pool):
            triggered = await update_degradation_state(
                app.pool,
                metrics=metrics,
                project_id=resolved_project,
            )
        return {
            "success": True,
            "triggered": triggered,
            "triggered_count": len(triggered),
        }
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_degradation_check", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_degradation_check", e)


@mcp.tool()
async def weft_degradation_list(
    ctx: Context,
    trigger_type: str | None = None,
    action: str | None = None,
    status: str | None = None,
    project_id: str | None = None,
    limit: int = 20,
) -> dict:
    """List degradation policies with optional filters.

    trigger_type: 'low_confidence', 'api_error', 'context_decay', or 'budget_breach'.
    action: 'pause', 'escalate', 'restart', or 'restrict'.
    status: 'active', 'disabled', or 'fired'."""
    try:
        from weft.degradation import list_policies
        from weft.models import (
            DegradationAction,
            DegradationPolicyStatus,
            DegradationTriggerType,
        )

        tt_enum = None
        if trigger_type is not None:
            valid = [t.value for t in DegradationTriggerType]
            if trigger_type not in valid:
                return _input_error_response(
                    "weft_degradation_list",
                    ValueError(f"Invalid trigger_type '{trigger_type}'. Valid: {valid}"),
                )
            tt_enum = DegradationTriggerType(trigger_type)

        action_enum = None
        if action is not None:
            valid = [a.value for a in DegradationAction]
            if action not in valid:
                return _input_error_response(
                    "weft_degradation_list",
                    ValueError(f"Invalid action '{action}'. Valid: {valid}"),
                )
            action_enum = DegradationAction(action)

        status_enum = None
        if status is not None:
            valid = [s.value for s in DegradationPolicyStatus]
            if status not in valid:
                return _input_error_response(
                    "weft_degradation_list",
                    ValueError(f"Invalid status '{status}'. Valid: {valid}"),
                )
            status_enum = DegradationPolicyStatus(status)

        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)

        async with acquire(app.pool):
            policies = await list_policies(
                app.pool,
                trigger_type=tt_enum,
                action=action_enum,
                status=status_enum,
                project_id=resolved_project,
                limit=limit,
            )
        return {
            "success": True,
            "policies": [p.to_dict() for p in policies],
            "count": len(policies),
        }
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_degradation_list", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_degradation_list", e)


@mcp.tool()
async def weft_budget_check(
    ctx: Context,
    daily_limit_usd: float = 10.0,
) -> dict:
    """Check today's spending against a daily budget limit.

    Returns within_budget (bool), pct_used (0-100+), remaining_usd,
    and daily_spent_usd. Default daily limit is $10.

    daily_limit_usd: the budget ceiling for today (in USD)."""
    try:
        from weft.cost_tracking import check_budget

        if daily_limit_usd < 0:
            return _input_error_response(
                "weft_budget_check",
                ValueError("daily_limit_usd must be non-negative"),
            )

        app: AppContext = ctx.request_context.lifespan_context
        async with acquire(app.pool):
            status = await check_budget(app.pool, daily_limit_usd)
        return status.to_dict()
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_budget_check", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_budget_check", e)


# --- Proactive trigger tools ---


@mcp.tool()
async def weft_trigger_create(
    ctx: Context,
    name: str,
    condition_type: str,
    condition: dict,
    action: str,
    cooldown_hours: float | None = None,
    max_fires: int | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
) -> dict:
    """Create a proactive trigger that fires when its condition is met.

    condition_type: 'time', 'threshold', 'event', or 'absence'.
    condition: dict with type-specific params:
      time: {'trigger_at': '<ISO datetime>'}
      threshold: {'metric': '<name>', 'threshold': <number>}
      event: {'event_name': '<name>'}
      absence: {'absence_hours': <positive number>}
    action: description of what should happen when triggered.
    cooldown_hours: minimum hours between firings (None = no cooldown).
    max_fires: stop after N firings (None = unlimited)."""
    try:
        from weft.models import TriggerConditionType, TriggerCreate
        from weft.triggers import create_trigger

        valid_types = [t.value for t in TriggerConditionType]
        if condition_type not in valid_types:
            return _input_error_response(
                "weft_trigger_create",
                ValueError(f"Invalid condition_type '{condition_type}'. Valid: {valid_types}"),
            )

        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)

        create = TriggerCreate(
            name=name,
            condition_type=TriggerConditionType(condition_type),
            condition=condition,
            action=action,
            cooldown_hours=cooldown_hours,
            max_fires=max_fires,
            project_id=resolved_project,
            agent_id=agent_id,
        )
        async with acquire(app.pool):
            trigger = await create_trigger(app.pool, create)
            return {"success": True, "trigger": trigger.to_dict()}
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_trigger_create", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_trigger_create", e)


@mcp.tool()
async def weft_trigger_list(
    ctx: Context,
    condition_type: str | None = None,
    status: str | None = None,
    project_id: str | None = None,
    limit: int = 50,
) -> dict:
    """List proactive triggers with optional filters.

    condition_type: 'time', 'threshold', 'event', or 'absence'.
    status: 'enabled', 'disabled', or 'fired'."""
    try:
        from weft.models import TriggerConditionType, TriggerStatus
        from weft.triggers import list_triggers

        ct_enum = None
        if condition_type is not None:
            valid = [t.value for t in TriggerConditionType]
            if condition_type not in valid:
                return _input_error_response(
                    "weft_trigger_list",
                    ValueError(f"Invalid condition_type '{condition_type}'. Valid: {valid}"),
                )
            ct_enum = TriggerConditionType(condition_type)

        st_enum = None
        if status is not None:
            valid = [s.value for s in TriggerStatus]
            if status not in valid:
                return _input_error_response(
                    "weft_trigger_list",
                    ValueError(f"Invalid status '{status}'. Valid: {valid}"),
                )
            st_enum = TriggerStatus(status)

        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)

        async with acquire(app.pool):
            triggers = await list_triggers(
                app.pool,
                condition_type=ct_enum,
                status=st_enum,
                project_id=resolved_project,
                limit=limit,
            )
        return {
            "count": len(triggers),
            "triggers": [t.to_dict() for t in triggers],
        }
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_trigger_list", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_trigger_list", e)


@mcp.tool()
async def weft_trigger_due(
    ctx: Context,
    condition_type: str | None = None,
    project_id: str | None = None,
) -> dict:
    """Get triggers whose conditions are met and cooldown has elapsed.

    Returns enabled triggers ready to fire, sorted oldest-first.
    condition_type: optional filter — 'time', 'threshold', 'event', or 'absence'."""
    try:
        from weft.models import TriggerConditionType
        from weft.triggers import get_triggers_due

        ct_enum = None
        if condition_type is not None:
            valid = [t.value for t in TriggerConditionType]
            if condition_type not in valid:
                return _input_error_response(
                    "weft_trigger_due",
                    ValueError(f"Invalid condition_type '{condition_type}'. Valid: {valid}"),
                )
            ct_enum = TriggerConditionType(condition_type)

        app: AppContext = ctx.request_context.lifespan_context
        resolved_project = await _resolve_project_id(ctx, project_id)

        async with acquire(app.pool):
            due = await get_triggers_due(
                app.pool,
                condition_type=ct_enum,
                project_id=resolved_project,
            )
        return {
            "count": len(due),
            "triggers": [t.to_dict() for t in due],
        }
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_trigger_due", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_trigger_due", e)


@mcp.tool()
async def weft_trigger_fire(
    ctx: Context,
    trigger_id: str,
) -> dict:
    """Record that a trigger has fired. Increments fire_count and sets last_fired_at.

    If max_fires is reached, transitions status to 'fired' (one-shot complete).
    Returns the updated trigger."""
    try:
        from weft.triggers import record_fire

        app: AppContext = ctx.request_context.lifespan_context
        async with acquire(app.pool):
            trigger = await record_fire(app.pool, trigger_id)
            return {"success": True, "trigger": trigger.to_dict()}
    except LookupError as e:
        return _input_error_response("weft_trigger_fire", e)
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_trigger_fire", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_trigger_fire", e)


@mcp.tool()
async def weft_trigger_delete(
    ctx: Context,
    trigger_id: str,
) -> dict:
    """Permanently delete a proactive trigger."""
    try:
        from weft.triggers import delete_trigger

        app: AppContext = ctx.request_context.lifespan_context
        async with acquire(app.pool):
            deleted = await delete_trigger(app.pool, trigger_id)
            return {
                "trigger_id": trigger_id,
                "deleted": deleted,
            }
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_trigger_delete", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_trigger_delete", e)


# ---------------------------------------------------------------------------
# Workspace tools — shared-brain primitive. Memories tagged with workspace_id
# are readable by every workspace member (RLS subquery in migrations.py:1299).
# ---------------------------------------------------------------------------


@mcp.tool()
async def weft_workspace_create(
    ctx: Context,
    name: str,
    description: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """Create a shared workspace. Caller becomes the owner and an admin member.

    Use the returned ``id`` as the ``workspace_id`` parameter on
    ``weft_remember`` to scope memories to this shared brain. Add other
    user_ids with ``weft_workspace_add_member``."""
    try:
        from weft.workspaces import create_workspace
        app: AppContext = ctx.request_context.lifespan_context
        caller_uid = resolve_caller_user_id()
        async with acquire(app.pool):
            ws = await create_workspace(
                app.pool,
                name=name,
                created_by=caller_uid,
                description=description,
                metadata=metadata,
            )
            return ws.to_dict()
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_workspace_create", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_workspace_create", e)


@mcp.tool()
async def weft_workspace_add_member(
    ctx: Context,
    workspace_id: str,
    user_id: str,
    role: str = "member",
) -> dict:
    """Add a user to a workspace. Only the workspace owner can add members."""
    try:
        from weft.workspaces import add_member
        app: AppContext = ctx.request_context.lifespan_context
        caller_uid = resolve_caller_user_id()
        async with acquire(app.pool):
            member = await add_member(
                app.pool,
                workspace_id=workspace_id,
                user_id=user_id,
                added_by=caller_uid,
                role=role,
            )
            return member.to_dict()
    except PermissionError as e:
        return {"error": "Permission denied", "detail": str(e), "tool": "weft_workspace_add_member"}
    except LookupError as e:
        return {"error": "Not found", "detail": str(e), "tool": "weft_workspace_add_member"}
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_workspace_add_member", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_workspace_add_member", e)


@mcp.tool()
async def weft_workspace_remove_member(
    ctx: Context,
    workspace_id: str,
    user_id: str,
) -> dict:
    """Remove a user from a workspace. Only the owner can remove members.
    The owner cannot be removed — delete the workspace instead."""
    try:
        from weft.workspaces import remove_member
        app: AppContext = ctx.request_context.lifespan_context
        caller_uid = resolve_caller_user_id()
        async with acquire(app.pool):
            removed = await remove_member(
                app.pool,
                workspace_id=workspace_id,
                user_id=user_id,
                removed_by=caller_uid,
            )
            return {"workspace_id": workspace_id, "user_id": user_id, "removed": removed}
    except PermissionError as e:
        return {"error": "Permission denied", "detail": str(e), "tool": "weft_workspace_remove_member"}
    except LookupError as e:
        return {"error": "Not found", "detail": str(e), "tool": "weft_workspace_remove_member"}
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_workspace_remove_member", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_workspace_remove_member", e)


@mcp.tool()
async def weft_workspace_list(ctx: Context) -> dict:
    """List all workspaces the caller is a member of, newest first.
    Includes workspaces where the caller is the owner."""
    try:
        from weft.workspaces import list_workspaces_for_user, list_members
        app: AppContext = ctx.request_context.lifespan_context
        caller_uid = resolve_caller_user_id()
        async with acquire(app.pool):
            workspaces = await list_workspaces_for_user(app.pool, caller_uid)
            out = []
            for ws in workspaces:
                members = await list_members(app.pool, ws.id)
                d = ws.to_dict()
                d["members"] = [m.to_dict() for m in members]
                d["is_owner"] = ws.created_by == caller_uid
                out.append(d)
            return {"workspaces": out, "count": len(out)}
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_workspace_list", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_workspace_list", e)


# ---------------------------------------------------------------------------
# Tracker tools — open-loop primitive (Wick Phase 3, weft_v2_spec.md §3).
# State machine: open ∈ {in_progress, awaiting_reply, blocked};
#                terminal ∈ {done, abandoned}. Nudges fire on schedule;
#                snooze suppresses; dismiss bumps; close terminates.
# ---------------------------------------------------------------------------


def _parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


def _parse_interval(value: str | int | None) -> timedelta | None:
    """Accept ISO duration ('PT1H'), shorthand ('30d', '2w', '3h'), or seconds."""
    if value is None:
        return None
    if isinstance(value, int):
        return timedelta(seconds=value)
    s = value.strip().lower()
    if s.endswith("d"):
        return timedelta(days=int(s[:-1]))
    if s.endswith("w"):
        return timedelta(weeks=int(s[:-1]))
    if s.endswith("h"):
        return timedelta(hours=int(s[:-1]))
    if s.endswith("m"):
        # treat 'm' as minutes here (review_after uses 'm' for months but
        # interval semantics differ — minutes is the more useful unit for
        # nudge intervals)
        return timedelta(minutes=int(s[:-1]))
    if s.endswith("s"):
        return timedelta(seconds=int(s[:-1]))
    # Fall back to ISO 8601 duration
    raise ValueError(f"unrecognized interval: {value!r}")


@mcp.tool()
async def weft_tracker_create(
    ctx: Context,
    kind: TrackerKindLiteral,
    title: str,
    project_id: str | None = None,
    entity_id: str | None = None,
    state: TrackerStateLiteral = "in_progress",
    context: dict | None = None,
    nudge_mode: NudgeModeLiteral = "none",
    nudge_after: str | None = None,
    nudge_interval: str | None = None,
) -> dict:
    """Create a tracker — a lifecycle-aware open loop.

    Use trackers for things memories can't track well: pitches awaiting
    reply, follow-ups, shopping lists, meal plans, long-running orchestrator
    traces. Memories are blob-shaped facts; trackers carry state that changes
    and can be nudged.

    nudge_after: ISO timestamp when the nudge should first fire.
    nudge_interval: '7d' / '2w' / '3h' / '30m' — interval between recurring nudges."""
    try:
        from weft.trackers import create_tracker
        app: AppContext = ctx.request_context.lifespan_context
        create = TrackerCreate(
            kind=TrackerKind(kind),
            title=title,
            project_id=project_id,
            entity_id=entity_id,
            state=TrackerState(state),
            context=context or {},
            nudge_mode=NudgeMode(nudge_mode),
            nudge_after=_parse_dt(nudge_after),
            nudge_interval=_parse_interval(nudge_interval),
        )
        async with acquire(app.pool):
            tr = await create_tracker(app.pool, create)
            return tr.to_dict()
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_tracker_create", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_tracker_create", e)


@mcp.tool()
async def weft_tracker_get(ctx: Context, tracker_id: str) -> dict:
    """Fetch a single tracker by ID."""
    try:
        from weft.trackers import get_tracker
        app: AppContext = ctx.request_context.lifespan_context
        async with acquire(app.pool):
            tr = await get_tracker(app.pool, tracker_id)
            if tr is None:
                return {"error": "Not found", "tracker_id": tracker_id}
            return tr.to_dict()
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_tracker_get", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_tracker_get", e)


@mcp.tool()
async def weft_tracker_update(
    ctx: Context,
    tracker_id: str,
    title: str | None = None,
    state: TrackerStateLiteral | None = None,
    state_note: str | None = None,
    context: dict | None = None,
    nudge_mode: NudgeModeLiteral | None = None,
    nudge_after: str | None = None,
    nudge_interval: str | None = None,
) -> dict:
    """Update tracker fields. State transitions append to state_history.
    Cannot transition out of terminal states (done, abandoned)."""
    try:
        from weft.trackers import update_tracker
        app: AppContext = ctx.request_context.lifespan_context
        async with acquire(app.pool):
            tr = await update_tracker(
                app.pool, tracker_id,
                title=title,
                state=TrackerState(state) if state else None,
                state_note=state_note,
                context=context,
                nudge_mode=NudgeMode(nudge_mode) if nudge_mode else None,
                nudge_after=_parse_dt(nudge_after),
                nudge_interval=_parse_interval(nudge_interval),
            )
            return tr.to_dict()
    except LookupError as e:
        return {"error": "Not found", "detail": str(e), "tool": "weft_tracker_update"}
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_tracker_update", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_tracker_update", e)


@mcp.tool()
async def weft_tracker_close(
    ctx: Context,
    tracker_id: str,
    final_state: TrackerStateLiteral = "done",
    note: str | None = None,
) -> dict:
    """Terminal close. final_state must be 'done' or 'abandoned'."""
    try:
        from weft.trackers import close_tracker
        app: AppContext = ctx.request_context.lifespan_context
        async with acquire(app.pool):
            tr = await close_tracker(
                app.pool, tracker_id,
                final_state=TrackerState(final_state),
                note=note,
            )
            return tr.to_dict()
    except LookupError as e:
        return {"error": "Not found", "detail": str(e), "tool": "weft_tracker_close"}
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_tracker_close", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_tracker_close", e)


@mcp.tool()
async def weft_tracker_dismiss(ctx: Context, tracker_id: str) -> dict:
    """One-click "thanks, I know" — bumps last_touch, rolls a recurring
    nudge forward by nudge_interval, or silences a once-mode tracker."""
    try:
        from weft.trackers import dismiss_tracker
        app: AppContext = ctx.request_context.lifespan_context
        async with acquire(app.pool):
            tr = await dismiss_tracker(app.pool, tracker_id)
            return tr.to_dict()
    except LookupError as e:
        return {"error": "Not found", "detail": str(e), "tool": "weft_tracker_dismiss"}
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_tracker_dismiss", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_tracker_dismiss", e)


@mcp.tool()
async def weft_tracker_snooze(
    ctx: Context,
    tracker_id: str,
    until: str,
) -> dict:
    """Suppress nudges until ISO timestamp. Mode unchanged — resumes after."""
    try:
        from weft.trackers import snooze_tracker
        app: AppContext = ctx.request_context.lifespan_context
        async with acquire(app.pool):
            tr = await snooze_tracker(
                app.pool, tracker_id, _parse_dt(until),
            )
            return tr.to_dict()
    except LookupError as e:
        return {"error": "Not found", "detail": str(e), "tool": "weft_tracker_snooze"}
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_tracker_snooze", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_tracker_snooze", e)


@mcp.tool()
async def weft_tracker_list(
    ctx: Context,
    kind: TrackerKindLiteral | None = None,
    state: TrackerStateLiteral | None = None,
    open_only: bool = False,
    project_id: str | None = None,
    entity_id: str | None = None,
    context_filter: dict[str, str] | None = None,
    since: str | None = None,
    limit: int = 100,
) -> dict:
    """List trackers, newest-touch first. Filter by kind, state, scope.

    context_filter: match top-level keys in the JSONB context column by
    string equality (one ``context->>key = value`` clause per pair, ANDed).
    Use this to partition the kind=trace catchall — e.g.
    ``{"wick_kind": "authority_skip"}`` isolates Wick authority-skip events.
    since: ISO8601 datetime; bounds results to created_at >= since.
    """
    try:
        from weft.trackers import list_trackers
        since_dt = None
        if since is not None:
            since_dt = datetime.fromisoformat(since)
            if since_dt.tzinfo is None:
                since_dt = since_dt.replace(tzinfo=timezone.utc)
        app: AppContext = ctx.request_context.lifespan_context
        async with acquire(app.pool):
            trs = await list_trackers(
                app.pool,
                kind=TrackerKind(kind) if kind else None,
                state=TrackerState(state) if state else None,
                open_only=open_only,
                project_id=project_id,
                entity_id=entity_id,
                context_filter=context_filter,
                since=since_dt,
                limit=limit,
            )
            return {"trackers": [t.to_dict() for t in trs], "count": len(trs)}
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_tracker_list", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_tracker_list", e)


@mcp.tool()
async def weft_tracker_due(
    ctx: Context,
    limit: int = 100,
) -> dict:
    """Trackers with a nudge due now: open-state, non-snoozed, past nudge_after.
    Use this for the daily-brief "open loops" section."""
    try:
        from weft.trackers import due_trackers
        app: AppContext = ctx.request_context.lifespan_context
        async with acquire(app.pool):
            trs = await due_trackers(app.pool, limit=limit)
            return {"trackers": [t.to_dict() for t in trs], "count": len(trs)}
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_tracker_due", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_tracker_due", e)


# --- List sugar — context.items helpers for kind=list / shopping_list / meal_plan / pantry ---


@mcp.tool()
async def weft_list_append(
    ctx: Context,
    tracker_id: str,
    text: str,
    checked: bool = False,
    link: str | None = None,
) -> dict:
    """Append an item to a list-shaped tracker's context.items.
    Item shape: {text, checked, link?}."""
    try:
        from weft.trackers import list_append
        app: AppContext = ctx.request_context.lifespan_context
        item: dict = {"text": text, "checked": checked}
        if link is not None:
            item["link"] = link
        async with acquire(app.pool):
            tr = await list_append(app.pool, tracker_id, item)
            return tr.to_dict()
    except LookupError as e:
        return {"error": "Not found", "detail": str(e), "tool": "weft_list_append"}
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_list_append", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_list_append", e)


@mcp.tool()
async def weft_list_check(
    ctx: Context,
    tracker_id: str,
    index: int,
    checked: bool = True,
) -> dict:
    """Toggle the checked flag on the item at ``index`` (0-based)."""
    try:
        from weft.trackers import list_check
        app: AppContext = ctx.request_context.lifespan_context
        async with acquire(app.pool):
            tr = await list_check(app.pool, tracker_id, index, checked=checked)
            return tr.to_dict()
    except LookupError as e:
        return {"error": "Not found", "detail": str(e), "tool": "weft_list_check"}
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_list_check", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_list_check", e)


@mcp.tool()
async def weft_list_remove(
    ctx: Context,
    tracker_id: str,
    index: int,
) -> dict:
    """Remove the item at ``index`` from a list-shaped tracker."""
    try:
        from weft.trackers import list_remove
        app: AppContext = ctx.request_context.lifespan_context
        async with acquire(app.pool):
            tr = await list_remove(app.pool, tracker_id, index)
            return tr.to_dict()
    except LookupError as e:
        return {"error": "Not found", "detail": str(e), "tool": "weft_list_remove"}
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_list_remove", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_list_remove", e)


# ---------------------------------------------------------------------------
# Phase 2.5 / L6 — bearer-token management over MCP.
#
# Mirror of the ``weft tokens`` CLI group. Every tool here is supervisor-only:
# the whole point of the credential model is that an agent-mode caller cannot
# escalate, so letting agent-mode mint or revoke tokens would round-trip the
# entire defense to zero. The Phase-2 caller-mode gate is enforced in the
# tool body, not at the MCP boundary, matching ``weft_quarantine_review``.
#
# Tokens are first-class auth credentials: each row binds a user_id to a
# caller_mode at issuance time. Plaintext is returned ONCE on issue and then
# only the SHA-256 hash is persisted. Revocation always takes the full hash —
# we never accept plaintext on the revoke path because that re-introduces the
# leak vector L1 was designed to close.
# ---------------------------------------------------------------------------


_SUPERVISOR_ONLY_TOKEN_TOOLS_ERROR = (
    "weft_token_* tools are supervisor-only. Agent-mode callers cannot "
    "mint, list, or revoke credentials (Phase 2.5 / L6)."
)


def _supervisor_gate(tool_name: str) -> dict | None:
    """Return an error dict if the caller is in agent mode, else None.

    Centralized so each token tool's first three lines look the same and a
    future audit can grep for callers of this gate."""
    from weft.auth import is_agent_caller
    if is_agent_caller():
        logger.warning(
            "agent-mode caller blocked from %s (Phase 2.5 / L6 supervisor gate)",
            tool_name,
        )
        return {"error": _SUPERVISOR_ONLY_TOKEN_TOOLS_ERROR, "tool": tool_name}
    return None


def _parse_expires_in_spec(spec: str | None) -> "timedelta | None":
    """Mirror of weft.cli._parse_expires_in. Accepts 'Nd' / 'Nh' / 'Nm'.

    Lives here as a private helper so the MCP layer doesn't import from
    the click-based CLI module (cli.py pulls in heavy deps and click
    decorators we don't want in the MCP path)."""
    if not spec:
        return None
    spec = spec.strip().lower()
    units = {"d": "days", "h": "hours", "m": "minutes"}
    if spec[-1] not in units or not spec[:-1].isdigit():
        raise ValueError(f"expires_in must be N(d|h|m), got {spec!r}")
    return timedelta(**{units[spec[-1]]: int(spec[:-1])})


@mcp.tool()
async def weft_token_issue(
    ctx: Context,
    user_id: str,
    caller_mode: Literal["supervisor", "agent"],
    label: str | None = None,
    expires_in: str | None = None,
) -> dict:
    """Mint a bearer token bound to ``(user_id, caller_mode)``.

    Supervisor-only. The plaintext token is returned **once** in the
    ``token`` field — store it immediately (1Password, env var, secret
    manager) because there is no recovery path. Only the SHA-256
    ``token_hash`` is persisted; future revocation goes through that hash.

    Parameters
    ----------
    user_id:
        The user this credential authenticates as. Cannot be empty.
    caller_mode:
        ``'supervisor'`` (full trust, can downgrade to agent via header for
        testing) or ``'agent'`` (lower trust, agent floor — the header
        cannot escalate). Stamped into the row at issuance and read by
        :class:`UserIdentityMiddleware` on every request.
    label:
        Free-text operator note (e.g. ``'wick-runtime'``,
        ``'face-2026-04'``). Surfaces in ``weft_token_list``.
    expires_in:
        Optional ``Nd`` / ``Nh`` / ``Nm`` spec, e.g. ``'30d'``. Omit for
        a non-expiring REDACTED

    Returns
    -------
    dict with: ``token`` (plaintext, one-time), ``token_hash``,
    ``user_id``, ``caller_mode``, ``label``, ``expires_at`` (ISO or null).
    """
    blocked = _supervisor_gate("weft_token_issue")
    if blocked is not None:
        return blocked

    try:
        from weft.credentials import issue_token
        delta = _parse_expires_in_spec(expires_in)
        app: AppContext = ctx.request_context.lifespan_context
        plaintext, row = await issue_token(
            app.pool,
            user_id=user_id,
            caller_mode=caller_mode,
            label=label,
            expires_in=delta,
        )
        return {
            "token": plaintext,
            "token_hash": row.token_hash,
            "user_id": row.user_id,
            "caller_mode": row.caller_mode,
            "label": row.label,
            "expires_at": row.expires_at.isoformat() if row.expires_at else None,
            "warning": (
                "Plaintext token will not be shown again — store it now. "
                "Use the hash for revocation."
            ),
        }
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_token_issue", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_token_issue", e)


@mcp.tool()
async def weft_token_list(
    ctx: Context,
    user_id: str,
    include_revoked: bool = False,
) -> dict:
    """List a user's tokens, newest first. Supervisor-only.

    Each row carries the ``token_hash`` (full 64-char SHA-256 — pass to
    ``weft_token_revoke``), caller_mode, label, created/last-used/expires
    timestamps, and a derived ``status`` of ``active`` / ``revoked`` /
    ``expired``. Plaintext is never returned — that ship sailed at issuance.

    Defaults to live rows; pass ``include_revoked=True`` for audit /
    forensics. Expired-but-not-revoked rows are always included so
    operators see what aged out."""
    blocked = _supervisor_gate("weft_token_list")
    if blocked is not None:
        return blocked

    try:
        from datetime import datetime, timezone
        from weft.credentials import list_tokens
        app: AppContext = ctx.request_context.lifespan_context
        rows = await list_tokens(
            app.pool, user_id, include_revoked=include_revoked,
        )
        now = datetime.now(timezone.utc)
        items = []
        for r in rows:
            if r.revoked_at is not None:
                status = "revoked"
            elif r.expires_at is not None and r.expires_at <= now:
                status = "expired"
            else:
                status = "active"
            items.append({
                "token_hash": r.token_hash,
                "caller_mode": r.caller_mode,
                "label": r.label,
                "created_at": r.created_at.isoformat(),
                "last_used_at": (
                    r.last_used_at.isoformat() if r.last_used_at else None
                ),
                "expires_at": r.expires_at.isoformat() if r.expires_at else None,
                "revoked_at": r.revoked_at.isoformat() if r.revoked_at else None,
                "status": status,
            })
        return {"count": len(items), "tokens": items}
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_token_list", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_token_list", e)


@mcp.tool()
async def weft_token_revoke(
    ctx: Context,
    token_hash: str,
) -> dict:
    """Revoke a token by its full 64-char SHA-256 hash. Supervisor-only.

    Mirrors the CLI: partial hashes are deliberately rejected — the
    ``weft_token_list`` response surfaces the full hash; revocation
    requires the full value to keep operator intent unambiguous and to
    avoid prefix-collision footguns.

    Idempotent: ``revoked: false`` for unknown hashes and rows already
    revoked. The two are not distinguished — same reason
    ``lookup_token`` collapses unknown / revoked / expired into a single
    None: probe-resistance."""
    blocked = _supervisor_gate("weft_token_revoke")
    if blocked is not None:
        return blocked

    if len(token_hash) != 64:
        return {
            "error": "Invalid input",
            "detail": (
                f"token_hash must be the full 64-char SHA-256 hex; "
                f"got {len(token_hash)} chars"
            ),
            "tool": "weft_token_revoke",
        }

    try:
        from weft.credentials import revoke_token
        app: AppContext = ctx.request_context.lifespan_context
        flipped = await revoke_token(app.pool, token_hash)
        return {
            "token_hash": token_hash,
            "revoked": flipped,
            "detail": (
                "Token revoked." if flipped
                else "No live token matched (unknown hash or already revoked)."
            ),
        }
    except _DB_ERRORS as e:
        return _db_error_response("weft_token_revoke", e)


@mcp.tool()
async def weft_turn_append(
    ctx: Context,
    episode_id: str,
    role: str,
    content: str,
    occurred_at: str | None = None,
    trace_id: str | None = None,
) -> dict:
    """Append one dialogue turn to an existing episode (turn-tier write).

    Wick's per-completion fire-and-forget hook: each user/assistant exchange
    becomes one turn so the raw trace is queryable later. Race-safe under
    concurrent appenders to the same episode (advisory lock keyed by
    episode_id; turn_index assigned inside the same transaction).

    role: one of "user", "assistant", "tool", "system".
    occurred_at: ISO-8601 timestamp; defaults to now() if omitted.
    trace_id: optional Wick run_id for cross-system correlation.

    Returns ``{turn_id, episode_id, turn_index, occurred_at}``. Embedding
    is computed inline before insert, matching the ``weft_remember`` →
    ``store_memory`` pattern; on embedding failure the turn is still
    written (searchable by turn_index/timestamp, not vector) and the
    response carries a ``warning`` field.
    """
    try:
        cid = set_correlation_id()
        logger.debug("weft_turn_append start [%s]", cid)
        app: AppContext = ctx.request_context.lifespan_context

        # Validate role early so a typo from Wick surfaces as an input
        # error, not a 422-ish enum failure later.
        from weft.models import EpisodeTurnCreate, TurnRole
        try:
            role_enum = TurnRole(role)
        except ValueError:
            valid = ", ".join(r.value for r in TurnRole)
            raise ValueError(
                f"role must be one of [{valid}]; got {role!r}"
            )

        parsed_at: datetime | None = None
        if occurred_at is not None:
            try:
                parsed_at = datetime.fromisoformat(occurred_at)
            except ValueError as exc:
                raise ValueError(
                    f"occurred_at must be ISO-8601 (e.g. "
                    f"'2026-05-02T12:34:56+00:00'); got {occurred_at!r} "
                    f"({exc})"
                )
            if parsed_at.tzinfo is None:
                parsed_at = parsed_at.replace(tzinfo=timezone.utc)

        create = EpisodeTurnCreate(
            episode_id=episode_id,
            role=role_enum,
            content=content,
            occurred_at=parsed_at,
            trace_id=trace_id,
        )

        # Verify the episode exists before locking the row — gives Wick a
        # clean "episode not found" instead of a foreign-key violation
        # surfaced as a generic DB error.
        async with acquire(app.pool):
            episode_row = await app.pool.fetchrow(
                "SELECT id FROM episodes WHERE id = $1", episode_id,
            )
            if episode_row is None:
                raise ValueError(f"episode not found: {episode_id!r}")

        embedding = None
        embedding_failed = False
        try:
            embedding = await app.embedding.embed(content)
        except Exception as embed_err:
            logger.warning(
                "Embedding failed for weft_turn_append, storing without "
                "vector: %s", embed_err,
            )
            embedding_failed = True

        from weft.episode_turns import append_turn
        async with acquire(app.pool):
            turn = await append_turn(app.pool, create, embedding=embedding)

        result = {
            "turn_id": turn.id,
            "episode_id": turn.episode_id,
            "turn_index": turn.turn_index,
            "occurred_at": turn.occurred_at.isoformat(),
        }
        if embedding_failed:
            result["warning"] = (
                "Turn saved but embedding failed — not searchable by "
                "semantic similarity until next re-embed cycle."
            )
        return result
    except _INPUT_ERRORS as e:
        return _input_error_response("weft_turn_append", e)
    except _DB_ERRORS as e:
        return _db_error_response("weft_turn_append", e)


@mcp.tool()
async def weft_fsck(ctx: Context) -> dict:
    """List orphan memories: active memories reachable ONLY by vector cosine.

    Orphans are memories with NO topic tags AND NO entity mentions AND NO
    episode membership. They hide in the vector index but cannot be recalled
    through tag/entity/episode navigators — a leading indicator of future
    recall misses.

    Returns: {
        "orphan_count": int,
        "orphans": [
            {"memory_id": str, "reason": str},
            ...
        ]
    }

    The reason field always reads "vector-only reachable" for now.
    Collections do not exist yet, so that edge is vacuously absent.
    """
    try:
        cid = set_correlation_id()
        logger.debug("weft_fsck start [%s]", cid)
        app: AppContext = ctx.request_context.lifespan_context
        caller_uid = resolve_caller_user_id()

        async with acquire(app.pool):
            orphans = await list_orphan_memories(app.pool, user_id=caller_uid)

        return {
            "orphan_count": len(orphans),
            "orphans": orphans,
        }
    except _DB_ERRORS as e:
        return _db_error_response("weft_fsck", e)
