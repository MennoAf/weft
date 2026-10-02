"""Question-type-aware retrieval policy for the LongMemEval adapter.

The default ``search_hybrid(... limit=10)`` is fundamentally a "best-k matches"
operation. That works for single-session questions, but it systematically
under-recalls on:

  * ``multi-session`` counting questions ("how many weddings", "how many tanks")
    where the gold answer requires *enumerating* all relevant sessions, not
    just the top match.
  * ``temporal-reasoning`` questions with multiple time anchors ("how many
    days between X and Y") where each anchor lives in a different session and
    all must be retrieved.

This module centralizes the recall policy so the dispatch surface lives in
one place. Today the knobs exposed are ``top_k`` per question type and
``tier`` (belief / turns / auto). Later extension points (already factored
into ``RetrievalPolicy``):

  * Time-window filtering via the v45 ``episode_turns.list_turns_in_range``
    for temporal-reasoning questions.
  * Entity-anchored enumeration via ``weft_entity_context`` for
    "how many X" questions.
  * Multi-query expansion (one query per anchor candidate) with union/dedup.

Keep this module thin: it owns the policy table and the dispatch function,
nothing more. ``adapter.py`` calls ``retrieve()`` and gets ``MemoryRecall``s.

When ``tier='turns'``, the dispatcher calls ``recall_turns`` /
``temporal_anchor`` over ``episode_turns`` and shims the returned
``EpisodeTurn``s into ``MemoryRecall``-shaped objects so the Reader stays
unchanged. RRF rank inside the turn tier already orders results; we
synthesize a descending pseudo-similarity from rank order so the Reader's
relevance score has the same monotonic shape as the belief path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import asyncpg

from weft.db.connection import set_user_context_value
from weft.embeddings.base import EmbeddingProvider
from weft.episode_turns import recall_turns
from weft.models import (
    EpisodeTurn,
    Memory,
    MemoryRecall,
    MemorySource,
    MemoryStatus,
    MemoryType,
)
from weft.store import search_hybrid
from weft.turn_recall import (
    TemporalWindow,
    multi_session_query_variant,
    route_query_to_tier,
    temporal_anchor,
)
from weft.views.belief_query import BeliefClaimResult, search_belief_claims
from benchmarks.longmemeval.task_shape import TaskShape, derive_task_shape

logger = logging.getLogger(__name__)


Tier = Literal["belief", "turns", "auto", "belief-view", "replay", "production-belief"]

# Owner identity benchmark turns + claims are written under. Mirrors
# adapter.BENCHMARK_USER_ID; duplicated here to avoid a router→adapter import
# cycle (adapter imports router). The adapter passes its canonical constant
# into retrieve(), so this default only matters for direct/standalone calls.
_BENCHMARK_USER_ID = "longmemeval-bench"


def _resolve_turn_tier_expansion_slots(
    manifest_retrieval: dict | None,
    requested: int,
) -> int:
    """Reconcile the manifest-pinned turn-tier expansion depth with the run.

    Mirrors the dataset-checksum drift refusal: a manifest that pins
    ``retrieval.turn_tier_expansion_slots`` must match the runner invocation
    exactly; a manifest without the pin uses the requested value as-is.
    """
    pinned = None
    if manifest_retrieval:
        pinned = manifest_retrieval.get("turn_tier_expansion_slots")
    if pinned is None:
        return requested
    if isinstance(pinned, bool) or not isinstance(pinned, int) or pinned < 0:
        raise ValueError(
            "manifest turn_tier_expansion_slots must be a non-negative int, "
            f"got {pinned!r}"
        )
    if pinned != requested:
        raise ValueError(
            "turn_tier_expansion_slots drift: manifest pins "
            f"{pinned}, runner invoked with {requested}"
        )
    return pinned


# ----------------------------------------------------------------------
# Policy
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RetrievalPolicy:
    """Per-question-type retrieval knobs.

    ``top_k`` is the count fed to the Reader after over-fetch + project
    filtering. ``overfetch_multiplier`` controls how aggressively we widen
    the hybrid-search candidate pool before post-filtering to the sandbox
    project_id; benchmarks need strict project isolation, and over-fetch
    protects against losing relevant hits to global-scope leakage.
    """

    top_k: int
    overfetch_multiplier: int = 4
    candidate_sql_limit: int | None = None
    fusion_candidate_limit: int | None = None


@dataclass(slots=True)
class RetrievalDiagnostics:
    """Observational cause breakdown for one benchmark retrieval."""

    path: str = "unknown"
    initial_empty: bool = False
    retry_attempted: bool = False
    retry_rescued: bool = False
    fallback_attempted: bool = False
    fallback_rescued: bool = False
    final_empty: bool = False
    vector_gold_ranks: dict[str, int] = field(default_factory=dict)
    keyword_gold_ranks: dict[str, int] = field(default_factory=dict)
    vector_candidate_count: int = 0
    keyword_candidate_count: int = 0
    anchor_result_counts: dict[str, int] = field(default_factory=dict)
    anchor_candidate_ids: set[str] = field(default_factory=set)
    anchor_diagnostics: list[dict] = field(default_factory=list)
    query_variant_diagnostics: list[dict] = field(default_factory=list)
    indexed_turn_ids: list[str] = field(default_factory=list)
    gold_session_turn_ids: list[str] = field(default_factory=list)
    session_selector_applied: bool = False
    session_selector_reason: str | None = None
    session_selector_pool_count: int = 0
    session_selector_selected_session_ids: list[str] = field(default_factory=list)
    session_selector_selected_turn_ids: list[str] = field(default_factory=list)
    session_centroid_applied: bool = False
    session_centroid_reason: str | None = None
    session_centroid_pool_count: int = 0
    session_centroid_candidate_count: int = 0
    session_centroid_session_count: int = 0
    session_centroid_vector_dimensions: int = 0
    session_centroid_invalid_turn_ids: list[str] = field(default_factory=list)
    session_centroid_selected_session_ids: list[str] = field(default_factory=list)
    session_centroid_selected_turn_ids: list[str] = field(default_factory=list)

    def to_dict(
        self,
        *,
        gold_turn_ids: list[str],
        top_k: int,
        recall_hit: bool = False,
    ) -> dict:
        """Return stable JSON-safe diagnostics and aggregate classifications."""
        vector = set(self.vector_gold_ranks)
        keyword = set(self.keyword_gold_ranks)
        gold = set(gold_turn_ids)
        both_absent = sorted(gold - vector - keyword)
        vector_only = sorted(vector - keyword)
        keyword_only = sorted(keyword - vector)
        both_present_below_window = sorted(
            tid for tid in gold & vector & keyword
            if self.vector_gold_ranks[tid] > top_k
            and self.keyword_gold_ranks[tid] > top_k
        )
        anchor_gold_present = sorted(gold & self.anchor_candidate_ids)
        if recall_hit:
            cause = "hit"
        elif self.final_empty:
            cause = "empty_unrecovered"
        elif self.path == "temporal_anchor":
            cause = (
                "anchor_no_gold_candidate"
                if not anchor_gold_present
                else "anchor_partial_candidate_coverage"
            )
        elif both_absent:
            cause = "candidate_generation_gap"
        elif both_present_below_window:
            cause = "ranking_window_gap"
        else:
            cause = "candidate_fusion_or_ranking_gap"
        return {
            "cause": cause,
            "path": self.path,
            "initial_empty": self.initial_empty,
            "retry_attempted": self.retry_attempted,
            "retry_rescued": self.retry_rescued,
            "fallback_attempted": self.fallback_attempted,
            "fallback_rescued": self.fallback_rescued,
            "final_empty": self.final_empty,
            "vector_candidate_count": self.vector_candidate_count,
            "keyword_candidate_count": self.keyword_candidate_count,
            "anchor_count": len(self.anchor_result_counts),
            "anchor_result_counts": dict(self.anchor_result_counts),
            "anchor_gold_present_count": len(anchor_gold_present),
            "gold_vector_present_count": len(vector & gold),
            "gold_keyword_present_count": len(keyword & gold),
            "both_halves_absent_count": len(both_absent),
            "vector_only_count": len(vector_only),
            "keyword_only_count": len(keyword_only),
            "both_present_below_top_k_count": len(both_present_below_window),
            "anchor_gold_present": anchor_gold_present,
            "gold_vector_present": sorted(vector & gold),
            "gold_keyword_present": sorted(keyword & gold),
            "both_halves_absent": both_absent,
            "vector_only": vector_only,
            "keyword_only": keyword_only,
            "both_present_below_top_k": both_present_below_window,
            "anchor_diagnostics": list(self.anchor_diagnostics),
            "query_variant_diagnostics": list(self.query_variant_diagnostics),
            "session_selector_applied": self.session_selector_applied,
            "session_selector_reason": self.session_selector_reason,
            "session_selector_pool_count": self.session_selector_pool_count,
            "session_selector_selected_session_ids": list(
                self.session_selector_selected_session_ids
            ),
            "session_selector_selected_turn_ids": list(
                self.session_selector_selected_turn_ids
            ),
            "session_centroid_applied": self.session_centroid_applied,
            "session_centroid_reason": self.session_centroid_reason,
            "session_centroid_pool_count": self.session_centroid_pool_count,
            "session_centroid_candidate_count": self.session_centroid_candidate_count,
            "session_centroid_session_count": self.session_centroid_session_count,
            "session_centroid_vector_dimensions": self.session_centroid_vector_dimensions,
            "session_centroid_invalid_turn_ids": list(
                self.session_centroid_invalid_turn_ids
            ),
            "session_centroid_selected_session_ids": list(
                self.session_centroid_selected_session_ids
            ),
            "session_centroid_selected_turn_ids": list(
                self.session_centroid_selected_turn_ids
            ),
            "indexed_turn_ids": list(self.indexed_turn_ids),
            "indexed_turn_count": len(self.indexed_turn_ids),
            "gold_session_turn_ids": list(self.gold_session_turn_ids),
            "gold_session_turn_count": len(self.gold_session_turn_ids),
            "ground_truth_derivation": {
                "indexed": "manifest turn_ids (all materialized turns)",
                "gold_session": "turn_session_map values intersect gold_session_ids",
            },
        }


# Defaults are chosen from the failure analysis on the
# 20260430T204018Z extracted run:
#
#   multi-session    26.3% empty/uncertain  →  widen top_k
#   temporal         21.1%                  →  widen top_k (time-window
#                                              filter is a follow-up)
#   single-session-* 10–17%                 →  keep top_k=10 baseline
#   knowledge-update  7.7%                  →  keep top_k=10 baseline
_DEFAULT_POLICIES: dict[str, RetrievalPolicy] = {
    "multi-session": RetrievalPolicy(top_k=30),
    "temporal-reasoning": RetrievalPolicy(top_k=30),
    "single-session-user": RetrievalPolicy(top_k=10),
    "single-session-assistant": RetrievalPolicy(top_k=10),
    "single-session-preference": RetrievalPolicy(top_k=10),
    "knowledge-update": RetrievalPolicy(top_k=10),
}

_FALLBACK_POLICY = RetrievalPolicy(top_k=10)

# LongMemEval evidence supports session-aware reranking only for these flat
# question families. Keep the gate explicit: temporal, knowledge-update,
# assistant, abstention-only, and unknown types retain baseline ranking.
_SESSION_RERANK_TYPES = frozenset({
    "multi-session",
    "single-session-user",
    "single-session-preference",
})
SESSION_RERANK_POOL_LIMIT = 60


def session_rerank_enabled_for(question_type: str) -> bool:
    """Return whether the validated session-aware rerank policy applies.

    Abstention variants use the same retrieval shape as their base type. The
    caller still needs a complete turn-to-session map for the selector to do
    any work; this helper intentionally answers only the type-policy question.
    """
    return question_type.removesuffix("_abs") in _SESSION_RERANK_TYPES


def session_rerank_pool_limit_for(question_type: str) -> int | None:
    """Return the validated candidate pool for eligible question types."""
    return SESSION_RERANK_POOL_LIMIT if session_rerank_enabled_for(question_type) else None


def policy_for(question_type: str) -> RetrievalPolicy:
    """Look up the retrieval policy for a question type.

    The ``_abs`` suffix on abstention questions is stripped before lookup —
    the retrieval shape is identical, only the Reader's refusal-licensed
    system prompt changes.
    """
    base = question_type.removesuffix("_abs")
    return _DEFAULT_POLICIES.get(base, _FALLBACK_POLICY)


# ----------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------


def _turn_to_recall(
    turn: EpisodeTurn,
    *,
    rank: int,
    total: int,
    project_id: str,
) -> MemoryRecall:
    """Wrap an ``EpisodeTurn`` in a ``MemoryRecall`` for the Reader.

    The Reader formats memories as ``[i] (relevance=X.XX) <content>``. We
    synthesize a descending pseudo-similarity from rank order so the
    relevance display stays monotonic without inventing fake scores. The
    occurred_at timestamp is rendered into the content as a date prefix so
    temporal-reasoning Reads don't lose anchor information.
    """
    similarity = max(0.0, 1.0 - (rank / max(total, 1)))
    role_label = turn.role.value if hasattr(turn.role, "value") else str(turn.role)
    date_str = turn.occurred_at.date().isoformat()
    content = f"Session date: {date_str}\n{role_label}: {turn.content}"
    memory = Memory(
        id=turn.id,
        type=MemoryType.fact,
        topic=[f"longmemeval/{project_id}"],
        content=content,
        source=MemorySource.conversation,
        confidence=1.0,
        token_count=turn.token_count,
        created_at=turn.created_at,
        updated_at=turn.created_at,
        accessed_at=turn.created_at,
        project_id=project_id,
        status=MemoryStatus.active,
    )
    return MemoryRecall(memory=memory, similarity=similarity)


def _render_claim_value(value: object) -> str:
    """Render a claim's JSONB value as compact prose for the Reader.

    Dicts become ``k=v`` pairs; scalars pass through. Mirrors the default
    rendering in ``weft.views.belief_query._render_value`` so the Reader sees
    the same shape whether the claim arrives via the MCP tool or this harness.
    """
    if isinstance(value, dict):
        return ", ".join(f"{k}={v}" for k, v in value.items())
    return str(value)


def _claim_to_recall(
    claim: BeliefClaimResult,
    *,
    rank: int,
    total: int,
    project_id: str,
) -> MemoryRecall:
    """Wrap a ``BeliefClaimResult`` in a ``MemoryRecall`` for the Reader.

    The belief-view returns supersession-collapsed *current* claims, so the
    occurred_at date is rendered into the content as the as-of anchor — this
    is exactly the signal that resolves knowledge-update questions (the Reader
    sees only the active value, not the superseded one). Pseudo-similarity is
    synthesized from rank order, matching ``_turn_to_recall`` so the relevance
    display stays monotonic across tiers.
    """
    similarity = max(0.0, 1.0 - (rank / max(total, 1)))
    date_str = claim.occurred_at.date().isoformat()
    content = (
        f"As of {date_str} — {claim.attribute}: {_render_claim_value(claim.value)} "
        f"(source: {claim.source_provenance})"
    )
    memory = Memory(
        id=claim.claim_id,
        type=MemoryType.fact,
        topic=[f"longmemeval/{project_id}"],
        content=content,
        source=MemorySource.conversation,
        confidence=claim.detector_confidence,
        token_count=0,
        created_at=claim.occurred_at,
        updated_at=claim.occurred_at,
        accessed_at=claim.occurred_at,
        project_id=project_id,
        status=MemoryStatus.active,
    )
    return MemoryRecall(memory=memory, similarity=similarity)


def _make_centroid_callback(
    *,
    diagnostics: RetrievalDiagnostics | None,
    query_embedding: Sequence[float],
    top_k: int,
    representation_pool_limit: int | None,
    max_sessions: int | None,
    turn_session_map: Mapping[str, str] | None,
    session_centroid_selector: Callable[..., object] | None,
) -> Callable[[list[dict[str, object]]], None]:
    """Build the benchmark-only callback that selects centroid-ranked turns.

    The callback runs after ``recall_turns`` has produced its final ranked
    overfetch pool. It stores only selected IDs in diagnostics; the router
    applies those IDs to the returned ``EpisodeTurn`` objects afterward. This
    keeps the callback observational and preserves the production model shape.
    """
    if representation_pool_limit is None or not turn_session_map:
        raise ValueError("centroid callback requires pool limit and session mapping")
    if session_centroid_selector is None:
        raise ValueError("centroid callback requires a selector")

    from benchmarks.longmemeval.ab_compare import RepresentationTurn

    def callback(raw_rows: list[dict[str, object]]) -> None:
        bounded_rows = raw_rows[:representation_pool_limit]
        records: list[RepresentationTurn] = []
        for row in bounded_rows:
            turn_id = row.get("turn_id")
            rank = row.get("rank")
            if turn_id is None or not isinstance(rank, int):
                if diagnostics is not None:
                    diagnostics.session_centroid_reason = "invalid_raw_row"
                    diagnostics.session_centroid_pool_count = len(raw_rows)
                return
            turn_id = str(turn_id)
            session_id = turn_session_map.get(turn_id)
            # A partial identity map would make the treatment's session scores
            # incomparable with the baseline. Reject the whole opt-in arm,
            # rather than silently selecting from a biased subset.
            if not isinstance(session_id, str) or not session_id:
                if diagnostics is not None:
                    diagnostics.session_centroid_reason = "incomplete_session_mapping"
                    diagnostics.session_centroid_pool_count = len(raw_rows)
                return
            vector = row.get("vector")
            if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
                vector = ()
            records.append(RepresentationTurn(turn_id, session_id, tuple(vector), rank))
        try:
            selection = session_centroid_selector(
                records,
                query_vector=query_embedding,
                top_k=top_k,
                max_sessions=max_sessions,
            )
        except Exception as exc:
            # raw-row callback failures are swallowed by recall_turns; record a
            # stable reason here so the report distinguishes a rejected
            # treatment from an ordinary no-candidate result.
            if diagnostics is not None:
                diagnostics.session_centroid_reason = (
                    f"selector_error:{type(exc).__name__}"
                )
                diagnostics.session_centroid_pool_count = len(raw_rows)
            return
        if diagnostics is not None:
            diagnostics.session_centroid_applied = True
            diagnostics.session_centroid_reason = getattr(selection, "reason", "selected")
            diagnostics.session_centroid_pool_count = len(raw_rows)
            diagnostics.session_centroid_candidate_count = getattr(
                selection, "candidate_count", len(records),
            )
            diagnostics.session_centroid_session_count = getattr(
                selection, "session_count", 0,
            )
            diagnostics.session_centroid_vector_dimensions = getattr(
                selection, "vector_dimensions", 0,
            )
            diagnostics.session_centroid_invalid_turn_ids = list(
                getattr(selection, "invalid_turn_ids", ()),
            )
            diagnostics.session_centroid_selected_session_ids = list(
                getattr(selection, "selected_session_ids", ()),
            )
            diagnostics.session_centroid_selected_turn_ids = list(
                getattr(selection, "selected_turn_ids", ()),
            )

    return callback


def _apply_centroid_selection(
    turns: list[EpisodeTurn],
    *,
    top_k: int,
    diagnostics: RetrievalDiagnostics | None,
) -> list[EpisodeTurn]:
    """Apply callback-selected IDs and always restore the Reader cap.

    The raw-row callback is observational and deliberately fail-closed: if a
    selector rejects malformed vectors or the callback is unavailable, the
    baseline-ranked pool remains the source of truth.  It must still be
    truncated to ``top_k`` here; otherwise an overfetch pool could leak into
    the Reader-facing result after an opt-in treatment fails.
    """
    if diagnostics is None or not diagnostics.session_centroid_applied:
        return turns[:top_k]
    selected_ids = diagnostics.session_centroid_selected_turn_ids
    by_id = {turn.id: turn for turn in turns}
    selected = [by_id[turn_id] for turn_id in selected_ids if turn_id in by_id]
    if not selected:
        if diagnostics.session_centroid_reason is None:
            diagnostics.session_centroid_reason = "no_selected_rows"
        return turns[:top_k]
    return selected[:top_k]


def _select_session_diverse_turns(
    turns: list[EpisodeTurn],
    *,
    top_k: int,
    turn_session_map: Mapping[str, str] | None,
    diagnostics: RetrievalDiagnostics | None = None,
) -> list[EpisodeTurn]:
    """Select a deterministic, session-diverse prefix from a ranked pool.

    The benchmark materializes one Weft episode for an entire LongMemEval
    question, so ``EpisodeTurn.episode_id`` is *not* the source-session key.
    The benchmark manifest's ``turn_id -> source_session_id`` mapping must be
    supplied explicitly; without it this opt-in treatment is a safe no-op.

    Selection is intentionally conservative: take the best-ranked turn from
    each source session first, then fill any remaining slots in the original
    ranked order.  Thus a baseline hit remains eligible while concentrated
    turns from one session cannot consume the whole Reader cap when the raw
    candidate pool contains other sessions.
    """
    pool = list(turns)
    if diagnostics is not None:
        diagnostics.session_selector_pool_count = len(pool)

    if top_k <= 0:
        if diagnostics is not None:
            diagnostics.session_selector_reason = "non_positive_top_k"
        return []
    if not pool or len(pool) <= top_k:
        if diagnostics is not None:
            diagnostics.session_selector_applied = bool(pool)
            diagnostics.session_selector_reason = "pool_within_cap"
            diagnostics.session_selector_selected_turn_ids = [
                turn.id for turn in pool[:top_k]
            ]
            diagnostics.session_selector_selected_session_ids = [
                turn_session_map[turn.id]
                for turn in pool[:top_k]
                if turn_session_map is not None and turn.id in turn_session_map
            ]
        return pool[:top_k]
    if not turn_session_map:
        if diagnostics is not None:
            diagnostics.session_selector_reason = "missing_session_mapping"
        return pool[:top_k]

    missing = [turn.id for turn in pool if turn.id not in turn_session_map]
    if missing:
        if diagnostics is not None:
            diagnostics.session_selector_reason = "incomplete_session_mapping"
        return pool[:top_k]

    selected: list[EpisodeTurn] = []
    selected_ids: set[str] = set()
    selected_sessions: list[str] = []
    seen_sessions: set[str] = set()

    # First pass: one best-ranked turn per source session.
    for turn in pool:
        session_id = turn_session_map[turn.id]
        if session_id in seen_sessions:
            continue
        seen_sessions.add(session_id)
        selected.append(turn)
        selected_ids.add(turn.id)
        selected_sessions.append(session_id)
        if len(selected) >= top_k:
            break

    # Second pass: restore the strongest remaining turns if there are fewer
    # sessions than the Reader cap.
    if len(selected) < top_k:
        for turn in pool:
            if turn.id in selected_ids:
                continue
            selected.append(turn)
            selected_ids.add(turn.id)
            if len(selected) >= top_k:
                break

    if diagnostics is not None:
        diagnostics.session_selector_applied = True
        diagnostics.session_selector_reason = "diversified"
        diagnostics.session_selector_selected_session_ids = selected_sessions
        diagnostics.session_selector_selected_turn_ids = [
            turn.id for turn in selected
        ]
    return selected[:top_k]


async def _retrieve_turns(
    pool: asyncpg.Pool,
    embedder: EmbeddingProvider,
    *,
    question: str,
    question_type: str,
    project_id: str,
    policy: RetrievalPolicy,
    expected_turn_count: int = 0,
    as_of: datetime | None = None,
    gold_turn_ids: list[str] | None = None,
    diagnostics: RetrievalDiagnostics | None = None,
    candidate_sql_limit: int | None = None,
    fusion_candidate_limit: int | None = None,
    use_anchor_local_variant: bool = True,
    include_embedded_temporal_variant: bool = False,
    temporal_window: TemporalWindow | None = None,
    use_multi_session_query_variant: bool = False,
    use_session_selector: bool = False,
    turn_session_map: Mapping[str, str] | None = None,
    session_selector_pool_limit: int | None = None,
    use_session_centroid_representation: bool = False,
    representation_pool_limit: int | None = None,
    representation_max_sessions: int | None = None,
    session_centroid_selector: Callable[..., object] | None = None,
    turn_tier_expansion_slots: int = 0,
) -> list[MemoryRecall]:
    """Turn-tier retrieval path. Returns Reader-compatible MemoryRecall list.

    Splits behavior by question type:
      * temporal-reasoning (and any temporal marker hit) → ``temporal_anchor``
        so multi-anchor queries get per-anchor turn pulls.
      * everything else → single ``recall_turns`` over ``episode_turns``.

    Uses a transaction-scoped connection with explicit ``SET LOCAL
    app.user_id`` so RLS policies see the benchmark identity regardless
    of pool state. This eliminates the 0-return-under-sustained-load
    bug where ``app.user_id`` was lost on a pooled connection.

    When ``expected_turn_count > 0``, the 0-return path triggers
    diagnostics on the SAME connection that produced the empty result.

    ``as_of`` is an optional fixed timestamp for deterministic recency
    reranking in A/B comparisons. When None (production), wall-clock
    time is used.
    """
    # A policy may carry benchmark-only candidate controls when callers do not
    # pass the individual overrides explicitly.  Explicit arguments win, while
    # ``None`` retains the production default in ``recall_turns``.
    if candidate_sql_limit is None:
        candidate_sql_limit = policy.candidate_sql_limit
    if fusion_candidate_limit is None:
        fusion_candidate_limit = policy.fusion_candidate_limit
    if use_session_selector:
        if session_selector_pool_limit is None:
            raise ValueError(
                "session_selector_pool_limit is required when session selector is enabled"
            )
        if session_selector_pool_limit < policy.top_k:
            raise ValueError(
                "session_selector_pool_limit must be at least policy.top_k"
            )
    if use_session_centroid_representation:
        if representation_pool_limit is None:
            raise ValueError(
                "representation_pool_limit is required when session centroid representation is enabled"
            )
        if representation_pool_limit < policy.top_k:
            raise ValueError(
                "representation_pool_limit must be at least policy.top_k"
            )
        if not turn_session_map:
            raise ValueError(
                "turn_session_map is required when session centroid representation is enabled"
            )
        if session_centroid_selector is None:
            raise ValueError(
                "session_centroid_selector is required when session centroid representation is enabled"
            )

    base_type = question_type.removesuffix("_abs")
    use_anchor = base_type == "temporal-reasoning" or (
        route_query_to_tier(question) == "turns"
        and base_type != "multi-session"
    )

    if use_anchor:
        if diagnostics is not None:
            diagnostics.path = "temporal_anchor"
        # Anchor probes must share the same transaction-scoped identity and
        # connection as the flat path. asyncpg connections are sequential, so
        # temporal_anchor deliberately runs probes one at a time.
        def _anchor_diag(anchor: str, vector_rows: list, keyword_rows: list,
                         final_turns: list[EpisodeTurn]) -> None:
            if diagnostics is None:
                return
            gold = set(gold_turn_ids or [])
            vector_ids = [row["id"] for row in vector_rows]
            keyword_ids = [row["id"] for row in keyword_rows]
            final_ids = [turn.id for turn in final_turns]
            diagnostics.anchor_diagnostics.append({
                "anchor": anchor,
                "vector_candidate_count": len(vector_rows),
                "keyword_candidate_count": len(keyword_rows),
                "vector_gold_present": sorted(gold.intersection(vector_ids)),
                "keyword_gold_present": sorted(gold.intersection(keyword_ids)),
                "final_union_ids": final_ids,
            })
            diagnostics.vector_candidate_count += len(vector_rows)
            diagnostics.keyword_candidate_count += len(keyword_rows)
            # Candidate coverage is observational over both raw halves; do
            # not let the per-anchor output cap hide a gold candidate.
            diagnostics.anchor_candidate_ids.update(vector_ids)
            diagnostics.anchor_candidate_ids.update(keyword_ids)

        async with pool.acquire() as conn:
            async with conn.transaction():
                await set_user_context_value(conn, _BENCHMARK_USER_ID)
                anchored = await temporal_anchor(
                    pool, question,
                    project_id=project_id,
                    top_k_per_anchor=max(1, policy.top_k // 2),
                    embedder=embedder,
                    executor=conn,
                    as_of=as_of,
                    diag_callback=_anchor_diag,
                    candidate_sql_limit=policy.candidate_sql_limit,
                    fusion_candidate_limit=policy.fusion_candidate_limit,
                    use_anchor_local_variant=use_anchor_local_variant,
                    include_embedded_temporal_variant=include_embedded_temporal_variant,
                    expansion_slots=turn_tier_expansion_slots,
                )
                if temporal_window is not None:
                    # Keep the normal temporal retrieval as the primary list.
                    # The date-bounded probe is strictly additive and runs on
                    # the same scoped connection/transaction.
                    windowed = await temporal_anchor(
                        pool, question,
                        project_id=project_id,
                        top_k_per_anchor=max(1, policy.top_k // 2),
                        embedder=embedder,
                        since=temporal_window.since,
                        until=temporal_window.until,
                        executor=conn,
                        as_of=as_of,
                        diag_callback=_anchor_diag,
                        candidate_sql_limit=policy.candidate_sql_limit,
                        fusion_candidate_limit=policy.fusion_candidate_limit,
                        use_anchor_local_variant=use_anchor_local_variant,
                        include_embedded_temporal_variant=include_embedded_temporal_variant,
                        expansion_slots=turn_tier_expansion_slots,
                    )
                    for anchor, window_turns in windowed.items():
                        baseline_turns = anchored.setdefault(anchor, [])
                        seen_window = {turn.id for turn in baseline_turns}
                        baseline_turns.extend(
                            turn for turn in window_turns
                            if turn.id not in seen_window
                        )
                seen: set[str] = set()
                ordered: list[EpisodeTurn] = []
                for turns in anchored.values():
                    for t in turns:
                        if t.id in seen:
                            continue
                        seen.add(t.id)
                        ordered.append(t)
                        if len(ordered) >= policy.top_k:
                            break
                    if len(ordered) >= policy.top_k:
                        break
                turns_list = ordered[: policy.top_k]
        if diagnostics is not None:
            diagnostics.anchor_result_counts = {
                anchor: len(turns) for anchor, turns in anchored.items()
            }
            diagnostics.anchor_candidate_ids.update(
                turn.id for turns in anchored.values() for turn in turns
            )
            diagnostics.final_empty = not turns_list
            if diagnostics.final_empty and anchored:
                diagnostics.fallback_attempted = True
    else:
        if diagnostics is not None:
            diagnostics.path = "flat"
        # Non-anchor path: acquire one connection, set identity explicitly
        # via SET LOCAL, run recall through it, and diagnose on 0-return.
        # The multi-session treatment is deliberately isolated here: temporal
        # and single-session questions retain the one-query baseline path.
        query_variant = None
        if use_multi_session_query_variant and base_type == "multi-session":
            query_variant = multi_session_query_variant(question)
        turns_list = await _recall_turns_scoped(
            pool, question,
            project_id=project_id,
            top_k=policy.top_k,
            embedder=embedder,
            expected_turn_count=expected_turn_count,
            as_of=as_of,
            gold_turn_ids=gold_turn_ids,
            diagnostics=diagnostics,
            candidate_sql_limit=policy.candidate_sql_limit,
            fusion_candidate_limit=policy.fusion_candidate_limit,
            query_variant=query_variant,
            result_limit=(
                representation_pool_limit
                if use_session_centroid_representation
                else session_selector_pool_limit
                if use_session_selector
                else None
            ),
            use_session_centroid_representation=use_session_centroid_representation,
            representation_pool_limit=representation_pool_limit,
            representation_max_sessions=representation_max_sessions,
            turn_session_map=turn_session_map,
            session_centroid_selector=session_centroid_selector,
            expansion_slots=turn_tier_expansion_slots,
        )
        if use_session_centroid_representation:
            turns_list = _apply_centroid_selection(
                turns_list,
                top_k=policy.top_k,
                diagnostics=diagnostics,
            )
        elif use_session_selector:
            turns_list = _select_session_diverse_turns(
                turns_list,
                top_k=policy.top_k,
                turn_session_map=turn_session_map,
                diagnostics=diagnostics,
            )

    total = len(turns_list)
    return [
        _turn_to_recall(t, rank=i, total=total, project_id=project_id)
        for i, t in enumerate(turns_list)
    ]


async def _recall_turns_scoped(
    pool: asyncpg.Pool,
    query: str,
    *,
    project_id: str,
    top_k: int,
    embedder: EmbeddingProvider,
    expected_turn_count: int = 0,
    as_of: datetime | None = None,
    gold_turn_ids: list[str] | None = None,
    diagnostics: RetrievalDiagnostics | None = None,
    candidate_sql_limit: int | None = None,
    fusion_candidate_limit: int | None = None,
    query_variant: str | None = None,
    result_limit: int | None = None,
    raw_row_callback: Callable[[list[dict[str, object]]], None] | None = None,
    use_session_centroid_representation: bool = False,
    representation_pool_limit: int | None = None,
    representation_max_sessions: int | None = None,
    turn_session_map: Mapping[str, str] | None = None,
    session_centroid_selector: Callable[..., object] | None = None,
    expansion_slots: int = 0,
) -> list[EpisodeTurn]:
    """Run one or two flat probes on one explicitly scoped connection.

    The primary query keeps the existing retry and same-connection diagnostic
    behavior. ``query_variant`` is an optional additive probe used only by
    the benchmark treatment: it runs after the primary query in the same
    transaction, then its turns are appended by first-seen ID and capped at
    the existing ``top_k``. The primary result and diagnostics therefore
    remain the authoritative first arm.
    """
    # Per-half diagnostic: capture where gold turns ranked in each half.
    # The callback fires inside recall_turns after both halves complete
    # but before RRF fusion.
    half_diag: dict = {"vector_ranks": {}, "keyword_ranks": {}}

    def _diag_cb(vec_rows: list, kw_rows: list) -> None:
        if not gold_turn_ids:
            return
        gold_set = set(gold_turn_ids)
        half_diag["vector_ranks"] = {
            r["id"]: i + 1 for i, r in enumerate(vec_rows) if r["id"] in gold_set
        }
        half_diag["keyword_ranks"] = {
            r["id"]: i + 1 for i, r in enumerate(kw_rows) if r["id"] in gold_set
        }
        half_diag["vector_candidate_ids"] = [r["id"] for r in vec_rows]
        half_diag["keyword_candidate_ids"] = [r["id"] for r in kw_rows]
        if diagnostics is not None:
            diagnostics.vector_gold_ranks = dict(half_diag["vector_ranks"])
            diagnostics.keyword_gold_ranks = dict(half_diag["keyword_ranks"])
            diagnostics.vector_candidate_count = len(vec_rows)
            diagnostics.keyword_candidate_count = len(kw_rows)

    async with pool.acquire() as conn:
        async with conn.transaction():
            # Set identity explicitly — this is the transaction-scoped
            # SET LOCAL that production (acquire()) uses. The setup
            # callback may or may not have set it; SET LOCAL here
            # guarantees it for this transaction.
            await set_user_context_value(conn, _BENCHMARK_USER_ID)

            query_embedding = await embedder.embed(query)
            centroid_callback = None
            if use_session_centroid_representation:
                centroid_callback = _make_centroid_callback(
                    diagnostics=diagnostics,
                    query_embedding=query_embedding,
                    top_k=top_k,
                    representation_pool_limit=representation_pool_limit,
                    max_sessions=representation_max_sessions,
                    turn_session_map=turn_session_map,
                    session_centroid_selector=session_centroid_selector,
                )
            turns_list = await recall_turns(
                pool, query,  # pool arg is unused when executor is set
                project_id=project_id,
                top_k=top_k,
                embedding=query_embedding,
                executor=conn,
                as_of=as_of,
                diag_callback=_diag_cb,
                raw_row_callback=centroid_callback or raw_row_callback,
                candidate_sql_limit=candidate_sql_limit,
                fusion_candidate_limit=fusion_candidate_limit,
                result_limit=result_limit,
                expansion_slots=expansion_slots,
            )

            if not turns_list:
                if diagnostics is not None:
                    diagnostics.initial_empty = True
                    diagnostics.retry_attempted = True
                # Diagnose on the SAME connection before releasing it.
                # This observes the exact GUC/RLS state that produced
                # the 0-return.
                logger.warning(
                    "recall_turns returned 0 turns for q (project_id=%s, "
                    "top_k=%d, expected_turn_count=%d) — retrying on same connection",
                    project_id, top_k, expected_turn_count,
                )

                # Retry with the same embedding on the SAME connection.
                # The first query already computed it above; reusing it keeps
                # the retry deterministic and avoids an undefined external
                # ``embedding`` variable.
                turns_list = await recall_turns(
                    pool, query,
                    project_id=project_id,
                    top_k=top_k,
                    embedding=query_embedding,
                    executor=conn,
                    as_of=as_of,
                    diag_callback=_diag_cb,
                    raw_row_callback=centroid_callback or raw_row_callback,
                    candidate_sql_limit=candidate_sql_limit,
                    fusion_candidate_limit=fusion_candidate_limit,
                    result_limit=result_limit,
                    expansion_slots=expansion_slots,
                )
                if turns_list and diagnostics is not None:
                    diagnostics.retry_rescued = True
                if not turns_list:
                    logger.warning(
                        "recall_turns still 0 after retry (project_id=%s) — "
                        "falling through to belief fallback",
                        project_id,
                    )

            if query_variant is not None:
                variant_diag: dict = {
                    "query": query_variant,
                    "vector_gold_ranks": {},
                    "keyword_gold_ranks": {},
                    "vector_candidate_count": 0,
                    "keyword_candidate_count": 0,
                }

                def _variant_diag(vec_rows: list, kw_rows: list) -> None:
                    gold = set(gold_turn_ids or [])
                    variant_diag["vector_gold_ranks"] = {
                        row["id"]: i + 1
                        for i, row in enumerate(vec_rows)
                        if row["id"] in gold
                    }
                    variant_diag["keyword_gold_ranks"] = {
                        row["id"]: i + 1
                        for i, row in enumerate(kw_rows)
                        if row["id"] in gold
                    }
                    variant_diag["vector_candidate_count"] = len(vec_rows)
                    variant_diag["keyword_candidate_count"] = len(kw_rows)

                variant_turns: list[EpisodeTurn] = []
                try:
                    variant_embedding = await embedder.embed(query_variant)
                    variant_turns = await recall_turns(
                        pool, query_variant,
                        project_id=project_id,
                        top_k=top_k,
                        embedding=variant_embedding,
                        executor=conn,
                        as_of=as_of,
                        diag_callback=_variant_diag,
                        candidate_sql_limit=candidate_sql_limit,
                        fusion_candidate_limit=fusion_candidate_limit,
                        expansion_slots=expansion_slots,
                    )
                except Exception as exc:
                    # The optional arm must never turn a valid baseline into
                    # a failed retrieval. Keep the failure visible separately.
                    variant_diag["error"] = type(exc).__name__
                    logger.warning(
                        "multi-session query variant failed (project_id=%s): %s",
                        project_id, exc,
                    )

                seen = {turn.id for turn in turns_list}
                added_turns: list[EpisodeTurn] = []
                for turn in variant_turns:
                    if turn.id in seen:
                        continue
                    seen.add(turn.id)
                    added_turns.append(turn)
                turns_list.extend(added_turns)
                turns_list = turns_list[:top_k]
                variant_diag["retrieved_turn_ids"] = [
                    turn.id for turn in variant_turns
                ]
                variant_diag["added_turn_ids"] = [
                    turn.id for turn in added_turns[:top_k]
                ]
                if diagnostics is not None:
                    diagnostics.query_variant_diagnostics.append(variant_diag)

    if diagnostics is not None:
        diagnostics.final_empty = not turns_list

    # Per-half miss diagnostic: if gold turns were provided and any were
    # absent from a half's candidates, log which half missed them and at
    # what rank the present ones were.
    if gold_turn_ids and half_diag.get("vector_ranks") is not None:
        _log_per_half_miss(
            project_id, gold_turn_ids, half_diag, top_k,
        )

    return turns_list


def _log_per_half_miss(
    project_id: str,
    gold_turn_ids: list[str],
    half_diag: dict,
    top_k: int,
) -> None:
    """Log where gold turns ranked in each half (vector vs. keyword).

    Classifies each gold turn as:
      - PRESENT in a half: at rank N (1-indexed) within the candidate_limit
      - ABSENT from a half: not in the candidate pool at all

    This is the key signal for Phase 4 fusion weight tuning: if gold turns
    are present in the vector half but absent from keyword (or vice versa),
    the failing half needs improvement, not the fusion weights.
    """
    gold_set = set(gold_turn_ids)
    vec_ranks: dict = half_diag.get("vector_ranks", {})
    kw_ranks: dict = half_diag.get("keyword_ranks", {})
    vec_cand_ids: set = set(half_diag.get("vector_candidate_ids", []))
    kw_cand_ids: set = set(half_diag.get("keyword_candidate_ids", []))

    for tid in gold_turn_ids:
        v_rank = vec_ranks.get(tid)
        k_rank = kw_ranks.get(tid)
        v_status = f"rank={v_rank}" if v_rank else ("absent" if tid not in vec_cand_ids else "no-embedding")
        k_status = f"rank={k_rank}" if k_rank else ("absent" if tid not in kw_cand_ids else "no-match")

        # Only log if the gold turn was not in the top_k of BOTH halves
        # (it's in the fused result and likely a hit — no need to log)
        if v_rank and v_rank <= top_k and k_rank and k_rank <= top_k:
            continue

        logger.info(
            "per-half-miss: project=%s gold_turn=%s vector=[%s] keyword=[%s] "
            "top_k=%d — %s",
            project_id, tid, v_status, k_status, top_k,
            "both-halves-missed" if not v_rank and not k_rank
            else "vector-only" if v_rank and not k_rank
            else "keyword-only" if k_rank and not v_rank
            else "both-present-below-top_k",
        )


async def _retrieve_belief(
    pool: asyncpg.Pool,
    embedder: EmbeddingProvider,
    *,
    question: str,
    project_id: str,
    policy: RetrievalPolicy,
    user_id: str = _BENCHMARK_USER_ID,
) -> list[MemoryRecall]:
    """Belief-tier hybrid recall over ``memories``, project-isolated.

    Over-fetches then post-filters to the exact project_id because
    ``search_hybrid`` treats ``project_id=None`` rows as global and lets them
    leak alongside scoped hits.
    """
    query_embedding = await embedder.embed(question)
    raw = await search_hybrid(
        pool,
        question,
        query_embedding,
        limit=policy.top_k * policy.overfetch_multiplier,
        project_id=project_id,
        user_id=user_id,
    )
    memories = [r for r in raw if r.memory.project_id == project_id]
    return memories[: policy.top_k]


async def retrieve(
    pool: asyncpg.Pool,
    embedder: EmbeddingProvider,
    *,
    question: str,
    project_id: str,
    question_type: str | None = None,
    task_shape: TaskShape | None = None,
    policy: RetrievalPolicy | None = None,
    tier: Tier = "belief",
    user_id: str = _BENCHMARK_USER_ID,
    expected_turn_count: int = 0,
    as_of: datetime | None = None,
    gold_turn_ids: list[str] | None = None,
    diagnostics: RetrievalDiagnostics | None = None,
    candidate_sql_limit: int | None = None,
    fusion_candidate_limit: int | None = None,
    use_anchor_local_variant: bool = True,
    include_embedded_temporal_variant: bool = False,
    temporal_window: TemporalWindow | None = None,
    use_multi_session_query_variant: bool = False,
    use_session_selector: bool | None = None,
    turn_session_map: Mapping[str, str] | None = None,
    session_selector_pool_limit: int | None = None,
    use_session_centroid_representation: bool = False,
    representation_pool_limit: int | None = None,
    representation_max_sessions: int | None = None,
    session_centroid_selector: Callable[..., object] | None = None,
    turn_tier_expansion_slots: int = 0,
) -> list[MemoryRecall]:
    """Run the question-type-appropriate retrieval and return ranked memories.

    The benchmark sandbox semantics require strict project_id isolation:
    Weft's ``search_hybrid`` treats ``project_id=None`` memories as global
    and lets them leak alongside scoped hits, so we over-fetch and
    post-filter to the exact project_id.

    ``tier``:
      * ``'belief'`` (default) — current behavior, hybrid recall over
        ``memories``. Used when the haystack was ingested via raw or
        extracted modes.
      * ``'turns'`` — query ``episode_turns`` directly. Use when the
        haystack was ingested via mode='turns'. Multi-anchor temporal
        queries get per-anchor splitting via ``temporal_anchor``.
      * ``'auto'`` — for multi-session and temporal-reasoning questions
        use turns; everything else stays on belief. Caller is responsible
        for ingesting both shapes if 'auto' is requested.
      * ``'belief-view'`` — query the materialized ``belief_claims`` view
        (supersession-collapsed current claims) and fall back to turn-tier
        recall when no claim matches. Use with ``--mode turns`` + a prior
        ``materialize_question`` pass. ``belief_claims`` has no ``project_id``
        column, so sandbox isolation here relies on per-question cleanup of
        claims plus the ``user_id`` filter — safe under the default
        sequential, cleanup-on path.
      * ``'replay'`` — identical READ to ``'belief-view'``; the adapter
        additionally drives the recall-gap replay loop (enqueue + drain) after
        materialization, so the claim view also holds the multi-turn aggregate
        (``replay-`` stamped) claims the single-turn detector cannot produce.
        Same sandbox-isolation caveats as ``belief-view``.

    ``user_id`` scopes the belief-view claim lookup (no effect on other tiers,
    which isolate by ``project_id``).

    ``expected_turn_count`` is passed to the turn-tier diagnostic path —
    when set, a 0-return triggers connection-state diagnostics that compare
    RLS-visible count against this ground-truth.

    ``as_of`` is an optional fixed timestamp for deterministic recency
    reranking in A/B comparisons. When None (production), wall-clock time.

    ``use_session_selector`` is benchmark-only and applies only to the flat
    path. It requires the manifest's source-session mapping and overfetches
    ``session_selector_pool_limit`` fused turns before restoring ``top_k``.
    """
    explicit_task_shape = task_shape is not None
    task_shape = task_shape or derive_task_shape(question)
    runtime_shape = task_shape.task_shape
    shape_question_type = {
        "temporal": "temporal-reasoning",
        "temporal-multi": "temporal-reasoning",
        "multi-session": "multi-session",
        "single-session": "single-session-user",
    }.get(runtime_shape, "single-session-user")
    if explicit_task_shape:
        # The adapter always supplies task_shape. From this boundary onward,
        # runtime policy is shape/content-derived and question_type is ignored.
        question_type = shape_question_type
    else:
        # Compatibility for direct standalone integrations that still call
        # retrieve(question_type=...). This branch is intentionally unreachable
        # from the adapter, which passes task_shape on every runtime path.
        question_type = question_type or shape_question_type
    if policy is None:
        policy = (
            RetrievalPolicy(top_k=task_shape.top_k, overfetch_multiplier=4)
            if explicit_task_shape
            else policy_for(question_type)
        )
    if candidate_sql_limit is None:
        candidate_sql_limit = policy.candidate_sql_limit
    if fusion_candidate_limit is None:
        fusion_candidate_limit = policy.fusion_candidate_limit

    # The selector is shape-derived for explicit runtime calls and retains the
    # legacy question_type gate only for no-shape standalone callers. In both
    # cases a source-session map is required for a meaningful selection.
    if use_session_selector is None:
        if explicit_task_shape:
            use_session_selector = bool(
                turn_session_map and runtime_shape in {"multi-session", "single-session"}
            )
        else:
            use_session_selector = bool(
                turn_session_map and session_rerank_enabled_for(question_type)
            )
    elif not explicit_task_shape and not session_rerank_enabled_for(question_type):
        use_session_selector = False
    if use_session_selector and session_selector_pool_limit is None:
        session_selector_pool_limit = (
            SESSION_RERANK_POOL_LIMIT
            if not explicit_task_shape
            else max(policy.top_k, SESSION_RERANK_POOL_LIMIT)
        )

    if tier == "auto":
        if explicit_task_shape:
            tier = task_shape.routing_class
        else:
            tier = (
                "turns"
                if question_type.removesuffix("_abs")
                in ("multi-session", "temporal-reasoning")
                else "belief"
            )

    if tier == "production-belief":
        # Corrected pilot arm: active belief claims first, then the same
        # project-scoped hybrid-memory fallback as the public belief path.
        # Claims have no project column, so this path intentionally uses only
        # the benchmark's isolated global user partition and cleans claims per
        # question before the next question starts.
        claims = await search_belief_claims(
            pool,
            query=question,
            user_id=user_id,
            scope="global",
            limit=policy.top_k,
        )
        if claims:
            total = len(claims)
            return [
                _claim_to_recall(c, rank=i, total=total, project_id=project_id)
                for i, c in enumerate(claims)
            ]
        return await _retrieve_belief(
            pool,
            embedder,
            question=question,
            project_id=project_id,
            policy=policy,
            user_id=user_id,
        )

    if tier in ("belief-view", "replay"):
        # 'replay' reads the SAME claim view as 'belief-view' — the difference is
        # purely on the WRITE side: the adapter ran the replay loop (enqueue +
        # drain) before this recall, so belief_claims now also holds the
        # multi-turn aggregate (replay-stamped) claims. Both tiers fall back to
        # turn recall on an empty claim match (augment-not-gate).
        claims = await search_belief_claims(
            pool,
            query=question,
            user_id=user_id,
            scope="global",
            limit=policy.top_k,
        )
        if claims:
            total = len(claims)
            return [
                _claim_to_recall(c, rank=i, total=total, project_id=project_id)
                for i, c in enumerate(claims)
            ]
        return await _retrieve_turns(
            pool, embedder,
            question=question,
            question_type=question_type,
            project_id=project_id,
            policy=policy,
            expected_turn_count=expected_turn_count,
            as_of=as_of,
            gold_turn_ids=gold_turn_ids,
            diagnostics=diagnostics,
            candidate_sql_limit=candidate_sql_limit,
            fusion_candidate_limit=fusion_candidate_limit,
            use_anchor_local_variant=use_anchor_local_variant,
            include_embedded_temporal_variant=include_embedded_temporal_variant,
            temporal_window=temporal_window,
            use_multi_session_query_variant=use_multi_session_query_variant,
            use_session_selector=use_session_selector,
            turn_session_map=turn_session_map,
            session_selector_pool_limit=session_selector_pool_limit,
            use_session_centroid_representation=use_session_centroid_representation,
            representation_pool_limit=representation_pool_limit,
            representation_max_sessions=representation_max_sessions,
            session_centroid_selector=session_centroid_selector,
            turn_tier_expansion_slots=turn_tier_expansion_slots,
        )

    if tier == "turns":
        turns = await _retrieve_turns(
            pool, embedder,
            question=question,
            question_type=question_type,
            project_id=project_id,
            policy=policy,
            expected_turn_count=expected_turn_count,
            as_of=as_of,
            gold_turn_ids=gold_turn_ids,
            diagnostics=diagnostics,
            candidate_sql_limit=candidate_sql_limit,
            fusion_candidate_limit=fusion_candidate_limit,
            use_anchor_local_variant=use_anchor_local_variant,
            include_embedded_temporal_variant=include_embedded_temporal_variant,
            temporal_window=temporal_window,
            use_multi_session_query_variant=use_multi_session_query_variant,
            use_session_selector=use_session_selector,
            turn_session_map=turn_session_map,
            session_selector_pool_limit=session_selector_pool_limit,
            use_session_centroid_representation=use_session_centroid_representation,
            representation_pool_limit=representation_pool_limit,
            representation_max_sessions=representation_max_sessions,
            session_centroid_selector=session_centroid_selector,
            turn_tier_expansion_slots=turn_tier_expansion_slots,
        )
        # Never-miss fallback — mirrors the same resilience added to
        # weft_recall (empty turns tier → belief recall). A question the
        # single hard-regex router shapes as a turn query, but for which the
        # turn substrate holds nothing, still surfaces an answer if the belief
        # substrate has one. Fires ONLY on an empty turns result, so it can
        # never displace a real turn hit. (No-op when the haystack was ingested
        # turns-only, since there are no belief rows to fall back to — the
        # safety net only pays off on dual-shape ingests.)
        if turns:
            return turns
        if diagnostics is not None:
            diagnostics.fallback_attempted = True
        fallback = await _retrieve_belief(
            pool, embedder,
            question=question, project_id=project_id, policy=policy,
            user_id=user_id,
        )
        if diagnostics is not None:
            diagnostics.fallback_rescued = bool(fallback)
        return fallback

    return await _retrieve_belief(
        pool, embedder,
        question=question, project_id=project_id, policy=policy,
    )
