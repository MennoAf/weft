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

from dataclasses import dataclass
from typing import Literal

import asyncpg

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
from weft.turn_recall import route_query_to_tier, temporal_anchor
from weft.views.belief_query import BeliefClaimResult, search_belief_claims


Tier = Literal["belief", "turns", "auto", "belief-view"]

# Owner identity benchmark turns + claims are written under. Mirrors
# adapter.BENCHMARK_USER_ID; duplicated here to avoid a router→adapter import
# cycle (adapter imports router). The adapter passes its canonical constant
# into retrieve(), so this default only matters for direct/standalone calls.
_BENCHMARK_USER_ID = "longmemeval-bench"


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


async def _retrieve_turns(
    pool: asyncpg.Pool,
    embedder: EmbeddingProvider,
    *,
    question: str,
    question_type: str,
    project_id: str,
    policy: RetrievalPolicy,
) -> list[MemoryRecall]:
    """Turn-tier retrieval path. Returns Reader-compatible MemoryRecall list.

    Splits behavior by question type:
      * temporal-reasoning (and any temporal marker hit) → ``temporal_anchor``
        so multi-anchor queries get per-anchor turn pulls.
      * everything else → single ``recall_turns`` over ``episode_turns``.
    """
    base_type = question_type.removesuffix("_abs")
    use_anchor = base_type == "temporal-reasoning" or (
        route_query_to_tier(question) == "turns"
        and base_type != "multi-session"
    )

    if use_anchor:
        anchored = await temporal_anchor(
            pool, question,
            project_id=project_id,
            top_k_per_anchor=max(1, policy.top_k // 2),
            embedder=embedder,
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
    else:
        query_embedding = await embedder.embed(question)
        turns_list = await recall_turns(
            pool, question,
            project_id=project_id,
            top_k=policy.top_k,
            embedding=query_embedding,
        )

    total = len(turns_list)
    return [
        _turn_to_recall(t, rank=i, total=total, project_id=project_id)
        for i, t in enumerate(turns_list)
    ]


async def retrieve(
    pool: asyncpg.Pool,
    embedder: EmbeddingProvider,
    *,
    question: str,
    question_type: str,
    project_id: str,
    policy: RetrievalPolicy | None = None,
    tier: Tier = "belief",
    user_id: str = _BENCHMARK_USER_ID,
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

    ``user_id`` scopes the belief-view claim lookup (no effect on other tiers,
    which isolate by ``project_id``).
    """
    policy = policy or policy_for(question_type)

    if tier == "auto":
        base_type = question_type.removesuffix("_abs")
        tier = "turns" if base_type in ("multi-session", "temporal-reasoning") else "belief"

    if tier == "belief-view":
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
        # No claim matched — augment-not-gate: fall through to the turn
        # substrate so non-belief questions still get answered.
        return await _retrieve_turns(
            pool, embedder,
            question=question,
            question_type=question_type,
            project_id=project_id,
            policy=policy,
        )

    if tier == "turns":
        return await _retrieve_turns(
            pool, embedder,
            question=question,
            question_type=question_type,
            project_id=project_id,
            policy=policy,
        )

    query_embedding = await embedder.embed(question)
    raw = await search_hybrid(
        pool,
        question,
        query_embedding,
        limit=policy.top_k * policy.overfetch_multiplier,
        project_id=project_id,
    )
    memories = [r for r in raw if r.memory.project_id == project_id]
    return memories[: policy.top_k]
