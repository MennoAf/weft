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
one place. Today the only knob exposed is ``top_k`` per question type. Later
extension points (already factored into ``RetrievalPolicy``):

  * Time-window filtering via the v45 ``episode_turns.list_turns_in_range``
    for temporal-reasoning questions.
  * Entity-anchored enumeration via ``weft_entity_context`` for
    "how many X" questions.
  * Multi-query expansion (one query per anchor candidate) with union/dedup.

Keep this module thin: it owns the policy table and the dispatch function,
nothing more. ``adapter.py`` calls ``retrieve()`` and gets ``MemoryRecall``s.
"""

from __future__ import annotations

from dataclasses import dataclass

import asyncpg

from weft.embeddings.base import EmbeddingProvider
from weft.models import MemoryRecall
from weft.store import search_hybrid


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


async def retrieve(
    pool: asyncpg.Pool,
    embedder: EmbeddingProvider,
    *,
    question: str,
    question_type: str,
    project_id: str,
    policy: RetrievalPolicy | None = None,
) -> list[MemoryRecall]:
    """Run the question-type-appropriate retrieval and return ranked memories.

    The benchmark sandbox semantics require strict project_id isolation:
    Weft's ``search_hybrid`` treats ``project_id=None`` memories as global
    and lets them leak alongside scoped hits, so we over-fetch and
    post-filter to the exact project_id.
    """
    policy = policy or policy_for(question_type)

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
