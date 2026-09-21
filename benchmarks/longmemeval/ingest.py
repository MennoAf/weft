#!/usr/bin/env python3
"""
ingest.py — Two ingest modes for loading a question's haystack into Weft.

The architectural question this benchmark exists to answer:

    Does Weft's belief-shaped extraction (compress → store) preserve enough
    fidelity for multi-session and temporal questions, vs. raw dialogue-trace
    storage (write → store)?

Two modes are provided so the benchmark can answer it empirically:

* RAW mode: each session is written as one ``MemoryType.fact`` memory whose
  content is the rendered conversation. Embedding generated locally via
  FastEmbed. No LLM extraction — full fidelity, no compression.

* EXTRACTED mode: each session is fed through Weft's existing
  ``ingest_pipeline.process``, which classifies intents via Claude and
  routes to memories/entities/etc. This is what production Weft does for
  Slack/Obsidian/conversation ingest.

Both modes write under a per-question ``project_id`` so haystacks never bleed
between questions.

Author:  Jason Bauman
Version: 0.1.0
Date:    2026-04-30
Python:  >= 3.12

Dependencies:
    weft (this repo) — store, ingest_pipeline, embeddings, models

Usage:
    See adapter.py — this module is invoked per-question as part of the harness.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Literal

import asyncpg

from weft.embeddings.base import EmbeddingProvider
from weft.episode_turns import _short_id
from weft.episodes import create_episode
from weft.ingest_pipeline import IngestItem, process as pipeline_process
from weft.tokens import estimate_tokens
from weft.models import (
    EpisodeCreate,
    MemoryCreate,
    MemorySource,
    MemoryType,
    TurnRole,
)
from weft.store import store_memory

from benchmarks.longmemeval.dataset import Instance, Session

logger = logging.getLogger(__name__)

# Retry config for batch embedding. The OpenAI SDK retries internally
# (max_retries=2), but when those exhaust, the ingest code falls back to
# per-turn embedding — which sends N individual API calls instead of 1,
# causing 429 cascades that leave turns without vectors. This wrapper adds
# one more retry layer with exponential backoff before the per-turn fallback.
_BATCH_RETRY_ATTEMPTS = 3
_BATCH_RETRY_BASE_DELAY = 2.0  # seconds, doubles each attempt
_PER_TURN_DELAY = 0.1  # seconds between per-turn fallback calls


async def _embed_batch_with_retry(
    embedder: EmbeddingProvider, texts: list[str], *, question_id: str, chunk_label: str,
) -> list[list[float]]:
    """Wrap embed_batch with exponential backoff on transient failures."""
    last_exc: Exception | None = None
    for attempt in range(_BATCH_RETRY_ATTEMPTS):
        try:
            return await embedder.embed_batch(texts)
        except Exception as exc:
            last_exc = exc
            if attempt < _BATCH_RETRY_ATTEMPTS - 1:
                delay = _BATCH_RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "embed_batch attempt %d/%d failed for chunk %s (q=%s): %s — retrying in %.1fs",
                    attempt + 1, _BATCH_RETRY_ATTEMPTS, chunk_label, question_id, exc, delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.warning(
                    "embed_batch exhausted %d retries for chunk %s (q=%s): %s — falling back per-turn",
                    _BATCH_RETRY_ATTEMPTS, chunk_label, question_id, last_exc,
                )
    assert last_exc is not None
    raise last_exc


IngestMode = Literal["raw", "extracted", "turns"]


def project_id_for(question_id: str) -> str:
    """Stable, greppable project_id namespace per benchmark question.

    Prefix lets us bulk-clean: ``DELETE FROM memories WHERE project_id LIKE 'lme_%'``.
    """
    return f"lme_{question_id}"


def expected_turn_count(instance: Instance) -> int:
    """Count turns that turn-mode ingestion will actually persist."""
    return sum(
        1
        for session in instance.sessions
        for turn in session.turns
        if turn.role in _ROLE_MAP
    )


_WEEKDAY_PAREN = re.compile(r"\s*\([A-Za-z]+\)\s*")


def _parse_session_date(date_str: str) -> datetime:
    """Parse the dataset's session date into a UTC datetime.

    LongMemEval session dates appear in two flavors:
      - bare day:        ``2023/04/10``
      - day + weekday + clock: ``2023/04/10 (Mon) 17:50``

    Strip any ``(Weekday)`` annotation, then try each accepted format.
    """
    cleaned = _WEEKDAY_PAREN.sub(" ", date_str).strip()
    for fmt in ("%Y/%m/%d %H:%M", "%Y/%m/%d", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(cleaned, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return datetime.fromisoformat(cleaned).replace(tzinfo=timezone.utc)


async def _ingest_session_raw(
    pool: asyncpg.Pool,
    embedder: EmbeddingProvider,
    session: Session,
    project_id: str,
) -> None:
    """Write one session as a single ``fact`` memory, full fidelity.

    Content includes the session date so the Reader can do temporal reasoning
    even when memories are returned out of chronological order.
    """
    text = session.to_text()
    embedding = await embedder.embed(text)
    create = MemoryCreate(
        type=MemoryType.fact,
        content=text,
        topic=[f"longmemeval/{project_id}"],
        source=MemorySource.conversation,
        confidence=1.0,  # Ground-truth dataset; not LLM-extracted.
        project_id=project_id,
    )
    await store_memory(pool, create, embedding=embedding)


async def _ingest_session_extracted(
    pool: asyncpg.Pool,
    embedder: EmbeddingProvider,
    session: Session,
    project_id: str,
) -> None:
    """Run one session through Weft's production ingest pipeline.

    The pipeline classifies intent (person_fact / decision / reminder / etc.),
    extracts entities and dates, and writes one or more derived memories.
    May produce zero memories if the LLM finds nothing worth storing.
    """
    item = IngestItem(
        text=session.to_text(),
        source="longmemeval",
        timestamp=_parse_session_date(session.date),
        metadata={"benchmark": "longmemeval", "session_id": session.session_id},
    )
    result = await pipeline_process(
        item, pool, embedding_provider=embedder, project_id=project_id,
    )
    if result.errors:
        logger.warning(
            "extracted-ingest errors for session %s: %s",
            session.session_id, result.errors,
        )


_ROLE_MAP: dict[str, TurnRole] = {
    "user": TurnRole.user,
    "assistant": TurnRole.assistant,
    "system": TurnRole.system,
    "tool": TurnRole.tool,
}

# OpenAI's embeddings endpoint accepts up to 2048 inputs per call; 100 is a
# conservative chunk size that keeps individual requests responsive and
# avoids hitting the 8192-token-per-input ceiling on long-turn batches.
_EMBED_BATCH_SIZE = 100
_EMBEDDING_DIMENSIONS = 768


async def _ingest_haystack_turns(
    pool: asyncpg.Pool,
    embedder: EmbeddingProvider,
    instance: Instance,
    project_id: str,
    *,
    turn_session_map: dict[str, str] | None = None,
    turn_content_map: dict[str, str] | None = None,
) -> None:
    """Write the haystack as one episode of dialogue turns (turn-tier).

    Creates a single episode for the question, then appends every
    user/assistant turn from every session as one ``episode_turns`` row
    with ``occurred_at`` set from the session date. Each turn carries its
    own embedding so ``recall_turns`` can run hybrid (vector + BM25) over
    the dialogue trace.

    Embeddings are computed in batches of ``_EMBED_BATCH_SIZE`` so a
    single haystack of ~500 turns finishes in a handful of API calls
    instead of 500 sequential awaits. On batch failure we fall back to
    per-turn embed so a single bad input doesn't poison the whole
    question's ingest.

    Branch A of the 2026-05-02 roadmap uses this path to test whether the
    turn tier preserves enough fidelity to claw back the multi-session
    and single-session-assistant losses observed under belief-tier
    extraction (see ``project_longmemeval_baseline.md``).

    Args:
        turn_session_map: Optional dict that, if provided, will be
            populated in-place with ``{turn_id: session_id}`` for every
            turn written. Used by the adapter's recall@10 instrumentation
            to map retrieved turn IDs back to the source session ID
            without persisting the lineage to the episode_turns table.
        turn_content_map: Optional dict that, if provided, will be
            populated in-place with ``{turn_id: content}`` for every turn
            written. Used by the adapter's turn-level recall@k
            instrumentation to check whether the gold answer text appears
            in any of the retrieved turns' content.
    """
    episode = await create_episode(
        pool,
        EpisodeCreate(
            title=f"longmemeval/{instance.question_id}",
            summary=f"Haystack for question {instance.question_id} ({instance.question_type})",
            project_id=project_id,
        ),
    )

    # Flatten the haystack into a single list of (role, content, occurred_at)
    # so embedding calls can batch across session boundaries. Track
    # session_id alongside each turn so we can populate the side-map
    # after rows are assigned IDs in _bulk_append_turns.
    pending: list[tuple[TurnRole, str, datetime]] = []
    pending_session_ids: list[str] = []
    for session in instance.sessions:
        occurred_at = _parse_session_date(session.date)
        for turn in session.turns:
            role = _ROLE_MAP.get(turn.role)
            if role is None:
                logger.warning(
                    "skipping turn with unknown role %r in session %s",
                    turn.role, session.session_id,
                )
                continue
            pending.append((role, turn.content, occurred_at))
            pending_session_ids.append(session.session_id)

    embeddings: list[list[float] | None] = [None] * len(pending)
    for start in range(0, len(pending), _EMBED_BATCH_SIZE):
        chunk = pending[start : start + _EMBED_BATCH_SIZE]
        texts = [content for _, content, _ in chunk]
        try:
            batch_vecs = await _embed_batch_with_retry(
                embedder, texts,
                question_id=instance.question_id,
                chunk_label=f"{start}-{start + len(chunk)}",
            )
            if len(batch_vecs) != len(texts):
                raise ValueError(
                    f"embed_batch returned {len(batch_vecs)} vectors for "
                    f"{len(texts)} texts"
                )
            for i, vec in enumerate(batch_vecs):
                if vec is not None and len(vec) != _EMBEDDING_DIMENSIONS:
                    logger.warning(
                        "invalid embedding dimension %d for turn %d (q=%s); "
                        "retrying individually",
                        len(vec), start + i, instance.question_id,
                    )
                    continue
                embeddings[start + i] = vec
            invalid = [
                i for i, vec in enumerate(batch_vecs)
                if vec is None or len(vec) != _EMBEDDING_DIMENSIONS
            ]
            for i in invalid:
                try:
                    vec = await embedder.embed(texts[i])
                    await asyncio.sleep(_PER_TURN_DELAY)
                    if len(vec) == _EMBEDDING_DIMENSIONS:
                        embeddings[start + i] = vec
                    else:
                        logger.warning(
                            "per-turn embedding dimension %d invalid (q=%s); "
                            "storing without vector",
                            len(vec), instance.question_id,
                        )
                except Exception as inner_exc:
                    logger.warning(
                        "embed failed on turn %d (q=%s): %s — storing without vector",
                        start + i, instance.question_id, inner_exc,
                    )
        except Exception as exc:
            logger.warning(
                "embed_batch failed for chunk %d-%d (q=%s): %s — falling back per-turn",
                start, start + len(chunk), instance.question_id, exc,
            )
            for i, text in enumerate(texts):
                try:
                    vec = await embedder.embed(text)
                    await asyncio.sleep(_PER_TURN_DELAY)
                    if len(vec) == _EMBEDDING_DIMENSIONS:
                        embeddings[start + i] = vec
                    else:
                        logger.warning(
                            "per-turn embedding dimension %d invalid (q=%s); "
                            "storing without vector",
                            len(vec), instance.question_id,
                        )
                except Exception as inner_exc:
                    logger.warning(
                        "embed failed on turn %d (q=%s): %s — storing without vector",
                        start + i, instance.question_id, inner_exc,
                    )

    turn_ids = await _bulk_append_turns(
        pool, episode.id, pending, embeddings,
        turn_content_map=turn_content_map,
    )
    if turn_session_map is not None:
        for turn_id, session_id in zip(turn_ids, pending_session_ids):
            turn_session_map[turn_id] = session_id


async def _bulk_append_turns(
    pool: asyncpg.Pool,
    episode_id: str,
    pending: list[tuple[TurnRole, str, datetime]],
    embeddings: list[list[float] | None],
    *,
    turn_content_map: dict[str, str] | None = None,
) -> list[str]:
    """Bulk-insert all turns of one episode in a single executemany call.

    The production ``append_turn`` path takes a per-row advisory lock and
    derives ``turn_index`` from ``MAX(...) + 1`` so concurrent writers to
    the same episode serialize cleanly. Benchmark ingest is single-writer
    per question with a fresh episode, so neither guard is needed; we can
    pre-allocate sequential ``turn_index`` values and skip the lock,
    dropping ingest from one round-trip per turn (~0.5s × N) to one
    prepared-statement batch.

    The pgvector codec is registered on the pool (init=_pgvector_codec_init)
    so list[float] parameters are encoded into the vector type natively.
    Pass the list straight through; explicit ``::vector`` casts collide
    with the codec's binary encoding under ``executemany``.

    Args:
        turn_content_map: Optional dict that, if provided, will be
            populated in-place with ``{turn_id: content}`` for every turn
            inserted. Used by the adapter's turn-level recall@k
            instrumentation. Populated alongside ``turn_ids`` so
            the content is available without a DB round-trip.

    Returns:
        List of generated turn IDs in the same order as ``pending``. Used
        by the adapter to build the in-memory ``{turn_id: session_id}``
        and ``{turn_id: content}`` side-maps for recall@10 instrumentation.
    """
    if not pending:
        return []

    rows: list[tuple] = []
    turn_ids: list[str] = []
    for idx, ((role, content, occurred_at), embedding) in enumerate(zip(pending, embeddings)):
        turn_id = f"et-{_short_id()}"
        turn_ids.append(turn_id)
        if turn_content_map is not None:
            turn_content_map[turn_id] = content
        token_count = estimate_tokens(content)
        rows.append(
            (
                turn_id,
                episode_id,
                idx,
                role.value,
                content,
                occurred_at,
                embedding,
                token_count,
            )
        )

    async with pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO episode_turns (
                id, episode_id, turn_index, role, content,
                occurred_at, embedding, token_count
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            rows,
        )
    return turn_ids


async def load_haystack(
    pool: asyncpg.Pool,
    embedder: EmbeddingProvider,
    instance: Instance,
    mode: IngestMode,
    *,
    turn_session_map: dict[str, str] | None = None,
    turn_content_map: dict[str, str] | None = None,
) -> int:
    """Load all sessions for one benchmark question into Weft.

    Args:
        pool: asyncpg pool connected to the Weft Postgres instance.
        embedder: Embedding provider (typically FastEmbed BGE-small).
        instance: The benchmark question + its haystack.
        mode: "raw" (write sessions verbatim), "extracted"
            (run Weft's LLM extraction pipeline), or "turns"
            (write each conversational turn as an ``episode_turns`` row
            for turn-tier hybrid recall — Branch A of the roadmap).
        turn_session_map: Optional dict that, in turn-mode only, will be
            populated in-place with ``{turn_id: session_id}`` for every
            turn written. Ignored in raw/extracted modes (those paths
            don't have stable per-row IDs that map back to the Reader's
            view). The adapter passes a fresh dict per question to drive
            recall@10 instrumentation without persisting lineage.
        turn_content_map: Optional dict that, in turn-mode only, will be
            populated in-place with ``{turn_id: content}`` for every turn
            written. Ignored in raw/extracted modes. Used alongside
            ``turn_session_map`` to power turn-level recall@k
            instrumentation (gold-answer text-match against retrieved
            turn contents).

    Returns:
        Count of sessions ingested. Memory count may be higher (extracted
        mode produces multiple memories per session) or zero per session
        (LLM may decide nothing is worth storing).

    Raises:
        ValueError: If mode is not one of the supported literals.
    """
    project_id = project_id_for(instance.question_id)
    if mode == "turns":
        await _ingest_haystack_turns(
            pool, embedder, instance, project_id,
            turn_session_map=turn_session_map,
            turn_content_map=turn_content_map,
        )
        return len(instance.sessions)

    handler = {
        "raw": _ingest_session_raw,
        "extracted": _ingest_session_extracted,
    }.get(mode)
    if handler is None:
        raise ValueError(f"unknown ingest mode: {mode!r}")

    for session in instance.sessions:
        await handler(pool, embedder, session, project_id)
    return len(instance.sessions)


async def cleanup_haystack(pool: asyncpg.Pool, instance: Instance) -> int:
    """Delete every memory + episode + belief claim written for one question.

    Useful between dev iterations so the database does not grow unboundedly.
    Returns the count of memory rows deleted; episodes/turns also pruned
    (turns cascade from episodes via the v45 ``ON DELETE CASCADE`` FK).

    Belief claims are deleted FIRST, while the turns they reference still
    exist: ``belief_claims`` has no ``project_id`` column (the table is
    user-partitioned, not project-partitioned), so the only way to scope a
    delete to this question is by overlap between ``evidence_turn_ids`` and
    the question's turn ids. After the episode cascade removes those turns the
    linkage is gone, so order matters. Without this step claims accumulate
    across questions under the shared benchmark ``user_id`` and leak into
    later belief-view recalls.
    """
    project_id = project_id_for(instance.question_id)

    # Delete claims anchored to this question's turns before the cascade drops
    # the turns. COALESCE to an empty array so a question with no turns (raw /
    # extracted mode) is a no-op rather than a NULL-overlap surprise.
    await pool.execute(
        """
        DELETE FROM belief_claims
        WHERE evidence_turn_ids && (
            SELECT COALESCE(array_agg(et.id), ARRAY[]::text[])
            FROM episode_turns et
            JOIN episodes e ON et.episode_id = e.id
            WHERE e.project_id = $1
        )
        """,
        project_id,
    )

    # Hard delete (not Weft's soft-delete) so re-runs start clean.
    count = await pool.fetchval(
        "SELECT COUNT(*) FROM memories WHERE project_id = $1", project_id,
    )
    await pool.execute("DELETE FROM memories WHERE project_id = $1", project_id)
    await pool.execute("DELETE FROM episodes WHERE project_id = $1", project_id)
    return int(count or 0)


# ═══════════════════════════════════════════════════════════════
# RUN COMMANDS
# ═══════════════════════════════════════════════════════════════
#
# Library module — invoked from adapter.py. No standalone CLI.
#
# ═══════════════════════════════════════════════════════════════
