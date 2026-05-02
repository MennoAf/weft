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

import logging
import re
from datetime import datetime, timezone
from typing import Literal

import asyncpg

from weft.embeddings.base import EmbeddingProvider
from weft.ingest_pipeline import IngestItem, process as pipeline_process
from weft.models import MemoryCreate, MemorySource, MemoryType
from weft.store import store_memory

from benchmarks.longmemeval.dataset import Instance, Session

logger = logging.getLogger(__name__)


IngestMode = Literal["raw", "extracted"]


def project_id_for(question_id: str) -> str:
    """Stable, greppable project_id namespace per benchmark question.

    Prefix lets us bulk-clean: ``DELETE FROM memories WHERE project_id LIKE 'lme_%'``.
    """
    return f"lme_{question_id}"


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


async def load_haystack(
    pool: asyncpg.Pool,
    embedder: EmbeddingProvider,
    instance: Instance,
    mode: IngestMode,
) -> int:
    """Load all sessions for one benchmark question into Weft.

    Args:
        pool: asyncpg pool connected to the Weft Postgres instance.
        embedder: Embedding provider (typically FastEmbed BGE-small).
        instance: The benchmark question + its haystack.
        mode: "raw" (write sessions verbatim) or "extracted"
            (run Weft's LLM extraction pipeline).

    Returns:
        Count of sessions ingested. Memory count may be higher (extracted
        mode produces multiple memories per session) or zero per session
        (LLM may decide nothing is worth storing).

    Raises:
        ValueError: If mode is not one of the supported literals.
    """
    project_id = project_id_for(instance.question_id)
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
    """Delete every memory written for one benchmark question.

    Useful between dev iterations so the database does not grow unboundedly.
    Returns the number of rows deleted.
    """
    project_id = project_id_for(instance.question_id)
    # Hard delete (not Weft's soft-delete) so re-runs start clean.
    count = await pool.fetchval(
        "SELECT COUNT(*) FROM memories WHERE project_id = $1", project_id,
    )
    await pool.execute("DELETE FROM memories WHERE project_id = $1", project_id)
    return int(count or 0)


# ═══════════════════════════════════════════════════════════════
# RUN COMMANDS
# ═══════════════════════════════════════════════════════════════
#
# Library module — invoked from adapter.py. No standalone CLI.
#
# ═══════════════════════════════════════════════════════════════
