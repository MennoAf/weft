"""Auto re-embed rows with NULL embeddings after dimension migration.

Provides reembed_table() for per-table batch re-embedding and
auto_reembed() as the top-level orchestrator called from
ensure_vector_dimensions after a dimension migration.

Also hosts the per-tier embedder lookup helper
(``resolve_episode_embedder``) so future tiers can run on a different
provider than memory embeddings without touching the migration files.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import asyncpg

if TYPE_CHECKING:
    from weft.config import WeftConfig
    from weft.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)

# Table → text column used to generate embeddings. Single-column case;
# the SQL projection is just the bare column name.
TABLE_TEXT_COLUMNS: dict[str, str] = {
    "memories": "content",
    "behaviors": "action",
    "entities": "description",
    # Episodes use a composite expression — see TABLE_TEXT_EXPRESSIONS.
    # The mapping value here is the column we filter on for "has text"
    # (title is NOT NULL on episodes, so this filter is effectively a no-op,
    # but keeping the entry preserves the symmetry with auto_reembed's
    # default-to-all-tables behaviour).
    "episodes": "title",
}

# Optional per-table override of the SQL expression that produces the
# text input fed to the embedder. When absent, the column from
# TABLE_TEXT_COLUMNS is used directly. When present, this expression
# replaces the bare column name in the SELECT clause and is exposed
# under the alias ``embed_text`` so the row dict has a stable key.
TABLE_TEXT_EXPRESSIONS: dict[str, str] = {
    # P2.1: episodes embed title + summary. summary is nullable; coalesce
    # to empty string so a missing summary doesn't null out the whole
    # expression.
    "episodes": "title || ' ' || COALESCE(summary, '')",
    # RC2: memories embed content + topics. MUST byte-match the Python helper
    # weft.store.embed_text_for_memory(content, topic) so the generic NULL-embed
    # re-embed path agrees with the write path and the composition backfill.
    "memories": "content || ' ' || array_to_string(COALESCE(topic, '{}'), ' ')",
}

# Known safe table names — reject anything else.
_ALLOWED_TABLES = frozenset(TABLE_TEXT_COLUMNS.keys())


# ---------------------------------------------------------------------------
# Per-tier embedder selection
# ---------------------------------------------------------------------------


def resolve_episode_embedder(config: "WeftConfig") -> "EmbeddingProvider":
    """Resolve the embedding provider used for the episode tier.

    Reads the ``WEFT_EPISODE_EMBEDDER`` env var. When set, instantiates
    the named provider with the same model/dimensions as the memory
    embedder (callers can extend this with WEFT_EPISODE_EMBEDDER_MODEL
    if a future tier needs full provider+model independence). When unset,
    returns a provider built from ``config.embedding`` — the same one
    memory embeddings use, so a single-key local install just works.

    Factored out here (not inlined in the migration) so additional tiers
    (turns, entities, …) can grow analogous helpers without each one
    re-implementing the env-var-with-fallback pattern.
    """
    from weft.embeddings import get_provider

    override = os.environ.get("WEFT_EPISODE_EMBEDDER")
    provider_name = override or config.embedding.provider
    return get_provider(
        provider_name,
        model_name=config.embedding.model,
        dimensions=config.embedding.dimensions,
    )


async def reembed_table(
    pool: asyncpg.Pool,
    table: str,
    provider: EmbeddingProvider,
    batch_size: int = 100,
    force: bool = False,
) -> int:
    """Re-embed rows in a single table.

    By default only processes rows with NULL embeddings (used after
    dimension migration). With force=True, re-embeds all rows regardless
    (used by the CLI ``weft re-embed`` command).

    Returns the number of rows successfully re-embedded.
    """
    if table not in _ALLOWED_TABLES:
        raise ValueError(
            f"Unknown table {table!r}. Allowed: {sorted(_ALLOWED_TABLES)}"
        )

    text_col = TABLE_TEXT_COLUMNS[table]
    # Composite expressions (e.g. episodes "title || ' ' || COALESCE(summary, '')")
    # are aliased to ``embed_text`` so the row dict key is stable across
    # the simple-column and expression cases.
    text_expr = TABLE_TEXT_EXPRESSIONS.get(table, text_col)
    text_key = "embed_text" if table in TABLE_TEXT_EXPRESSIONS else text_col

    # Fetch rows needing embeddings
    where = f"WHERE {text_col} IS NOT NULL"
    if not force:
        where = f"WHERE embedding IS NULL AND {text_col} IS NOT NULL"
    rows = await pool.fetch(
        f"SELECT id, {text_expr} AS {text_key} FROM {table} {where}"  # noqa: S608
    )

    if not rows:
        logger.info("reembed_skip_empty", extra={"table": table})
        return 0

    total = len(rows)
    logger.info("reembed_starting", extra={"table": table, "rows": total})

    embedded = 0
    for i in range(0, total, batch_size):
        batch = rows[i : i + batch_size]
        texts = [r[text_key] for r in batch]
        ids = [r["id"] for r in batch]

        try:
            embeddings = await provider.embed_batch(texts)
        except Exception:
            logger.exception(
                "reembed_batch_failed",
                extra={"table": table, "batch_start": i, "batch_size": len(batch)},
            )
            continue

        if len(embeddings) != len(texts):
            logger.warning(
                "reembed_length_mismatch",
                extra={
                    "table": table,
                    "expected": len(texts),
                    "got": len(embeddings),
                },
            )
            # Only update rows we got embeddings for
            ids = ids[: len(embeddings)]

        async with pool.acquire() as conn:
            async with conn.transaction():
                for row_id, emb in zip(ids, embeddings):
                    await conn.execute(
                        f"UPDATE {table} SET embedding = $1::vector WHERE id = $2",  # noqa: S608
                        emb,
                        row_id,
                    )

        embedded += len(ids)
        logger.info(
            "reembed_progress",
            extra={"table": table, "progress": f"{embedded}/{total}"},
        )

    logger.info(
        "reembed_complete",
        extra={"table": table, "embedded": embedded, "total": total},
    )
    return embedded


async def backfill_memory_composition(
    pool: asyncpg.Pool,
    provider: EmbeddingProvider,
    *,
    batch_size: int = 100,
) -> int:
    """Re-embed memories whose embedding predates the current embed-text composition.

    RC2 backfill: selects rows where ``embed_composition_version < current`` and
    re-embeds them from ``embed_text_for_memory(content, topic)`` — the SAME
    Python helper the write path uses, so write-path/backfill parity is
    automatic (no SQL-expression mirror to drift). Updates the embedding and
    stamps ``embed_composition_version`` to current in one transaction per batch.

    Idempotent and resumable: already-current rows are skipped, so a re-run only
    processes whatever remains. Returns the number of rows re-embedded.

    AC5 of the recall-completeness PRD: after this completes,
    ``count(*) WHERE embed_composition_version < current`` is 0.
    """
    from weft.store import EMBED_COMPOSITION_VERSION, embed_text_for_memory

    rows = await pool.fetch(
        "SELECT id, content, topic FROM memories "
        "WHERE embed_composition_version < $1 AND content IS NOT NULL",
        EMBED_COMPOSITION_VERSION,
    )
    if not rows:
        logger.info("backfill_memory_composition: nothing stale")
        return 0

    total = len(rows)
    logger.info("backfill_memory_composition_starting", extra={"rows": total})
    done = 0
    for i in range(0, total, batch_size):
        batch = rows[i : i + batch_size]
        texts = [embed_text_for_memory(r["content"], r["topic"]) for r in batch]
        ids = [r["id"] for r in batch]
        try:
            embeddings = await provider.embed_batch(texts)
        except Exception:
            logger.exception(
                "backfill_memory_composition_batch_failed",
                extra={"batch_start": i, "batch_size": len(batch)},
            )
            continue
        if len(embeddings) != len(ids):
            ids = ids[: len(embeddings)]
        async with pool.acquire() as conn:
            async with conn.transaction():
                for row_id, emb in zip(ids, embeddings):
                    await conn.execute(
                        "UPDATE memories SET embedding = $1::vector, "
                        "embed_composition_version = $2 WHERE id = $3",
                        emb,
                        EMBED_COMPOSITION_VERSION,
                        row_id,
                    )
        done += len(ids)
        logger.info(
            "backfill_memory_composition_progress",
            extra={"progress": f"{done}/{total}"},
        )

    logger.info("backfill_memory_composition_complete", extra={"embedded": done})
    return done


async def auto_reembed(
    pool: asyncpg.Pool,
    provider: EmbeddingProvider,
    tables: list[str] | None = None,
    batch_size: int = 100,
) -> dict[str, int]:
    """Re-embed NULL embeddings across multiple tables.

    If tables is None, processes all known tables.
    Returns {table: rows_embedded} dict.
    """
    if tables is None:
        tables = list(TABLE_TEXT_COLUMNS.keys())

    results: dict[str, int] = {}
    for table in tables:
        count = await reembed_table(pool, table, provider, batch_size)
        results[table] = count

    total = sum(results.values())
    if total > 0:
        logger.info(
            "auto_reembed_complete",
            extra={"results": results, "total": total},
        )
    return results
