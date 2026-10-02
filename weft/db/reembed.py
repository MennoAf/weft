"""Auto re-embed rows with NULL embeddings after dimension migration.

Provides reembed_table() for per-table batch re-embedding and
auto_reembed() as the top-level orchestrator called from
ensure_vector_dimensions after a dimension migration.

Also hosts the per-tier embedder lookup helper
(``resolve_episode_embedder``) so future tiers can run on a different
provider than memory embeddings without touching the migration files.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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


@dataclass(frozen=True)
class ReembedProfile:
    """Immutable identity for one embedding materialization."""

    profile_id: str
    provider: str
    model: str
    dimensions: int
    composition: dict[str, Any]

    @classmethod
    def from_provider(
        cls, provider: "EmbeddingProvider", *, composition: dict[str, Any] | None = None
    ) -> "ReembedProfile":
        provider_name = str(provider.provider_name)
        model = str(getattr(provider, "model_name", getattr(provider, "_model_name", "unknown")))
        dims = int(provider.dimensions)
        composition = dict(composition or {})
        identity = json.dumps(
            {"provider": provider_name, "model": model, "dimensions": dims,
             "composition": composition},
            sort_keys=True, separators=(",", ":"),
        )
        profile_id = "emb-" + hashlib.sha256(identity.encode()).hexdigest()[:32]
        return cls(profile_id, provider_name, model, dims, composition)

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "provider": self.provider,
            "model": self.model,
            "dimensions": self.dimensions,
            "composition": self.composition,
        }


async def _ensure_profile(pool: asyncpg.Pool, profile: ReembedProfile) -> None:
    await pool.execute(
        "INSERT INTO embedding_profiles "
        "(profile_id, provider, model, dimensions, composition, state) "
        "VALUES ($1, $2, $3, $4, $5::jsonb, 'pending') "
        "ON CONFLICT (profile_id) DO UPDATE SET provider = EXCLUDED.provider, "
        "model = EXCLUDED.model, dimensions = EXCLUDED.dimensions, "
        "composition = EXCLUDED.composition",
        profile.profile_id, profile.provider, profile.model, profile.dimensions,
        json.dumps(profile.composition),
    )


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return dict(json.loads(value))
    return dict(value or {})


def _report(row: Any) -> dict[str, Any]:
    cursor = _json_object(row["cursor"])
    return {
        "run_id": row["run_id"],
        "target_profile_id": row["target_profile_id"],
        "status": row["status"],
        "completed": bool(row["completed"]),
        "cursor": cursor,
        "embedded_rows": row["embedded_rows"],
        "total_rows": row["total_rows"],
        "error": row["error"],
    }


async def _set_run(
    pool: asyncpg.Pool, run_id: str, *, status: str, cursor: dict[str, int],
    embedded_rows: int, error: str | None = None, completed: bool = False,
) -> Any:
    return await pool.fetchrow(
        "UPDATE embedding_reembed_runs SET status = $2, cursor = $3::jsonb, "
        "embedded_rows = $4, error = $5, completed = $6, updated_at = now(), "
        "completed_at = CASE WHEN $6 THEN now() ELSE completed_at END "
        "WHERE run_id = $1 RETURNING *",
        run_id, status, json.dumps(cursor), embedded_rows, error, completed,
    )


async def _promote_reembed(pool: asyncpg.Pool, run_id: str) -> Any:
    async with pool.acquire() as conn:
        async with conn.transaction():
            run = await conn.fetchrow(
                "SELECT * FROM embedding_reembed_runs WHERE run_id = $1 FOR UPDATE", run_id
            )
            if run is None:
                raise ValueError(f"Unknown re-embed run: {run_id}")
            profile_id = run["target_profile_id"]
            # Verification is deliberately performed immediately before the
            # swap in this transaction; no incomplete target can be promoted.
            for table in run["tables"]:
                text_col = TABLE_TEXT_COLUMNS[table]
                missing = await conn.fetchval(
                    f"SELECT COUNT(*) FROM {table} "
                    f"WHERE {text_col} IS NOT NULL AND (embedding_target IS NULL "
                    "OR embedding_target_profile_id <> $1)", profile_id  # noqa: S608
                )
                if missing:
                    raise ValueError(f"re-embed run {run_id} is not complete ({table}: {missing})")
                await conn.execute(
                    f"UPDATE {table} SET embedding = embedding_target, "
                    "embedding_profile_id = embedding_target_profile_id, "
                    "embedding_target = NULL, embedding_target_profile_id = NULL "
                    "WHERE embedding_target_profile_id = $1",  # noqa: S608
                    profile_id,
                )
            await conn.execute(
                "UPDATE embedding_profiles SET state = 'retired' "
                "WHERE state = 'active' AND profile_id <> $1", profile_id
            )
            await conn.execute(
                "UPDATE embedding_profiles SET state = 'active', activated_at = now() "
                "WHERE profile_id = $1", profile_id
            )
            await conn.execute(
                "UPDATE embedding_profile_state SET active_profile_id = $1, "
                "target_profile_id = NULL, updated_at = now() WHERE state_key = 'default'",
                profile_id,
            )
            return await conn.fetchrow(
                "UPDATE embedding_reembed_runs SET status = 'promoted', completed = TRUE, "
                "completed_at = now(), updated_at = now() WHERE run_id = $1 RETURNING *", run_id
            )


async def run_reembed(
    pool: asyncpg.Pool,
    provider: "EmbeddingProvider",
    *,
    tables: list[str] | None = None,
    batch_size: int = 100,
    composition: dict[str, Any] | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Run a bounded, resumable target materialization and atomic promotion."""
    import uuid

    tables = list(tables or TABLE_TEXT_COLUMNS)
    if not tables or any(table not in _ALLOWED_TABLES for table in tables):
        raise ValueError(f"Unknown table; allowed: {sorted(_ALLOWED_TABLES)}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive")
    profile = ReembedProfile.from_provider(provider, composition=composition)
    await _ensure_profile(pool, profile)
    run_id = "reembed-" + uuid.uuid4().hex
    cursor = {table: 0 for table in tables}
    total_rows = 0
    for table in tables:
        total_rows += await pool.fetchval(
            f"SELECT COUNT(*) FROM {table} WHERE {TABLE_TEXT_COLUMNS[table]} IS NOT NULL"  # noqa: S608
        )
    await pool.execute(
        "INSERT INTO embedding_reembed_runs "
        "(run_id, target_profile_id, tables, batch_size, cursor, total_rows) "
        "VALUES ($1, $2, $3, $4, $5::jsonb, $6)",
        run_id, profile.profile_id, tables, batch_size, json.dumps(cursor), total_rows,
    )
    await pool.execute(
        "INSERT INTO embedding_profile_state (state_key, target_profile_id) "
        "VALUES ('default', $1) ON CONFLICT (state_key) DO UPDATE SET "
        "target_profile_id = EXCLUDED.target_profile_id, updated_at = now()",
        profile.profile_id,
    )
    await pool.execute(
        "UPDATE embedding_profiles SET state = 'pending' WHERE profile_id = $1",
        profile.profile_id,
    )

    embedded_rows = 0
    batches = 0
    try:
        for table in tables:
            text_col = TABLE_TEXT_COLUMNS[table]
            text_expr = TABLE_TEXT_EXPRESSIONS.get(table, text_col)
            text_key = "embed_text" if table in TABLE_TEXT_EXPRESSIONS else text_col
            while True:
                rows = await pool.fetch(
                    f"SELECT id, {text_expr} AS {text_key} FROM {table} "
                    f"WHERE {text_col} IS NOT NULL ORDER BY id OFFSET $1 LIMIT $2",  # noqa: S608
                    cursor[table], batch_size,
                )
                if not rows:
                    break
                embeddings = await provider.embed_batch([row[text_key] for row in rows])
                if len(embeddings) != len(rows):
                    raise ValueError("provider returned a partial embedding batch")
                async with pool.acquire() as conn:
                    async with conn.transaction():
                        for row, embedding in zip(rows, embeddings):
                            await conn.execute(
                                f"UPDATE {table} SET embedding_target = $1::vector, "
                                "embedding_target_profile_id = $2 WHERE id = $3",  # noqa: S608
                                embedding, profile.profile_id, row["id"],
                            )
                            await conn.execute(
                                "INSERT INTO embedding_profile_vectors "
                                "(profile_id, table_name, row_id, embedding) "
                                "VALUES ($1, $2, $3, $4::vector) "
                                "ON CONFLICT (profile_id, table_name, row_id) "
                                "DO UPDATE SET embedding = EXCLUDED.embedding",
                                profile.profile_id, table, row["id"], embedding,
                            )
                cursor[table] += len(rows)
                embedded_rows += len(rows)
                batches += 1
                await _set_run(
                    pool, run_id, status="running", cursor=cursor,
                    embedded_rows=embedded_rows,
                )
                if max_batches is not None and batches >= max_batches:
                    return _report(await _set_run(
                        pool, run_id, status="interrupted", cursor=cursor,
                        embedded_rows=embedded_rows,
                    ))
        promoted = await _promote_reembed(pool, run_id)
        return _report(promoted)
    except asyncio.CancelledError:
        interrupted = await _set_run(
            pool, run_id, status="interrupted", cursor=cursor,
            embedded_rows=embedded_rows, error="operation cancelled",
        )
        raise
    except Exception as exc:
        failed = await _set_run(
            pool, run_id, status="failed", cursor=cursor,
            embedded_rows=embedded_rows, error=str(exc),
        )
        await pool.execute(
            "UPDATE embedding_profiles SET state = 'failed' WHERE profile_id = $1",
            profile.profile_id,
        )
        return _report(failed)


async def resume_reembed(
    pool: asyncpg.Pool,
    provider: "EmbeddingProvider",
    run_id: str,
    *,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Continue a persisted run using the same target profile identity."""
    row = await pool.fetchrow(
        "SELECT * FROM embedding_reembed_runs WHERE run_id = $1", run_id
    )
    if row is None:
        raise ValueError(f"Unknown re-embed run: {run_id}")
    if row["status"] == "promoted":
        return _report(row)
    if require_complete:
        raise ValueError(f"re-embed run {run_id} is not complete")
    cursor = _json_object(row["cursor"])
    profile = await pool.fetchrow(
        "SELECT provider, model, dimensions, composition FROM embedding_profiles "
        "WHERE profile_id = $1", row["target_profile_id"]
    )
    if profile is None:
        raise ValueError(f"Target profile for run {run_id} is missing")
    expected = ReembedProfile(
        row["target_profile_id"], profile["provider"], profile["model"],
        profile["dimensions"], _json_object(profile["composition"]),
    )
    actual = ReembedProfile.from_provider(provider, composition=expected.composition)
    if actual.profile_id != expected.profile_id:
        raise ValueError("provider identity does not match the re-embed target profile")
    # Continue the existing run without creating another run record.
    embedded_rows = row["embedded_rows"]
    batches = 0
    try:
        for table in row["tables"]:
            text_col = TABLE_TEXT_COLUMNS[table]
            text_expr = TABLE_TEXT_EXPRESSIONS.get(table, text_col)
            text_key = "embed_text" if table in TABLE_TEXT_EXPRESSIONS else text_col
            while True:
                rows = await pool.fetch(
                    f"SELECT id, {text_expr} AS {text_key} FROM {table} "
                    f"WHERE {text_col} IS NOT NULL ORDER BY id OFFSET $1 LIMIT $2",  # noqa: S608
                    cursor.get(table, 0), row["batch_size"],
                )
                if not rows:
                    break
                embeddings = await provider.embed_batch([item[text_key] for item in rows])
                if len(embeddings) != len(rows):
                    raise ValueError("provider returned a partial embedding batch")
                async with pool.acquire() as conn:
                    async with conn.transaction():
                        for item, embedding in zip(rows, embeddings):
                            await conn.execute(
                                f"UPDATE {table} SET embedding_target = $1::vector, "
                                "embedding_target_profile_id = $2 WHERE id = $3",  # noqa: S608
                                embedding, expected.profile_id, item["id"],
                            )
                            await conn.execute(
                                "INSERT INTO embedding_profile_vectors "
                                "(profile_id, table_name, row_id, embedding) VALUES "
                                "($1, $2, $3, $4::vector) ON CONFLICT "
                                "(profile_id, table_name, row_id) DO UPDATE SET embedding = EXCLUDED.embedding",
                                expected.profile_id, table, item["id"], embedding,
                            )
                cursor[table] = cursor.get(table, 0) + len(rows)
                embedded_rows += len(rows)
                batches += 1
                await _set_run(pool, run_id, status="running", cursor=cursor,
                               embedded_rows=embedded_rows)
        return _report(await _promote_reembed(pool, run_id))
    except asyncio.CancelledError:
        await _set_run(
            pool, run_id, status="interrupted", cursor=cursor,
            embedded_rows=embedded_rows, error="operation cancelled",
        )
        raise
    except Exception as exc:
        return _report(await _set_run(
            pool, run_id, status="failed", cursor=cursor,
            embedded_rows=embedded_rows, error=str(exc),
        ))


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
