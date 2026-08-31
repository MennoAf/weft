"""Seed memory bootstrapping — load starter memories into a fresh Weft install."""

from __future__ import annotations

import importlib.resources
import logging

import asyncpg
import yaml

from weft.auth import current_user_id
from weft.db.connection import acquire
from weft.embeddings.base import EmbeddingProvider
from weft.models import MemoryCreate, MemorySource, MemoryType
from weft.schema import SYSTEM_GLOBAL_USER_ID
from weft.store import embed_text_for_memory, store_memory

logger = logging.getLogger(__name__)


def load_seeds() -> list[dict]:
    """Load seed definitions from the bundled YAML file.

    Returns an empty list if the file is missing or unparseable.
    """
    try:
        ref = importlib.resources.files("weft.data").joinpath("seeds.yaml")
        text = ref.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        if not isinstance(data, list):
            logger.warning("seeds.yaml did not contain a list — skipping")
            return []
        return data
    except FileNotFoundError:
        logger.warning("seeds.yaml not found in package data — skipping seed load")
        return []
    except Exception as exc:
        logger.warning("Failed to load seeds.yaml: %s", exc)
        return []


async def seed_memories(
    pool: asyncpg.Pool,
    embedding: EmbeddingProvider,
    *,
    force: bool = False,
) -> int:
    """Insert seed memories if the store is empty (or if *force* is True).

    Returns the number of memories successfully stored.
    """
    seeds = load_seeds()
    if not seeds:
        return 0

    if not force:
        count = await pool.fetchval(
            "SELECT COUNT(*) FROM memories WHERE status = 'active'"
        )
        if count > 0:
            logger.info("Memory store already has %d active memories — skipping seed", count)
            return 0

    stored = 0
    # Seeds are system-owned: write under the SYSTEM_GLOBAL sentinel rather
    # than the operator's user_id. Set the user_id contextvar that the
    # ``acquire()`` machinery reads, so every downstream INSERT through
    # store_memory lands on SYSTEM_GLOBAL with the right RLS scope.
    token = current_user_id.set(SYSTEM_GLOBAL_USER_ID)
    try:
        for entry in seeds:
            try:
                create = MemoryCreate(
                    type=MemoryType(entry["type"]),
                    content=entry["content"].strip(),
                    topic=entry.get("topic", []),
                    source=MemorySource(entry.get("source", "seed")),
                    confidence=entry.get("confidence", 0.9),
                    pinned=entry.get("pinned", False),
                    project_id=entry.get("project_id"),
                )
                vec = await embedding.embed(
                    embed_text_for_memory(create.content, create.topic)
                )
                async with acquire(pool):
                    await store_memory(pool, create, embedding=vec)
                stored += 1
            except Exception as exc:
                logger.warning("Failed to seed memory: %s", exc)
                continue
    finally:
        current_user_id.reset(token)

    logger.info("Seeded %d/%d memories", stored, len(seeds))
    return stored
