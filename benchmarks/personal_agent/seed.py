#!/usr/bin/env python3
"""
seed.py — Seed the PAAH manifest through the REAL write path.

Every member fact is stored via ``weft_remember`` (not ``store_memory``
directly), so embedding, topic tagging, validation, and the pre-insert dedup
check are all exercised — the same code an agent hits. The trade-off is that
dedup can collapse near-duplicate facts; ``seed_corpus`` therefore records the
ACTUAL stored memory id for every member and verifies stored cardinality
against the manifest, so any collapse surfaces as a ``SeedResult`` with
``dedup_collisions`` rather than silently corrupting the ground truth.

Seeding runs under the dedicated PAAH synthetic identity (``PAAH_USER_ID`` /
``PAAH_PROJECT_ID``) so it never touches Jason's real corpus or the
enumeration_eval sandbox.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import asyncpg

from weft.auth import current_user_id

from benchmarks.personal_agent.context import build_app_context, make_ctx
from benchmarks.personal_agent.manifest import (
    PAAH_PROJECT_ID,
    PAAH_USER_ID,
    Collection,
    get_collections,
)
from benchmarks.personal_agent.temporal_manifest import (
    PAAH_TEMPORAL_PROJECT_ID,
    get_turn_specs,
)
from benchmarks.personal_agent.entity_manifest import (
    PAAH_ENTITY_PROJECT_ID,
    EntitySpec,
    get_brief_entity,
    get_distractor_entity,
)
from benchmarks.personal_agent.agenda_manifest import (
    PAAH_AGENDA_PROJECT_ID,
    AgendaSpec,
    get_agenda_specs,
)

logger = logging.getLogger(__name__)


@dataclass
class SeedResult:
    """Outcome of seeding one collection through the real write path."""

    collection: Collection
    stored_ids: list[str] = field(default_factory=list)
    stored_contents: list[str] = field(default_factory=list)
    dedup_collisions: list[str] = field(default_factory=list)

    @property
    def intended(self) -> int:
        return self.collection.cardinality

    @property
    def stored(self) -> int:
        # Ground truth = distinct member rows actually persisted.
        return len(set(self.stored_ids))

    @property
    def clean(self) -> bool:
        """True when every intended member persisted as its own row."""
        return self.stored == self.intended and not self.dedup_collisions


async def _remember_member(ctx, content: str, topic_tag: str) -> tuple[str, bool]:
    """Store one member via weft_remember. Returns (memory_id, was_dedup).

    ``check_contradictions=False`` — contradiction handling can archive rows and
    perturb cardinality, and these are independent facts, not competing claims.
    ``pinned=False`` — keep dedup and normal recall ranking active so the eval
    reflects ordinary facts (pinning would both disable dedup and boost recall).
    """
    from weft.mcp.tools import weft_remember

    result = await weft_remember(
        ctx,
        content=content,
        type="fact",
        topic=[topic_tag],
        source="conversation",
        confidence=0.9,
        project_id=PAAH_PROJECT_ID,
        check_contradictions=False,
        pinned=False,
    )
    was_dedup = bool(result.get("dedup", {}).get("is_duplicate")) if isinstance(
        result.get("dedup"), dict
    ) else ("dedup" in result)
    return result["id"], was_dedup


async def seed_corpus(pool: asyncpg.Pool) -> list[SeedResult]:
    """Seed every manifest collection and verify stored cardinality.

    Returns one SeedResult per collection. Raises nothing on dedup collapse —
    the caller inspects ``SeedResult.clean`` / ``dedup_collisions`` and decides
    whether the ground truth is trustworthy.
    """
    app = await build_app_context(pool)
    ctx = make_ctx(app)

    results: list[SeedResult] = []
    token = current_user_id.set(PAAH_USER_ID)
    try:
        for collection in get_collections():
            res = SeedResult(collection=collection)
            for content in collection.members:
                mem_id, was_dedup = await _remember_member(
                    ctx, content, collection.topic_tag
                )
                if was_dedup:
                    res.dedup_collisions.append(content)
                    logger.warning(
                        "seed_corpus: dedup collapsed member in %s: %r",
                        collection.name,
                        content,
                    )
                res.stored_ids.append(mem_id)
                res.stored_contents.append(content)
            logger.info(
                "seed_corpus: %s intended=%d stored=%d clean=%s",
                collection.name,
                res.intended,
                res.stored,
                res.clean,
            )
            results.append(res)
    finally:
        current_user_id.reset(token)

    return results


@dataclass
class SeedTurnsResult:
    """Outcome of seeding the temporal dialogue trace through weft_turn_append."""

    episode_id: str
    turn_ids: dict[str, str] = field(default_factory=dict)  # spec.key -> turn id
    intended: int = 0

    @property
    def stored(self) -> int:
        return len(set(self.turn_ids.values()))

    @property
    def clean(self) -> bool:
        return self.stored == self.intended


async def seed_turns(pool: asyncpg.Pool) -> SeedTurnsResult:
    """Seed the dated dialogue trace through the REAL weft_turn_append path.

    Creates one episode, then appends every TurnSpec as a 'user' turn with its
    known occurred_at. Returns the spec.key -> turn_id map so the temporal
    harness can assert anchor turns by id. Scoped to PAAH_TEMPORAL_PROJECT_ID
    under PAAH_USER_ID so it never shares rows with the enumeration corpus.
    """
    from weft.mcp.tools import weft_episode_create, weft_turn_append

    app = await build_app_context(pool)
    ctx = make_ctx(app)

    specs = get_turn_specs()
    result = SeedTurnsResult(episode_id="", intended=len(specs))
    token = current_user_id.set(PAAH_USER_ID)
    try:
        episode = await weft_episode_create(
            ctx,
            title="Jim Boblaw's June dialogue trace",
            project_id=PAAH_TEMPORAL_PROJECT_ID,
        )
        result.episode_id = episode["id"]
        for spec in specs:
            turn = await weft_turn_append(
                ctx,
                episode_id=result.episode_id,
                role="user",
                content=spec.content,
                occurred_at=spec.occurred_at,
            )
            result.turn_ids[spec.key] = turn["turn_id"]
        logger.info(
            "seed_turns: episode=%s intended=%d stored=%d clean=%s",
            result.episode_id, result.intended, result.stored, result.clean,
        )
    finally:
        current_user_id.reset(token)

    return result


@dataclass
class SeedEntityResult:
    """Outcome of seeding one entity + its linked facts through the real path."""

    spec: EntitySpec
    entity_id: str
    fact_ids: list[str] = field(default_factory=list)
    dedup_collisions: list[str] = field(default_factory=list)

    @property
    def intended(self) -> int:
        return self.spec.cardinality

    @property
    def stored(self) -> int:
        return len(set(self.fact_ids))

    @property
    def clean(self) -> bool:
        return self.stored == self.intended and not self.dedup_collisions


@dataclass
class SeedEntityBriefResult:
    """The brief entity plus a distractor, for the entity-brief shape."""

    brief: SeedEntityResult
    distractor: SeedEntityResult


async def _seed_one_entity(ctx, spec: EntitySpec) -> SeedEntityResult:
    """Create an entity, remember each fact, and link it to the entity."""
    from weft.mcp.tools import weft_entity_create

    entity = await weft_entity_create(
        ctx,
        name=spec.name,
        entity_type=spec.entity_type,
        description=spec.description,
        project_id=PAAH_ENTITY_PROJECT_ID,
    )
    res = SeedEntityResult(spec=spec, entity_id=entity["id"])
    for content in spec.facts:
        mem_id, was_dedup = await _remember_member(ctx, content, spec.name.lower())
        if was_dedup:
            res.dedup_collisions.append(content)
        await _link_fact(ctx, res.entity_id, mem_id)
        res.fact_ids.append(mem_id)
    return res


async def _link_fact(ctx, entity_id: str, memory_id: str) -> None:
    from weft.mcp.tools import weft_entity_link

    await weft_entity_link(ctx, entity_id=entity_id, memory_id=memory_id)


async def seed_entity_brief(pool: asyncpg.Pool) -> SeedEntityBriefResult:
    """Seed the brief entity + a distractor entity through the REAL graph path.

    Every fact is a weft_remember memory linked to its entity via
    weft_entity_link — the same edges an agent builds. Returns both entities'
    ids + fact ids so the harness can measure recall@links by memory id.
    """
    from weft.mcp.tools import weft_remember  # noqa: F401 (ensures import path valid)

    app = await build_app_context(pool)
    ctx = make_ctx(app)

    token = current_user_id.set(PAAH_USER_ID)
    try:
        brief = await _seed_one_entity(ctx, get_brief_entity())
        distractor = await _seed_one_entity(ctx, get_distractor_entity())
        logger.info(
            "seed_entity_brief: brief=%s facts=%d/%d clean=%s | distractor facts=%d",
            brief.spec.name, brief.stored, brief.intended, brief.clean,
            distractor.stored,
        )
    finally:
        current_user_id.reset(token)

    return SeedEntityBriefResult(brief=brief, distractor=distractor)


@dataclass
class SeedAgendaTracker:
    """One seeded tracker plus its manifest spec (the ground-truth fate)."""

    spec: AgendaSpec
    tracker_id: str
    created_ok: bool = True
    close_applied: bool = False
    snooze_applied: bool = False


@dataclass
class SeedAgendaResult:
    """Outcome of seeding the agenda trackers through the real lifecycle path."""

    trackers: list[SeedAgendaTracker] = field(default_factory=list)
    # Anchor captured at seed time so the harness reasons about the same "now".
    seeded_at: datetime | None = None

    @property
    def intended(self) -> int:
        return len(self.trackers)

    def by_key(self, key: str) -> SeedAgendaTracker:
        for t in self.trackers:
            if t.spec.key == key:
                return t
        raise KeyError(key)

    @property
    def due_trackers(self) -> list[SeedAgendaTracker]:
        """The open loops that must appear in the due queue (ground truth)."""
        return [t for t in self.trackers if t.spec.expected_due]

    @property
    def due_ids(self) -> set[str]:
        return {t.tracker_id for t in self.due_trackers}

    @property
    def excluded_ids(self) -> set[str]:
        return {t.tracker_id for t in self.trackers if not t.spec.expected_due}

    @property
    def keep_pushing(self) -> SeedAgendaTracker:
        """The longest-overdue due tracker — the 'thing I keep pushing' anchor.

        due_trackers orders by nudge_after ASC, so this is the one that must
        surface FIRST in the due queue.
        """
        return min(self.due_trackers, key=lambda t: t.spec.nudge_after_days)

    @property
    def clean(self) -> bool:
        """Every spec created, and every close/snooze it declared was applied."""
        if len(self.trackers) != len(get_agenda_specs()):
            return False
        for t in self.trackers:
            if not (t.created_ok and t.tracker_id):
                return False
            if t.spec.close_as is not None and not t.close_applied:
                return False
            if t.spec.snooze_days is not None and not t.snooze_applied:
                return False
        return True


async def seed_agenda(pool: asyncpg.Pool) -> SeedAgendaResult:
    """Seed the agenda trackers through the REAL tracker lifecycle path.

    Each tracker is created via weft_tracker_create with a nudge_after resolved
    from its day-offset against a single ``now`` anchor; snoozed specs go through
    weft_tracker_snooze and terminal specs through weft_tracker_close, so the
    real state machine (state_history, snooze_until) is exercised — not a fixture
    INSERT. Scoped to PAAH_AGENDA_PROJECT_ID under PAAH_USER_ID.
    """
    from weft.mcp.tools import (
        weft_tracker_close,
        weft_tracker_create,
        weft_tracker_snooze,
    )

    app = await build_app_context(pool)
    ctx = make_ctx(app)

    now = datetime.now(timezone.utc)
    result = SeedAgendaResult(seeded_at=now)
    token = current_user_id.set(PAAH_USER_ID)
    try:
        for spec in get_agenda_specs():
            nudge_after = (
                (now + timedelta(days=spec.nudge_after_days)).isoformat()
                if spec.nudge_after_days is not None
                else None
            )
            created = await weft_tracker_create(
                ctx,
                kind=spec.kind,
                title=spec.title,
                project_id=PAAH_AGENDA_PROJECT_ID,
                state=spec.state,
                nudge_mode=spec.nudge_mode,
                nudge_after=nudge_after,
                nudge_interval=spec.nudge_interval,
            )
            tid = created.get("id", "")
            seeded = SeedAgendaTracker(spec=spec, tracker_id=tid, created_ok=bool(tid))

            if tid and spec.snooze_days is not None:
                snoozed = await weft_tracker_snooze(
                    ctx,
                    tracker_id=tid,
                    until=(now + timedelta(days=spec.snooze_days)).isoformat(),
                )
                seeded.snooze_applied = "error" not in snoozed

            if tid and spec.close_as is not None:
                closed = await weft_tracker_close(
                    ctx, tracker_id=tid, final_state=spec.close_as,
                )
                seeded.close_applied = "error" not in closed

            result.trackers.append(seeded)

        logger.info(
            "seed_agenda: seeded=%d due=%d excluded=%d clean=%s",
            result.intended, len(result.due_ids), len(result.excluded_ids),
            result.clean,
        )
    finally:
        current_user_id.reset(token)

    return result
