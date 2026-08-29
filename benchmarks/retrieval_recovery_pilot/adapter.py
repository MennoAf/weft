"""Provider-free live snapshot adapter for the retrieval-recovery pilot.

The adapter intentionally owns the synthetic namespace and the label->database-ID
mapping.  It calls the same MCP recall function used by clients, but injects a
small deterministic embedding provider so no network/provider call is possible.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping
from unittest.mock import AsyncMock, MagicMock

from weft.auth import current_user_id
from weft.cache import NullCache
from weft.config import WeftConfig
from weft.db.connection import acquire, get_db
from weft.episode_turns import _row_to_turn
from weft.episodes import create_episode
from weft.mcp.server import AppContext
from weft.mcp.tools import weft_recall
from weft.models import EpisodeCreate, EpisodeTurnCreate, MemoryCreate, MemorySource, MemoryType, RelationType, TurnRole
from weft.store import add_relationship, store_memory

from .evaluator import Arm, FrozenCase, RecallResult, Scope


class PilotEmbedding:
    """Stable local 768-dimensional embedding; no provider/network access."""

    provider_name = "retrieval-recovery-pilot-frozen"
    dimensions = 768

    @staticmethod
    def _vector(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [((digest[i % len(digest)] / 255.0) - 0.5) for i in range(768)]

    async def embed(self, text: str) -> list[float]:
        return self._vector(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]


@dataclass(frozen=True, slots=True)
class SnapshotNamespace:
    user_id: str
    project_id: str
    run_id: str


@dataclass(slots=True)
class FrozenSnapshot:
    namespace: SnapshotNamespace
    labels: dict[str, str] = field(default_factory=dict)
    created_memory_ids: list[str] = field(default_factory=list)
    created_turn_ids: list[str] = field(default_factory=list)
    created_episode_ids: list[str] = field(default_factory=list)

    def stable_id(self, label: str) -> str:
        try:
            return self.labels[label]
        except KeyError as exc:
            raise ValueError(f"unmapped pilot fixture label: {label}") from exc

    def scope_for(self, case: FrozenCase) -> dict[str, Any]:
        scope = case.scope.to_dict()
        scope["user"] = self.namespace.user_id
        if scope.get("project") is not None:
            scope["project"] = self.namespace.project_id
        return scope


class LivePilotAdapter:
    """Seed and query one isolated synthetic snapshot through ``weft_recall``."""

    def __init__(self, pool: Any, *, namespace: SnapshotNamespace) -> None:
        self.pool = pool
        self.snapshot = FrozenSnapshot(namespace)
        self._ctx: Any | None = None

    def context(self) -> Any:
        if self._ctx is None:
            app = AppContext(
                pool=self.pool,
                cache=NullCache(),
                embedding=PilotEmbedding(),
                config=WeftConfig(),
            )
            ctx = MagicMock()
            ctx.request_context.lifespan_context = app
            ctx.list_roots = AsyncMock(return_value=[])
            self._ctx = ctx
        return self._ctx

    async def _append_turn_scoped(self, create: EpisodeTurnCreate) -> Any:
        """Insert a pilot turn on the current acquire()-scoped connection."""
        conn = get_db(self.pool)
        row = await conn.fetchrow(
                """
                INSERT INTO episode_turns (
                    id, episode_id, turn_index, role, content, occurred_at,
                    embedding, trace_id, source_session_id, token_count, user_id
                )
                SELECT
                    $1, $2, COALESCE(MAX(turn_index), -1) + 1, $3, $4, $5,
                    $6::vector, $7, $8, $9,
                    nullif(current_setting('app.user_id', true), '')
                FROM episode_turns WHERE episode_id = $2
                RETURNING *
                """,
                f"et-{uuid.uuid4().hex[:16]}",
                create.episode_id,
                create.role.value,
                create.content,
                create.occurred_at,
                None,
                create.trace_id,
                create.source_session_id,
                max(1, len(create.content.split())),
            )
        if row is None:
            raise RuntimeError(f"pilot turn insert returned no row for {create.episode_id}")
        return _row_to_turn(row)

    async def seed(self) -> FrozenSnapshot:
        """Insert exactly the synthetic rows needed by the checked-in cases."""
        ns = self.snapshot.namespace
        token = current_user_id.set(ns.user_id)
        try:
            memory_specs = {
                "memory:pilot-procedural-gold": ("export configuration uses pilot_config.yaml and the export command", MemorySource.documentation, ns.project_id),
                "memory:pilot-identifier-gold": ("pilot_config.yaml is configured in the deployment code", MemorySource.code, ns.project_id),
                "memory:pilot-alpha-gold": ("Alpha uses batch exports with a weekly schedule", MemorySource.conversation, ns.project_id),
                "memory:pilot-beta-gold": ("Beta uses streaming exports with a daily schedule", MemorySource.conversation, ns.project_id),
                "memory:pilot-unknown-gold": ("The pilot launch date is not disclosed", MemorySource.conversation, ns.project_id),
                "memory:pilot-conflict-a": ("The pilot was approved on 2026-01-10", MemorySource.conversation, ns.project_id),
                "memory:pilot-conflict-b": ("The pilot was approved on 2026-02-10", MemorySource.conversation, ns.project_id),
                "memory:pilot-scope-gold": ("This project contains the retrieval recovery pilot", MemorySource.code, ns.project_id),
                "memory:pilot-outside-scope": ("This belongs to a different project and must not leak", MemorySource.code, f"{ns.project_id}-other"),
            }
            # Keep pressure deterministic: these rows deliberately share the
            # high-frequency query vocabulary but contain no pilot gold facts.
            # Case limits below make the control a real top-k slice without
            # removing any gold evidence from the indexed snapshot.
            for index in range(24):
                memory_specs[f"memory:pilot-distractor-{index:02d}"] = (
                    (
                        f"Export configuration pilot note {index}: "
                        + ("configure export settings and schedule details are archived. " * (index + 1))
                    ),
                    MemorySource.conversation,
                    ns.project_id,
                )
            for label, (content, source, project_id) in memory_specs.items():
                async with acquire(self.pool):
                    memory = await store_memory(
                        self.pool,
                        MemoryCreate(type=MemoryType.fact, content=content, source=source, confidence=0.95, project_id=project_id),
                        embedding=None,
                    )
                self.snapshot.labels[label] = f"memory:{memory.id}"
                self.snapshot.created_memory_ids.append(memory.id)

            # Conflict detection is structural and must be seeded explicitly;
            # never rely on text inference in this pilot fixture.  Both IDs are
            # available now, and the active context authenticates the edge.
            async with acquire(self.pool):
                await add_relationship(
                    self.pool,
                    self.snapshot.stable_id("memory:pilot-conflict-a").split(":", 1)[1],
                    self.snapshot.stable_id("memory:pilot-conflict-b").split(":", 1)[1],
                    RelationType.contradicts,
                )

            async with acquire(self.pool):
                chronology = await create_episode(
                    self.pool,
                    EpisodeCreate(title="retrieval recovery pilot chronology", project_id=ns.project_id),
                    embedding=None,
                )
            self.snapshot.created_episode_ids.append(chronology.id)
            turn_specs = {
                "turn:pilot-turn-gold": ("I said the pilot needs a frozen snapshot and no paid provider.", datetime(2026, 1, 5, tzinfo=timezone.utc)),
                "turn:pilot-demo-gold": ("The pilot demo happened on 2026-01-12.", datetime(2026, 1, 12, tzinfo=timezone.utc)),
                "turn:pilot-retro-gold": ("The pilot retro happened on 2026-02-02.", datetime(2026, 2, 2, tzinfo=timezone.utc)),
            }
            for label, (content, occurred_at) in turn_specs.items():
                async with acquire(self.pool):
                    turn = await self._append_turn_scoped(
                        EpisodeTurnCreate(episode_id=chronology.id, role=TurnRole.user, content=content, occurred_at=occurred_at),
                    )
                self.snapshot.labels[label] = f"turn:{turn.id}"
                self.snapshot.created_turn_ids.append(turn.id)
            return self.snapshot
        finally:
            current_user_id.reset(token)

    def materialize_cases(self, cases: tuple[FrozenCase, ...]) -> tuple[FrozenCase, ...]:
        """Replace only runtime gold labels; the checked-in fixture stays safe."""
        materialized: list[FrozenCase] = []
        for case in cases:
            values = [self.snapshot.stable_id(label) for label in case.gold_evidence_ids]
            materialized.append(FrozenCase(
                case.case_id, case.query,
                Scope.from_mapping(self.snapshot.scope_for(case)), tuple(values),
                case.expected_outcome, case.category, case.mandatory_branches,
                case.retrieval_limit,
            ))
        return tuple(materialized)

    async def recall(self, case: FrozenCase, arm: Arm) -> RecallResult:
        scope = self.snapshot.scope_for(case)
        started = time.perf_counter()
        response = await weft_recall(
            self.context(),
            query=case.query,
            project_id=scope.get("project"),
            user_id=scope.get("user"),
            mode="keyword",
            retrieval_mode=scope.get("retrieval_mode", "face"),
            tier="turns" if case.category in {"turn_only", "chronology"} else "belief",
            limit=case.retrieval_limit,
            threshold=0.0,
            recovery_mode=arm.value,
            enumeration_compatibility=False,
        )
        # Keep the additive recovery block for evaluator scoring and diagnostics.
        # Legacy canonicalization removes recovery separately, so preserving it
        # here cannot turn recovery metadata into a legacy drift signal.
        response = _normalize_response(response)
        baseline_ids = tuple(_response_ids(response, self.snapshot))
        recovery = _normalize_recovery(response.get("recovery", response.get("recovery_block")))
        return RecallResult(
            legacy_response=response,
            baseline_ids=baseline_ids,
            recovery=recovery,
            latency_ms=(time.perf_counter() - started) * 1000,
            error_category=response.get("error"),
            scope=scope,
        )

    async def cleanup(self) -> None:
        """Delete only IDs this adapter created; never truncate or broad-delete."""
        if not self.snapshot.created_memory_ids and not self.snapshot.created_episode_ids:
            return
        token = current_user_id.set(self.snapshot.namespace.user_id)
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    if self.snapshot.created_memory_ids:
                        await conn.execute("DELETE FROM memories WHERE id = ANY($1::text[]) AND user_id = $2", self.snapshot.created_memory_ids, self.snapshot.namespace.user_id)
                    if self.snapshot.created_episode_ids:
                        await conn.execute("DELETE FROM episodes WHERE id = ANY($1::text[]) AND user_id = $2", self.snapshot.created_episode_ids, self.snapshot.namespace.user_id)
        finally:
            current_user_id.reset(token)


def _response_ids(response: Mapping[str, Any], snapshot: FrozenSnapshot) -> list[str]:
    ids: list[str] = []
    for key in ("results", "turns"):
        values = response.get(key, ())
        if not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, Mapping):
                continue
            raw = value.get("id") or value.get("memory_id") or value.get("turn_id")
            if not raw:
                continue
            raw = str(raw)
            ids.append(raw if raw.startswith(("memory:", "turn:")) else (f"turn:{raw}" if key == "turns" else f"memory:{raw}"))
    return list(dict.fromkeys(ids))


def _normalize_response(response: Mapping[str, Any]) -> dict[str, Any]:
    """Strip volatile fields so control/treatment legacy payloads compare."""
    value = json.loads(json.dumps(response, default=str))
    # Recovery is additive metadata used by the evaluator; _canonical_legacy()
    # removes it when comparing the frozen legacy payload.
    value.pop("latency_ms", None)
    value.pop("request_id", None)
    return value


def _normalize_recovery(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    output = json.loads(json.dumps(value, default=str))
    candidates = output.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            stable = candidate.get("stable_id")
            if stable and not str(stable).startswith(("memory:", "turn:", "claim:")):
                prefix = "turn:" if candidate.get("turn_id") else "memory:"
                candidate["stable_id"] = prefix + str(stable)
            for key, prefix in (("memory_id", "memory:"), ("turn_id", "turn:")):
                if candidate.get(key) and not str(candidate[key]).startswith(prefix):
                    candidate[key] = str(candidate[key]).split(":", 1)[-1]
    return output


def fixture_hash(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


__all__ = ["FrozenSnapshot", "LivePilotAdapter", "PilotEmbedding", "SnapshotNamespace", "fixture_hash"]
