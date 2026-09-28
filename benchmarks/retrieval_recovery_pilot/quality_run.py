"""CLI for the isolated, provider-free paired answer-quality pilot."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from benchmarks.longmemeval.dataset import Instance, load_split
from weft.auth import current_user_id
from weft.db.connection import acquire
from weft.mcp.tools import weft_answer
from weft.models import MemoryCreate, MemorySource, MemoryType
from weft.store import get_memory, store_memory

from .adapter import LivePilotAdapter, PilotEmbedding, SnapshotNamespace, fixture_hash
from .quality_evaluator import (
    QUALITY_PILOT_VERSION,
    AnswerQualityPilot,
    QualityCase,
    load_quality_cases,
    write_quality_report,
)

PACKAGE = Path(__file__).parent
FIXTURE = PACKAGE / "quality_fixtures.json"
SOURCE_FIXTURE = PACKAGE / "fixtures.json"
CONFIG = Path(__file__)
LONGMEMEVAL_DATASET = Path(__file__).parents[1] / "longmemeval/data/longmemeval_s_cleaned.json"
LONGMEMEVAL_SNAPSHOT = Path(__file__).parents[1] / "longmemeval/snapshots/baseline_v1_local"
RUN_VERSION = "retrieval-recovery-answer-quality-live-v1"


class QualityMappingError(RuntimeError):
    """Raised when LongMemEval evidence cannot be proven canonical and bounded."""


LONGMEMEVAL_MAX_CASES = 8
LONGMEMEVAL_MAX_SESSIONS_PER_CASE = 500
LONGMEMEVAL_MAX_TURNS_PER_SESSION = 128
LONGMEMEVAL_MAX_TOTAL_TURNS = 8000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_turn_id(question_id: str, session_id: str, turn_index: int) -> str:
    digest = hashlib.sha256(f"{question_id}:{session_id}:{turn_index}".encode()).hexdigest()[:16]
    return f"lme-turn-{digest}"


def _longmemeval_instances(cases: tuple[QualityCase, ...], dataset: Path) -> dict[str, Instance]:
    selected = [case for case in cases if case.source == "longmemeval"]
    if len(selected) > LONGMEMEVAL_MAX_CASES:
        raise QualityMappingError(
            f"LongMemEval selection exceeds bounded pilot limit: {len(selected)}/{LONGMEMEVAL_MAX_CASES}"
        )
    try:
        instances = {instance.question_id: instance for instance in load_split(dataset)}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise QualityMappingError(f"LongMemEval cleaned dataset is unreadable: {exc}") from exc
    missing = [case.longmemeval_question_id for case in selected if case.longmemeval_question_id not in instances]
    if missing:
        raise QualityMappingError(f"LongMemEval dataset lacks selected question mapping: {missing}")
    total_turns = 0
    for case in selected:
        instance = instances[str(case.longmemeval_question_id)]
        if not case.gold_session_ids or not set(case.gold_session_ids).issubset({s.session_id for s in instance.sessions}):
            raise QualityMappingError(f"LongMemEval gold session mapping is incomplete for {case.case_id}")
        if len(instance.sessions) > LONGMEMEVAL_MAX_SESSIONS_PER_CASE:
            raise QualityMappingError(f"LongMemEval case {case.case_id} exceeds bounded session limit")
        for session in instance.sessions:
            if not session.turns or len(session.turns) > LONGMEMEVAL_MAX_TURNS_PER_SESSION:
                raise QualityMappingError(f"LongMemEval session {session.session_id} exceeds bounded turn limits")
            total_turns += len(session.turns)
    if total_turns > LONGMEMEVAL_MAX_TOTAL_TURNS:
        raise QualityMappingError(f"LongMemEval selection exceeds bounded turn limit: {total_turns}/{LONGMEMEVAL_MAX_TOTAL_TURNS}")
    return {str(case.longmemeval_question_id): instances[str(case.longmemeval_question_id)] for case in selected}


def _longmemeval_preflight(
    cases: tuple[QualityCase, ...], *, dataset: Path = LONGMEMEVAL_DATASET,
    snapshot: Path = LONGMEMEVAL_SNAPSHOT, allow_materialize: bool = False,
) -> dict[str, Any]:
    """Validate source rows and choose a verified snapshot or bounded materialization.

    The checked-in ``baseline_v1_local`` archive is for the M split.  It must
    never be used for the S quality fixture merely because question IDs overlap.
    When the archive does not match the selected cleaned dataset, the runner
    uses only the bounded source rows validated by ``_longmemeval_instances``.
    """
    selected = [case for case in cases if case.source == "longmemeval"]
    if not selected:
        return {"status": "not_selected", "selected_cases": 0}
    if not dataset.exists():
        raise QualityMappingError(f"LongMemEval dataset missing: {dataset}")
    expected_checksum = _sha256(dataset)
    manifest_path = snapshot / "manifest.json"
    marker_path = snapshot / ".complete"
    instances: dict[str, Instance] | None = None
    if manifest_path.exists() and marker_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise QualityMappingError(f"canonical LongMemEval manifest is unreadable: {exc}") from exc
        if manifest.get("dataset_checksum") == expected_checksum:
            questions = {str(item.get("question_id")): item for item in manifest.get("questions", []) if isinstance(item, dict)}
            for case in selected:
                item = questions.get(str(case.longmemeval_question_id))
                if item is None or item.get("project_id") != f"lme_{case.longmemeval_question_id}" or not item.get("turn_ids") or set(item.get("turn_ids", ())) != set(item.get("turn_session_map", {})):
                    raise QualityMappingError(f"canonical snapshot mapping is incomplete for {case.case_id}")
            return {"status": "verified", "snapshot": str(snapshot), "dataset_checksum": expected_checksum, "selected_cases": len(selected), "question_ids": [case.longmemeval_question_id for case in selected]}
        reason = "canonical snapshot dataset checksum does not match cleaned S dataset"
    else:
        reason = "canonical LongMemEval snapshot manifest or .complete marker is absent"
    if not allow_materialize:
        raise QualityMappingError(reason)
    instances = _longmemeval_instances(cases, dataset)
    return {"status": "materialize", "dataset_checksum": expected_checksum, "selected_cases": len(selected), "question_ids": [case.longmemeval_question_id for case in selected], "source_turns": sum(len(s.turns) for i in instances.values() for s in i.sessions), "snapshot": str(snapshot), "reason": reason}


def readiness(output: Path) -> dict[str, Any]:
    dsn = os.environ.get("RETRIEVAL_RECOVERY_QUALITY_DSN") or os.environ.get("RETRIEVAL_RECOVERY_PILOT_DSN")
    cases = load_quality_cases(FIXTURE)
    try:
        longmemeval = _longmemeval_preflight(cases)
    except QualityMappingError as exc:
        longmemeval = {"status": "blocked", "blocker": str(exc)}
    return {
        "status": "ready" if dsn and longmemeval.get("status") in {"verified", "not_selected"} else "blocked", "mode": "readiness", "writes": False,
        "longmemeval_mapping": longmemeval,
        "provider_calls": 0, "fixture": str(FIXTURE), "fixture_hash": fixture_hash(str(FIXTURE)),
        "source_fixture_hash": fixture_hash(str(SOURCE_FIXTURE)), "longmemeval_dataset": "benchmarks/longmemeval/data/longmemeval_s_cleaned.json",
        "longmemeval_provenance": "bounded embedded subset; gold answer/session labels; authority IDs unavailable",
        "config_hash": _sha256(CONFIG), "cases": len(cases), "expected_rows": len(cases) * 2, "database": "configured" if dsn else "not configured",
        "blocker": None if dsn else "RETRIEVAL_RECOVERY_QUALITY_DSN (or RETRIEVAL_RECOVERY_PILOT_DSN) is not set; readiness performs no writes",
        "recommendation": "Run with --seed-and-run only against an isolated test Postgres DSN" if dsn else "Start repository test Postgres and pass --dsn explicitly",
    }


def _blocked_report(reason: str, output: Path) -> Path:
    cases = load_quality_cases(FIXTURE)
    payload = {
        "version": RUN_VERSION, "status": "blocked", "pilot_version": QUALITY_PILOT_VERSION,
        "provider_calls": 0,
        "provenance": {"fixture_hash": fixture_hash(str(FIXTURE)), "source_fixture_hash": fixture_hash(str(SOURCE_FIXTURE)), "longmemeval_dataset": "benchmarks/longmemeval/data/longmemeval_s_cleaned.json", "config_hash": _sha256(CONFIG), "snapshot": None},
        "denominator": {"cases": len(cases), "arms": 2, "rows": 0, "expected_rows": len(cases) * 2, "complete": False},
        "metrics": {}, "materiality": {"decision": "not_material", "reasons": ["incomplete_denominator"]},
        "blocker": reason, "recommendation": "No materiality claim; provide a matching complete canonical LongMemEval snapshot or bounded materialization mapping.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


class LiveQualityAdapter:
    """Call only the explicit deterministic ``weft_answer`` seam."""

    def __init__(self, base: LivePilotAdapter) -> None:
        self.base = base
        self.snapshot = base.snapshot
        self.longmemeval_namespace = SnapshotNamespace(
            user_id=base.snapshot.namespace.user_id,
            project_id=f"{base.snapshot.namespace.project_id}-longmemeval",
            run_id=base.snapshot.namespace.run_id,
        )
        self.longmemeval_mapping: dict[str, dict[str, Any]] = {}
        self.longmemeval_memory_ids: list[str] = []

    async def cleanup(self) -> None:
        if self.longmemeval_memory_ids:
            token = current_user_id.set(self.longmemeval_namespace.user_id)
            try:
                async with self.base.pool.acquire() as conn:
                    await conn.execute("DELETE FROM memories WHERE id = ANY($1::text[]) AND user_id = $2", self.longmemeval_memory_ids, self.longmemeval_namespace.user_id)
            finally:
                current_user_id.reset(token)
        await self.base.cleanup()

    async def materialize_longmemeval(
        self, cases: tuple[QualityCase, ...], dataset: Path,
    ) -> dict[str, dict[str, Any]]:
        """Materialize selected source turns as canonical, scoped memories.

        This is deliberately separate from ``LivePilotAdapter.seed``.  Every
        row retains deterministic source-session/turn provenance in its topic
        tags and content, and every inserted ID is re-read through ``get_memory``
        before it is admitted to the authoritative mapping.
        """
        instances = _longmemeval_instances(cases, dataset)
        selected = [case for case in cases if case.source == "longmemeval"]
        mapping: dict[str, dict[str, Any]] = {}
        created_ids: list[str] = []
        token = current_user_id.set(self.longmemeval_namespace.user_id)
        try:
            for case in selected:
                question_id = str(case.longmemeval_question_id)
                instance = instances[question_id]
                source_turns: list[dict[str, str]] = []
                gold_ids: list[str] = []
                for session in instance.sessions:
                    for turn_index, turn in enumerate(session.turns):
                        source_turn_id = _source_turn_id(question_id, session.session_id, turn_index)
                        content = (
                            f"LongMemEval source question {question_id}; "
                            f"source_session_id={session.session_id}; "
                            f"source_turn_id={source_turn_id}; date={session.date}; "
                            f"{turn.role}: {turn.content}"
                        )
                        topic = [
                            "longmemeval",
                            f"longmemeval/question:{question_id}",
                            f"longmemeval/session:{session.session_id}",
                            f"longmemeval/turn:{source_turn_id}",
                        ]
                        async with acquire(self.base.pool):
                            memory = await store_memory(
                                self.base.pool,
                                MemoryCreate(
                                    type=MemoryType.fact,
                                    content=content,
                                    topic=topic,
                                    source=MemorySource.conversation,
                                    confidence=1.0,
                                    project_id=self.longmemeval_namespace.project_id,
                                ),
                                embedding=await PilotEmbedding().embed(content),
                            )
                        created_ids.append(memory.id)
                        self.longmemeval_memory_ids.append(memory.id)
                        async with acquire(self.base.pool):
                            verified = await get_memory(self.base.pool, memory.id)
                        if verified is None or verified.id != memory.id or verified.project_id != self.longmemeval_namespace.project_id or f"longmemeval/turn:{source_turn_id}" not in verified.topic or f"longmemeval/session:{session.session_id}" not in verified.topic:
                            raise QualityMappingError(f"canonical LongMemEval memory re-read failed for {question_id}/{source_turn_id}")
                        rendered_id = f"memory:{verified.id}"
                        source_turns.append({"source_turn_id": source_turn_id, "session_id": session.session_id, "memory_id": verified.id})
                        if session.session_id in case.gold_session_ids:
                            gold_ids.append(rendered_id)
                if not gold_ids:
                    raise QualityMappingError(f"canonical LongMemEval mapping has no gold memory IDs for {case.case_id}")
                mapping[question_id] = {
                    "question_id": question_id,
                    "project_id": self.longmemeval_namespace.project_id,
                    "user_id": self.longmemeval_namespace.user_id,
                    "source_session_ids": sorted({row["session_id"] for row in source_turns}),
                    "source_turns": source_turns,
                    "verified_memory_ids": [row["memory_id"] for row in source_turns],
                    "gold_evidence_ids": list(dict.fromkeys(gold_ids)),
                    "authoritative_ids_available": True,
                }
            self.longmemeval_mapping = mapping
            return mapping
        except Exception:
            if created_ids:
                async with self.base.pool.acquire() as conn:
                    await conn.execute("DELETE FROM memories WHERE id = ANY($1::text[]) AND user_id = $2", created_ids, self.longmemeval_namespace.user_id)
            raise
        finally:
            current_user_id.reset(token)

    def materialize_cases(self, cases: tuple[QualityCase, ...]) -> tuple[QualityCase, ...]:
        result = []
        for case in cases:
            if case.source == "longmemeval":
                question_id = str(case.longmemeval_question_id)
                mapped = self.longmemeval_mapping.get(question_id)
                if mapped is None:
                    raise QualityMappingError("LongMemEval cases cannot execute against the synthetic LivePilotAdapter namespace")
                result.append(QualityCase.from_mapping({**case.to_dict(), "gold_evidence_ids": mapped["gold_evidence_ids"], "authoritative_ids_available": True, "scope": {"user": mapped["user_id"], "project": mapped["project_id"], "retrieval_mode": "all"}}))
                continue
            ids = tuple(self.snapshot.stable_id(label) for label in case.gold_evidence_ids)
            scope = self.snapshot.scope_for(case.frozen())
            result.append(QualityCase.from_mapping({**case.to_dict(), "gold_evidence_ids": list(ids), "scope": scope}))
        return tuple(result)

    async def answer(self, case: QualityCase, arm: str) -> dict[str, Any]:
        scope = case.scope.to_dict()
        token = current_user_id.set(scope.get("user"))
        try:
            response = await weft_answer(
                self.base.context(), question=case.query, project_id=scope.get("project"),
                retrieval_mode=scope.get("retrieval_mode", "face"), limit=case.retrieval_limit,
                recovery_mode=arm,
            )
            return response
        finally:
            current_user_id.reset(token)


async def _run_live(output: Path, dsn: str) -> Path:
    cases = load_quality_cases(FIXTURE)
    mapping = _longmemeval_preflight(cases, allow_materialize=True)
    if mapping.get("status") == "blocked":
        raise QualityMappingError(mapping.get("blocker", "LongMemEval mapping is unavailable"))
    import asyncpg
    from weft.db.connection import _pgvector_codec_init, register_pgvector_codec
    from weft.db.migrations import run_migrations

    run_id = uuid.uuid4().hex[:12]
    namespace = SnapshotNamespace(
        user_id=f"retrieval-recovery-quality-user-{run_id}",
        project_id=f"retrieval-recovery-quality-project-{run_id}", run_id=run_id,
    )
    async def setup(conn: Any) -> None:
        await conn.execute(f"SET app.user_id = '{namespace.user_id}'")
    pool = await asyncpg.create_pool(dsn, min_size=2, max_size=5, init=_pgvector_codec_init, setup=setup)
    base = LivePilotAdapter(pool, namespace=namespace)
    adapter = LiveQualityAdapter(base)
    cases = load_quality_cases(FIXTURE)
    try:
        await run_migrations(pool)
        await register_pgvector_codec(pool)
        await base.seed()
        lme_mapping = await adapter.materialize_longmemeval(cases, LONGMEMEVAL_DATASET)
        materialized = adapter.materialize_cases(cases)
        report = await AnswerQualityPilot(materialized, provenance={
            "run_version": RUN_VERSION, "pilot_version": QUALITY_PILOT_VERSION,
            "snapshot": {"user_id": namespace.user_id, "project_id": namespace.project_id, "run_id": run_id},
            "longmemeval_mapping": lme_mapping,
            "fixture_hash": fixture_hash(str(FIXTURE)), "source_fixture_hash": fixture_hash(str(SOURCE_FIXTURE)),
            "longmemeval_dataset": "benchmarks/longmemeval/data/longmemeval_s_cleaned.json",
            "longmemeval_provenance": "bounded source-session/turn materialization into canonical memories; IDs re-read through get_memory",
            "embedding_provider": "retrieval-recovery-pilot-frozen", "provider_calls": 0,
            "seam": "weft_answer", "recovery_arms": ["off", "deterministic"],
            "guards": ["denominator", "baseline_controls", "no_change_controls", "scope", "legacy", "citations", "unknown", "conflict"],
        }).evaluate(adapter.answer)
        payload = report.to_dict()
        payload["status"] = "complete"
        payload["run_version"] = RUN_VERSION
        payload["provider_calls"] = sum(int(row.get("provider_calls", 0)) for row in report.rows)
        payload["snapshot_ids"] = {"memory": list(base.snapshot.created_memory_ids), "turn": list(base.snapshot.created_turn_ids), "episode": list(base.snapshot.created_episode_ids)}
        payload["longmemeval_mapping"] = lme_mapping
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return output
    finally:
        await adapter.cleanup()
        await pool.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PACKAGE / "runs/quality-latest.json")
    parser.add_argument("--seed-and-run", action="store_true", help="required before any database write")
    parser.add_argument("--dsn", help="isolated test Postgres DSN; overrides environment")
    args = parser.parse_args(argv)
    if not args.seed_and_run:
        payload = readiness(args.output)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["status"] == "ready" else 2
    dsn = args.dsn or os.environ.get("RETRIEVAL_RECOVERY_QUALITY_DSN") or os.environ.get("RETRIEVAL_RECOVERY_PILOT_DSN")
    if not dsn:
        path = _blocked_report("No explicit isolated test Postgres DSN; --seed-and-run refused database writes.", args.output)
        print(json.dumps({"status": "blocked", "report": str(path), "provider_calls": 0}, indent=2))
        return 2
    try:
        path = asyncio.run(_run_live(args.output, dsn))
    except Exception as exc:
        path = _blocked_report(f"Live quality pilot failed before a complete report: {type(exc).__name__}: {exc}", args.output)
        print(json.dumps({"status": "blocked", "report": str(path), "provider_calls": 0}, indent=2))
        return 2
    print(json.dumps({"status": "complete", "report": str(path), "provider_calls": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
