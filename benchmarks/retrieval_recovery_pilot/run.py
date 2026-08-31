"""CLI for the provider-free retrieval-recovery pilot.

Readiness is the default and performs no database writes. ``--seed-and-run`` is
required for writes and requires an explicit DSN; it is intended for an isolated
test Postgres database, never an ambient production deployment.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.recall_contract.artifact_io import publish_json_atomic
from .adapter import LivePilotAdapter, SnapshotNamespace, fixture_hash
from .evaluator import Arm, PilotReport, RecoveryPilot, load_cases

PACKAGE = Path(__file__).parent
FIXTURE = PACKAGE / "fixtures.json"
CONFIG = PACKAGE / "run.py"
RUN_VERSION = "retrieval-recovery-pilot-live-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def readiness(output: Path) -> dict[str, Any]:
    dsn = os.environ.get("RETRIEVAL_RECOVERY_PILOT_DSN")
    return {
        "status": "ready" if dsn else "blocked",
        "mode": "readiness",
        "writes": False,
        "provider_calls": 0,
        "fixture": str(FIXTURE),
        "fixture_hash": fixture_hash(str(FIXTURE)),
        "config_hash": _sha256(CONFIG),
        "database": "configured" if dsn else "not configured",
        "blocker": None if dsn else "RETRIEVAL_RECOVERY_PILOT_DSN is not set; readiness performs no writes",
        "recommendation": "Run with --seed-and-run only against an isolated test Postgres DSN" if dsn else "Start the repository test Postgres and pass its DSN explicitly",
    }


def _blocked_report(reason: str, output: Path) -> Path:
    payload = {
        "version": RUN_VERSION,
        "status": "blocked",
        "pilot_version": "retrieval-recovery-pilot-v1",
        "provider_calls": 0,
        "provenance": {
            "fixture_hash": fixture_hash(str(FIXTURE)),
            "config_hash": _sha256(CONFIG),
            "snapshot": None,
        },
        "denominator": {"cases": 8, "arms": 2, "rows": 0, "expected_rows": 16, "complete": False},
        "metrics": {},
        "blocker": reason,
        "recommendation": "No lift claim; rerun --seed-and-run with an isolated test Postgres DSN.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    return publish_json_atomic(output, payload)


async def _run_live(output: Path, dsn: str) -> Path:
    import asyncpg
    from weft.db.connection import _pgvector_codec_init, register_pgvector_codec
    from weft.db.migrations import run_migrations

    run_id = uuid.uuid4().hex[:12]
    namespace = SnapshotNamespace(
        user_id=f"retrieval-recovery-pilot-user-{run_id}",
        project_id=f"retrieval-recovery-pilot-project-{run_id}",
        run_id=run_id,
    )

    async def _pilot_setup(conn):
        """Set the isolated pilot identity on every pool acquire."""
        await conn.execute(f"SET app.user_id = '{namespace.user_id}'")

    # Match the established isolated test fixture pool shape. The recovery
    # telemetry and async query logger can use separate connections while a
    # request holds one; this is benchmark isolation only, not production
    # pool sizing. Codec registration remains post-migration below.
    pool = await asyncpg.create_pool(
        dsn,
        min_size=2,
        max_size=5,
        init=_pgvector_codec_init,
        setup=_pilot_setup,
    )
    adapter = LivePilotAdapter(pool, namespace=namespace)
    cases = load_cases(FIXTURE)
    try:
        await run_migrations(pool)
        await register_pgvector_codec(pool)
        await adapter.seed()
        materialized = adapter.materialize_cases(cases)
        report = await RecoveryPilot(
            materialized,
            provenance={
                "run_version": RUN_VERSION,
                "snapshot": namespace.__dict__ if hasattr(namespace, "__dict__") else {"user_id": namespace.user_id, "project_id": namespace.project_id, "run_id": run_id},
                "fixture_hash": fixture_hash(str(FIXTURE)),
                "config_hash": _sha256(CONFIG),
                "embedding_provider": "retrieval-recovery-pilot-frozen",
                "provider_calls": 0,
                "legacy_control_treatment_kwargs_identical": True,
                "guards": ["scope", "legacy", "unknown", "conflict"],
            },
        ).evaluate(adapter.recall)
        payload = report.to_dict()
        payload["status"] = "complete"
        payload["materiality"] = _materiality(report)
        payload["snapshot_ids"] = {
            "memory": list(adapter.snapshot.created_memory_ids),
            "turn": list(adapter.snapshot.created_turn_ids),
            "episode": list(adapter.snapshot.created_episode_ids),
        }
        return publish_json_atomic(output, payload)
    finally:
        await adapter.cleanup()
        await pool.close()


def _materiality(report: PilotReport) -> dict[str, Any]:
    """Apply the safety *and* efficacy gates for a promotion claim.

    Safety guards are necessary but deliberately not sufficient: a complete,
    fail-closed zero-lift run is useful evidence, but it is not material
    evidence of recovery benefit.  The thresholds are predeclared for this
    eight-case pilot and are emitted with their observed values so a report
    cannot hide why promotion was blocked.
    """
    treatment = report.metrics.get("treatment", {})
    guards = {
        "legacy_baseline_drift_zero": treatment.get("legacy_baseline_drift", 0) == 0,
        "scope_violations_zero": treatment.get("scope_violations", 0) == 0,
        "unknown_false_sufficiency_zero": treatment.get("unknown_false_sufficiency", 0) == 0,
        "conflict_preservation": treatment.get("conflict_preservation", 0) >= 1,
    }
    complete_denominator = report.denominator.get("rows") == (
        report.denominator.get("cases", 0) * report.denominator.get("arms", 0)
    )
    baseline_miss_opportunities = sum(
        1 for row in report.rows
        if row.get("arm") == "deterministic" and float(row.get("baseline_recall", 1.0)) < 1.0
    )
    rescued_cases = int(treatment.get("rescued_cases", 0))
    rescue_rate = float(treatment.get("rescue_rate", 0.0))
    absolute_lift = float(treatment.get("absolute_lift", 0.0))
    efficacy = {
        "minimum_baseline_miss_opportunities": 3,
        "minimum_true_complete_rescues": 1,
        "minimum_rescue_rate": 0.20,
        "require_positive_absolute_lift": True,
        "baseline_miss_opportunities": baseline_miss_opportunities,
        "rescued_cases": rescued_cases,
        "rescue_rate": rescue_rate,
        "absolute_lift": absolute_lift,
    }
    efficacy_checks = {
        "enough_baseline_miss_opportunities": baseline_miss_opportunities >= efficacy["minimum_baseline_miss_opportunities"],
        "true_complete_rescue_or_rate": (
            rescued_cases >= efficacy["minimum_true_complete_rescues"]
            or rescue_rate >= efficacy["minimum_rescue_rate"]
        ),
        "positive_absolute_lift": absolute_lift > 0.0,
    }
    reasons = [
        "incomplete_denominator" if not complete_denominator else None,
        "safety_guard_failed" if not all(guards.values()) else None,
        "fewer_than_three_baseline_miss_opportunities" if not efficacy_checks["enough_baseline_miss_opportunities"] else None,
        "no_true_complete_rescue_or_rescue_rate_below_20_percent" if not efficacy_checks["true_complete_rescue_or_rate"] else None,
        "zero_or_negative_absolute_lift" if not efficacy_checks["positive_absolute_lift"] else None,
    ]
    reasons = [reason for reason in reasons if reason is not None]
    material = complete_denominator and all(guards.values()) and all(efficacy_checks.values())
    return {
        "decision": "material" if material else "not_material",
        "complete_denominator": complete_denominator,
        "guards": guards,
        "guards_passed": all(guards.values()),
        "efficacy_gate": {"checks": efficacy_checks, "values": efficacy, "passed": all(efficacy_checks.values())},
        "absolute_lift": absolute_lift,
        "reasons": reasons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/retrieval_recovery_pilot/runs/latest.json"))
    parser.add_argument("--seed-and-run", action="store_true", help="required before any database write")
    parser.add_argument("--dsn", help="isolated test Postgres DSN; overrides WEFT_DATABASE_URL")
    args = parser.parse_args(argv)
    if not args.seed_and_run:
        payload = readiness(args.output)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["status"] == "ready" else 2
    dsn = args.dsn or os.environ.get("RETRIEVAL_RECOVERY_PILOT_DSN")
    if not dsn:
        path = _blocked_report("No explicit isolated test Postgres DSN (--dsn or RETRIEVAL_RECOVERY_PILOT_DSN); --seed-and-run refused database writes.", args.output)
        print(json.dumps({"status": "blocked", "report": str(path), "provider_calls": 0}, indent=2))
        return 2
    try:
        path = asyncio.run(_run_live(args.output, dsn))
    except Exception as exc:  # CLI emits a truthful blocked artifact, never fake scores.
        path = _blocked_report(f"Live pilot failed before a complete report: {type(exc).__name__}: {exc}", args.output)
        print(json.dumps({"status": "blocked", "report": str(path), "provider_calls": 0}, indent=2))
        return 2
    print(json.dumps({"status": "complete", "report": str(path), "provider_calls": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
