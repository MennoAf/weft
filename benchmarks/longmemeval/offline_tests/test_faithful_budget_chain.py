"""Standalone, offline regressions for faithful budget ledger carry-forward."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.longmemeval.faithful_budget import (
    BUDGET_LEDGER_SCHEMA,
    BudgetExceeded,
    BudgetLedger,
    LedgerBindingError,
    import_prior_ledger,
)


V2_SCHEMA = "weft.longmemeval.faithful-budget.v2"


def _write_legacy(path: Path, *, estimate: float, phase: str = "calibration") -> Path:
    path.write_text(json.dumps({
        "schema": V2_SCHEMA,
        "reservations": [{
            "reservation_id": "legacy-reservation",
            "model": "gpt-5.6-luna",
            "estimated_usd": estimate,
            "actual_usd": None,
            "phase": phase,
            "status": "unknown",
        }],
    }), encoding="utf-8")
    return path


def _v3_from_prior(path: Path, prior: Path, *, binding: str) -> BudgetLedger:
    carry = import_prior_ledger(prior)
    return BudgetLedger(path, binding={"run": binding}, carry_forward=carry)


def test_v3_chain_flattens_reservations_without_double_counting(tmp_path: Path) -> None:
    legacy = _write_legacy(tmp_path / "v2.json", estimate=1.25)
    first_path = tmp_path / "first-v3.json"
    first = _v3_from_prior(first_path, legacy, binding="first")
    first_reservation = first.reserve("gpt-4o", 100_000, 10_000, phase="run")
    first.finalize(first_reservation.reservation_id, unknown=True, error="ambiguous")

    second_path = tmp_path / "second-v3.json"
    second = _v3_from_prior(second_path, first_path, binding="second")
    second_reservation = second.reserve("gpt-4o", 50_000, 5_000, phase="run")
    second.finalize(second_reservation.reservation_id, unknown=True, error="ambiguous")

    imported = import_prior_ledger(second_path)
    expected = 1.25 + first_reservation.estimated_usd + second_reservation.estimated_usd
    assert imported["reservation_count"] == 3
    assert imported["imported_usd"] == pytest.approx(expected)
    assert imported["imported_calibration_usd"] == pytest.approx(1.25)
    assert len({row["reservation_id"] for row in imported["reservations"]}) == 3
    assert [row["charge_usd"] for row in imported["reservations"]][0] == pytest.approx(1.25)


def test_v3_chain_rejects_tampered_predecessor_and_carry_aggregate(tmp_path: Path) -> None:
    legacy = _write_legacy(tmp_path / "v2.json", estimate=1.0)
    first_path = tmp_path / "first-v3.json"
    _v3_from_prior(first_path, legacy, binding="first")
    second_path = tmp_path / "second-v3.json"
    _v3_from_prior(second_path, first_path, binding="second")

    first_doc = json.loads(first_path.read_text(encoding="utf-8"))
    first_doc["reservations"].append({
        "reservation_id": "unbound-extra", "model": "gpt-4o", "estimated_usd": 0.1,
        "actual_usd": None, "phase": "run", "status": "unknown", "error": None,
    })
    first_path.write_text(json.dumps(first_doc), encoding="utf-8")
    with pytest.raises(LedgerBindingError, match="hash mismatch"):
        import_prior_ledger(second_path)

    # Restore the predecessor, then rebuild its successor before separately
    # corrupting the declared imported total.
    _v3_from_prior(first_path, legacy, binding="first")
    second_rebuilt_path = tmp_path / "second-rebuilt-v3.json"
    _v3_from_prior(second_rebuilt_path, first_path, binding="second")
    second_doc = json.loads(second_rebuilt_path.read_text(encoding="utf-8"))
    second_doc["carry_forward"]["imported_usd"] += 0.5
    second_rebuilt_path.write_text(json.dumps(second_doc), encoding="utf-8")
    with pytest.raises(LedgerBindingError, match="imported_usd mismatch"):
        import_prior_ledger(second_rebuilt_path)


def test_v3_chain_rejects_duplicate_ids_and_unverifiable_plain_v3(tmp_path: Path) -> None:
    legacy = _write_legacy(tmp_path / "v2.json", estimate=0.5)
    v3_path = tmp_path / "v3.json"
    ledger = _v3_from_prior(v3_path, legacy, binding="first")
    reservation = ledger.reserve("gpt-4o", 100, 100)
    ledger.finalize(reservation.reservation_id, unknown=True)
    doc = json.loads(v3_path.read_text(encoding="utf-8"))
    doc["reservations"].append(dict(doc["reservations"][0]))
    v3_path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(LedgerBindingError, match="duplicate or invalid"):
        import_prior_ledger(v3_path)

    plain_v3 = tmp_path / "plain-v3.json"
    plain_v3.write_text(json.dumps({
        "schema": BUDGET_LEDGER_SCHEMA,
        "carry_forward": {},
        "reservations": [],
    }), encoding="utf-8")
    with pytest.raises(LedgerBindingError, match="lacks a verifiable"):
        import_prior_ledger(plain_v3)


def test_carry_forward_counts_against_total_and_calibration_caps(tmp_path: Path) -> None:
    legacy_total = _write_legacy(tmp_path / "legacy-total.json", estimate=19.999, phase="run")
    total_ledger = _v3_from_prior(tmp_path / "total-v3.json", legacy_total, binding="total")
    with pytest.raises(BudgetExceeded):
        total_ledger.reserve("gpt-4o", 1_000, 0, phase="run")

    legacy_calibration = _write_legacy(
        tmp_path / "legacy-calibration.json", estimate=4.999, phase="calibration"
    )
    calibration_ledger = _v3_from_prior(
        tmp_path / "calibration-v3.json", legacy_calibration, binding="calibration"
    )
    with pytest.raises(BudgetExceeded):
        calibration_ledger.reserve("gpt-4o", 1_000, 0, phase="calibration")
