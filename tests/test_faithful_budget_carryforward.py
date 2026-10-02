"""Focused regressions for faithful S36 v3 carry-forward accounting."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.longmemeval.faithful_budget import (
    BudgetExceeded,
    BudgetLedger,
    LedgerBindingError,
    Pricing,
    import_prior_ledger,
)
from benchmarks.longmemeval.faithful_s36 import RunPaths, prepare_run
from tests.test_longmemeval_faithful_s36 import _write_fixture_dataset


V1 = {
    "schema": "weft.longmemeval.faithful-budget.v2",
    "max_budget_usd": 20.0,
    "calibration_budget_usd": 5.0,
    "pricing": {
        "luna_input_usd_per_million": 0.4,
        "luna_cache_write_usd_per_million": 0.4,
        "luna_output_usd_per_million": 1.8,
        "gpt4o_input_usd_per_million": 2.5,
        "gpt4o_output_usd_per_million": 10.0,
    },
    "reservations": [{
        "reservation_id": "old-1", "model": "gpt-5.6-luna",
        "estimated_usd": 0.0070176, "actual_usd": 0.0017066,
        "phase": "calibration", "status": "completed",
    }],
}


def _prior(path: Path) -> Path:
    path.write_text(json.dumps(V1), encoding="utf-8")
    return path


def test_new_pricing_and_conservative_import_charge(tmp_path: Path) -> None:
    prior = import_prior_ledger(_prior(tmp_path / "v1.json"))
    assert Pricing().luna_cache_write_usd_per_million == pytest.approx(0.5)
    assert prior["imported_usd"] == pytest.approx(0.0070176)
    assert prior["imported_calibration_usd"] == pytest.approx(0.0070176)


def test_import_is_bound_and_rejects_missing_or_changed_source(tmp_path: Path) -> None:
    source = _prior(tmp_path / "v1.json")
    carry = import_prior_ledger(source)
    source.write_text(json.dumps({**V1, "reservations": []}), encoding="utf-8")
    assert import_prior_ledger(source)["imported_usd"] == pytest.approx(0.0)
    with pytest.raises(LedgerBindingError):
        import_prior_ledger(tmp_path / "missing.json")
    assert carry["source_sha256"]


def test_double_initialization_and_prior_v3_import_are_rejected(tmp_path: Path) -> None:
    source = _prior(tmp_path / "v1.json")
    carry = import_prior_ledger(source)
    ledger = BudgetLedger(tmp_path / "new.json", binding={"run": "v2"}, carry_forward=carry)
    assert ledger.summary()["carried_forward_usd"] == pytest.approx(0.0070176)
    with pytest.raises(LedgerBindingError):
        BudgetLedger(tmp_path / "new.json", binding={"run": "v2"}, carry_forward={})
    source.write_text(json.dumps({**V1, "schema": "weft.longmemeval.faithful-budget.v3"}), encoding="utf-8")
    with pytest.raises(LedgerBindingError):
        import_prior_ledger(source)
    source.write_text(json.dumps({**V1, "carry_forward": {"imported_usd": 1}}), encoding="utf-8")
    with pytest.raises(LedgerBindingError):
        import_prior_ledger(source)


def test_import_counts_against_both_caps_before_reservation(tmp_path: Path) -> None:
    source = _prior(tmp_path / "v1.json")
    carry = import_prior_ledger(source)
    ledger = BudgetLedger(tmp_path / "new.json", max_budget_usd=0.0075,
                          binding={"run": "v2"}, carry_forward=carry)
    with pytest.raises(BudgetExceeded):
        ledger.reserve("gpt-4o", 1_000, 0, phase="run")
    calibration_source = tmp_path / "cal-source.json"
    calibration_doc = {**V1, "reservations": [{**V1["reservations"][0], "estimated_usd": 5.0}]}
    calibration_source.write_text(json.dumps(calibration_doc), encoding="utf-8")
    cal_carry = import_prior_ledger(calibration_source)
    ledger2 = BudgetLedger(tmp_path / "cal.json", max_budget_usd=20.0,
                           binding={"run": "v2"}, carry_forward=cal_carry)
    with pytest.raises(BudgetExceeded):
        ledger2.reserve("gpt-4o", 100, 0, phase="calibration")


def test_prepare_repeated_is_idempotent_and_binds_source(tmp_path: Path) -> None:
    dataset, manifest = _write_fixture_dataset(tmp_path)
    prior = _prior(tmp_path / "v1.json")
    paths = RunPaths.from_root(tmp_path / "artifacts")
    first = prepare_run(dataset, manifest, paths=paths, prior_ledger=prior)
    original_manifest = paths.manifest.read_text(encoding="utf-8")
    second = prepare_run(dataset, manifest, paths=paths, prior_ledger=prior)
    assert first.binding == second.binding
    assert paths.manifest.read_text(encoding="utf-8") == original_manifest
    checkpoint = json.loads(paths.checkpoint.read_text(encoding="utf-8"))
    assert checkpoint["carry_forward"]["import_sum_usd"] == "0.0070176"
    assert json.loads(paths.ledger.read_text(encoding="utf-8"))["carry_forward"]["imported_usd"] == pytest.approx(0.0070176)
