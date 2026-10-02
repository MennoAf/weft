from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/local_docker_acceptance.py"
_spec = importlib.util.spec_from_file_location("local_acceptance_final_v3", RUNNER)
assert _spec and _spec.loader
acceptance = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = acceptance
_spec.loader.exec_module(acceptance)


def _receipt(**kwargs):
    receipt = acceptance.AcceptanceReceipt("run", "project", "synthetic", "owner", "image", "compose", 0.0, **kwargs)
    receipt.acceptance_status = "succeeded"
    receipt.acceptance_rc = kwargs.get("acceptance_rc", 0)
    receipt.cleanup_status = kwargs.get("cleanup_status", "succeeded")
    receipt.cleanup_rc = kwargs.get("cleanup_rc", 0)
    receipt.status = kwargs.get("status", "passed")
    return receipt


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"acceptance_rc": 1, "status": "failed"}, 1),
        ({"acceptance_rc": 130, "first_signal": "SIGINT", "status": "failed"}, 130),
        ({"first_signal": "SIGTERM", "cleanup_rc": 1}, 143),
        ({"cleanup_rc": 1, "cleanup_status": "failed"}, 1),
        ({}, 0),
    ],
)
def test_result_precedence_table(kwargs, expected):
    receipt = _receipt(**kwargs)
    assert acceptance._select_return_code(receipt) == expected


def test_marker_digests_are_nonrecursive(tmp_path: Path) -> None:
    receipt = _receipt()
    marker = acceptance._publish_terminal(receipt, tmp_path / "receipt.json", tmp_path / "status.txt")
    payload = json.loads((tmp_path / "receipt.json").read_text())
    marker_payload = json.loads(marker.read_text())
    assert "receipt_sha256" not in payload
    assert marker_payload["receipt_sha256"] == hashlib.sha256((tmp_path / "receipt.json").read_bytes()).hexdigest()
    assert marker_payload["checkpoint_sha256"] == hashlib.sha256((tmp_path / ".receipt.json.checkpoint.json").read_bytes()).hexdigest()
    assert marker_payload["status_sha256"] == hashlib.sha256((tmp_path / "status.txt").read_bytes()).hexdigest()


def test_publication_rejects_existing_targets(tmp_path: Path) -> None:
    receipt = _receipt()
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text("existing")
    with pytest.raises(FileExistsError):
        acceptance._publish_terminal(receipt, receipt_path)


def test_publication_failure_table_preserves_no_retry_and_invalid_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A required write failure is fail-closed and never retries or emits a marker."""
    receipt = _receipt()
    receipt_path = tmp_path / "receipt.json"
    writes: list[Path] = []
    original = acceptance._atomic_bytes

    def fail_receipt(path: Path, content: bytes) -> None:
        writes.append(path)
        if path == receipt_path:
            raise OSError("receipt disk full")
        original(path, content)

    monkeypatch.setattr(acceptance, "_atomic_bytes", fail_receipt)
    with pytest.raises(OSError, match="disk full"):
        acceptance._publish_terminal(receipt, receipt_path)
    assert writes.count(receipt_path) == 1
    assert not receipt_path.exists()
    assert not receipt_path.with_name(f".{receipt_path.name}.complete.json").exists()


def test_normal_publication_replaces_provisional_checkpoint_after_parent_fix(tmp_path: Path) -> None:
    """A same-generation accepting checkpoint is replaced before marker publish."""
    receipt = _receipt()
    checkpoint = tmp_path / ".receipt.json.checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "run_id": receipt.run_id,
                "generation": receipt.generation,
                "lifecycle_status": "accepting",
            }
        ),
        encoding="utf-8",
    )
    receipt.checkpoint_path = str(checkpoint)
    receipt_path = tmp_path / "receipt.json"
    marker = acceptance._publish_terminal(receipt, receipt_path)
    assert marker.exists()
    assert json.loads(checkpoint.read_text())["lifecycle_status"] == "terminal"
    assert json.loads(marker.read_text())["terminal_complete"] is True


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param("not-json", id="malformed-json"),
        pytest.param([], id="non-object-json"),
        pytest.param({"generation": "missing-run", "lifecycle_status": "accepting"}, id="missing-run-id"),
        pytest.param({"run_id": "other", "generation": "generation", "lifecycle_status": "accepting"}, id="foreign-run-id"),
        pytest.param({"run_id": "run", "lifecycle_status": "accepting"}, id="missing-generation"),
        pytest.param({"run_id": "run", "generation": "other", "lifecycle_status": "accepting"}, id="foreign-generation"),
        pytest.param({"run_id": "run", "generation": "generation", "lifecycle_status": "finalizing"}, id="finalizing"),
        pytest.param({"run_id": "run", "generation": "generation", "lifecycle_status": "terminal"}, id="terminal"),
        pytest.param({"run_id": "run", "generation": "generation", "lifecycle_status": "incomplete"}, id="incomplete"),
        pytest.param({"run_id": "run", "generation": "generation", "lifecycle_status": "unknown"}, id="unknown-lifecycle"),
        pytest.param({"run_id": "run", "generation": "generation"}, id="missing-lifecycle"),
    ],
)
def test_publication_rejects_non_provisional_checkpoint_states_without_overwrite(
    tmp_path: Path, payload
) -> None:
    """Only this run's explicitly accepting checkpoint is replaceable."""
    receipt = _receipt(generation="generation")
    receipt_path = tmp_path / "receipt.json"
    checkpoint = tmp_path / ".receipt.json.checkpoint.json"
    if isinstance(payload, str):
        checkpoint.write_text(payload, encoding="utf-8")
    else:
        checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    before = checkpoint.read_bytes()
    receipt.checkpoint_path = str(checkpoint)

    with pytest.raises(FileExistsError, match="provisional output"):
        acceptance._publish_terminal(receipt, receipt_path)

    assert checkpoint.read_bytes() == before
    assert not receipt_path.exists()
    assert not receipt_path.with_name(f".{receipt_path.name}.complete.json").exists()


def test_publication_rejects_colliding_checkpoint_target(tmp_path: Path) -> None:
    receipt = _receipt()
    receipt_path = tmp_path / "receipt.json"
    receipt.checkpoint_path = str(receipt_path)

    with pytest.raises(FileExistsError, match="targets must be distinct"):
        acceptance._publish_terminal(receipt, receipt_path)

    assert not receipt_path.exists()


def test_publication_rejects_symlink_checkpoint_target(tmp_path: Path) -> None:
    receipt = _receipt()
    receipt_path = tmp_path / "receipt.json"
    checkpoint = tmp_path / ".receipt.json.checkpoint.json"
    target = tmp_path / "outside.json"
    target.write_text("do not overwrite", encoding="utf-8")
    checkpoint.symlink_to(target)
    receipt.checkpoint_path = str(checkpoint)

    with pytest.raises(FileExistsError, match="regular file"):
        acceptance._publish_terminal(receipt, receipt_path)

    assert target.read_text(encoding="utf-8") == "do not overwrite"
    assert not receipt_path.exists()


def test_publication_entry_exhausted_fails_before_mutation_or_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receipt = _receipt()
    receipt.budget_ledger = {"deadline": 10.0, "publication_reserve": 1.0}
    receipt.lifecycle_status = "accepting"
    receipt_path = tmp_path / "receipt.json"
    writes: list[Path] = []
    monkeypatch.setattr(acceptance.time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(acceptance, "_atomic_bytes", lambda path, content: writes.append(path))

    with pytest.raises(acceptance.AcceptanceTimeout, match="publication deadline exhausted"):
        acceptance._publish_terminal(receipt, receipt_path)

    assert receipt.lifecycle_status == "accepting"
    assert writes == []
    assert not receipt_path.exists()
    assert not (tmp_path / ".receipt.json.complete.json").exists()


@pytest.mark.parametrize(
    ("budget_initialized", "budget_ledger", "expected_error"),
    [
        pytest.param(True, {}, "publication deadline unavailable", id="owner-initialized-empty"),
        pytest.param(True, None, "publication deadline unavailable", id="owner-initialized-none"),
        pytest.param(True, False, "publication deadline unavailable", id="owner-initialized-falsey"),
        pytest.param(True, ["malformed"], "publication deadline unavailable", id="owner-initialized-wrong-shape"),
        pytest.param(
            True,
            {"publication_reserve": 60.0},
            "publication deadline unavailable",
            id="owner-initialized-missing-deadline",
        ),
        pytest.param(
            True,
            {"deadline": float("inf")},
            "publication deadline unavailable",
            id="owner-initialized-nonfinite-deadline",
        ),
        pytest.param(
            False,
            {"deadline": 100.0, "publication_reserve": 1.0},
            "publication budget ledger unavailable",
            id="owner-uninitialized",
        ),
        pytest.param(
            True,
            {"deadline": 100.0, "publication_reserve": 1.0},
            None,
            id="owner-valid-ledger",
        ),
    ],
)
def test_owner_publication_requires_initialized_nonempty_finite_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    budget_initialized: bool,
    budget_ledger,
    expected_error: str | None,
) -> None:
    receipt = _receipt(publication_owner=True, budget_initialized=budget_initialized)
    receipt.budget_ledger = budget_ledger
    receipt.lifecycle_status = "accepting"
    receipt_path = tmp_path / "receipt.json"
    checkpoint = tmp_path / ".receipt.json.checkpoint.json"
    writes: list[Path] = []
    original = acceptance._atomic_bytes

    def record_write(path: Path, content: bytes) -> None:
        writes.append(path)
        original(path, content)

    monkeypatch.setattr(acceptance.time, "monotonic", lambda: 1.0)
    monkeypatch.setattr(acceptance, "_atomic_bytes", record_write)

    if expected_error is not None:
        with pytest.raises(acceptance.AcceptanceTimeout, match=expected_error):
            acceptance._publish_terminal(receipt, receipt_path)
        assert receipt.lifecycle_status == "accepting"
        assert writes == []
        assert not receipt_path.exists()
        assert not checkpoint.exists()
        assert not receipt_path.with_name(f".{receipt_path.name}.complete.json").exists()
        return

    marker = acceptance._publish_terminal(receipt, receipt_path)
    assert receipt.lifecycle_status == "terminal"
    assert writes == [checkpoint, receipt_path, marker]
    assert marker.exists()


def test_legacy_empty_ledger_compatibility_remains_non_owner(tmp_path: Path) -> None:
    legacy_receipt = _receipt()
    marker = acceptance._publish_terminal(legacy_receipt, tmp_path / "legacy.json")
    assert marker.exists()


def test_main_budget_initialization_failure_does_not_publish_terminal_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An owner fallback after budget setup failure is never terminal evidence."""
    receipt_path = tmp_path / "receipt.json"
    checkpoint_path = tmp_path / ".receipt.json.checkpoint.json"
    marker_path = tmp_path / ".receipt.json.complete.json"
    args = SimpleNamespace(
        image="image",
        compose_file=tmp_path / "compose.yml",
        receipt=receipt_path,
        status=None,
        timeout=10.0,
        total_timeout=40.0,
        finalization_reserve=30.0,
        compose_cleanup_reserve=22.0,
        publication_reserve=1.0,
        compose_cleanup_timeout=1.0,
        cleanup_kill_grace=0.0,
        project_name="project",
        run_id="run",
        docker_executable=None,
    )
    writes: list[Path] = []

    def fail_budget(*_args, **_kwargs):
        raise acceptance.AcceptanceFailure("budget initialization failed")

    monkeypatch.setattr(acceptance, "parse_args", lambda _argv=None: args)
    monkeypatch.setattr(acceptance, "_budget_ledger", fail_budget)
    monkeypatch.setattr(acceptance, "_docker_compose_prefix", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Compose must not start")))
    monkeypatch.setattr(acceptance, "_atomic_bytes", lambda path, _content: writes.append(path))
    monkeypatch.setattr(acceptance, "_freeze_cli_signals", lambda _receipt: None)
    handlers_before = {
        sig: acceptance.signal.getsignal(sig)
        for sig in (acceptance.signal.SIGTERM, acceptance.signal.SIGINT)
    }

    result = acceptance.main([])

    captured = capsys.readouterr()
    handlers_after = {
        sig: acceptance.signal.getsignal(sig)
        for sig in (acceptance.signal.SIGTERM, acceptance.signal.SIGINT)
    }
    assert result == 1
    assert handlers_after == handlers_before
    payload = json.loads(captured.out)
    assert payload["primary_error"] == "AcceptanceFailure: budget initialization failed"
    assert payload["selected_return_code"] == 1
    assert payload["lifecycle_status"] == "incomplete"
    assert "publication budget ledger unavailable" in captured.err
    assert writes == []
    assert not receipt_path.exists()
    assert not checkpoint_path.exists()
    assert not marker_path.exists()


def test_publication_checks_deadline_between_writes_and_never_emits_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _receipt()
    receipt.budget_ledger = {"deadline": 3.0, "publication_reserve": 1.0}
    receipt_path = tmp_path / "receipt.json"
    checkpoint = tmp_path / ".receipt.json.checkpoint.json"
    receipt.checkpoint_path = str(checkpoint)
    clock = [1.0]
    writes: list[Path] = []
    original = acceptance._atomic_bytes

    def advancing_write(path: Path, content: bytes) -> None:
        writes.append(path)
        original(path, content)
        if path == receipt_path:
            clock[0] = 3.0

    monkeypatch.setattr(acceptance.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(acceptance, "_atomic_bytes", advancing_write)

    with pytest.raises(acceptance.AcceptanceTimeout, match="status write"):
        acceptance._publish_terminal(receipt, receipt_path, tmp_path / "status.txt")

    assert writes == [checkpoint, receipt_path]
    assert not (tmp_path / ".receipt.json.complete.json").exists()
    assert receipt_path.exists() and checkpoint.exists()


def test_publication_positive_reserve_writes_each_terminal_target_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _receipt()
    receipt.budget_ledger = {"deadline": 100.0, "publication_reserve": 1.0}
    clock = [1.0]
    writes: list[Path] = []
    original = acceptance._atomic_bytes

    def record_write(path: Path, content: bytes) -> None:
        writes.append(path)
        original(path, content)

    monkeypatch.setattr(acceptance.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(acceptance, "_atomic_bytes", record_write)
    marker = acceptance._publish_terminal(receipt, tmp_path / "receipt.json", tmp_path / "status.txt")

    assert marker.exists()
    assert writes == [
        tmp_path / ".receipt.json.checkpoint.json",
        tmp_path / "receipt.json",
        tmp_path / "status.txt",
        tmp_path / ".receipt.json.complete.json",
    ]


def test_final_checkpoint_exhausted_has_no_retry_or_late_checkpoint_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finalization admission refuses an exhausted ledger before checkpoint write."""
    args = SimpleNamespace(
        image="image",
        compose_file=tmp_path / "compose.yml",
        receipt=tmp_path / "receipt.json",
        timeout=10.0,
        total_timeout=40.0,
        finalization_reserve=30.0,
        compose_cleanup_reserve=22.0,
        publication_reserve=1.0,
        compose_cleanup_timeout=1.0,
        cleanup_kill_grace=0.0,
        project_name="project",
        run_id="run",
        docker_executable=None,
    )
    clock = iter((10.0, 50.0))
    writes: list[Path] = []

    def fail_before_external_work(*_args, **_kwargs):
        raise acceptance.AcceptanceFailure("pre-resource test failure")

    monkeypatch.setattr(acceptance.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(acceptance, "_docker_compose_prefix", fail_before_external_work)
    monkeypatch.setattr(acceptance, "_write_checkpoint", lambda path, receipt: writes.append(path))
    monkeypatch.setattr(acceptance, "_cleanup_resources", lambda *args, **kwargs: None)

    receipt = acceptance.run_acceptance(args)

    assert writes == [args.receipt.with_name(".receipt.json.checkpoint.json")]
    assert receipt.status == "failed"
    assert any("final checkpoint failed" in warning for warning in receipt.warnings)
    assert not args.receipt.exists()
    assert not args.receipt.with_name(".receipt.json.complete.json").exists()
