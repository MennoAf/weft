"""Pure v3 lifecycle ledger tests; no subprocesses, Docker, or network."""
from __future__ import annotations

import importlib.util
import io
import subprocess
import sys
import urllib.error
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/local_docker_acceptance.py"
_spec = importlib.util.spec_from_file_location("local_acceptance_lifecycle_v3", RUNNER)
assert _spec and _spec.loader
acceptance = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = acceptance
_spec.loader.exec_module(acceptance)


def _args(**overrides):
    values = dict(
        total_timeout=2400,
        finalization_reserve=240,
        compose_cleanup_reserve=120,
        publication_reserve=60,
        compose_cleanup_timeout=90,
        cleanup_kill_grace=1,
    )
    values.update(overrides)
    return Namespace(**values)


def test_reserved_deadline_ledger() -> None:
    ledger = acceptance._budget_ledger(_args(), 100.0)
    assert ledger["deadline"] > ledger["acceptance_deadline"]
    assert ledger["memory_cleanup_deadline"] == ledger["acceptance_deadline"] + 60
    assert ledger["compose_cleanup_deadline"] == ledger["deadline"] - 60
    assert ledger["memory_window"] == 60


def test_memory_exhaustion_preserves_down_and_publication_reserves() -> None:
    with pytest.raises(acceptance.AcceptanceFailure, match="fit compose reserve"):
        acceptance._budget_ledger(_args(compose_cleanup_timeout=100, cleanup_kill_grace=1), 0)


def test_all_operation_classes_receive_remaining_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every operation cap is bounded by the same injected aggregate deadline."""
    receipt = acceptance.AcceptanceReceipt("run", "project", "synthetic", "owner", "image", "compose", 0)
    receipt.budget_ledger = {"operation_deadline": 107.0}
    monkeypatch.setattr(acceptance, "_ACTIVE_RECEIPT", receipt)
    monkeypatch.setattr(acceptance.time, "monotonic", lambda: 100.0)
    caps = {"compose": 120.0, "health": 3.0, "mcp": 15.0, "restart": 30.0, "memory_cleanup": 15.0, "publication": 60.0}
    admitted = {name: acceptance._active_operation_timeout(cap) for name, cap in caps.items()}
    assert admitted == {"compose": 7.0, "health": 3.0, "mcp": 7.0, "restart": 7.0, "memory_cleanup": 7.0, "publication": 7.0}
    monkeypatch.setattr(acceptance.time, "monotonic", lambda: 107.0)
    with pytest.raises(acceptance.AcceptanceTimeout):
        acceptance._active_operation_timeout(1.0)


def test_compose_admits_after_checkpoint_and_refuses_exhausted_slot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Compose must not launch when its phase checkpoint consumes the budget."""
    now = [100.0]
    receipt = acceptance.AcceptanceReceipt("run", "project", "synthetic", "owner", "image", "compose", 0)
    receipt.budget_ledger = {"operation_deadline": 107.0}
    checkpoint = tmp_path / "checkpoint.json"
    calls: list[float] = []

    def checkpoint_write(_path, _receipt):
        now[0] = 107.0

    monkeypatch.setattr(acceptance, "_ACTIVE_RECEIPT", receipt)
    monkeypatch.setattr(acceptance, "_ACTIVE_CHECKPOINT", checkpoint)
    monkeypatch.setattr(acceptance.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(acceptance, "_write_checkpoint", checkpoint_write)
    monkeypatch.setattr(
        acceptance,
        "_run_compose_process",
        lambda *args, **kwargs: calls.append(args[5]) or subprocess.CompletedProcess(args, 0),
    )

    with pytest.raises(acceptance.AcceptanceTimeout, match="deadline exhausted"):
        acceptance._run_compose(["docker", "compose"], Path("compose.yml"), "project", ["up", "-d"], {}, 120.0)
    assert calls == []


def test_compose_cap_is_recomputed_at_post_checkpoint_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The process primitive receives the current, not pre-checkpoint, allowance."""
    now = [100.0]
    receipt = acceptance.AcceptanceReceipt("run", "project", "synthetic", "owner", "image", "compose", 0)
    receipt.budget_ledger = {"operation_deadline": 107.0}
    checkpoint = tmp_path / "checkpoint.json"
    calls: list[float] = []
    monkeypatch.setattr(acceptance, "_ACTIVE_RECEIPT", receipt)
    monkeypatch.setattr(acceptance, "_ACTIVE_CHECKPOINT", checkpoint)
    monkeypatch.setattr(acceptance.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(acceptance, "_write_checkpoint", lambda *_: now.__setitem__(0, 103.0))
    monkeypatch.setattr(
        acceptance,
        "_run_compose_process",
        lambda *args, **kwargs: calls.append(args[5]) or subprocess.CompletedProcess(args, 0),
    )

    acceptance._run_compose(["docker", "compose"], Path("compose.yml"), "project", ["up", "-d"], {}, 120.0)
    assert calls == [4.0]


def test_negative_auth_admits_after_checkpoint_and_uses_remaining_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The negative probe must use the selected remaining deadline, not args.timeout."""
    now = [100.0]
    receipt = acceptance.AcceptanceReceipt("run", "project", "synthetic", "owner", "image", "compose", 0)
    receipt.budget_ledger = {"operation_deadline": 107.0}
    checkpoint = tmp_path / "checkpoint.json"
    observed: list[float] = []

    def fake_urlopen(_request, *, timeout):
        observed.append(timeout)
        raise urllib.error.HTTPError("http://localhost/mcp", 401, "unauthorized", {}, io.BytesIO(b""))

    monkeypatch.setattr(acceptance, "_ACTIVE_RECEIPT", receipt)
    monkeypatch.setattr(acceptance, "_ACTIVE_CHECKPOINT", checkpoint)
    monkeypatch.setattr(acceptance.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(acceptance, "_write_checkpoint", lambda *_: now.__setitem__(0, 103.0))
    monkeypatch.setattr(acceptance.urllib.request, "urlopen", fake_urlopen)

    acceptance._phase_http(
        receipt, checkpoint, "http_negative_auth", lambda: acceptance._negative_auth_probe("http://localhost", 120.0)
    )
    assert observed == [4.0]


def test_negative_auth_refuses_after_checkpoint_exhaustion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An exhausted negative-auth phase must not invoke urllib at all."""
    now = [100.0]
    receipt = acceptance.AcceptanceReceipt("run", "project", "synthetic", "owner", "image", "compose", 0)
    receipt.budget_ledger = {"operation_deadline": 107.0}
    checkpoint = tmp_path / "checkpoint.json"
    calls: list[float] = []

    monkeypatch.setattr(acceptance, "_ACTIVE_RECEIPT", receipt)
    monkeypatch.setattr(acceptance, "_ACTIVE_CHECKPOINT", checkpoint)
    monkeypatch.setattr(acceptance.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(acceptance, "_write_checkpoint", lambda *_: now.__setitem__(0, 107.0))
    monkeypatch.setattr(
        acceptance.urllib.request,
        "urlopen",
        lambda *args, **kwargs: calls.append(kwargs["timeout"]),
    )

    with pytest.raises(acceptance.AcceptanceTimeout, match="deadline exhausted"):
        acceptance._phase_http(
            receipt, checkpoint, "http_negative_auth", lambda: acceptance._negative_auth_probe("http://localhost", 120.0)
        )
    assert calls == []


class _ChunkedResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = iter(chunks)

    def read(self, size: int) -> bytes:
        assert size == acceptance.HTTP_READ_CHUNK_SIZE
        return next(self._chunks)


class _RecordingResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = iter(chunks)
        self.read_sizes: list[int] = []

    def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        return next(self._chunks)

    def close(self) -> None:
        """Provide the close hook expected by urllib.error.HTTPError cleanup."""



def test_http_error_body_cap_stops_chunked_read_and_preserves_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An oversized HTTP error body stops at the output cap without draining."""
    response = _RecordingResponse(
        [b"x" * acceptance.HTTP_READ_CHUNK_SIZE] * 8 + [b"suffix-never-read"]
    )
    error = urllib.error.HTTPError("http://example.test/mcp", 503, "overloaded", {}, response)

    def fake_urlopen(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(acceptance.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(acceptance.AcceptanceFailure) as raised:
        acceptance._http_json("http://example.test/mcp", {}, {}, 120.0)

    assert raised.value.__cause__ is error
    assert error.code == 503
    assert "HTTP 503" in str(raised.value)
    assert "[TRUNCATED]" in str(raised.value)
    assert "suffix-never-read" not in str(raised.value)
    assert response.read_sizes == [acceptance.HTTP_READ_CHUNK_SIZE] * 8


def test_http_error_truncation_sanitizes_secret_at_cap_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registered secret split by the raw cap cannot leak its retained prefix."""
    secret = "boundary-secret"
    response = _RecordingResponse([b"prefix-" + secret.encode("ascii") + b"-suffix-never-read"])
    error = urllib.error.HTTPError("http://example.test/mcp", 500, "failed", {}, response)
    monkeypatch.setattr(acceptance, "HTTP_ERROR_BODY_MAX_BYTES", 12)
    monkeypatch.setattr(acceptance, "_REDACTION_VALUES", {secret})
    monkeypatch.setattr(acceptance.urllib.request, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(error))

    with pytest.raises(acceptance.AcceptanceFailure) as raised:
        acceptance._http_json("http://example.test/mcp", {}, {}, 120.0)

    message = str(raised.value)
    assert "boundary-secret" not in message
    assert "bound" not in message
    assert "suffix-never-read" not in message
    assert "[TRUNCATED]" in message
    assert response.read_sizes == [12]


def test_http_error_body_eof_checks_deadline_before_return(monkeypatch: pytest.MonkeyPatch) -> None:
    """EOF is not accepted as a completed diagnostic after deadline exhaustion."""
    receipt = acceptance.AcceptanceReceipt("run", "project", "synthetic", "owner", "image", "compose", 0)
    receipt.budget_ledger = {"operation_deadline": 105.0}
    response = _RecordingResponse([b"partial", b""])
    clock = iter([100.0, 101.0, 105.0])
    monkeypatch.setattr(acceptance, "_ACTIVE_RECEIPT", receipt)
    monkeypatch.setattr(acceptance.time, "monotonic", lambda: next(clock))

    with pytest.raises(acceptance.AcceptanceTimeout, match="deadline exhausted"):
        acceptance._read_http_error_body(response, 120.0)

    assert response.read_sizes == [acceptance.HTTP_READ_CHUNK_SIZE] * 2


def test_http_error_body_deadline_refuses_next_chunk_without_drain(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deadline at the next read boundary prevents further error-body reads."""
    receipt = acceptance.AcceptanceReceipt("run", "project", "synthetic", "owner", "image", "compose", 0)
    receipt.budget_ledger = {"operation_deadline": 105.0}
    response = _RecordingResponse([b"partial", b"suffix-never-read"])
    clock = iter([100.0, 105.0])
    monkeypatch.setattr(acceptance, "_ACTIVE_RECEIPT", receipt)
    monkeypatch.setattr(acceptance.time, "monotonic", lambda: next(clock))

    with pytest.raises(acceptance.AcceptanceTimeout, match="deadline exhausted"):
        acceptance._read_http_error_body(response, 120.0)

    assert response.read_sizes == [acceptance.HTTP_READ_CHUNK_SIZE]


def test_http_body_completion_is_admitted_before_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    """A body completing at the aggregate cutoff cannot become a payload."""
    receipt = acceptance.AcceptanceReceipt("run", "project", "synthetic", "owner", "image", "compose", 0)
    receipt.budget_ledger = {"operation_deadline": 105.0}
    clock = iter([100.0, 101.0, 102.0, 105.0])
    decoded: list[bytes] = []
    response = _ChunkedResponse([b'{"ok":', b"true}", b""])
    monkeypatch.setattr(acceptance, "_ACTIVE_RECEIPT", receipt)
    monkeypatch.setattr(acceptance.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(acceptance, "_decode_http_payload", lambda body: decoded.append(body) or {"ok": True})

    with pytest.raises(acceptance.AcceptanceTimeout, match="deadline exhausted"):
        acceptance._read_http_body(response, 120.0)
    assert decoded == []


def test_http_body_chunks_complete_before_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bounded body that completes with allowance remaining is returned."""
    receipt = acceptance.AcceptanceReceipt("run", "project", "synthetic", "owner", "image", "compose", 0)
    receipt.budget_ledger = {"operation_deadline": 105.0}
    clock = iter([100.0, 101.0, 102.0, 103.0])
    monkeypatch.setattr(acceptance, "_ACTIVE_RECEIPT", receipt)
    monkeypatch.setattr(acceptance.time, "monotonic", lambda: next(clock))

    assert acceptance._read_http_body(_ChunkedResponse([b"first", b"second", b""]), 120.0) == b"firstsecond"


def test_cleanup_grace_is_bound_to_actual_process_primitive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configured grace reaches the Compose process seam, not just receipt metadata."""
    calls: list[dict] = []
    monkeypatch.setattr(acceptance, "_run_compose", lambda *args, **kwargs: (calls.append(kwargs) or subprocess.CompletedProcess(args, 0, "", "")))
    receipt = acceptance.AcceptanceReceipt("run", "project", "synthetic", "owner", "image", "compose", 0, budget_ledger={"compose_cleanup_deadline": 200.0})
    monkeypatch.setattr(acceptance.time, "monotonic", lambda: 100.0)
    acceptance._cleanup_resources(["docker"], Path("compose.yml"), "project", {}, 20.0, None, [], receipt, cleanup_deadline=150.0, down_timeout=20.0, cleanup_kill_grace=4.5)
    assert calls and calls[0]["kill_grace"] == 4.5
    assert receipt.cleanup_status == "succeeded" and receipt.cleanup_rc == 0


def _signal_receipt() -> acceptance.AcceptanceReceipt:
    """Build the minimal receipt used by mocked cutoff tests."""
    return acceptance.AcceptanceReceipt("run", "project", "synthetic", "owner", "image", "compose", 0)


def _mock_signal_cutoff(monkeypatch: pytest.MonkeyPatch, pending: set[int] | None = None) -> list[set[int]]:
    """Mock POSIX cutoff primitives without changing this process's real mask."""
    mask_calls: list[set[int]] = []
    monkeypatch.setattr(acceptance, "_SIGNALS_FROZEN", False)
    monkeypatch.setattr(acceptance, "_PENDING_SIGNALS", [])
    monkeypatch.setattr(
        acceptance.signal,
        "pthread_sigmask",
        lambda _how, signals: mask_calls.append(set(signals)) or set(),
    )
    monkeypatch.setattr(acceptance.signal, "sigpending", lambda: pending or set())
    return mask_calls


def test_handler_then_freeze_records_one_observation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A finalization handler is queued and recorded only by the cutoff."""
    receipt = _signal_receipt()
    monkeypatch.setattr(acceptance, "_ACTIVE_RECEIPT", receipt)
    monkeypatch.setattr(acceptance, "_FINALIZING", True)
    _mock_signal_cutoff(monkeypatch)

    acceptance._interrupt_handler(acceptance.signal.SIGINT, None)
    assert receipt.signal_count == 0
    acceptance._freeze_cli_signals(receipt)
    assert receipt.first_signal == "SIGINT"
    assert receipt.signal_count == 1
    assert acceptance._PENDING_SIGNALS == []


def test_mixed_repeated_handler_observations_saturate_and_coalesce(monkeypatch: pytest.MonkeyPatch) -> None:
    """Queue storage and receipt counts saturate without claiming delivery count."""
    receipt = _signal_receipt()
    monkeypatch.setattr(acceptance, "_ACTIVE_RECEIPT", receipt)
    monkeypatch.setattr(acceptance, "_FINALIZING", True)
    _mock_signal_cutoff(monkeypatch)

    for signum in [acceptance.signal.SIGTERM, acceptance.signal.SIGINT, acceptance.signal.SIGTERM] * 8:
        acceptance._interrupt_handler(signum, None)
    assert len(acceptance._PENDING_SIGNALS) == acceptance.SIGNAL_COUNT_CAP
    acceptance._freeze_cli_signals(receipt)
    assert receipt.first_signal == "SIGTERM"
    assert receipt.signal_count == acceptance.SIGNAL_COUNT_CAP
    acceptance._freeze_cli_signals(receipt)
    assert receipt.signal_count == acceptance.SIGNAL_COUNT_CAP
    assert acceptance._PENDING_SIGNALS == []


def test_pending_posix_signal_is_merged_once_and_after_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    """Queued handler observations precede one coalesced POSIX pending signal."""
    receipt = _signal_receipt()
    monkeypatch.setattr(acceptance, "_ACTIVE_RECEIPT", receipt)
    monkeypatch.setattr(acceptance, "_FINALIZING", True)
    mask_calls = _mock_signal_cutoff(monkeypatch, {acceptance.signal.SIGTERM})

    acceptance._interrupt_handler(acceptance.signal.SIGINT, None)
    acceptance._freeze_cli_signals(receipt)
    acceptance._freeze_cli_signals(receipt)
    assert receipt.first_signal == "SIGINT"
    assert receipt.signal_count == 2
    assert len(mask_calls) == 1


def test_primary_failure_and_frozen_payload_are_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    """Signal cutoff never masks an existing primary failure or mutates later."""
    receipt = _signal_receipt()
    receipt.acceptance_rc = 1
    receipt.status = "failed"
    receipt.error = "primary failure"
    monkeypatch.setattr(acceptance, "_ACTIVE_RECEIPT", receipt)
    monkeypatch.setattr(acceptance, "_FINALIZING", True)
    _mock_signal_cutoff(monkeypatch)

    acceptance._interrupt_handler(acceptance.signal.SIGTERM, None)
    acceptance._freeze_cli_signals(receipt)
    frozen = (receipt.first_signal, receipt.signal_count, receipt.error, receipt.status)
    assert acceptance._select_return_code(receipt) == 1
    with pytest.raises(acceptance.AcceptanceInterrupted, match="SIGINT"):
        acceptance._interrupt_handler(acceptance.signal.SIGINT, None)
    assert (receipt.first_signal, receipt.signal_count, receipt.error, receipt.status) == frozen


@pytest.mark.parametrize(
    ("missing_api", "expected_error"),
    (
        ("pthread_sigmask", "POSIX pthread_sigmask is required for CLI acceptance"),
        ("sigpending", "POSIX sigpending is required for CLI acceptance"),
    ),
)
def test_cli_rejects_missing_signal_api_before_external_or_publication_work(
    missing_api: str,
    expected_error: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Missing POSIX signal APIs fail before handlers, workflow, or publication."""
    case_dir = tmp_path / f"missing-{missing_api}"
    case_dir.mkdir()
    receipt_path = case_dir / "receipt.json"
    status_path = case_dir / "status.txt"
    calls: list[str] = []

    with monkeypatch.context() as isolated:
        isolated.delattr(acceptance.signal, missing_api, raising=False)
        isolated.setattr(acceptance.signal, "getsignal", lambda *_: calls.append("getsignal"))
        isolated.setattr(acceptance.signal, "signal", lambda *_: calls.append("signal"))
        isolated.setattr(acceptance, "run_acceptance", lambda *_: calls.append("run_acceptance"))
        isolated.setattr(acceptance, "_freeze_cli_signals", lambda *_: calls.append("freeze"))
        isolated.setattr(acceptance, "_publish_terminal", lambda *_: calls.append("publish"))

        result = acceptance.main(["--receipt", str(receipt_path), "--status", str(status_path)])

    assert result == 2
    assert calls == []
    assert not receipt_path.exists()
    assert not status_path.exists()
    assert not (case_dir / ".receipt.json.checkpoint.json").exists()
    assert not (case_dir / ".receipt.json.complete.json").exists()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"FAIL: {expected_error}\n"


def test_library_run_is_mask_neutral_while_cli_cutoff_blocks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Library finalization does not mask; only the CLI cutoff invokes pthread_sigmask."""
    args = Namespace(
        image="image", compose_file=tmp_path / "compose.yml", receipt=tmp_path / "receipt.json",
        timeout=10.0, project_name="project", run_id="run", docker_executable=None,
        total_timeout=40.0, finalization_reserve=30.0, compose_cleanup_reserve=22.0,
        publication_reserve=1.0, compose_cleanup_timeout=1.0, cleanup_kill_grace=0.0,
    )
    mask_calls: list[object] = []
    monkeypatch.setattr(acceptance, "_budget_ledger", lambda *_: (_ for _ in ()).throw(acceptance.AcceptanceFailure("no budget")))
    monkeypatch.setattr(acceptance, "_write_checkpoint", lambda *_: None)
    monkeypatch.setattr(acceptance.signal, "pthread_sigmask", lambda *args: mask_calls.append(args))
    with pytest.raises(acceptance.AcceptanceFailure, match="no budget"):
        acceptance.run_acceptance(args)
    assert mask_calls == []

    receipt = _signal_receipt()
    monkeypatch.setattr(acceptance, "_SIGNALS_FROZEN", False)
    monkeypatch.setattr(acceptance, "_PENDING_SIGNALS", [])
    monkeypatch.setattr(acceptance.signal, "sigpending", lambda: set())
    acceptance._freeze_cli_signals(receipt)
    assert mask_calls and mask_calls[-1][0] == acceptance.signal.SIG_BLOCK
