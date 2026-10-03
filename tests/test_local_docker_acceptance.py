"""Host-only safety tests for the local Docker RC contract."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from weft.config import load_config

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "docker-compose.local.yml"
RUNNER = ROOT / "scripts/local_docker_acceptance.py"
_spec = importlib.util.spec_from_file_location("local_docker_acceptance", RUNNER)
assert _spec is not None and _spec.loader is not None
acceptance = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = acceptance
_spec.loader.exec_module(acceptance)


def test_local_compose_has_private_dependencies_and_loopback_app_port() -> None:
    document = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    services = document["services"]
    assert set(services) == {"postgres", "redis", "migrate", "bootstrap", "app"}
    assert "ports" not in services["postgres"]
    assert "ports" not in services["redis"]
    assert services["app"]["ports"] == ["127.0.0.1:${WEFT_LOCAL_PORT:-18000}:8000"]
    assert services["migrate"]["depends_on"]["postgres"]["condition"] == "service_healthy"
    assert services["bootstrap"]["depends_on"]["migrate"]["condition"] == "service_completed_successfully"
    assert services["app"]["depends_on"]["bootstrap"]["condition"] == "service_completed_successfully"
    assert services["app"]["depends_on"]["redis"]["condition"] == "service_healthy"
    migration_environment = services["migrate"]["environment"]
    assert migration_environment["WEFT_OWNER_DATABASE_URL"].endswith("@postgres:5432/weft")
    bootstrap_environment = services["bootstrap"]["environment"]
    assert bootstrap_environment["WEFT_DATABASE_URL"].endswith("@postgres:5432/weft")
    environment = services["app"]["environment"]
    assert environment["WEFT_DATABASE_URL"].startswith("postgresql://weft_app:")
    assert environment["WEFT_DATABASE_URL"].endswith("@postgres:5432/weft")
    assert "WEFT_OWNER_DATABASE_URL" not in environment
    assert environment["WEFT_REDIS_URL"] == "redis://redis:6379"
    assert environment["WEFT_MIGRATION_MODE"] == "verify"
    assert environment["WEFT_OUTBOUND_CONNECTOR"] == "none"
    assert environment["WEFT_QUARANTINE_REVIEW_ENABLED"] == "0"
    assert "WEFT_LOCAL_API_KEY:?" in environment["WEFT_API_KEY"]
    assert "WEFT_LOCAL_USER_ID:?" in environment["WEFT_DEFAULT_USER_ID"]


def test_local_compose_has_no_literal_secret_or_production_override() -> None:
    text = COMPOSE.read_text(encoding="utf-8")
    assert "weft_local" in text
    assert "WEFT_LOCAL_API_KEY:?" in text
    assert "WEFT_API_KEY: local" not in text
    assert "ANTHROPIC_API_KEY" not in text
    assert "WEFT_MIGRATION_MODE" not in (ROOT / "Dockerfile").read_text(encoding="utf-8")


def test_dockerignore_blocks_private_and_nested_workspace_inputs() -> None:
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    for entry in (".env", ".loom/", ".polytoken/", "artifacts/", ".ci-rc-worktree/", ".rc-candidate-worktree/"):
        assert entry in ignored


def test_acceptance_runner_is_stdlib_only_and_redacts_secrets() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    imports = {node.names[0].name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) and node.names}
    assert imports <= {"argparse", "dataclasses", "hashlib", "json", "math", "os", "re", "secrets", "selectors", "shutil", "signal", "socket", "subprocess", "sys", "time", "urllib", "uuid", "pathlib", "typing"}
    assert "ANTHROPIC_API_KEY" not in RUNNER.read_text(encoding="utf-8")
    receipt_source = RUNNER.read_text(encoding="utf-8")
    for marker in ('"cleanup_scope"', '"unique_resource_cleanup"', '"runtime_db_role_isolation"', '"runtime_privilege_contract"', '"future_default_acl_probe_setup"', "create-future-probes"):
        assert marker in receipt_source


def test_local_runtime_allowlist_is_explicit_and_unknowns_fail_closed() -> None:
    helper = (ROOT / "weft/local_bootstrap.py").read_text(encoding="utf-8")
    assert "GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.{table" not in helper
    assert "GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES" not in helper
    assert "REVOKE ALL ON ALL TABLES IN SCHEMA public" in helper
    assert "REVOKE ALL ON ALL SEQUENCES IN SCHEMA public" in helper
    for marker in ('"weft_tokens"', '"weft_metadata"', '"workspaces"', '"workspace_members"', '"calibration_records"', "d.deptype IN ('a', 'i')", '"oauth_clients"'):
        assert marker in helper


def test_quarantine_env_override_is_local_and_preserves_explicit_text_provider(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text('[text_generation]\nprovider = "openai"\n', encoding="utf-8")
    monkeypatch.setattr("weft.config.CONFIG_PATH", config_path)
    monkeypatch.setenv("WEFT_QUARANTINE_REVIEW_ENABLED", "0")
    monkeypatch.delenv("WEFT_ENV", raising=False)
    assert load_config().quarantine_review.enabled is False
    monkeypatch.setenv("WEFT_QUARANTINE_REVIEW_ENABLED", "1")
    assert load_config().quarantine_review.enabled is True


def test_receipt_schema_is_json_and_never_mentions_bearer_fields() -> None:
    encoded = json.dumps({"schema": "weft.local-docker-acceptance.v1", "status": "failed", "checks": {}, "warnings": [], "cleanup_scope": {"compose_project": "weft-rc-example"}})
    assert json.loads(encoded)["schema"].endswith(".v1")
    assert "Authorization" not in encoded and "WEFT_LOCAL_API_KEY" not in encoded


def test_recall_rejects_echoed_query_without_actual_evidence() -> None:
    with pytest.raises(acceptance.AcceptanceFailure, match="omitted tracked memory"):
        acceptance._assert_recall({"query": "sentinel", "results": []}, "weft-memory", "sentinel", "local-project", "recall")


def test_prime_rejects_empty_response_without_authoritative_handoff() -> None:
    with pytest.raises(acceptance.AcceptanceFailure, match="omitted authoritative handoff"):
        acceptance._assert_prime_handoff({"handoff": []}, "weft-handoff", "Acceptance handoff", "local-project", "prime")


def test_memory_assertion_rejects_missing_project_ownership_evidence() -> None:
    with pytest.raises(acceptance.AcceptanceFailure, match="synthetic project scope"):
        acceptance._assert_memory({"id": "weft-memory", "content": "sentinel"}, "sentinel", "local-project")


def test_cleanup_failure_forces_failed_receipt(monkeypatch: pytest.MonkeyPatch) -> None:
    receipt = acceptance.AcceptanceReceipt(
        "cleanup-test", "weft-rc-cleanup-test", "local-project", "owner", "image",
        "docker-compose.local.yml", 0.0, status="passed",
        budget_ledger={"compose_cleanup_deadline": 10.0},
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(acceptance.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(acceptance, "_mcp_initialize", lambda *a, **k: (_ for _ in ()).throw(acceptance.AcceptanceFailure("app unavailable")))
    def fail_down(prefix, compose_file, project_name, args, env, timeout, **kwargs):
        calls.append(args); return subprocess.CompletedProcess(args, 1, "", "down failed")
    monkeypatch.setattr(acceptance, "_run_compose", fail_down)
    acceptance._cleanup_resources(["docker", "compose"], Path("docker-compose.local.yml"), receipt.project_name, {"WEFT_LOCAL_PORT": "18000"}, 1.0, "bearer", [], receipt)
    assert receipt.status == "failed" and receipt.warnings and "unique_resource_cleanup" not in receipt.checks
    assert receipt.cleanup_status == "failed" and receipt.cleanup_rc == 1
    assert calls == [["down", "--volumes", "--remove-orphans"]]


def test_output_redaction_handles_split_chunks_and_caps() -> None:
    redactor = acceptance._OutputRedactor()
    assert redactor.feed(b"Authorization: Bearer SPLIT_SECRET") == ""
    output = redactor.feed(b"x " + b"A" * (acceptance.MAX_OUTPUT_BYTES + 10)) + redactor.finish()
    assert "SPLIT_SECRET" not in output and "[REDACTED]" in output


def test_run_compose_kills_process_group_and_does_not_wait_for_held_pipe(tmp_path: Path) -> None:
    child_pid = tmp_path / "child.pid"
    fixture = tmp_path / "fake-compose.py"
    fixture.write_text(textwrap.dedent(f"""
        import pathlib, subprocess, sys, time
        child = subprocess.Popen([sys.executable, '-c', "import time; time.sleep(30)"])
        pathlib.Path({str(child_pid)!r}).write_text(str(child.pid))
        print('WEFT_LOCAL_API_KEY=split-secret', flush=True)
        time.sleep(30)
    """), encoding="utf-8")
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired) as exc:
        acceptance._run_compose_process([sys.executable, str(fixture)], Path("compose.yml"), "isolated", [], os.environ.copy(), 0.1)
    assert time.monotonic() - started < 5 and "secret" not in str(exc.value)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not child_pid.exists(): time.sleep(0.01)
    assert child_pid.exists()
    # os.killpg is asynchronous: the signaled descendant must still be
    # scheduled to die and, as an orphan, reaped before its PID stops
    # answering kill(pid, 0). A single-shot check races that window, so poll
    # for actual exit within a bounded window; passing still requires the
    # descendant to be provably gone.
    descendant = int(child_pid.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 10
    while True:
        try:
            os.kill(descendant, 0)
        except ProcessLookupError:
            break
        if time.monotonic() >= deadline:
            pytest.fail(f"same-session descendant {descendant} survived group cancellation")
        time.sleep(0.05)


def test_checkpoint_is_atomic_and_phase_failure_is_durable(tmp_path: Path) -> None:
    receipt = acceptance.AcceptanceReceipt("r", "p", "s", "o", "i", "c", 0.0)
    checkpoint = tmp_path / "checkpoint.json"
    with pytest.raises(RuntimeError, match="boom"):
        acceptance._phase(receipt, checkpoint, "fault", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert saved["phases"][-1]["status"] == "failed" and saved["phases"][-1]["error"] == "boom"


def test_cleanup_failure_does_not_mask_primary_error(monkeypatch: pytest.MonkeyPatch) -> None:
    receipt = acceptance.AcceptanceReceipt("r", "p", "s", "o", "i", "c", 0.0, error="primary")
    monkeypatch.setattr(acceptance, "_mcp_initialize", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("mcp cleanup")))
    monkeypatch.setattr(acceptance, "_run_compose", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down cleanup")))
    acceptance._cleanup_resources(["docker"], Path("compose.yml"), "p", {"WEFT_LOCAL_PORT": "1"}, 0.1, "secret", [], receipt)
    assert receipt.error == "primary" and receipt.status == "failed"


def test_signal_handler_is_interrupt_exception() -> None:
    with pytest.raises(acceptance.AcceptanceInterrupted, match="SIGTERM"): acceptance._interrupt_handler(signal.SIGTERM, None)
    with pytest.raises(acceptance.AcceptanceInterrupted, match="SIGINT"): acceptance._interrupt_handler(signal.SIGINT, None)


def test_receipt_exposes_phase_verifier_and_checkpoint_fields() -> None:
    receipt = acceptance.AcceptanceReceipt("r", "p", "s", "o", "i", "c", 0.0)
    assert receipt.to_dict()["phases"] == [] and receipt.to_dict()["verifier"] == {}


def test_partial_startup_still_attempts_unique_project_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    def compose(prefix, compose_file, project_name, args, env, timeout, **kwargs):
        calls.append(args)
        if args == ["up", "-d"]: raise subprocess.TimeoutExpired(args, timeout)
        return subprocess.CompletedProcess(args, 0, "", "")
    monkeypatch.setattr(acceptance, "_docker_compose_prefix", lambda: ["docker", "compose"])
    monkeypatch.setattr(acceptance, "_run_compose", compose)
    monkeypatch.setattr(acceptance, "_mcp_initialize", lambda *a, **k: ("session", {}))
    receipt = acceptance.run_acceptance(Namespace(image="image", compose_file=Path("docker-compose.local.yml"), receipt=Path("receipt.json"), timeout=1.0))
    assert receipt.status == "failed" and calls == [["up", "-d"], ["down", "--volumes", "--remove-orphans"]]
    assert receipt.cleanup_status == "succeeded"
    assert receipt.checks["unique_resource_cleanup"] == "passed"


def test_runner_accepts_wrapper_project_identity_and_timeout_validation() -> None:
    parsed = acceptance.parse_args(["--project-name", "weft-rc-fixed", "--timeout", "0.25"])
    assert parsed.project_name == "weft-rc-fixed" and parsed.timeout == 0.25
    for value in ("0", "nan", "inf", "-1"):
        with pytest.raises(SystemExit): acceptance.parse_args(["--timeout", value])
    with pytest.raises(SystemExit): acceptance.parse_args(["--image", "../../danger"])


def test_final_wrapper_is_bounded_scoped_and_never_deletes_image() -> None:
    launcher = (ROOT / "scripts/local_docker_acceptance_round2.sh").read_text(encoding="utf-8")
    assert 'exec env -i' in launcher
    assert 'RUNNER="$SOURCE_DIR/scripts/local_docker_acceptance.py"' in launcher
    assert '--project-name "$PROJECT"' in launcher
    assert 'PYTHON="${WEFT_ACCEPTANCE_PYTHON:-$(command -v python3 || true)}"' in launcher
    assert 'DOCKER="${WEFT_ACCEPTANCE_DOCKER:-$(command -v docker || true)}"' in launcher
    assert '[[ "$DOCKER" = /* && -f "$DOCKER" && -x "$DOCKER" ]]' in launcher
    assert 'PATH="$DOCKER_DIR"' in launcher
    assert '"$PYTHON" "$RUNNER"' in launcher
    assert 'trap ' not in launcher and 'cleanup()' not in launcher
    assert 'start_new_session=True' not in launcher and 'os.killpg(process.pid' not in launcher
    assert "docker image rm" not in launcher and "docker system prune" not in launcher
    assert "docker container rm -f" not in launcher


def test_wrapper_delegates_to_runner_and_propagates_runner_status(tmp_path: Path) -> None:
    """Round2 forwards one scoped invocation; the runner owns lifecycle cleanup."""
    job = tmp_path / "job"
    source = tmp_path / "source"
    source.mkdir()
    (source / "docker-compose.local.yml").write_text("services: {}\n", encoding="utf-8")
    (source / "scripts").mkdir()
    argv_path = tmp_path / "runner-argv.json"
    (source / "scripts/local_docker_acceptance.py").write_text(textwrap.dedent(f"""\
        import json
        import sys
        from pathlib import Path
        Path({str(argv_path)!r}).write_text(json.dumps(sys.argv[1:]), encoding="utf-8")
        raise SystemExit(23)
    """), encoding="utf-8")
    fake = tmp_path / "docker"
    fake.write_text(f"#!{sys.executable}\nraise SystemExit(0)\n", encoding="utf-8")
    fake.chmod(0o755)
    env = {
        "PATH": "/usr/bin:/bin",
        "WEFT_ACCEPTANCE_PYTHON": sys.executable,
        "WEFT_ACCEPTANCE_DOCKER": str(fake),
        "WEFT_ACCEPTANCE_JOB_DIR": str(job),
        "WEFT_ACCEPTANCE_SOURCE_DIR": str(source),
        "WEFT_ACCEPTANCE_IMAGE": "safe:tag",
        "WEFT_ACCEPTANCE_TIMEOUT": "0.2",
    }
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/local_docker_acceptance_round2.sh")],
        env=env, capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 23, result.stderr + result.stdout
    forwarded = json.loads(argv_path.read_text(encoding="utf-8"))
    assert forwarded[forwarded.index("--image") + 1] == "safe:tag"
    assert forwarded[forwarded.index("--compose-file") + 1] == str(source / "docker-compose.local.yml")
    project = forwarded[forwarded.index("--project-name") + 1]
    assert project.startswith("weft-rc-")
    assert forwarded[forwarded.index("--receipt") + 1] == str(job / "receipt.json")
    assert not (job / "cleanup.log").exists()


def test_final_wrapper_does_not_own_runner_cleanup_or_signal_state() -> None:
    launcher = (ROOT / "scripts/local_docker_acceptance_round2.sh").read_text(encoding="utf-8")
    assert "runner_rc=$?" not in launcher
    assert "signal_status" not in launcher
    assert "signal_exit_code" not in launcher
    assert "trap " not in launcher
    assert "cleanup()" not in launcher
    assert "JOB_DIR/.cleanup-signal" not in launcher
