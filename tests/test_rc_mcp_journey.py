"""Provider-free contract tests for the RC-FL-13 MCP persistence journey.

These tests exercise the production acceptance runner's MCP and cleanup seams
with faithful structured responses.  They never start Docker, HTTP, Postgres,
Redis, providers, or hosted services.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/local_docker_acceptance.py"
_spec = importlib.util.spec_from_file_location("local_docker_acceptance_rc13", RUNNER)
assert _spec and _spec.loader
acceptance = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = acceptance
_spec.loader.exec_module(acceptance)


PROJECT = "local-docker-rc-run"
OLD_ID = "weft-preference-old"
NEW_ID = "weft-preference-new"
OLD_CONTENT = "RC13 prefers concise evidence"
NEW_CONTENT = "RC13 prefers verified concise evidence"
PREFERENCE_METADATA = {
    "schema_version": 1,
    "polarity": "positive",
    "strength": "hard",
    "subject": "local acceptance",
    "value": "concise evidence",
    "context": ["rc", "mcp-journey"],
}


def _entry(memory_id: str, content: str, *, status: str = "active", memory_type: str = "preference") -> dict:
    return {
        "id": memory_id,
        "content": content,
        "type": memory_type,
        "status": status,
        "project_id": PROJECT,
        "preference_metadata": PREFERENCE_METADATA if memory_type == "preference" else None,
    }


def test_preference_and_revision_assertions_reject_incomplete_or_wrong_state() -> None:
    with pytest.raises(acceptance.AcceptanceFailure, match="metadata"):
        acceptance._assert_preference(
            _entry(OLD_ID, OLD_CONTENT) | {"preference_metadata": None},
            OLD_CONTENT,
            PREFERENCE_METADATA,
            PROJECT,
        )

    with pytest.raises(acceptance.AcceptanceFailure, match="predecessor"):
        acceptance._assert_revision(
            {"new": _entry(NEW_ID, NEW_CONTENT), "superseded": _entry("wrong", OLD_CONTENT, status="archived")},
            OLD_ID,
            OLD_CONTENT,
            NEW_CONTENT,
            PROJECT,
            PREFERENCE_METADATA,
        )

    with pytest.raises(acceptance.AcceptanceFailure, match="metadata"):
        acceptance._assert_revision(
            {
                "new": _entry(NEW_ID, NEW_CONTENT),
                "superseded": _entry(OLD_ID, OLD_CONTENT, status="archived") | {"preference_metadata": None},
            },
            OLD_ID,
            OLD_CONTENT,
            NEW_CONTENT,
            PROJECT,
            PREFERENCE_METADATA,
        )

    with pytest.raises(acceptance.AcceptanceFailure, match="relationships"):
        acceptance._assert_revision_lineage({}, NEW_ID, OLD_ID, "lineage")

    with pytest.raises(acceptance.AcceptanceFailure, match="omitted successor lineage"):
        acceptance._assert_revision_lineage(
            {"relationships": [{"source_id": OLD_ID, "target_id": NEW_ID, "relation": "supersedes"}]},
            NEW_ID,
            OLD_ID,
            "lineage",
        )


def test_recall_negative_assertions_reject_superseded_deleted_and_boundary_leaks() -> None:
    with pytest.raises(acceptance.AcceptanceFailure, match="exposed isolated memory"):
        acceptance._assert_not_recalled(
            {"results": [{"id": OLD_ID, "content": "unrelated"}]},
            OLD_ID,
            OLD_CONTENT,
            "old-version",
        )
    with pytest.raises(acceptance.AcceptanceFailure, match="exposed isolated memory"):
        acceptance._assert_not_recalled(
            {"results": [{"id": "other", "content": NEW_CONTENT}]},
            NEW_ID,
            NEW_CONTENT,
            "deleted-version",
        )
    with pytest.raises(acceptance.AcceptanceFailure, match="exposed isolated memory"):
        acceptance._assert_not_recalled(
            {"results": [{"id": "other", "content": OLD_CONTENT}]},
            OLD_ID,
            OLD_CONTENT,
            "wrong-owner",
        )

    with pytest.raises(acceptance.AcceptanceFailure, match="did not delete"):
        acceptance._assert_deleted({"memory_id": NEW_ID, "deleted": False, "hard": True}, NEW_ID)


def _args(tmp_path: Path) -> Namespace:
    return Namespace(
        image="safe:tag",
        compose_file=tmp_path / "docker-compose.local.yml",
        receipt=tmp_path / "receipt.json",
        timeout=1.0,
        project_name="weft-rc-test",
        run_id="run",
        docker_executable=None,
        total_timeout=2400.0,
        finalization_reserve=240.0,
        compose_cleanup_reserve=120.0,
        publication_reserve=60.0,
        compose_cleanup_timeout=90.0,
        cleanup_kill_grace=1.0,
    )


def _install_workflow_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    broken_lineage: bool = False,
    reuse_post_restart_session: bool = False,
    initialize_before_restart: bool = False,
    events: list[tuple[str, str]] | None = None,
):
    compose_calls: list[list[str]] = []
    tool_calls: list[tuple[str, dict]] = []
    forget_calls: list[str] = []
    records: dict[str, dict] = {}
    handoff: dict | None = None
    owner_bearer: str | None = None
    restart_count = 0
    owner_initialize_count = 0
    cleanup_ready = False
    owner_sessions = iter(("session-1", "session-2", "cleanup-session"))

    def compose(_prefix, _compose_file, _project, operation, _env, _timeout, **_kwargs):
        nonlocal restart_count
        compose_calls.append(list(operation))
        if operation and operation[0] == "restart":
            restart_count += 1
            if events is not None:
                events.append(("restart", operation[1]))
            if initialize_before_restart and operation == ["restart", "postgres"]:
                # Simulate the orchestration bug this contract must reject:
                # initialize during the first restart, before redis and app.
                initialize("url", owner_bearer, 1.0)
        return subprocess.CompletedProcess(operation, 0, "", "")

    def initialize(_base_url, bearer, _timeout):
        nonlocal owner_bearer, owner_initialize_count
        if bearer == "distinct-owner-token":
            return "distinct-owner-session", {"MCP-Session-Id": "distinct-owner-session"}
        if owner_bearer is None:
            owner_bearer = bearer
        if bearer != owner_bearer:
            raise AssertionError("unknown bearer used for MCP initialize")
        owner_initialize_count += 1
        if events is not None:
            events.append(("initialize", f"owner-{owner_initialize_count}"))
        if owner_initialize_count == 2 and restart_count != 3 and not cleanup_ready:
            raise AssertionError("new owner session was initialized without all service restarts")
        if owner_initialize_count > 2 and restart_count != 3 and not cleanup_ready:
            raise AssertionError("cleanup session initialized before restart workflow completed")
        session_id = "session-1" if reuse_post_restart_session and owner_initialize_count == 2 else next(owner_sessions)
        return session_id, {"MCP-Session-Id": session_id}

    def stored(memory_id: str, content: str, bearer: str, *, status: str = "active", memory_type: str = "preference") -> dict:
        value = _entry(memory_id, content, status=status, memory_type=memory_type)
        value["owner"] = bearer
        return value

    def call(_base_url, _headers, bearer, tool, arguments, _request_id, _timeout):
        nonlocal handoff, cleanup_ready
        tool_calls.append((tool, dict(arguments)))
        if tool == "weft_token_issue":
            if bearer != owner_bearer:
                raise AssertionError("only the tracked owner may issue an isolation token")
            return {"token": "distinct-owner-token"}
        if tool == "weft_remember":
            memory_id = "weft-fact" if arguments.get("type") == "fact" else OLD_ID
            memory_type = arguments.get("type", "preference")
            value = stored(memory_id, arguments["content"], bearer, memory_type=memory_type)
            value["preference_metadata"] = arguments.get("preference_metadata")
            records[memory_id] = value
            return dict(value)
        if tool == "weft_recall":
            if bearer != owner_bearer:
                return {"results": []}
            query = arguments["query"]
            matches = [record for record in records.values() if not record.get("deleted")]
            matches = [record for record in matches if record.get("owner") == bearer and record.get("content") == query]
            if arguments.get("retrieval_mode") == "code":
                return {"results": []}
            return {"results": [dict(record) for record in matches if record.get("status") == "active"]}
        if tool == "weft_revise":
            old = records.get(arguments["memory_id"])
            if bearer != owner_bearer or old is None or old.get("owner") != bearer:
                raise AssertionError("revision attempted by an untracked owner")
            old["status"] = "archived"
            successor = stored(NEW_ID, arguments["new_content"], bearer)
            successor["preference_metadata"] = old.get("preference_metadata")
            records[NEW_ID] = successor
            return {"new": dict(successor), "superseded": dict(old)}
        if tool == "weft_relate":
            if broken_lineage:
                cleanup_ready = True
                return {"relationships": []}
            if bearer != owner_bearer or NEW_ID not in records or OLD_ID not in records:
                return {"relationships": []}
            return {"relationships": [{"source_id": NEW_ID, "target_id": OLD_ID, "relation": "supersedes"}]}
        if tool == "weft_forget":
            memory_id = arguments["memory_id"]
            record = records.get(memory_id)
            if bearer != owner_bearer or record is None or record.get("owner") != bearer or record.get("deleted"):
                return {"memory_id": memory_id, "deleted": False, "hard": arguments.get("hard") is True}
            record["deleted"] = True
            forget_calls.append(memory_id)
            return {"memory_id": memory_id, "deleted": True, "hard": arguments.get("hard") is True}
        if tool == "weft_handoff":
            if bearer != owner_bearer:
                return {"id": "weft-handoff", "stored": False}
            handoff = {"id": "weft-handoff", "stored": True, "owner": bearer, "summary": arguments["summary"]}
            return dict(handoff)
        if tool == "weft_prime":
            if bearer != owner_bearer or arguments.get("project_id") != PROJECT or handoff is None:
                return {"handoff": []}
            return {"handoff": [{"id": handoff["id"], "content": "Summary: " + handoff["summary"]}]}
        raise AssertionError(f"unexpected tool call: {tool}: {arguments}")

    monkeypatch.setattr(acceptance, "_docker_compose_prefix", lambda: ["docker", "compose"])
    monkeypatch.setattr(acceptance, "_run_compose", compose)
    monkeypatch.setattr(acceptance, "_mcp_initialize", initialize)
    monkeypatch.setattr(acceptance, "_mcp_call_checkpointed", call)
    monkeypatch.setattr(acceptance, "_wait_for_health", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(acceptance, "_negative_auth_probe", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(acceptance, "_parse_runtime_verifier_output", lambda *_args: {"status": "passed"})
    monkeypatch.setattr(acceptance, "_write_checkpoint", lambda *_args: None)
    monkeypatch.setattr(acceptance, "_free_port", lambda: 18001)
    return compose_calls, tool_calls, forget_calls


def test_full_mocked_rc13_journey_proves_calls_and_unique_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    compose_calls, tool_calls, forget_calls = _install_workflow_fakes(monkeypatch)
    receipt = acceptance.run_acceptance(_args(tmp_path))

    assert receipt.status == "passed", receipt.error
    assert receipt.checks["mcp_preference_capture"] == "passed"
    assert receipt.checks["mcp_revision_supersession"] == "passed"
    assert receipt.checks["mcp_old_version_excluded"] == "passed"
    assert receipt.checks["mcp_deletion_exclusion"] == "passed"
    assert receipt.checks["process_replacement_new_mcp_session"] == "passed"
    assert receipt.checks["memory_handoff_persistence"] == "passed"
    assert compose_calls.count(["down", "--volumes", "--remove-orphans"]) == 1
    # NEW_ID is deleted by the journey, while the fact, handoff, and archived
    # predecessor each receive exactly one normal cleanup attempt.
    assert forget_calls.count(NEW_ID) == 1
    assert forget_calls.count(OLD_ID) == 1
    assert len(forget_calls) == len(set(forget_calls))
    assert any(tool == "weft_remember" and args.get("type") == "preference" for tool, args in tool_calls)
    assert any(tool == "weft_revise" and args.get("memory_id") == OLD_ID for tool, args in tool_calls)
    assert any(tool == "weft_relate" and args.get("relation") == "supersedes" for tool, args in tool_calls)
    assert any(tool == "weft_forget" and args.get("hard") is True for tool, args in tool_calls)


def test_broken_lineage_fails_closed_and_still_cleans_registered_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    compose_calls, _tool_calls, forget_calls = _install_workflow_fakes(monkeypatch, broken_lineage=True)
    receipt = acceptance.run_acceptance(_args(tmp_path))

    assert receipt.status == "failed"
    assert "revision_lineage" in str(receipt.error) or "lineage" in str(receipt.error)
    assert compose_calls.count(["down", "--volumes", "--remove-orphans"]) == 1
    # The successor and predecessor were registered before lineage validation.
    assert NEW_ID in forget_calls and OLD_ID in forget_calls
    assert receipt.cleanup_status == "succeeded"


def test_cleanup_rejects_structured_deleted_false_and_still_tears_down(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    compose_calls: list[list[str]] = []
    forget_calls: list[str] = []
    receipt = acceptance.AcceptanceReceipt("run", "project", "synthetic", "owner", "image", "compose", 0.0)

    def compose(_prefix, _compose_file, _project, operation, _env, _timeout, **_kwargs):
        compose_calls.append(list(operation))
        return subprocess.CompletedProcess(operation, 0, "", "")

    monkeypatch.setattr(acceptance, "_mcp_initialize", lambda *_args, **_kwargs: ("cleanup", {}))

    def forget(_base_url, _headers, _bearer, _tool, arguments, _request_id, _timeout):
        forget_calls.append(arguments["memory_id"])
        return {"memory_id": arguments["memory_id"], "deleted": False, "hard": True}

    monkeypatch.setattr(acceptance, "_mcp_call_checkpointed", forget)
    monkeypatch.setattr(acceptance, "_run_compose_process", compose)
    acceptance._cleanup_resources(
        ["docker", "compose"],
        tmp_path / "compose.yml",
        "project",
        {"WEFT_LOCAL_PORT": "18001"},
        1.0,
        "owner-token",
        [NEW_ID],
        receipt,
        down_timeout=10.0,
    )

    assert forget_calls == [NEW_ID]
    assert compose_calls == [["down", "--volumes", "--remove-orphans"]]
    assert receipt.cleanup_status == "failed"
    assert receipt.cleanup_rc == 1
    assert any("did not delete" in error for error in receipt.cleanup_errors)


@pytest.mark.parametrize(
    ("reuse_post_restart_session", "initialize_before_restart", "expected_error"),
    [
        (True, False, "new MCP session identity"),
        (False, True, "all service restarts"),
    ],
    ids=("reused-session", "premature-initialize"),
)
def test_process_replacement_gates_post_restart_initialize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reuse_post_restart_session: bool,
    initialize_before_restart: bool,
    expected_error: str,
) -> None:
    events: list[tuple[str, str]] = []
    compose_calls, _tool_calls, _forget_calls = _install_workflow_fakes(
        monkeypatch,
        reuse_post_restart_session=reuse_post_restart_session,
        initialize_before_restart=initialize_before_restart,
        events=events,
    )
    receipt = acceptance.run_acceptance(_args(tmp_path))

    assert receipt.status == "failed"
    assert receipt.acceptance_status == "failed"
    assert expected_error in str(receipt.error)
    assert compose_calls.count(["down", "--volumes", "--remove-orphans"]) == 1
    if initialize_before_restart:
        restart_events = [event for event in events if event[0] == "restart"]
        assert restart_events == [("restart", "postgres")]
        assert events.index(("initialize", "owner-2")) > events.index(("restart", "postgres"))
        assert ("restart", "redis") not in events
        assert ("restart", "app") not in events
    else:
        restart_events = [event for event in events if event[0] == "restart"]
        assert restart_events == [
            ("restart", "postgres"),
            ("restart", "redis"),
            ("restart", "app"),
        ]
        assert events.index(("initialize", "owner-2")) > events.index(("restart", "app"))
