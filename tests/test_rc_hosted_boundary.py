from __future__ import annotations

import copy
import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from scripts import hosted_rc_smoke as smoke
from scripts import verify_rc_hosted_boundary as verifier

ROOT = Path(__file__).resolve().parents[1]


def _resigned(report: dict) -> dict:
    updated = copy.deepcopy(report)
    unsigned = dict(updated)
    unsigned.pop("receipt_digest")
    updated["receipt_digest"] = verifier.sha256_json(unsigned)
    return updated


def test_blocked_report_is_deterministic_and_source_bound() -> None:
    first = verifier.build_report(ROOT, mode="blocked", reason=verifier.BLOCKED_REASON)
    second = verifier.build_report(ROOT, mode="blocked", reason=verifier.BLOCKED_REASON)
    assert first == second
    assert first["status"] == "BLOCKED"
    assert first["external_evidence"] == {
        "candidate_pin": None,
        "production_pin": None,
        "deployment_trigger": None,
        "deployment_ref": None,
        "deployed_pin": None,
        "supplied": False,
    }
    assert first["leaf_flags"] == verifier._leaf_flags()
    assert first["source"]["candidate_pin"].startswith("source-bound:")
    assert len(first["receipt_digest"]) == 64
    verifier.validate_receipt(first, ROOT)


def test_blocked_status_requires_exact_reason_and_ready_hold_fail_closed() -> None:
    with pytest.raises(ValueError, match="exact reason"):
        verifier.build_report(ROOT, mode="blocked", reason="waiting")
    with pytest.raises(ValueError, match="requires external"):
        verifier.build_report(ROOT, mode="ready", reason=None)
    with pytest.raises(ValueError, match="requires external"):
        verifier.build_report(ROOT, mode="hold", reason=None)
    with pytest.raises(ValueError, match="one of"):
        verifier.build_report(ROOT, mode="unknown", reason=verifier.BLOCKED_REASON)


def test_receipt_rejects_extra_missing_and_resigned_mutations() -> None:
    report = verifier.build_report(ROOT, mode="blocked", reason=verifier.BLOCKED_REASON)
    mutations = []
    extra = copy.deepcopy(report)
    extra["unexpected"] = True
    mutations.append(extra)
    missing = copy.deepcopy(report)
    del missing["grounded_assumptions"]
    mutations.append(missing)
    for field, value in (
        ("candidate_pin", "source-bound:" + "0" * 64),
        ("assumptions", {"health": "changed"}),
        ("future_command", "evil command"),
        ("flags", {"network_calls": True, **{key: False for key in verifier._leaf_flags() if key != "network_calls"}}),
        ("external evidence", {"candidate_pin": "attacker", "production_pin": None, "deployment_trigger": None, "deployment_ref": None, "deployed_pin": None, "supplied": False}),
    ):
        mutated = copy.deepcopy(report)
        if field == "candidate_pin":
            mutated["source"]["candidate_pin"] = value
        elif field == "assumptions":
            mutated["grounded_assumptions"] = value
        elif field == "future_command":
            mutated["future_smoke"]["command"] = value
        elif field == "flags":
            mutated["leaf_flags"] = value
        else:
            mutated["external_evidence"] = value
        mutations.append(_resigned(mutated))
    for mutated in mutations:
        with pytest.raises(ValueError):
            verifier.validate_receipt(mutated, ROOT)


def test_receipt_digest_is_checked_only_after_exact_claim_validation() -> None:
    report = verifier.build_report(ROOT, mode="blocked", reason=verifier.BLOCKED_REASON)
    tampered = copy.deepcopy(report)
    tampered["reason"] = "stale"
    with pytest.raises(ValueError, match="exact BLOCKED"):
        verifier.validate_receipt(_resigned(tampered), ROOT)
    digest_tampered = copy.deepcopy(report)
    digest_tampered["receipt_digest"] = "0" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        verifier.validate_receipt(digest_tampered, ROOT)


def test_source_tamper_is_detected(tmp_path: Path) -> None:
    report = verifier.build_report(ROOT, mode="blocked", reason=verifier.BLOCKED_REASON)
    for relative in verifier.SOURCE_FILES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    (tmp_path / "pyproject.toml").write_bytes(b"tampered source")
    with pytest.raises(ValueError, match="candidate pin|source provenance"):
        verifier.validate_receipt(report, tmp_path)


def test_config_comparison_reports_missing_private_fly_without_inference() -> None:
    comparison = verifier.config_comparison(ROOT)
    assert comparison["current_present"] is False
    assert comparison["differences"] == "not-comparable-without-private-config"
    assert "production topology inferred" in comparison["observation"]


def test_secret_detection_covers_keys_assignments_and_safe_placeholders() -> None:
    pem_begin = "-----BEGIN " + "DSA PRIVATE KEY-----"
    encrypted_pem_begin = "-----BEGIN " + "ENCRYPTED PRIVATE KEY-----"
    assert verifier._has_credential_material(pem_begin)
    assert verifier._has_credential_material(encrypted_pem_begin)
    fly_name = "FLY_" + "API_TOKEN"
    supabase_name = "SUPABASE_" + "ACCESS_TOKEN"
    assert verifier._has_credential_material(fly_name + "=fly-live-abcdef123456")
    assert verifier._has_credential_material(supabase_name + "=sbp_live_abcdef123456")
    assert verifier._has_credential_material("GENERAL_" + "SECRET" + "=not-a-placeholder")
    assert not verifier._has_credential_material("FLY_API_TOKEN is the documented variable name")
    assert not verifier._has_credential_material("FLY_" + "API_TOKEN=<set-at-runtime>")
    assert not verifier._has_credential_material("SUPABASE_" + "ACCESS_TOKEN=PLACEHOLDER")
    assert not verifier._has_credential_material("DATABASE_URL" + "=postgres://localhost/weft")
    assert not verifier._has_credential_material("DATABASE_URL" + "=postgres://user:password@destination.example/weft")


def test_future_smoke_plan_is_safe_and_endpoint_is_strictly_canonical() -> None:
    endpoint = "https://rc.example.test/mcp"
    plan = smoke.build_plan(endpoint, ["rc.example.test"], 5.0)
    assert plan["mode"] == "plan"
    assert plan["endpoint"] == endpoint
    assert plan["health_endpoint"] == "https://rc.example.test/healthz"
    assert plan["authorization"]["required_flags"] == [
        "--execute",
        "--approve-hosted",
        "--approve-destructive",
    ]
    assert "credential" in plan["authorization"]["credential_reference"]
    for unsafe in (
        "http://rc.example.test/mcp",
        "https://evil.example.test/mcp",
        "https://rc.example.test:8443/mcp",
        "https://rc.example.test:443/mcp",
        "https://token:secret@rc.example.test/mcp",
        "https://rc.example.test/mcp/",
        "https://rc.example.test/mcp?x=1",
        "https://rc.example.test/mcp#fragment",
        "https://rc.example.test/not-mcp",
    ):
        with pytest.raises(smoke.SmokeError):
            smoke.build_plan(unsafe, ["rc.example.test"], 5.0)


def test_mocked_smoke_proves_exact_endpoint_scope_and_finally_cleanup() -> None:
    calls: list[tuple[str, dict | None, dict | None]] = []
    memory_id = "memory-synthetic-1"
    credential_seen: list[str] = []
    scope: dict[str, str] = {}

    class FakeTransport:
        def request(self, url: str, payload: dict | None = None, headers: dict | None = None):
            calls.append((url, payload, headers))
            if url.endswith("/healthz"):
                assert headers in (None, {})
                return smoke.HttpResponse(200, {}, {"status": "ok"})
            assert url == "https://rc.example.test/mcp"
            assert payload is not None
            method = payload["method"]
            if method == "initialize":
                assert headers == {"Accept": "application/json"}
                return smoke.HttpResponse(200, {"MCP-Session-Id": "session-1"}, {"result": {}})
            name = payload["params"]["name"]
            args = payload["params"]["arguments"]
            assert headers == {"MCP-Session-Id": "session-1", "MCP-Protocol-Version": "2025-06-18"}
            if name == "weft_remember":
                assert "user_id" not in args
                scope["project_id"] = args["project_id"]
                return smoke.HttpResponse(200, {}, {"result": {"structuredContent": {"id": memory_id, "content": args["content"], "project_id": args["project_id"]}}})
            if name == "weft_recall":
                assert "user_id" not in args
                assert args["project_id"] == scope["project_id"]
                return smoke.HttpResponse(200, {}, {"result": {"structuredContent": {"results": [{"id": memory_id, "content": args["query"], "project_id": args["project_id"]}]}}})
            if name == "weft_forget":
                assert args["memory_id"] == memory_id
                return smoke.HttpResponse(200, {}, {"result": {"structuredContent": {"memory_id": memory_id, "deleted": True, "hard": True}}})
            raise AssertionError(name)

    def factory(credential: str, timeout: float):
        credential_seen.append(credential)
        return FakeTransport()

    endpoint = "https://rc.example.test/mcp"
    receipt = smoke.run_smoke(endpoint, ["rc.example.test"], 5.0, factory, "opaque-token-not-logged")
    assert receipt["status"] == "passed"
    assert receipt["endpoint"] == endpoint
    assert receipt["health_endpoint"] == "https://rc.example.test/healthz"
    assert receipt["synthetic_project_id"].startswith("rc-fl-21-project-")
    assert receipt["synthetic_user_scope"].startswith("authenticated bearer owner")
    assert credential_seen == ["opaque-token-not-logged"]
    assert receipt["checks"] == {"health": "passed", "mcp_initialize": "passed", "remember": "passed", "recall": "passed", "cleanup": "passed"}
    assert receipt["tracked_count"] == 1
    assert [call[0] for call in calls] == [endpoint.replace("/mcp", "/healthz"), endpoint, endpoint, endpoint, endpoint]
    assert "opaque-token-not-logged" not in json.dumps(receipt)


def test_mocked_smoke_cleanup_runs_after_failure() -> None:
    calls: list[str] = []

    class FakeTransport:
        def request(self, url, payload=None, headers=None):
            if url.endswith("/healthz"):
                return smoke.HttpResponse(200, {}, {"status": "ok"})
            if payload["method"] == "initialize":
                return smoke.HttpResponse(200, {"MCP-Session-Id": "session-1"}, {"result": {}})
            name = payload["params"]["name"]
            calls.append(name)
            if name == "weft_remember":
                return smoke.HttpResponse(200, {}, {"result": {"structuredContent": {"id": "tracked", "content": payload["params"]["arguments"]["content"], "project_id": payload["params"]["arguments"]["project_id"]}}})
            if name == "weft_recall":
                return smoke.HttpResponse(200, {}, {"result": {"structuredContent": {"results": []}}})
            if name == "weft_forget":
                return smoke.HttpResponse(200, {}, {"result": {"structuredContent": {"memory_id": "tracked", "deleted": True, "hard": True}}})
            raise AssertionError(name)

    receipt = smoke.run_smoke("https://rc.example.test/mcp", ["rc.example.test"], 5.0, lambda *_: FakeTransport(), "secret")
    assert receipt["status"] == "failed"
    assert calls[-1] == "weft_forget"
    assert receipt["checks"]["cleanup"] == "passed"


def test_urllib_transport_rejects_3xx_without_redirect_or_token_forwarding() -> None:
    token = "opaque-token-not-logged"
    transport = smoke.UrllibTransport(token, 5.0)
    seen: list[urllib.request.Request] = []

    def reject_redirect(request, timeout):
        seen.append(request)
        raise urllib.error.HTTPError(
            request.full_url,
            302,
            "redirect",
            {"Location": "https://evil.example.test/mcp"},
            None,
        )

    transport._opener.open = reject_redirect
    with pytest.raises(smoke.SmokeError, match="redirect"):
        transport.request("https://rc.example.test/mcp", {"jsonrpc": "2.0"})
    assert seen[0].get_header("Authorization") == "Bearer " + "opaque-token-not-logged"
    assert seen[0].full_url == "https://rc.example.test/mcp"
    assert "evil.example.test" not in seen[0].headers


def test_urllib_transport_does_not_authenticate_health_url() -> None:
    token = "opaque-token-not-logged"
    transport = smoke.UrllibTransport(token, 5.0)
    seen: list[urllib.request.Request] = []

    class Response:
        status = 200
        headers = {}

        def geturl(self):
            return "https://rc.example.test/healthz"

        def read(self, _limit):
            return b'{"status":"ok"}'

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def accept(request, timeout):
        seen.append(request)
        return Response()

    transport._opener.open = accept
    assert transport.request("https://rc.example.test/healthz").body == {"status": "ok"}
    assert seen[0].get_header("Authorization") is None


def _run_mocked_scenario(
    remember_builder,
    recall_builders,
    forget_builder=None,
):
    calls: list[tuple[str, dict | None, dict | None]] = []
    forgotten: list[str] = []
    recall_index = 0

    class FakeTransport:
        def request(self, url, payload=None, headers=None):
            nonlocal recall_index
            calls.append((url, payload, headers))
            if url.endswith("/healthz"):
                return smoke.HttpResponse(200, {}, {"status": "ok"})
            if payload["method"] == "initialize":
                return smoke.HttpResponse(200, {"MCP-Session-Id": "session-1"}, {"result": {}})
            name = payload["params"]["name"]
            args = payload["params"]["arguments"]
            if name == "weft_remember":
                body = remember_builder(args)
            elif name == "weft_recall":
                builder = recall_builders[min(recall_index, len(recall_builders) - 1)]
                recall_index += 1
                body = builder(args)
            elif name == "weft_forget":
                forgotten.append(args["memory_id"])
                body = forget_builder(args) if forget_builder else {"memory_id": args["memory_id"], "deleted": True, "hard": True}
            else:
                raise AssertionError(name)
            if isinstance(body, Exception):
                raise body
            return smoke.HttpResponse(200, {}, {"result": {"structuredContent": body}})

    receipt = smoke.run_smoke(
        "https://rc.example.test/mcp",
        ["rc.example.test"],
        5.0,
        lambda *_: FakeTransport(),
        "opaque-token-not-logged",
    )
    return receipt, calls, forgotten


def test_malformed_remember_with_safe_id_is_forgotten_before_validation() -> None:
    def remember(args):
        return {"id": "tracked-safe", "content": "wrong", "project_id": args["project_id"]}

    receipt, _calls, forgotten = _run_mocked_scenario(remember, [])
    assert receipt["status"] == "failed"
    assert forgotten == ["tracked-safe"]
    assert receipt["cleanup"]["recovery_attempted"] is False
    assert receipt["cleanup"]["forgotten_ids"] == ["tracked-safe"]


def test_missing_remember_id_recovers_exact_candidate_and_forgets_it() -> None:
    def remember(args):
        return {"content": args["content"], "project_id": args["project_id"]}

    def recovery(args):
        return {"results": [{"id": "recovered", "content": args["query"], "project_id": args["project_id"]}]}

    receipt, calls, forgotten = _run_mocked_scenario(remember, [recovery])
    assert receipt["status"] == "failed"
    assert forgotten == ["recovered"]
    assert receipt["cleanup"]["recovery_attempted"] is True
    assert receipt["cleanup"]["recovered_ids"] == ["recovered"]
    assert calls[-2][1]["params"]["name"] == "weft_recall"
    assert calls[-2][1]["params"]["arguments"]["limit"] == smoke.RECOVERY_LIMIT


def test_invalid_remember_id_recovers_exact_candidate_and_forgets_it() -> None:
    def remember(args):
        return {"id": "not safe/id", "content": args["content"], "project_id": args["project_id"]}

    def recovery(args):
        return {"results": [{"id": "recovered-invalid", "content": args["query"], "project_id": args["project_id"]}]}

    receipt, _calls, forgotten = _run_mocked_scenario(remember, [recovery])
    assert receipt["status"] == "failed"
    assert forgotten == ["recovered-invalid"]
    assert receipt["cleanup"]["recovery_attempted"] is True


def test_missing_remember_id_without_exact_match_is_cleanup_incomplete() -> None:
    def remember(args):
        return {"content": args["content"], "project_id": args["project_id"]}

    receipt, _calls, forgotten = _run_mocked_scenario(remember, [lambda _args: {"results": []}])
    assert receipt["status"] == "failed"
    assert forgotten == []
    assert any("cleanup_incomplete" in error for error in receipt["errors"])
    assert receipt["cleanup"]["uncertainty"] == "incomplete"


def test_recovery_never_forgets_unrelated_recalled_ids() -> None:
    def remember(args):
        return {"id": None, "content": args["content"], "project_id": args["project_id"]}

    def recovery(args):
        return {
            "results": [
                {"id": "unrelated", "content": "different", "project_id": args["project_id"]},
                {"id": "wrong-project", "content": args["query"], "project_id": "other"},
            ]
        }

    receipt, _calls, forgotten = _run_mocked_scenario(remember, [recovery])
    assert receipt["status"] == "failed"
    assert forgotten == []
    assert receipt["cleanup"]["recovered_ids"] == []


def test_recall_failure_still_cleans_up_safe_id_and_records_success() -> None:
    def remember(args):
        return {"id": "recall-failed", "content": args["content"], "project_id": args["project_id"]}

    receipt, _calls, forgotten = _run_mocked_scenario(
        remember,
        [RuntimeError("recall unavailable")],
    )
    assert receipt["status"] == "failed"
    assert forgotten == ["recall-failed"]
    assert any("recall failed" in error for error in receipt["errors"])
    assert receipt["cleanup"]["errors"] == []
    assert receipt["checks"]["cleanup"] == "passed"


def test_forget_failure_is_recorded_as_cleanup_incomplete() -> None:
    def remember(args):
        return {"id": "forget-failed", "content": args["content"], "project_id": args["project_id"]}

    def forget(_args):
        return {"memory_id": "different-id", "deleted": True, "hard": True}

    receipt, _calls, forgotten = _run_mocked_scenario(remember, [], forget)
    assert receipt["status"] == "failed"
    assert forgotten == ["forget-failed"]
    assert any("cleanup_incomplete" in error for error in receipt["cleanup"]["errors"])
    assert receipt["cleanup"]["uncertainty"] == "incomplete"


def test_checked_receipt_matches_current_source_bytes_and_exact_blocked_reason() -> None:
    text = (ROOT / "evidence/rc-finish-line/hosted-boundary.md").read_text(encoding="utf-8")
    payload = text.split("```json\n", 1)[1].split("\n```", 1)[0]
    checked = json.loads(payload)
    expected = verifier.build_report(ROOT, mode="blocked", reason=verifier.BLOCKED_REASON)
    verifier.validate_receipt(checked, ROOT)
    assert checked == expected
    assert checked["reason"] == "external status not yet supplied"


def test_cleanup_receipt_never_contains_bearer_credential() -> None:
    receipt, _calls, _forgotten = _run_mocked_scenario(
        lambda args: {"id": "safe", "content": args["content"], "project_id": args["project_id"]},
        [lambda _args: {"results": []}],
    )
    assert "opaque-token-not-logged" not in json.dumps(receipt)
