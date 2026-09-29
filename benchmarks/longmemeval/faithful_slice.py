"""One synthetic, local-only faithful LongMemEval vertical slice.

This module intentionally does not repair or import the S36 runner.  It keeps a
small, durable case state and one cumulative budget ledger while using the real
public Weft MCP functions for memory writes, recall, and handoff.  Paid model
boundaries are injected fakes in tests; no provider is constructed here.
"""
from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import math
import os
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Protocol

from benchmarks.longmemeval.faithful_budget import Pricing

LUNA_MODEL = "gpt-5.6-luna"
GPT4O_MODEL = "gpt-4o-2024-08-06"
TOTAL_CAP_USD = 20.0
CALIBRATION_CAP_USD = 5.0
QUESTION_TYPE = "multi-session"
QUESTION = "What trip is planned, and what travel constraint matters?"
REFERENCE_ANSWER = "A coastal train trip in October is planned; overnight flights are disliked."


class SliceError(RuntimeError):
    """Base class for slice failures."""


class AmbiguousSliceError(SliceError):
    """A provider or durable write outcome is unknown; replay is forbidden."""


class BudgetExceeded(SliceError):
    """A request would cross a cumulative total or calibration ceiling."""


class InFlightError(SliceError):
    """A prior invocation was interrupted and cannot be safely replayed."""


class ResponsesClient(Protocol):
    async def create(self, **kwargs: Any) -> Any: ...


class PublicGateway(Protocol):
    async def call(self, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]: ...


class SliceLedger:
    """One atomic JSON ledger with a total cap and calibration sub-ceiling."""

    def __init__(self, path: Path, *, binding: Mapping[str, str]) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.binding = dict(binding)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked():
            data = self._read() if self.path.exists() else None
            if data is None:
                self._write({"schema": "faithful-slice-ledger.v1", "binding": self.binding, "reservations": []})
            elif data.get("binding") != self.binding:
                raise SliceError("ledger binding mismatch")

    @contextmanager
    def _locked(self):
        with self.lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read(self) -> dict[str, Any]:
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("binding") != self.binding:
            raise SliceError("invalid or mismatched ledger")
        return value

    def _write(self, value: Mapping[str, Any]) -> None:
        fd, name = tempfile.mkstemp(prefix=self.path.name + ".", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(value, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, self.path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def reserve(self, model: str, input_tokens: int, output_tokens: int, *, phase: str) -> str:
        if phase not in {"calibration", "run"}:
            raise ValueError("phase must be calibration or run")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (input_tokens, output_tokens)):
            raise ValueError("token counts must be non-negative integers")
        estimate = Pricing().reserve_cost(model, input_tokens, output_tokens)
        with self._locked():
            data = self._read()
            rows = data["reservations"]
            total = sum(float(r["estimated_usd"]) for r in rows)
            calibration = sum(float(r["estimated_usd"]) for r in rows if r["phase"] == "calibration")
            if total + estimate > TOTAL_CAP_USD + 1e-12:
                raise BudgetExceeded("request exceeds $20 cumulative cap")
            if phase == "calibration" and calibration + estimate > CALIBRATION_CAP_USD + 1e-12:
                raise BudgetExceeded("request exceeds $5 calibration sub-ceiling")
            rid = str(uuid.uuid4())
            rows.append({"reservation_id": rid, "model": model, "phase": phase, "estimated_usd": estimate, "status": "reserved"})
            self._write(data)
            return rid

    def finalize(self, reservation_id: str, *, usage: Mapping[str, int] | None = None, error: str | None = None, unknown: bool = False) -> None:
        with self._locked():
            data = self._read()
            for row in data["reservations"]:
                if row["reservation_id"] != reservation_id:
                    continue
                if row["status"] != "reserved":
                    raise SliceError("reservation already finalized")
                row["status"] = "unknown" if unknown or usage is None else "completed"
                row["error"] = error
                row["actual_usd"] = None if usage is None else Pricing().actual_cost(row["model"], usage)
                self._write(data)
                return
        raise SliceError("unknown reservation")

    def summary(self) -> dict[str, Any]:
        with self._locked():
            data = self._read()
        rows = data["reservations"]
        total = sum(float(r["estimated_usd"]) for r in rows)
        calibration = sum(float(r["estimated_usd"]) for r in rows if r["phase"] == "calibration")
        return {"total_reserved_usd": total, "calibration_reserved_usd": calibration, "total_cap_usd": TOTAL_CAP_USD, "calibration_cap_usd": CALIBRATION_CAP_USD, "attempt_count": len(rows), "unknown_count": sum(r["status"] == "unknown" for r in rows)}


def _value(obj: object, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, Mapping) else getattr(obj, key, default)


def _usage(response: object) -> dict[str, int]:
    raw = _value(response, "usage")
    if raw is None:
        raise AmbiguousSliceError("paid response omitted usage")
    values = {key: _value(raw, key, 0) for key in ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens")}
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values.values()):
        raise AmbiguousSliceError("paid response supplied invalid usage")
    return values


def _text(response: object) -> str:
    direct = _value(response, "output_text")
    if isinstance(direct, str):
        return direct.strip()
    return "".join(str(_value(item, "text", "")) for item in (_value(response, "output", ()) or ()) if _value(item, "type") in {"text", "output_text"}).strip()


def _calls(response: object) -> list[dict[str, Any]]:
    result = []
    for item in (_value(response, "output", ()) or ()):
        if _value(item, "type") != "function_call":
            continue
        args = _value(item, "arguments", {})
        if isinstance(args, str):
            args = json.loads(args)
        if not isinstance(args, dict):
            raise SliceError("function-call arguments must be an object")
        result.append({"call_id": _value(item, "call_id", _value(item, "id", "")), "name": _value(item, "name", ""), "arguments": args})
    return result


def _request_tokens(payload: Mapping[str, Any]) -> int:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return max(1, (len(encoded) + 4095) // 3)


class StrictGateway:
    """Invoke public tools with immutable owner/project/agent scope."""

    def __init__(self, app: Any, *, owner_id: str, project_id: str, agent_id: str = "faithful-slice") -> None:
        self.app, self.owner_id, self.project_id, self.agent_id = app, owner_id, project_id, agent_id
        self.calls: list[dict[str, Any]] = []

    async def call(self, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        allowed = {"weft_remember", "weft_recall", "weft_handoff"}
        if name not in allowed:
            raise SliceError(f"tool not allowed: {name}")
        args = dict(arguments)
        for key, expected in (("project_id", self.project_id), ("agent_id", self.agent_id)):
            if key in args and args[key] not in (None, expected):
                raise SliceError(f"{key} override rejected")
            args[key] = expected
        if name == "weft_recall":
            if args.get("user_id") not in (None, self.owner_id):
                raise SliceError("user_id override rejected")
            args["user_id"] = self.owner_id
        if name == "weft_remember":
            args["check_contradictions"] = args.get("check_contradictions", True)
            if args.get("workspace_id") is not None:
                raise SliceError("workspace scope is not allowed")
        from weft.auth import current_user_id
        from weft.mcp import tools
        token = current_user_id.set(self.owner_id)
        try:
            result = await getattr(tools, name)(self._ctx(), **args)
        finally:
            current_user_id.reset(token)
        if not isinstance(result, Mapping):
            raise SliceError(f"{name} returned non-object")
        self.calls.append({"name": name, "arguments": args, "result": dict(result)})
        return result

    def _ctx(self) -> Any:
        return SimpleNamespace(request_context=SimpleNamespace(lifespan_context=self.app), transport="stdio")


async def _paid_call(client: ResponsesClient, ledger: SliceLedger, *, input_items: list[dict[str, Any]], instructions: str, phase: str) -> tuple[Any, str]:
    payload = {"model": LUNA_MODEL, "input": input_items, "instructions": instructions, "tools": [], "max_output_tokens": 256}
    reservation = ledger.reserve(LUNA_MODEL, _request_tokens(payload), 256, phase=phase)
    try:
        response = await client.create(**payload)
    except Exception as exc:
        ledger.finalize(reservation, error=str(exc), unknown=True)
        raise AmbiguousSliceError("paid response outcome unknown; replay forbidden") from exc
    usage = _usage(response)
    ledger.finalize(reservation, usage=usage)
    return response, reservation


async def _loop(client: ResponsesClient, gateway: StrictGateway, ledger: SliceLedger, *, items: list[dict[str, Any]], instructions: str, allowed: set[str], phase: str) -> dict[str, Any]:
    current = list(items)
    tools: list[dict[str, Any]] = []
    for name in sorted(allowed):
        tools.append({"type": "function", "name": name, "parameters": {"type": "object"}})
    for _ in range(3):
        payload = {"model": LUNA_MODEL, "input": current, "instructions": instructions, "tools": tools, "max_output_tokens": 256}
        reservation = ledger.reserve(LUNA_MODEL, _request_tokens(payload), 256, phase=phase)
        try:
            response = await client.create(**payload)
        except Exception as exc:
            ledger.finalize(reservation, error=str(exc), unknown=True)
            raise AmbiguousSliceError("paid response outcome unknown; replay forbidden") from exc
        usage = _usage(response)
        ledger.finalize(reservation, usage=usage)
        calls = _calls(response)
        if not calls:
            return {"text": _text(response), "reservation_id": reservation, "tool_calls": len(gateway.calls)}
        current.extend({"type": "function_call", "call_id": call["call_id"], "name": call["name"], "arguments": json.dumps(call["arguments"], sort_keys=True)} for call in calls)
        for call in calls:
            if call["name"] not in allowed:
                raise SliceError(f"tool not allowed in phase: {call['name']}")
            result = await gateway.call(call["name"], call["arguments"])
            current.append({"type": "function_call_output", "call_id": call["call_id"], "output": json.dumps(result, sort_keys=True)})
    raise SliceError("paid tool loop exceeded three rounds")


def _state_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_official_judge_prompt(source_root: Path, *, question_type: str, question: str, answer: str, hypothesis: str, abstention: bool = False) -> str:
    """Render the preserved LongMemEval ``get_anscheck_prompt`` exactly."""
    from benchmarks.longmemeval.judge import _official_prompt_loader
    prompt_fn = _official_prompt_loader(Path(source_root))
    return prompt_fn(question_type, question, answer, hypothesis, abstention=abstention)


class SliceStore:
    """Atomic case state; an in-flight marker makes unknown work fail closed."""

    def __init__(self, path: Path, *, binding: Mapping[str, str]) -> None:
        self.path, self.binding = Path(path), dict(binding)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if value.get("binding") != self.binding:
            raise SliceError("case binding mismatch")
        if value.get("in_flight") is not None:
            raise InFlightError("case has unknown in-flight work; refusing replay")
        return value

    def save(self, state: dict[str, Any]) -> None:
        state["binding"] = self.binding
        state["state_hash"] = _state_hash({k: v for k, v in state.items() if k != "state_hash"})
        fd, name = tempfile.mkstemp(prefix=self.path.name + ".", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(state, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush(); os.fsync(handle.fileno())
            os.replace(name, self.path)
        finally:
            if os.path.exists(name): os.unlink(name)


async def run_slice(*, app: Any, root: Path, owner_id: str, project_id: str, client: ResponsesClient, judge_client: Any, official_source_root: Path) -> dict[str, Any]:
    """Run two chronological sessions, fresh normal recall, answer, and official judge."""
    binding = {"owner_id": owner_id, "project_id": project_id, "case": "synthetic-faithful-slice-v1"}
    store = SliceStore(Path(root) / "case.json", binding=binding)
    ledger = SliceLedger(Path(root) / "ledger.json", binding=binding)
    existing = store.load()
    if existing and existing.get("status") == "complete":
        return existing
    if existing:
        raise SliceError("incomplete state cannot be replayed")
    gateway = StrictGateway(app, owner_id=owner_id, project_id=project_id)
    state: dict[str, Any] = {"schema": "faithful-slice.v1", "status": "running", "owner_id": owner_id, "project_id": project_id, "in_flight": "write-session-early"}
    store.save(state)
    sessions = (("early", "2023/04/19", "The user keeps a paper atlas in the study.", "The user prefers the atlas for trip planning."), ("late", "2023/04/21", "The user plans a coastal train trip in October.", "The user dislikes overnight flights."))
    writer_results = []
    for sid, date, first, second in sessions:
        state["in_flight"] = f"write-{sid}"; store.save(state)
        result = await _loop(client, gateway, ledger, items=[{"role": "user", "content": f"Session {sid} ({date}): {first} {second}"}], instructions="Write only durable facts from this session using public memory tools.", allowed={"weft_remember"}, phase="calibration")
        writer_results.append({"session_id": sid, "date": date, "result": result, "tool_results": gateway.calls[-2:]})
        state["sessions"] = writer_results; state["in_flight"] = None; store.save(state)
    state["in_flight"] = "handoff"; store.save(state)
    handoff = await gateway.call("weft_handoff", {"summary": "Two synthetic sessions stored.", "in_progress": "Answering the October trip question.", "next_steps": "Use normal recall before answering.", "open_questions": "None."})
    state["handoff"] = handoff; state["in_flight"] = None; store.save(state)
    state["in_flight"] = "answer"; store.save(state)
    answer = await _loop(client, gateway, ledger, items=[{"role": "user", "content": f"Question: {QUESTION}"}], instructions="Answer only from fresh public recall context; do not use gold or hidden state.", allowed={"weft_recall"}, phase="run")
    answer_tool_results = [call for call in gateway.calls if call["name"] == "weft_recall"]
    state["answer"] = answer; state["answer_tool_results"] = answer_tool_results; state["in_flight"] = None; store.save(state)
    actual_hypothesis = answer["text"]
    official_prompt = load_official_judge_prompt(official_source_root, question_type=QUESTION_TYPE, question=QUESTION, answer=REFERENCE_ANSWER, hypothesis=actual_hypothesis)
    state["in_flight"] = "judge"; store.save(state)
    judge_payload = {"model": GPT4O_MODEL, "messages": [{"role": "user", "content": official_prompt}], "n": 1, "temperature": 0, "max_tokens": 10}
    reservation = ledger.reserve(GPT4O_MODEL, _request_tokens(judge_payload), 10, phase="run")
    try:
        completion = judge_client.chat.completions.create(**judge_payload)
    except Exception as exc:
        ledger.finalize(reservation, error=str(exc), unknown=True); raise AmbiguousSliceError("judge outcome unknown; replay forbidden") from exc
    raw = completion.choices[0].message.content.strip()
    ledger.finalize(reservation, usage={"input_tokens": _request_tokens(judge_payload), "output_tokens": len(raw.split()), "cached_input_tokens": 0, "reasoning_tokens": 0})
    if raw.lower() not in {"yes", "no", "yes.", "no."}:
        raise SliceError("official judge fake must return binary yes/no")
    state.update({"question_type": QUESTION_TYPE, "question": QUESTION, "reference_answer": REFERENCE_ANSWER, "hypothesis": actual_hypothesis, "judge": {"raw": raw, "label": raw.lower().startswith("yes"), "model": GPT4O_MODEL, "prompt": official_prompt, "prompt_sha256": hashlib.sha256(official_prompt.encode()).hexdigest()}, "ledger": ledger.summary(), "status": "complete", "in_flight": None})
    store.save(state)
    return state


__all__ = ["AmbiguousSliceError", "BudgetExceeded", "CALIBRATION_CAP_USD", "GPT4O_MODEL", "InFlightError", "LUNA_MODEL", "SliceError", "SliceLedger", "SliceStore", "StrictGateway", "TOTAL_CAP_USD", "load_official_judge_prompt", "run_slice"]
