#!/usr/bin/env python3
"""faithful_agent.py — Bounded Luna agent loop over public Weft tools.

The module implements the faithful S36 runtime seam without importing the
legacy Anthropic Reader. A fresh chronological session is given to a Luna
writer, which may make one optional public ``weft_remember`` write. The final
answer is generated from a fresh question context and injected public recall
results. Responses calls are budget-reserved before dispatch and never retried
when their outcome is ambiguous.

Author:  Jason Bauman
Version: 0.1.0
Date:    2026-09-21
Python:  >= 3.12

Dependencies:
    (stdlib only for the injected seam) — ``openai`` is lazy and optional for
    the explicit production adapter.

Usage:
    See bottom of file for run commands.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence

from benchmarks.longmemeval.dataset import Session
from benchmarks.longmemeval.task_shape import TaskShape
from benchmarks.longmemeval.faithful_budget import BudgetLedger, InvalidUsage, Reservation

LUNA_MODEL = "gpt-5.6-luna"
GPT6_LUNA_MODEL = "gpt-6-luna"
GPT4O_JUDGE_MODEL = "gpt-4o"
MAX_RETRIES = 0
MAX_TOOL_ROUNDS = 3
# Keep writer/answerer output bounded, but leave room for a complete tool
# argument object.  This is a request cap, not a retry budget.
MAX_OUTPUT_TOKENS = 512
FRESH_GPT6_MAX_OUTPUT_TOKENS = 2048
MAX_REQUEST_BYTES = 256_000
MAX_METADATA_VALUE_LENGTH = 120
PROVIDER_TIMEOUT_SECONDS = 90.0


class AgentExecutionError(RuntimeError):
    """Raised when the bounded agent loop cannot safely continue."""


class AmbiguousExecutionError(AgentExecutionError):
    """Raised when a provider attempt may have committed but its result is unknown."""


class NoAnthropicExecutionError(AgentExecutionError):
    """Raised when an Anthropic client or module is detected at runtime."""


class ResponsesClient(Protocol):
    """Minimal Responses-shaped client used by the agent and tests."""

    async def create(self, **kwargs: Any) -> Any:
        """Create one Responses request."""


class PublicToolGateway(Protocol):
    """Public Weft tool surface used by the benchmark-local loop."""

    async def call(self, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        """Invoke one public tool and return its JSON-compatible result."""


@dataclass(frozen=True, slots=True)
class AgentUsage:
    """Validated Responses usage counters; missing cache splits remain unknown."""

    input_tokens: int
    output_tokens: int
    cached_input_tokens: int | None = None
    cache_write_input_tokens: int | None = None
    reasoning_tokens: int = 0

    def as_dict(self) -> dict[str, int | None]:
        """Return usage in ledger-compatible form."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_input_tokens": self.cache_write_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Normalized function call extracted from a Responses output item."""

    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AgentResult:
    """Result of one bounded writer or answerer invocation."""

    text: str
    model: str
    calls: int
    tool_calls: int
    reservations: tuple[str, ...]
    disclosures: tuple[str, ...] = ()
    usage: tuple[AgentUsage, ...] = ()


@dataclass(frozen=True, slots=True)
class SessionWrite:
    """A session plus its date-visible writer instruction."""

    session_id: str
    session_date: str
    content: str
    instruction: str


@dataclass(slots=True)
class AgentPolicy:
    """Explicit policy that keeps the benchmark faithful and bounded."""

    allow_public_handoff: bool = True
    allow_background_tools: bool = False
    max_tool_rounds: int = MAX_TOOL_ROUNDS
    max_output_tokens: int = MAX_OUTPUT_TOKENS
    writer_can_write: bool = True
    allow_final_response_after_tool_rounds: bool = False

    def __post_init__(self) -> None:
        if self.max_tool_rounds <= 0 or self.max_output_tokens <= 0:
            raise ValueError("agent bounds must be positive")


def assert_no_anthropic_execution(client: object | None = None) -> None:
    """Reject an explicitly supplied Anthropic client, never module presence.

    The benchmark must not infer execution from ``sys.modules``: unrelated
    imports, test runners, and SDK dependencies can load that namespace.  The
    actual provider boundary is the injected client type/module and the
    Responses adapter's required ``responses.create`` interface.
    """
    if client is not None:
        identity = f"{type(client).__module__}.{type(client).__name__}".lower()
        if "anthropic" in identity or hasattr(client, "messages") and not hasattr(client, "responses"):
            raise NoAnthropicExecutionError(f"non-Responses client rejected: {identity}")


def _object_value(value: object, key: str, default: Any = None) -> Any:
    """Read a mapping key or object attribute uniformly."""
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _usage_from_response(response: object) -> AgentUsage | None:
    """Extract usage totals and optional cache details from a Responses result."""
    usage = _object_value(response, "usage")
    if usage is None:
        return None
    details = _object_value(usage, "input_tokens_details")
    raw_cached = _object_value(usage, "cached_input_tokens")
    if raw_cached is None and details is not None:
        raw_cached = _object_value(details, "cached_tokens")
    raw_cache_write = _object_value(usage, "cache_write_input_tokens")
    values: dict[str, int | None] = {
        "input_tokens": _object_value(usage, "input_tokens"),
        "output_tokens": _object_value(usage, "output_tokens"),
        "cached_input_tokens": raw_cached,
        "cache_write_input_tokens": raw_cache_write,
        "reasoning_tokens": _object_value(usage, "reasoning_tokens", 0),
    }
    required = ("input_tokens", "output_tokens", "reasoning_tokens")
    for key in required:
        raw = values[key]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise InvalidUsage(f"Responses usage {key} must be a non-negative integer")
    for key in ("cached_input_tokens", "cache_write_input_tokens"):
        raw = values[key]
        if raw is not None and (isinstance(raw, bool) or not isinstance(raw, int) or raw < 0):
            raise InvalidUsage(f"Responses usage {key} must be a non-negative integer when supplied")
    cached = values["cached_input_tokens"]
    cache_write = values["cache_write_input_tokens"]
    if cached is not None and cached > values["input_tokens"]:
        raise InvalidUsage("Responses cached input exceeds total input")
    if cache_write is not None and cache_write > values["input_tokens"] - (cached or 0):
        raise InvalidUsage("Responses cache-write input exceeds remaining input")
    return AgentUsage(**values)


def _bounded_metadata_value(value: object) -> str | None:
    """Return a short, non-content metadata value suitable for durable errors."""
    if value is None:
        return None
    text = str(value)
    return text[:MAX_METADATA_VALUE_LENGTH]


def _argument_hash(value: object) -> str:
    """Hash tool arguments without retaining their private/content values."""
    if isinstance(value, str):
        encoded = value.encode("utf-8", errors="replace")
    else:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _response_incomplete_metadata(response: object) -> dict[str, Any] | None:
    """Return bounded response metadata when the provider did not complete.

    The check intentionally runs before output text or function arguments are
    parsed.  It records only provider state and hashes of tool arguments, so
    a durable failure marker cannot leak prompt or memory content.
    """
    status = _object_value(response, "status")
    details = _object_value(response, "incomplete_details")
    if status is None and details is None:
        return None
    normalized_status = _bounded_metadata_value(status)
    if normalized_status == "completed" and details is None:
        return None
    reason = _object_value(details, "reason") if details is not None else None
    tools: list[dict[str, Any]] = []
    item_types: list[str] = []
    for item in (_object_value(response, "output", ()) or ())[:16]:
        item_type = _bounded_metadata_value(_object_value(item, "type"))
        if item_type is not None:
            item_types.append(item_type)
        if item_type == "function_call":
            tools.append({
                "call_id": _bounded_metadata_value(_object_value(item, "call_id", _object_value(item, "id", ""))),
                "name": _bounded_metadata_value(_object_value(item, "name", "")),
                "arguments_sha256": _argument_hash(_object_value(item, "arguments", "")),
            })
    return {
        "response_status": normalized_status,
        "incomplete_reason": _bounded_metadata_value(reason),
        "output_item_types": item_types,
        "tool_calls": tools,
    }


def _incomplete_error(metadata: Mapping[str, Any]) -> str:
    """Serialize only bounded response metadata for an operator-visible error."""
    return "incomplete Responses output; metadata=" + json.dumps(dict(metadata), sort_keys=True, separators=(",", ":"))


def _response_text(response: object) -> str:
    """Extract output text from common Responses SDK and test doubles."""
    output_text = _object_value(response, "output_text")
    if isinstance(output_text, str):
        return output_text.strip()
    pieces: list[str] = []
    for item in _object_value(response, "output", ()) or ():
        if _object_value(item, "type") == "message":
            for content in _object_value(item, "content", ()) or ():
                text = _object_value(content, "text")
                if isinstance(text, str):
                    pieces.append(text)
        elif _object_value(item, "type") in {"output_text", "text"}:
            text = _object_value(item, "text")
            if isinstance(text, str):
                pieces.append(text)
    return "".join(pieces).strip()


class IncompleteResponseError(AgentExecutionError):
    """Raised before parsing or executing an incomplete provider response."""

    def __init__(self, metadata: Mapping[str, Any]) -> None:
        self.metadata = dict(metadata)
        super().__init__(_incomplete_error(self.metadata))


def _tool_calls(response: object) -> list[ToolCall]:
    """Extract function calls without trusting provider-specific classes."""
    calls: list[ToolCall] = []
    for item in _object_value(response, "output", ()) or ():
        if _object_value(item, "type") != "function_call":
            continue
        raw_args = _object_value(item, "arguments", {})
        if isinstance(raw_args, str):
            import json
            try:
                raw_args = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                # Do not persist partial/private argument text in durable
                # checkpoint errors; the hash remains useful for correlation.
                raise AgentExecutionError(
                    "invalid tool arguments; arguments_sha256=" + _argument_hash(raw_args)
                ) from exc
        if not isinstance(raw_args, dict):
            raise AgentExecutionError("tool arguments must be a JSON object")
        call_id = _object_value(item, "call_id", _object_value(item, "id", ""))
        name = _object_value(item, "name", "")
        if not isinstance(call_id, str) or not call_id or not isinstance(name, str) or not name:
            raise AgentExecutionError("function call lacks call_id or name")
        calls.append(ToolCall(call_id, name, raw_args))
    return calls


def _is_finite_nonnegative(value: object) -> bool:
    """Return whether a numeric cost or token value is safe."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) >= 0


def _request_token_bound(payload: Mapping[str, Any]) -> int:
    """Conservatively bound serialized request tokens before any spend."""
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    # UTF-8 bytes are a lower-level bound than characters; add framing and
    # tokenizer variance rather than pretending byte/4 is an exact tokenizer.
    bounded_bytes = len(encoded) + 4096 + len(encoded) // 4
    tokens = max(1, (bounded_bytes + 2) // 3)
    if bounded_bytes > MAX_REQUEST_BYTES:
        raise AgentExecutionError("request exceeds conservative local context bound")
    return tokens


class BoundedJudge:
    """One-attempt GPT-4o judge using the same prompt supplied by the caller."""

    def __init__(self, client: ResponsesClient, ledger: BudgetLedger, *, phase: str = "run") -> None:
        assert_no_anthropic_execution(client)
        if phase not in {"calibration", "run"}:
            raise ValueError("phase must be calibration or run")
        self.client = client
        self.ledger = ledger
        self.phase = phase

    async def judge(self, prompt: str) -> tuple[bool, str, str]:
        """Return ``(label, raw_text, reservation_id)`` without retrying."""
        if not prompt.strip():
            raise ValueError("judge prompt must be non-empty")
        payload = {"model": GPT4O_JUDGE_MODEL, "input": [{"role": "user", "content": prompt}], "max_output_tokens": 32}
        estimate_input = _request_token_bound(payload)
        reservation = self.ledger.reserve(GPT4O_JUDGE_MODEL, estimate_input, 32, phase=self.phase)
        try:
            response = await self.client.create(
                model=GPT4O_JUDGE_MODEL,
                input=[{"role": "user", "content": prompt}],
                max_output_tokens=32,
            )
        except Exception as exc:
            self.ledger.finalize(reservation.reservation_id, error=str(exc), unknown=True)
            raise AmbiguousExecutionError(
                "judge outcome is unknown; reservation retained and replay is forbidden"
            ) from exc
        try:
            usage = _usage_from_response(response)
        except InvalidUsage as exc:
            self.ledger.finalize(
                reservation.reservation_id,
                error=f"judge returned invalid usage: {exc}",
                unknown=True,
            )
            raise AmbiguousExecutionError("judge usage was invalid; estimate retained and replay is forbidden") from exc
        if usage is None:
            self.ledger.finalize(
                reservation.reservation_id,
                error="judge omitted usage",
                unknown=True,
            )
            raise AmbiguousExecutionError("judge omitted usage; refusing to infer a label")
        incomplete = _response_incomplete_metadata(response)
        if incomplete is not None:
            self.ledger.finalize(
                reservation.reservation_id,
                usage=usage.as_dict(),
                error=_incomplete_error(incomplete),
                unknown=True,
            )
            raise IncompleteResponseError(incomplete)
        self.ledger.finalize(reservation.reservation_id, usage=usage.as_dict())
        raw = _response_text(response)
        normalized = raw.strip().lower()
        if normalized in {"yes", "true", "1", "yes."}:
            return True, raw, reservation.reservation_id
        if normalized in {"no", "false", "0", "no."}:
            return False, raw, reservation.reservation_id
        raise AgentExecutionError(f"judge returned non-binary label: {raw!r}")


class OpenAIResponsesClient:
    """Lazy OpenAI Responses adapter used only by explicit execution."""

    def __init__(self, client: object | None = None, *, timeout: float = PROVIDER_TIMEOUT_SECONDS) -> None:
        if timeout <= 0 or not math.isfinite(timeout):
            raise ValueError("provider timeout must be finite and positive")
        if client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise AgentExecutionError("openai is required for explicit execution") from exc
            # These are SDK-constructor options; responses.create does not
            # accept max_retries and must not receive it as a request field.
            client = AsyncOpenAI(max_retries=MAX_RETRIES, timeout=timeout)
        assert_no_anthropic_execution(client)
        responses = getattr(client, "responses", None)
        if responses is None or not hasattr(responses, "create"):
            raise AgentExecutionError("client does not expose responses.create")
        self._client = client

    async def create(self, **kwargs: Any) -> Any:
        """Forward one request to ``client.responses.create``."""
        assert_no_anthropic_execution(self._client)
        responses = getattr(self._client, "responses", None)
        if responses is None or not hasattr(responses, "create"):
            raise AgentExecutionError("client does not expose responses.create")
        return await responses.create(**kwargs)


class FaithfulAgent:
    """Bounded writer/answerer using only injected public tools."""

    def __init__(
        self,
        client: ResponsesClient,
        ledger: BudgetLedger,
        *,
        tools: PublicToolGateway | None = None,
        policy: AgentPolicy | None = None,
        model: str = LUNA_MODEL,
        phase: str = "run",
        retrieval_tier: str = "auto",
    ) -> None:
        assert_no_anthropic_execution(client)
        if "luna" not in model.lower():
            raise ValueError("FaithfulAgent writer/answerer must use the Luna model")
        if phase not in {"calibration", "run"}:
            raise ValueError("phase must be calibration or run")
        if retrieval_tier not in {"auto", "turns"}:
            raise ValueError("retrieval_tier must be auto or turns")
        self.client = client
        self.ledger = ledger
        self.phase = phase
        self.tools = tools
        self.policy = policy or AgentPolicy()
        self.model = model
        self.retrieval_tier = retrieval_tier

    async def _request(
        self,
        *,
        input_items: Sequence[Mapping[str, Any]],
        instructions: str,
        model: str | None = None,
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> tuple[Any, Reservation, AgentUsage | None]:
        """Reserve and dispatch one bounded Responses request."""
        selected_model = model or self.model
        payload = {
            "model": selected_model,
            "input": list(input_items),
            "instructions": instructions,
            "tools": list(tools),
            "max_output_tokens": self.policy.max_output_tokens,
        }
        estimate_input = _request_token_bound(payload)
        reservation = self.ledger.reserve(
            selected_model, estimate_input, self.policy.max_output_tokens, phase=self.phase
        )
        try:
            response = await self.client.create(
                model=selected_model,
                input=list(input_items),
                instructions=instructions,
                tools=list(tools),
                max_output_tokens=self.policy.max_output_tokens,
            )
        except Exception as exc:
            # A timeout or transport failure may have committed server-side.
            self.ledger.finalize(reservation.reservation_id, error=str(exc), unknown=True)
            raise AmbiguousExecutionError(
                "provider outcome is unknown; reservation retained and replay is forbidden"
            ) from exc
        try:
            usage = _usage_from_response(response)
        except InvalidUsage as exc:
            self.ledger.finalize(
                reservation.reservation_id,
                error=f"provider returned invalid usage: {exc}",
                unknown=True,
            )
            raise AmbiguousExecutionError("provider usage was invalid; estimate retained and replay is forbidden") from exc
        self.ledger.finalize(
            reservation.reservation_id,
            usage=None if usage is None else usage.as_dict(),
            error=None if usage is not None else "provider omitted usage",
            unknown=usage is None,
        )
        if usage is None:
            raise AmbiguousExecutionError(
                "provider omitted usage; reservation retained and replay is forbidden"
            )
        incomplete = _response_incomplete_metadata(response)
        if incomplete is not None:
            # The request itself has known usage, but its output is not a
            # complete model turn.  Never parse or execute any partial call.
            raise IncompleteResponseError(incomplete)
        return response, reservation, usage

    async def _loop(
        self,
        *,
        input_items: Sequence[Mapping[str, Any]],
        instructions: str,
        allowed_tools: Sequence[str],
        scope: Mapping[str, Any] | None = None,
    ) -> AgentResult:
        """Run a bounded Responses/tool loop with no automatic replay.

        ``scope`` is host-owned.  It is deliberately absent from the model
        schemas; a model may omit it (or return JSON null), but a non-null
        value remains visible to the gateway so immutable-scope enforcement
        can reject an attempted override.
        """
        definitions = tuple(_tool_definition(name) for name in allowed_tools)
        current = list(input_items)
        reservations: list[str] = []
        usages: list[AgentUsage] = []
        tool_count = 0
        disclosures: list[str] = []
        bound_scope = dict(scope or {})
        max_responses = self.policy.max_tool_rounds + int(
            self.policy.allow_final_response_after_tool_rounds
        )
        for response_index in range(max_responses):
            final_response_only = (
                self.policy.allow_final_response_after_tool_rounds
                and response_index == self.policy.max_tool_rounds
            )
            request_instructions = instructions
            request_tools = definitions
            if final_response_only:
                request_instructions = (
                    f"{instructions}\n"
                    "This is the final response. Do not call tools; finish with a concise "
                    "text response or acknowledgement."
                )
                request_tools = ()
            response, reservation, usage = await self._request(
                input_items=current, instructions=request_instructions, tools=request_tools
            )
            reservations.append(reservation.reservation_id)
            if usage is not None:
                usages.append(usage)
            calls = _tool_calls(response)
            if not calls:
                return AgentResult(
                    _response_text(response), self.model, len(reservations), tool_count,
                    tuple(reservations), tuple(disclosures), tuple(usages),
                )
            if response_index >= self.policy.max_tool_rounds:
                raise AgentExecutionError(
                    "Responses final turn requested tools after the bounded tool-round limit"
                )
            if self.tools is None:
                raise AgentExecutionError("model requested tools but no public gateway was supplied")
            current.extend(_response_items(response))
            for call in calls:
                if call.name not in allowed_tools:
                    raise AgentExecutionError(f"tool not permitted in this phase: {call.name}")
                arguments = dict(call.arguments)
                if call.name == "weft_recall":
                    arguments["tier"] = self.retrieval_tier
                for key, expected in bound_scope.items():
                    if key not in arguments or arguments[key] is None:
                        arguments[key] = expected
                try:
                    result = await self.tools.call(call.name, arguments)
                except Exception as exc:
                    # Preserve the attempted public call for the runner's
                    # durable failure evidence before propagating.  The
                    # gateway's successful-call audit remains authoritative.
                    calls_log = getattr(self.tools, "calls", None)
                    if isinstance(calls_log, list):
                        calls_log.append({
                            "name": call.name,
                            "arguments": dict(arguments),
                            "error": f"{type(exc).__name__}: {exc}",
                        })
                    raise
                tool_count += 1
                if call.name == "weft_remember" and not isinstance(result, Mapping):
                    raise AgentExecutionError("weft_remember returned a non-object result")
                current.append({
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": _json_text(result),
                })
        raise AgentExecutionError("Responses tool loop exceeded bounded rounds")

    async def write_session(self, session: Session, *, project_id: str, agent_id: str) -> AgentResult:
        """Offer one chronological session to the writer.

        The date is explicit in the instruction, and no question, answer, or
        answer-session label is present. Public handoff is optional and never
        silently enabled as a background operation.
        """
        session_text = session.to_text()
        instructions = (
            "You are the faithful LongMemEval memory writer.\n"
            f"The session date is {session.date!r}; preserve that provenance in any memory.\n"
            "Use only this session and do not infer future questions or gold answers.\n"
            "You may call weft_remember once when one durable user fact is clearly worth storing; "
            "otherwise return a short acknowledgement. Keep the memory body meaningful."
        )
        if not self.policy.writer_can_write:
            allowed: tuple[str, ...] = ()
        else:
            allowed = ("weft_remember",)
        items = ({
            "role": "user",
            "content": (
                f"Project: {project_id}\nAgent: {agent_id}\n"
                f"Session ID: {session.session_id}\n\n{session_text}"
            ),
        },)
        return await self._loop(
            input_items=items,
            instructions=instructions,
            allowed_tools=allowed,
            scope={"project_id": project_id, "agent_id": agent_id},
        )

    async def answer(
        self,
        *,
        question: str,
        question_date: str,
        task_shape: TaskShape | None,
        recalled_context: str | None,
        project_id: str,
        agent_id: str,
    ) -> AgentResult:
        """Answer from fresh public context without hidden routing or gold."""
        if not question.strip():
            raise ValueError("question must be non-empty")
        if recalled_context is not None and not recalled_context.strip():
            raise ValueError("recalled_context must be non-empty when supplied")
        instructions = (
            "You are the faithful LongMemEval answerer. Answer only from the supplied public "
            "recall context; do not use outside knowledge or gold labels. Be concise and say "
            "I don't know when evidence is insufficient.\n"
            f"Today's date: {question_date}. Do not disclose hidden routing metadata."
        )
        items = ({
            "role": "user",
            "content": (
                f"Project: {project_id}\nAgent: {agent_id}\nQuestion: {question}\n"
                f"Public recall context:\n{recalled_context or '(none yet; use the public tools if evidence is needed.)'}"
            ),
        },)
        return await self._loop(
            input_items=items,
            instructions=instructions,
            allowed_tools=("weft_prime", "weft_recall") if self.tools else (),
            scope={"project_id": project_id, "agent_id": agent_id},
        )


def _tool_definition(name: str) -> dict[str, Any]:
    """Return strict-enough benchmark-local public tool metadata."""
    schemas = {
        "weft_remember": {
            "type": "object",
            "properties": {
                "content": {"type": "string"}, "type": {"type": "string"},
                "topic": {"type": "array", "items": {"type": "string"}},
                "check_contradictions": {"type": "boolean"},
            }, "required": ["content"],
        },
        "weft_recall": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query"],
        },
        "weft_prime": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }
    if name not in schemas:
        raise ValueError(f"unknown public tool: {name}")
    return {"type": "function", "name": name, "parameters": schemas[name]}


def _response_items(response: object) -> list[Mapping[str, Any]]:
    """Serialize function-call output items for the next Responses turn."""
    items: list[Mapping[str, Any]] = []
    for item in _object_value(response, "output", ()) or ():
        if _object_value(item, "type") == "function_call":
            items.append({
                "type": "function_call",
                "call_id": _object_value(item, "call_id", _object_value(item, "id", "")),
                "name": _object_value(item, "name", ""),
                "arguments": _object_value(item, "arguments", "{}"),
            })
    return items


def _json_text(value: object) -> str:
    """Serialize tool output for a Responses function-call result."""
    import json
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


__all__ = [
    "AgentExecutionError", "AmbiguousExecutionError", "AgentPolicy", "AgentResult", "AgentUsage", "FaithfulAgent",
    "IncompleteResponseError",
    "GPT4O_JUDGE_MODEL", "LUNA_MODEL", "MAX_RETRIES", "NoAnthropicExecutionError",
    "OpenAIResponsesClient", "PublicToolGateway", "SessionWrite", "ToolCall",
    "assert_no_anthropic_execution",
]


# ═══════════════════════════════════════════════════════════════
# RUN COMMANDS
# ═══════════════════════════════════════════════════════════════
#
# 1. Install dependencies:
#    uv sync; uv pip install openai  # only for explicit network execution
# 2. Basic usage:
#    Use FaithfulAgent with an injected ResponsesClient and PublicToolGateway.
# 3. Production adapter:
#    OpenAIResponsesClient() is lazy and must be behind faithful_s36 guards.
# 4. Expected output:
#    AgentResult with bounded calls, reservations, usage, and disclosures.
#
# ═══════════════════════════════════════════════════════════════
