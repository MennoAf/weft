"""Tier-2 topic-digest synthesizer — Haiku narrative pass with provenance + cost cap.

Implements ``synthesize_digest(memories, *, budget_tokens)`` per the contract
specified in documents/prds/topic-digest-recall.md §Validation V4, V5, refined by
the cost-policy decision (Weft weft-d58f7350, loom-8e41000e).

The synthesizer:
  - Feeds a list of Memory objects to Haiku, asking it to produce a narrative
    "status" answer in which every sentence is grounded in the supplied memory ids.
  - Returns a provenance map ``{memory_id: [spans]}`` where keys are ONLY ids
    present in the input set (hallucinated ids are filtered out at parse time).
  - Abstains when the *projected* cost of the call would exceed
    MAX_SYNTH_COST_PER_CALL_USD — no call is made in that case. The result
    carries the projected cost and memory count so the caller can record the
    abstention (synthesis fire-rate is a health metric for deterministic recall).

Cost projection (Haiku 4.5 as of 2026):
  - Input pricing:  $1.00 / 1M tokens
  - Output pricing: $5.00 / 1M tokens

  The pre-call projection charges EXPECTED output (EXPECTED_OUTPUT_TOKENS), NOT
  the ``budget_tokens`` ceiling. ``budget_tokens`` is the *worst-case* output the
  model can emit — but it is already hard-capped by ``max_tokens`` on the API
  call, so projecting it against the cost cap double-guards an already-bounded
  quantity and (at the old $0.01 cap) consumed the entire budget before a single
  input token was counted, making the feature a no-op at defaults. Projecting
  expected output instead lets the cap headroom bound the genuinely UNBOUNDED
  cost — the input, since the Tier-1 gather has no limit on the topic[] path:

  projected = (estimated_input_tokens / 1M) * 1.00
            + (EXPECTED_OUTPUT_TOKENS    / 1M) * 5.00

  At cap = $0.10 this allows ~96K input tokens before abstaining — enough to
  synthesize the hottest real topics (CI ~431 memories, Weft, Loom); abstention
  is the pathological-topic backstop, not the default path. The ACTUAL cost is
  always computed from response usage after the call.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

from anthropic import AsyncAnthropic

from weft.models import Memory
from weft.tokens import estimate_tokens
from weft.views.belief_detector import _MODEL, _strip_fences

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

SYNTHESIZER_VERSION = "topic-synthesis-v1.0"

# Hard cost ceiling for the *projected* cost of a single synthesis call.
# Raised from $0.01 to $0.10 per the cost-policy decision (weft-d58f7350):
# synthesis is opt-in and fires deliberately, so a dime-scale ceiling for a
# deep overview of even the hottest topic is acceptable, and it keeps real
# topics off the abstain path (abstention is the pathological-case backstop).
# Defined as a named constant per V5: "no inline literal."
MAX_SYNTH_COST_PER_CALL_USD = 0.10

# Expected output size used in the PRE-CALL cost projection (NOT the worst-case
# budget_tokens ceiling — see module docstring). A narrative "status of this
# topic" answer realistically lands well under the budget; charging the full
# budget against the cap is what made the feature a no-op at defaults. The
# actual call is still hard-capped by budget_tokens via max_tokens.
EXPECTED_OUTPUT_TOKENS = 700

# Haiku pricing (as of 2026) — mirrors the derivation in belief_detector.py docstring.
_HAIKU_INPUT_PRICE_PER_TOKEN = 1.00 / 1_000_000   # $1.00 / 1M tokens
_HAIKU_OUTPUT_PRICE_PER_TOKEN = 5.00 / 1_000_000  # $5.00 / 1M tokens

# System prompt token overhead estimate (conservative).
_SYSTEM_PROMPT_TOKEN_OVERHEAD = 400


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


SynthesisStatus = Literal["synthesized", "abstained", "error", "empty"]


@dataclass
class SynthesisResult:
    """Outcome of a synthesis attempt.

    Carries enough for the caller to record telemetry on BOTH outcomes:
      - ``synthesized``: a Haiku call was made; ``content``/``provenance`` are
        populated and ``cost_usd`` is the actual incurred cost (from usage).
      - ``abstained``: projected cost exceeded the cap; no call was made.
        ``projected_cost_usd`` and ``memory_count`` let the caller log the
        abstention — synthesis fire-rate vs abstain-rate is the health signal
        for whether the deterministic recall path needs more structure.
      - ``error``: the call failed or the response could not be parsed.
      - ``empty``: the input memory set was empty; no attempt was made.
    """

    status: SynthesisStatus
    memory_count: int
    projected_cost_usd: float
    content: str | None = None
    provenance: dict[str, list[str]] | None = None
    cost_usd: float | None = None

    @property
    def synthesized(self) -> bool:
        return self.status == "synthesized"


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a memory synthesis engine. You are given a numbered list of memory \
records, each with an id and content. Produce a concise narrative summary that \
answers "what is the current status of this topic?" — written in plain prose, \
not a bullet list.

RULES
- Ground EVERY sentence in at least one memory id from the supplied list.
- After the narrative, emit a JSON provenance map under the key "provenance":
  an object whose keys are memory ids and whose values are arrays of short \
  verbatim spans (or sentence fragments) from the narrative that the memory \
  supports.
- ONLY cite memory ids that appear in the input — never invent ids.
- If no grounding is possible, emit an empty narrative and an empty provenance map.

OUTPUT FORMAT (return ONLY valid JSON, no markdown fences):
{
  "content": "<narrative prose>",
  "provenance": {
    "<memory_id>": ["<span1>", "<span2>"],
    ...
  }
}
"""


# ---------------------------------------------------------------------------
# Client singleton — same pattern as belief_detector.py
# ---------------------------------------------------------------------------

_client: AsyncAnthropic | None = None


def _get_client() -> AsyncAnthropic:
    """Lazy singleton for the Anthropic async client."""
    global _client
    if _client is None:
        _client = AsyncAnthropic()
    return _client


# ---------------------------------------------------------------------------
# Cost projection
# ---------------------------------------------------------------------------


def _projected_cost(input_tokens: int, output_tokens: int) -> float:
    """Return the projected USD cost for a single Haiku call."""
    return (
        input_tokens * _HAIKU_INPUT_PRICE_PER_TOKEN
        + output_tokens * _HAIKU_OUTPUT_PRICE_PER_TOKEN
    )


def _estimate_input_tokens(user_message: str) -> int:
    """Estimate total input tokens (system prompt overhead + user message)."""
    return _SYSTEM_PROMPT_TOKEN_OVERHEAD + estimate_tokens(user_message)


# ---------------------------------------------------------------------------
# Memory serialisation
# ---------------------------------------------------------------------------


def _render_memories(memories: list[Memory]) -> str:
    """Render the memory list as a numbered block for the prompt."""
    lines: list[str] = []
    for i, mem in enumerate(memories, start=1):
        lines.append(f"[{i}] id={mem.id}\n{mem.content}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def _parse_synthesis(
    raw: str,
    valid_ids: set[str],
) -> dict[str, Any] | None:
    """Parse the model's JSON response.

    Filters the provenance map to ONLY include ids present in valid_ids —
    any id the model invented or hallucinated is silently dropped (V4).

    Returns None on parse error.
    """
    raw = _strip_fences(raw)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning(
            "topic_synthesis.parse_error: error=%s raw=%r", exc, raw[:200]
        )
        return None

    if not isinstance(parsed, dict):
        logger.warning(
            "topic_synthesis.unexpected_shape: type=%s", type(parsed).__name__
        )
        return None

    content = parsed.get("content", "")
    if not isinstance(content, str):
        content = ""

    raw_provenance = parsed.get("provenance", {})
    if not isinstance(raw_provenance, dict):
        raw_provenance = {}

    # Filter provenance to ONLY ids in the input set (V4 hallucination guard).
    filtered_provenance: dict[str, list[str]] = {}
    for mem_id, spans in raw_provenance.items():
        if mem_id not in valid_ids:
            logger.debug(
                "topic_synthesis.provenance_hallucinated_id_dropped: id=%r", mem_id
            )
            continue
        if not isinstance(spans, list):
            spans = [str(spans)] if spans else []
        filtered_provenance[mem_id] = [str(s) for s in spans if s]

    return {
        "content": content,
        "provenance": filtered_provenance,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def synthesize_digest(
    memories: list[Memory],
    *,
    budget_tokens: int = 2000,
) -> SynthesisResult:
    """Synthesise a narrative digest over a set of memories.

    Args:
        memories: The memory objects to synthesise over. Must be non-empty
            for a meaningful result; an empty list returns an ``empty`` result.
        budget_tokens: Maximum output tokens for the Haiku call (the API
            ``max_tokens`` hard cap). NOTE: this is *not* used in the cost
            projection — the pre-call check projects ``EXPECTED_OUTPUT_TOKENS``
            instead (see module docstring), so a generous budget no longer
            forces abstention.

    Returns:
        A :class:`SynthesisResult`. ``status`` distinguishes the outcomes:
            - ``synthesized``: ``content`` is non-empty prose (V4-filtered
              ``provenance`` is a SUBSET of the input ids) and ``cost_usd`` is
              the actual incurred cost from usage.
            - ``abstained``: projected cost exceeded MAX_SYNTH_COST_PER_CALL_USD
              (V5); no call was made. ``projected_cost_usd`` + ``memory_count``
              are populated for telemetry.
            - ``error``: the LLM call failed or the response could not be parsed.
            - ``empty``: the input set was empty.
    """
    if not memories:
        logger.debug("topic_synthesis.empty_input")
        return SynthesisResult(status="empty", memory_count=0, projected_cost_usd=0.0)

    memory_count = len(memories)
    valid_ids: set[str] = {m.id for m in memories}

    # Render memories into a prompt-ready block.
    user_message = _render_memories(memories)

    # --- Cost pre-check (V5) ---
    # Project EXPECTED output, NOT the budget_tokens ceiling — the cap headroom
    # then bounds the unbounded input, not the already-max_tokens-bounded output.
    estimated_input = _estimate_input_tokens(user_message)
    projected = _projected_cost(estimated_input, EXPECTED_OUTPUT_TOKENS)
    if projected > MAX_SYNTH_COST_PER_CALL_USD:
        logger.warning(
            "topic_synthesis.cost_cap_exceeded: projected=%.6f cap=%.6f memories=%d; abstaining",
            projected,
            MAX_SYNTH_COST_PER_CALL_USD,
            memory_count,
        )
        return SynthesisResult(
            status="abstained",
            memory_count=memory_count,
            projected_cost_usd=projected,
        )

    # --- LLM call ---
    try:
        client = _get_client()
        response = await client.messages.create(
            model=_MODEL,
            max_tokens=budget_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        raw = response.content[0].text.strip()
        logger.debug("topic_synthesis.response: memories=%d raw=%s", memory_count, raw[:300])
    except Exception as exc:  # noqa: BLE001
        logger.warning("topic_synthesis.api_error: memories=%d error=%s", memory_count, exc)
        return SynthesisResult(
            status="error", memory_count=memory_count, projected_cost_usd=projected
        )

    # Compute actual call cost from usage metadata.
    usage = response.usage
    actual_cost = _projected_cost(usage.input_tokens, usage.output_tokens)

    # --- Parse + provenance filter (V4) ---
    parsed = _parse_synthesis(raw, valid_ids)
    if parsed is None:
        return SynthesisResult(
            status="error",
            memory_count=memory_count,
            projected_cost_usd=projected,
            cost_usd=actual_cost,
        )

    return SynthesisResult(
        status="synthesized",
        memory_count=memory_count,
        projected_cost_usd=projected,
        content=parsed["content"],
        provenance=parsed["provenance"],
        cost_usd=actual_cost,
    )
