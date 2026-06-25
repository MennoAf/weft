"""Tier-2 topic-digest synthesizer — Haiku narrative pass with provenance + cost cap.

Implements ``synthesize_digest(memories, *, budget_tokens)`` per the contract
specified in documents/prds/topic-digest-recall.md §Validation V4, V5.

The synthesizer:
  - Feeds a list of Memory objects to Haiku, asking it to produce a narrative
    "status" answer in which every sentence is grounded in the supplied memory ids.
  - Returns a provenance map ``{memory_id: [spans]}`` where keys are ONLY ids
    present in the input set (hallucinated ids are filtered out at parse time).
  - Abstains (returns None) when the projected cost of the call would exceed
    MAX_SYNTH_COST_PER_CALL_USD — no call is made in that case.

Cost calculation (Haiku 4.5 as of 2026):
  - Input pricing:  $1.00 / 1M tokens
  - Output pricing: $5.00 / 1M tokens
  Projected cost = (estimated_input_tokens / 1_000_000) * 1.00
                 + (budget_tokens / 1_000_000) * 5.00
  At budget_tokens=2000 and ~500-token system prompt + 1500 content tokens:
    = (2000 / 1_000_000) * 1.00 + (2000 / 1_000_000) * 5.00
    = 0.002 + 0.010 = $0.012
  Exceeds the cap at large inputs — abstention kicks in to prevent overspend.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from anthropic import AsyncAnthropic

from weft.models import Memory
from weft.tokens import estimate_tokens
from weft.views.belief_detector import _MODEL, _strip_fences

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

SYNTHESIZER_VERSION = "topic-synthesis-v1.0"

# Hard cost ceiling for a single synthesis call.
# Higher than the belief detector's per-call cap because narrative output
# is longer than a single claim extraction.
# Defined as a named constant per V5: "no inline literal."
MAX_SYNTH_COST_PER_CALL_USD = 0.01

# Haiku pricing (as of 2026) — mirrors the derivation in belief_detector.py docstring.
_HAIKU_INPUT_PRICE_PER_TOKEN = 1.00 / 1_000_000   # $1.00 / 1M tokens
_HAIKU_OUTPUT_PRICE_PER_TOKEN = 5.00 / 1_000_000  # $5.00 / 1M tokens

# System prompt token overhead estimate (conservative).
_SYSTEM_PROMPT_TOKEN_OVERHEAD = 400

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
) -> dict[str, Any] | None:
    """Synthesise a narrative digest over a set of memories.

    Args:
        memories: The memory objects to synthesise over. Must be non-empty
            for a meaningful result; an empty list returns None immediately.
        budget_tokens: Maximum output tokens for the Haiku call. Also used
            as the output-side of the cost projection — larger values allow
            longer narratives but increase the projected cost.

    Returns:
        A dict with keys:
            - ``content`` (str): The narrative prose. Non-empty when synthesis
              succeeds on a non-empty input.
            - ``provenance`` (dict[str, list[str]]): Map from memory id to
              citation spans. Keys are a SUBSET of the input memory ids (V4).
            - ``cost`` (float): Actual incurred USD cost of the call.
        Returns None when:
            - ``memories`` is empty.
            - Projected cost would exceed MAX_SYNTH_COST_PER_CALL_USD (V5).
            - The LLM call fails or the response cannot be parsed.
    """
    if not memories:
        logger.debug("topic_synthesis.empty_input: returning None")
        return None

    valid_ids: set[str] = {m.id for m in memories}

    # Render memories into a prompt-ready block.
    user_message = _render_memories(memories)

    # --- Cost pre-check (V5) ---
    estimated_input = _estimate_input_tokens(user_message)
    projected = _projected_cost(estimated_input, budget_tokens)
    if projected > MAX_SYNTH_COST_PER_CALL_USD:
        logger.warning(
            "topic_synthesis.cost_cap_exceeded: projected=%.6f cap=%.6f; abstaining",
            projected,
            MAX_SYNTH_COST_PER_CALL_USD,
        )
        return None

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
        logger.debug("topic_synthesis.response: memories=%d raw=%s", len(memories), raw[:300])
    except Exception as exc:  # noqa: BLE001
        logger.warning("topic_synthesis.api_error: memories=%d error=%s", len(memories), exc)
        return None

    # Compute actual call cost from usage metadata.
    usage = response.usage
    actual_cost = _projected_cost(usage.input_tokens, usage.output_tokens)

    # --- Parse + provenance filter (V4) ---
    parsed = _parse_synthesis(raw, valid_ids)
    if parsed is None:
        return None

    return {
        "content": parsed["content"],
        "provenance": parsed["provenance"],
        "cost": actual_cost,
    }
