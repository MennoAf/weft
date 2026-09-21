#!/usr/bin/env python3
"""
reader.py — Claude-based answer synthesis from recalled memories.

Stage 3 of the LongMemEval pipeline (indexing → retrieval → READING).

Given a benchmark question, its question_date, and the top-K memories Weft
returned, the Reader produces a short hypothesis string. The system prompt
is question-type-specific (e.g. abstention types are explicitly told they
may refuse) and is marked as a prompt-cache breakpoint so the same prompt
text is paid for once across the ~70 questions of each type per run.

Author:  Jason Bauman
Version: 0.1.0
Date:    2026-04-30
Python:  >= 3.12

Dependencies:
    anthropic>=0.42.0  — already a Weft dep

Usage:
    See adapter.py — invoked once per question after recall.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

try:
    from anthropic import AsyncAnthropic
except ImportError:  # Provider-injected tests and alternate providers need no SDK.
    AsyncAnthropic = None  # type: ignore[assignment,misc]

from weft.models import MemoryRecall
from weft.text_generation import GenerationRequest, TextGenerationProvider

logger = logging.getLogger(__name__)


# Sonnet is the default — Opus is reserved for abstention types where
# refusal-vs-answer judgment is the hardest call. Override per call if needed.
DEFAULT_MODEL = "claude-sonnet-4-6"
ABSTENTION_MODEL = "claude-sonnet-4-6"  # Sonnet handles refusal cleanly; bump to opus only if eval shows it's needed.
MAX_TOKENS_OUT = 256


# Per-question-type Reader instructions. Kept as constants so the
# Anthropic prompt cache can hash them stably across the 500-question run.
_BASE_INSTRUCTIONS = (
    "You are a careful question-answering assistant working from a memory store.\n"
    "You will receive (a) the user's question, (b) today's date, and (c) a list\n"
    "of memories retrieved from prior conversations.\n\n"
    "Rules:\n"
    "1. Answer ONLY from the supplied memories. Do not use outside knowledge.\n"
    "2. Be concise. Output the answer itself with no preamble or explanation.\n"
    "3. Pay attention to the dates on each memory and on today's date.\n"
    "4. Before answering, identify the exact evidence that answers the question. "
    "Ignore related facts that do not match the requested person, event, time, "
    "quantity, or scope.\n"
    "5. For questions involving multiple facts or events, make an explicit set "
    "of the matching items first, then calculate or synthesize from that set. "
    "Do not include nearby distractors.\n"
    "6. For current, previous, latest, or changed facts, compare the dated "
    "evidence and use the state appropriate to the wording; do not mix "
    "historical and current values.\n"
    "7. If the requested detail is absent or the matching evidence is "
    "insufficient, say so rather than guessing.\n"
)

_TYPE_INSTRUCTIONS: dict[str, str] = {
    "single-session-user": (
        "This question is about something the user said in a single past session. "
        "Find the matching session and answer directly."
    ),
    "single-session-assistant": (
        "This question is about something you (the assistant) said in a single "
        "past session. Find the matching session and answer directly."
    ),
    "single-session-preference": (
        "This question is about a user preference. Before answering, silently "
        "audit the retrieved evidence into: (a) hard constraints or explicit "
        "avoidances, (b) positive preferences, (c) soft preferences, (d) direct "
        "evidence, and (e) related-topic analogies. Keep this audit internal; "
        "the output must remain answer-only with no checklist or evidence report. "
        "Apply evidence in this order: hard constraints and avoidances first; "
        "then current, specific direct evidence; then current soft preferences; "
        "then a materially transferable analogy. Never recommend something that "
        "violates an explicit avoidance or hard constraint. Direct evidence "
        "outranks a weak analogy, and a weak analogy must not be presented as a "
        "stated preference. The relevant preference may have been stated about a "
        "RELATED topic in a prior session — preferences from a Seattle trip apply "
        "to a Miami trip; preferences about baking with one ingredient apply to "
        "baking with another — but only when the analogy is genuinely transferable "
        "and does not conflict with a more specific preference. When preference "
        "statements conflict, distinguish current from previous preferences using "
        "their dates and the question wording; do not merge or average incompatible "
        "preferences. When a preference IS present in memory, EXTRACT it and answer "
        "with that preference applied — do NOT give a generic answer when a signal "
        "exists. For an abstention variant, do not bridge a material evidence gap "
        "with analogy; follow the exact I don't know rule."
    ),
    "multi-session": (
        "This question requires synthesizing information across multiple past "
        "sessions. First identify every session or event that satisfies the "
        "question's exact constraints (including date, order, and scope), then "
        "combine only those matching memories. For totals, show the selected "
        "operands in your reasoning before giving the concise final answer."
    ),
    "knowledge-update": (
        "The relevant fact may have been UPDATED across sessions. If memories "
        "contradict, trust the most recent one based on session date. "
        "CRITICAL: frequency or volume of mentions does NOT determine the "
        "current value. Sort memories by session date; the LATEST date wins, "
        "even if older mentions are more numerous, more detailed, or more "
        "elaborately discussed."
    ),
    "temporal-reasoning": (
        "This question requires reasoning about WHEN events happened. Use the "
        "session dates and today's date to compute durations, ordering, or "
        "relative timing. Show only the final answer, not the calculation."
    ),
}

_ABSTENTION_SUFFIX = (
    "\n\nABSTENTION CASE: This question may not have an answer in the memories. "
    "If the SPECIFIC detail asked is not explicitly stated in the memories, "
    "respond with exactly: I don't know. Do NOT use general world knowledge "
    "to fill gaps. Do NOT infer plausible values from related-but-different "
    "memories — if the question asks about a film and the memories only "
    "mention a camera, you do not know about the film; if asked about a "
    "specific cost and the memories only describe a related event, you do "
    "not know the cost. Plausibility is not evidence."
)


def _system_prompt_for(task_shape: str = "single-session") -> str:
    """Build a prompt from a runtime shape or legacy family name.

    Runtime calls pass an explicit task shape. Legacy family names remain
    supported here for direct prompt-unit callers, but are not used by
    ``Reader.read_answer`` to select a model or runtime policy.
    """
    task_shape = getattr(task_shape, "task_shape", task_shape)
    abstention = isinstance(task_shape, str) and task_shape.endswith("_abs")
    family = task_shape.removesuffix("_abs") if isinstance(task_shape, str) else task_shape
    type_block = {
        "multi-session": _TYPE_INSTRUCTIONS["multi-session"],
        "temporal": _TYPE_INSTRUCTIONS["temporal-reasoning"],
        "temporal-multi": _TYPE_INSTRUCTIONS["temporal-reasoning"],
        "temporal-reasoning": _TYPE_INSTRUCTIONS["temporal-reasoning"],
        "single-session": "Answer the question from the supplied memories.",
        "single-session-user": _TYPE_INSTRUCTIONS["single-session-user"],
        "single-session-assistant": _TYPE_INSTRUCTIONS["single-session-assistant"],
        "single-session-preference": _TYPE_INSTRUCTIONS["single-session-preference"],
        "knowledge-update": _TYPE_INSTRUCTIONS["knowledge-update"],
    }.get(family, "Answer the question from the supplied memories.")
    suffix = _ABSTENTION_SUFFIX if abstention else ""
    return f"{_BASE_INSTRUCTIONS}\n{type_block}{suffix}"


def _format_memories(memories: list[MemoryRecall]) -> str:
    """Render recalled memories as a numbered context block for the Reader.

    Score is included so the Reader can break ties toward higher-relevance
    memories when multiple seem to address the question. Truncation is the
    caller's job — see ``read_answer`` ``top_k`` parameter.
    """
    if not memories:
        return "(no relevant memories retrieved)"
    lines: list[str] = []
    for i, recall in enumerate(memories, start=1):
        lines.append(
            f"[{i}] (relevance={recall.similarity:.2f}) {recall.memory.content}"
        )
    return "\n\n".join(lines)


def _format_result_rows(results: list) -> str:
    """Render a ``weft_recall`` response's ``results`` slice as numbered rows.

    Mirrors ``_format_memories`` but reads the serialized dict shape the tool
    returns (``r.to_dict()`` + ``similarity``) rather than ``MemoryRecall``
    objects. This is the limit-bounded relevance slice — for an enumeration ask
    it under/over-counts, which is exactly the gap ``format_recall_context``
    closes.
    """
    if not results:
        return "(no relevant memories retrieved)"
    lines: list[str] = []
    for i, r in enumerate(results, start=1):
        content = r.get("content", "") if isinstance(r, dict) else getattr(r, "content", "")
        sim = r.get("similarity") if isinstance(r, dict) else getattr(r, "similarity", None)
        prefix = f"[{i}] (relevance={sim:.2f})" if isinstance(sim, (int, float)) else f"[{i}]"
        lines.append(f"{prefix} {content}")
    return "\n\n".join(lines)


def format_recall_context(response: dict, *, use_enumeration: bool = True) -> str:
    """Render a ``weft_recall`` response as the agent-facing context block.

    A reading agent consumes THIS text, not the raw response dict — so this is
    where the enumeration consumption contract is actually kept or broken. When
    the response carries an ``enumeration`` answer (a complete-membership gather
    for a "how many / list all" ask), surface the corrected COUNT and the
    COMPLETE member list. Rendering only ``results`` — the limit-bounded
    relevance slice — silently drops that answer: it under-counts when
    membership exceeds the limit and over-counts via cross-collection pollution
    when it doesn't. ``response["count"]`` is already corrected upstream, but a
    render that ignores it hands the agent the wrong number anyway.

    ``use_enumeration=False`` reproduces the legacy results-only render. It is
    retained so the PAAH acceptance harness can prove the enumeration-aware
    branch closes a *real* gap (the ``False`` render undercounts) rather than a
    hypothetical one — the same find-then-close discipline the agenda shape
    used on ``weft_daily_brief``.
    """
    enum = response.get("enumeration") if use_enumeration else None
    if enum:
        members = enum.get("members", []) or []
        count = enum.get("count", len(members))
        target = enum.get("target", "items")
        shown = enum.get("similarity_count", len(response.get("results", []) or []))
        lines = [
            f"COUNT: {count} {target} — complete list of {count} follows; "
            f"`results` is only the top {shown} by relevance.",
        ]
        for i, m in enumerate(members, start=1):
            content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
            lines.append(f"[{i}] {content}")
        return "\n".join(lines)
    return _format_result_rows(response.get("results", []) or [])


@dataclass(frozen=True, slots=True)
class ReaderResponse:
    """Result of one Reader call."""

    hypothesis: str
    model: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int


class Reader:
    """Claude-based answer synthesizer with prompt caching.

    One instance is reused across the full benchmark run so the underlying
    HTTPX connection pool and cache hit rate are maximized.
    """

    def __init__(
        self,
        client: AsyncAnthropic | None = None,
        *,
        model: str = DEFAULT_MODEL,
        abstention_model: str = ABSTENTION_MODEL,
        provider: TextGenerationProvider | None = None,
    ):
        self._client = client if provider is None else None
        self._provider = provider
        if self._client is None and self._provider is None:
            if AsyncAnthropic is None:
                raise ValueError("Anthropic SDK is required when no Reader provider is injected")
            self._client = AsyncAnthropic()
        self._model = model
        self._abstention_model = abstention_model

    async def read_answer(
        self,
        *,
        question: str,
        question_date: str,
        question_type: str | None = None,
        task_shape: object = "single-session",
        memories: list[MemoryRecall] | None = None,
        top_k: int | None = 10,
    ) -> ReaderResponse:
        """Produce one hypothesis string for one benchmark question.

        Args:
            question: The natural-language question text.
            question_date: Today's date for the synthetic user (anchors
                temporal reasoning).
            question_type: One of the LongMemEval question type strings.
                Determines which instruction block is used and whether
                abstention is permitted.
            memories: Memories returned by Weft's hybrid recall, already
                ranked. Caller may pass more than ``top_k`` — extras are
                trimmed here.
            top_k: Maximum memories to feed the Reader. None = use all.

        Returns:
            ReaderResponse with the hypothesis and token accounting.
        """
        memories = list(memories or ())
        if top_k is not None and len(memories) > top_k:
            memories = memories[:top_k]

        system_prompt = _system_prompt_for(task_shape)
        model = self._model
        user_content = (
            f"Today's date: {question_date}\n\n"
            f"Question: {question}\n\n"
            f"Memories:\n{_format_memories(memories)}"
        )
        if self._provider is not None:
            generated = await self._provider.generate(
                GenerationRequest(
                    model=model,
                    system=system_prompt,
                    messages=({"role": "user", "content": user_content},),
                    max_tokens=MAX_TOKENS_OUT,
                )
            )
            return ReaderResponse(
                hypothesis=generated.text.strip(), model=generated.model,
                input_tokens=generated.input_tokens, cached_input_tokens=0,
                output_tokens=generated.output_tokens,
            )

        # System prompt is stable per runtime task shape → cacheable. The
        # `cache_control` marker tells Anthropic to keep the prefix warm
        # across calls for the 5-minute TTL window.
        system_blocks = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        user_content = (
            f"Today's date: {question_date}\n\n"
            f"Question: {question}\n\n"
            f"Memories:\n{_format_memories(memories)}"
        )

        response = await self._client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS_OUT,
            system=system_blocks,
            messages=[{"role": "user", "content": user_content}],
        )

        # Anthropic SDK exposes cache stats on `usage` when present.
        usage = response.usage
        cached = (
            getattr(usage, "cache_read_input_tokens", 0) or 0
        ) + (
            getattr(usage, "cache_creation_input_tokens", 0) or 0
        )
        text_parts = [
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ]
        hypothesis = "".join(text_parts).strip()
        return ReaderResponse(
            hypothesis=hypothesis,
            model=model,
            input_tokens=usage.input_tokens,
            cached_input_tokens=cached,
            output_tokens=usage.output_tokens,
        )


# ═══════════════════════════════════════════════════════════════
# RUN COMMANDS
# ═══════════════════════════════════════════════════════════════
#
# Library module. Requires ANTHROPIC_API_KEY in env. Invoked from adapter.py.
#
# ═══════════════════════════════════════════════════════════════
