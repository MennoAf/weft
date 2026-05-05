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

from anthropic import AsyncAnthropic

from weft.models import MemoryRecall

from benchmarks.longmemeval.dataset import ABSTENTION_TYPES

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
        "This question is about a user preference. The relevant preference may "
        "have been stated about a RELATED topic in a prior session — preferences "
        "from a Seattle trip apply to a Miami trip; preferences about baking "
        "with one ingredient apply to baking with another. Identify the "
        "preference signal even when the question's topic is not literal-match "
        "in memory, and answer in terms of how that preference applies to the "
        "current question. Abstain only if no preference is present anywhere "
        "in memory."
    ),
    "multi-session": (
        "This question requires synthesizing information across multiple past "
        "sessions. Combine the relevant memories before answering."
    ),
    "knowledge-update": (
        "The relevant fact may have been UPDATED across sessions. If memories "
        "contradict, trust the most recent one based on session date."
    ),
    "temporal-reasoning": (
        "This question requires reasoning about WHEN events happened. Use the "
        "session dates and today's date to compute durations, ordering, or "
        "relative timing. Show only the final answer, not the calculation."
    ),
}

_ABSTENTION_SUFFIX = (
    "\n\nABSTENTION CASE: This question may not have an answer in the memories. "
    "If the memories do not actually contain the requested information, respond "
    "with exactly: I don't know."
)


def _system_prompt_for(question_type: str) -> str:
    """Build the Reader's system prompt for a given question type.

    Stable for any given (question_type) input — important so the prompt
    cache can reuse the same hash across the run. Suffix for abstention
    types is appended deterministically.
    """
    base_type = question_type.removesuffix("_abs")
    type_block = _TYPE_INSTRUCTIONS.get(
        base_type, "Answer the question from the supplied memories.",
    )
    prompt = f"{_BASE_INSTRUCTIONS}\n{type_block}"
    if question_type in ABSTENTION_TYPES:
        prompt += _ABSTENTION_SUFFIX
    return prompt


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
    ):
        self._client = client or AsyncAnthropic()
        self._model = model
        self._abstention_model = abstention_model

    async def read_answer(
        self,
        *,
        question: str,
        question_date: str,
        question_type: str,
        memories: list[MemoryRecall],
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
        if top_k is not None and len(memories) > top_k:
            memories = memories[:top_k]

        system_prompt = _system_prompt_for(question_type)
        model = (
            self._abstention_model
            if question_type in ABSTENTION_TYPES
            else self._model
        )

        # System prompt is stable per question_type → cacheable. The
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
