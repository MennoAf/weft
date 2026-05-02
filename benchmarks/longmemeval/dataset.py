#!/usr/bin/env python3
"""
dataset.py — LongMemEval dataset loader and typed records.

Reads the upstream LongMemEval JSON splits (`longmemeval_oracle.json`,
`longmemeval_s.json`, `longmemeval_m.json`) into typed dataclasses so the
adapter doesn't have to chase string keys.

Author:  Jason Bauman
Version: 0.1.0
Date:    2026-04-30
Python:  >= 3.12

Dependencies:
    (stdlib only)

Usage:
    from benchmarks.longmemeval.dataset import load_split
    instances = load_split(Path("longmemeval_oracle.json"))
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


# Question types that explicitly permit refusal. The Reader's system prompt
# is instructed to abstain when the evidence does not support an answer.
ABSTENTION_TYPES: frozenset[str] = frozenset(
    {
        "single-session-user_abs",
        "single-session-assistant_abs",
        "single-session-preference_abs",
        "multi-session_abs",
        "knowledge-update_abs",
        "temporal-reasoning_abs",
    }
)


@dataclass(frozen=True, slots=True)
class Turn:
    """One conversational turn within a session."""

    role: str  # "user" | "assistant"
    content: str

    @classmethod
    def from_dict(cls, d: dict) -> "Turn":
        return cls(role=d["role"], content=d["content"])


@dataclass(frozen=True, slots=True)
class Session:
    """A timestamped multi-turn conversation. Unit of memory ingestion."""

    session_id: str
    date: str  # ISO-ish date string from the dataset (e.g. "2023/04/15")
    turns: tuple[Turn, ...]
    has_answer: bool = False  # True iff this session contains the gold evidence

    def to_text(self) -> str:
        """Render the session as a single text blob for embedding/storage.

        Format:
            Session date: <date>
            <role>: <content>
            <role>: <content>
            ...
        """
        lines = [f"Session date: {self.date}"]
        lines.extend(f"{turn.role}: {turn.content}" for turn in self.turns)
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class Instance:
    """One benchmark question with its full haystack of sessions."""

    question_id: str
    question_type: str
    question: str
    answer: str  # Gold answer — never given to the system, only the judge.
    question_date: str
    sessions: tuple[Session, ...]

    @property
    def is_abstention(self) -> bool:
        """True iff the question is an abstention type (refusal expected)."""
        return self.question_type in ABSTENTION_TYPES

    @classmethod
    def from_dict(cls, d: dict) -> "Instance":
        sessions: list[Session] = []
        # Upstream uses parallel arrays: haystack_session_ids[i],
        # haystack_dates[i], haystack_sessions[i]. answer_session_ids names
        # which sessions carry evidence (used only for has_answer flag).
        evidence_ids = set(d.get("answer_session_ids") or [])
        ids = d["haystack_session_ids"]
        dates = d["haystack_dates"]
        haystacks = d["haystack_sessions"]
        if not (len(ids) == len(dates) == len(haystacks)):
            raise ValueError(
                f"Malformed instance {d.get('question_id')}: "
                f"haystack arrays disagree in length "
                f"({len(ids)}, {len(dates)}, {len(haystacks)})"
            )
        for sid, date, turns in zip(ids, dates, haystacks):
            sessions.append(
                Session(
                    session_id=sid,
                    date=date,
                    turns=tuple(Turn.from_dict(t) for t in turns),
                    has_answer=sid in evidence_ids,
                )
            )
        return cls(
            question_id=d["question_id"],
            question_type=d["question_type"],
            question=d["question"],
            answer=d["answer"],
            question_date=d["question_date"],
            sessions=tuple(sessions),
        )


def load_split(path: Path) -> list[Instance]:
    """Load an entire LongMemEval split file from disk.

    Args:
        path: Path to one of `longmemeval_{oracle,s,m}.json` (or the
            `_cleaned.json` variants from the HuggingFace mirror).

    Returns:
        List of typed ``Instance`` records, ordered as in the file.

    Raises:
        FileNotFoundError: If the split file does not exist.
        ValueError: If any instance has malformed haystack arrays.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Instance.from_dict(item) for item in raw]


def iter_split(path: Path) -> Iterator[Instance]:
    """Streaming variant of ``load_split`` for large splits (LongMemEval_M).

    The upstream files are JSON arrays (not JSONL), so we still parse the
    full document in memory — but yielding one at a time means downstream
    code can drop a finished Instance for GC instead of holding the list.
    """
    for instance in load_split(path):
        yield instance


# ═══════════════════════════════════════════════════════════════
# RUN COMMANDS
# ═══════════════════════════════════════════════════════════════
#
# This module has no CLI of its own — it's a library. Used by adapter.py.
# Quick sanity check from a REPL:
#
#   uv run python -c "
#   from pathlib import Path
#   from benchmarks.longmemeval.dataset import load_split
#   xs = load_split(Path('data/longmemeval_oracle.json'))
#   print(f'{len(xs)} instances; first: {xs[0].question_type} / '
#         f'{len(xs[0].sessions)} sessions')
#   "
#
# ═══════════════════════════════════════════════════════════════
