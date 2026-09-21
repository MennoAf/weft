"""Provider-neutral, label-blind runtime shape for LongMemEval.

Dataset labels and gold fields are evaluation metadata. They must never decide
retrieval or generation behavior. This module derives the small runtime
routing contract from the question and permitted haystack text.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import re


@dataclass(frozen=True, slots=True)
class TaskShape:
    """The only task classification allowed across the runtime boundary."""

    task_shape: str
    routing_class: str
    top_k: int = 10

    def __post_init__(self) -> None:
        if not self.task_shape or not self.routing_class:
            raise ValueError("task shape and routing class must be non-empty")
        if self.top_k <= 0:
            raise ValueError("task shape top_k must be positive")


_DATE_RE = re.compile(
    r"\b(?:\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?|"
    r"january|february|march|april|may|june|july|august|"
    r"september|october|november|december|yesterday|today|tomorrow|"
    r"before|after|latest|earlier|recent|last)\b", re.IGNORECASE,
)
_MULTI_RE = re.compile(
    r"\b(?:how many|how much|all|every|each|between|compare|compared|"
    r"first|second|third|multiple|different|total|list|which sessions?)\b",
    re.IGNORECASE,
)


def _session_text(sessions: Iterable[object] | None) -> str:
    parts: list[str] = []
    for session in sessions or ():
        if isinstance(session, str):
            parts.append(session)
            continue
        if isinstance(session, Mapping):
            parts.extend(str(session.get(key, "")) for key in ("content", "text"))
            turns = session.get("turns", ())
            if isinstance(turns, Iterable) and not isinstance(turns, (str, bytes)):
                for turn in turns:
                    if isinstance(turn, Mapping):
                        parts.append(str(turn.get("content", "")))
                    else:
                        parts.append(str(getattr(turn, "content", turn)))
            continue
        turns = getattr(session, "turns", ())
        if isinstance(turns, Iterable) and not isinstance(turns, (str, bytes)):
            parts.extend(str(getattr(turn, "content", turn)) for turn in turns)
        else:
            content = getattr(session, "content", None)
            if content is not None:
                parts.append(str(content))
    return "\n".join(parts)


def derive_task_shape(
    question: str,
    sessions: Iterable[object] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> TaskShape:
    """Derive deterministic routing solely from permitted runtime inputs."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if metadata:
        forbidden = {"question_type", "answer", "answer_session_ids", "has_answer"}
        leaked = forbidden.intersection(metadata)
        if leaked:
            raise ValueError(f"gold metadata is not a runtime input: {sorted(leaked)}")
    text = f"{question}\n{_session_text(sessions)}"
    temporal = bool(_DATE_RE.search(text))
    multi = bool(_MULTI_RE.search(question))
    if temporal and multi:
        return TaskShape("temporal-multi", "turns", 30)
    if temporal:
        return TaskShape("temporal", "turns", 10)
    if multi:
        return TaskShape("multi-session", "turns", 30)
    return TaskShape("single-session", "belief", 10)


def routing_class_for(question: str, sessions: Iterable[object] | None = None) -> str:
    return derive_task_shape(question, sessions).routing_class


task_shape_for = derive_task_shape
shape_for = derive_task_shape

__all__ = ["TaskShape", "derive_task_shape", "routing_class_for", "task_shape_for", "shape_for"]
