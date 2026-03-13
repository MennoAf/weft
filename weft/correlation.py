"""Correlation ID for tracing tool invocations through store/cache/embedding layers.

Usage:
    from weft.correlation import correlation_id, set_correlation_id

    # At the tool boundary (in tools.py):
    set_correlation_id()  # generates a new short ID

    # In any module's logger output, the CorrelationFilter
    # (attached at server startup) automatically injects [req:xxxxx].
"""

from __future__ import annotations

import contextvars
import logging
import uuid

correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None,
)


def set_correlation_id(cid: str | None = None) -> str:
    """Set a correlation ID for the current context. Returns the ID."""
    cid = cid or uuid.uuid4().hex[:8]
    correlation_id.set(cid)
    return cid


class CorrelationFilter(logging.Filter):
    """Inject correlation_id into log records for structured tracing."""

    def filter(self, record: logging.LogRecord) -> bool:
        cid = correlation_id.get()
        record.correlation_id = cid or "-"  # type: ignore[attr-defined]
        return True
