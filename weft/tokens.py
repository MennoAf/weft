"""Token estimation for context budget management.

Provides a simple heuristic by default (~4 chars per token).
Uses tiktoken when available for more accurate counts.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_tiktoken_enc = None
_tiktoken_available: bool | None = None


def _get_tiktoken():
    """Lazy-load tiktoken encoder. Returns None if not installed."""
    global _tiktoken_enc, _tiktoken_available
    if _tiktoken_available is not None:
        return _tiktoken_enc
    try:
        import tiktoken

        _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
        _tiktoken_available = True
        logger.debug("Using tiktoken for token estimation")
    except ImportError:
        _tiktoken_available = False
        logger.debug("tiktoken not available, using heuristic")
    return _tiktoken_enc


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string.

    Uses tiktoken (cl100k_base) if available, otherwise falls back
    to a ~4 chars per token heuristic.
    """
    if not text:
        return 0
    enc = _get_tiktoken()
    if enc is not None:
        return len(enc.encode(text))
    return max(1, len(text) // 4)


def estimate_tokens_heuristic(text: str) -> int:
    """Always use the heuristic estimator (for testing/consistency)."""
    if not text:
        return 0
    return max(1, len(text) // 4)
