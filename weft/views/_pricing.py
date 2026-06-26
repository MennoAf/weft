"""Shared Haiku model id + pricing — single source of truth.

Both the belief-view detector (``belief_detector.py``) and the topic-digest
synthesizer (``topic_synthesis.py``) call the same Haiku model and derive USD
costs from the same per-token rates. Previously each module re-encoded those
rates — topic_synthesis as explicit constants, belief_detector inline in its
docstring math — so a price change had to be chased across two files. They now
import from here.

Haiku 4.5 pricing (as of 2026):
  - Input:  $1.00 / 1M tokens
  - Output: $5.00 / 1M tokens
"""

from __future__ import annotations

# Haiku model id used by both the detector and the synthesizer.
HAIKU_MODEL = "claude-haiku-4-5-20251001"

# Per-token Haiku rates.
HAIKU_INPUT_PRICE_PER_TOKEN = 1.00 / 1_000_000   # $1.00 / 1M tokens
HAIKU_OUTPUT_PRICE_PER_TOKEN = 5.00 / 1_000_000  # $5.00 / 1M tokens


def haiku_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Return the USD cost of a Haiku call for the given token counts."""
    return (
        input_tokens * HAIKU_INPUT_PRICE_PER_TOKEN
        + output_tokens * HAIKU_OUTPUT_PRICE_PER_TOKEN
    )
