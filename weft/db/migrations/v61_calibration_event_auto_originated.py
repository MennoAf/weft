"""Migration 61: add auto_originated provenance column to policy_calibration_events.

PROOF-integrity hardening (Sieve P1, loom-5a7ea3e7). The calibration loop's
aliveness PROOF metric — count_auto_originated_tier_changes — previously
identified auto-originated tier changes by matching ``reason LIKE
'auto-calibration%'``. But ``reason`` is free-text and user-writable via
update_policy_tier, so any caller could mint a row whose reason starts with
``auto-calibration`` and inflate the PROOF metric (make a dead loop look alive).

The fix is a dedicated boolean provenance column set TRUE *only* by the
auto-promotion path (_maybe_auto_promote → update_policy_tier(auto_originated=
True)). The count query switches to ``WHERE auto_originated = TRUE``, which is
not reachable from the free-text reason field. Existing rows default to FALSE:
historically the only writer of auto-calibration reasons was the genuine
auto-promotion path, but defaulting FALSE is the safe, non-spoofable choice —
the metric simply resumes counting from the next genuine auto-promotion.

A partial index supports the time-windowed PROOF query (auto_originated=TRUE
within a recent window) without scanning manual events.
"""

from __future__ import annotations

VERSION = 61
DESCRIPTION = "add auto_originated provenance column to policy_calibration_events"
SQL = r"""
ALTER TABLE policy_calibration_events
    ADD COLUMN IF NOT EXISTS auto_originated BOOLEAN NOT NULL DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_calibration_events_auto_originated
    ON policy_calibration_events (created_at DESC)
    WHERE auto_originated;
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
