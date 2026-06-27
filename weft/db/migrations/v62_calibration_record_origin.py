"""Migration 62: add origin trust dimension to calibration_records.

Trust-boundary hardening (Sieve P1, loom-173bd297). Auto-promotion drives an
action's autonomy tier all the way to ``always`` purely from the approval rate
over calibration_records, with no check that those approvals came from a
trusted origin. In the multi-tenant SaaS path this is a privilege-escalation
seam: an autonomous agent (untrusted writer, per the Phase-2 four-layer
defense) could mint its own approving calibration records via weft_calibrate
and thereby grant itself unattended autonomy.

The fix records WHO attested each calibration outcome. ``origin`` mirrors the
existing caller-mode trust tiers in weft/auth.py: 'supervisor' (Face / human /
Orchestrator — trusted) vs 'agent' (autonomous agent in a Wick container —
untrusted). record_calibration stamps it from get_caller_mode() at write time.
The promotion path then counts ONLY trusted-origin approvals, so forged
agent-origin approvals cannot drive a tier to 'always'. Demotion still counts
all origins — an untrusted rejection can still raise a human-review alert
(fail-safe direction).

Default 'supervisor' preserves the single-user / pre-Phase-2 behavior: every
historical row and every caller that doesn't set the agent caller-mode header
stays trusted, exactly as today. Only callers that explicitly opt into the
'agent' lower-trust tier produce untrusted records.
"""

from __future__ import annotations

VERSION = 62
DESCRIPTION = "add origin trust dimension to calibration_records"
SQL = r"""
ALTER TABLE calibration_records
    ADD COLUMN IF NOT EXISTS origin TEXT NOT NULL DEFAULT 'supervisor';

CREATE INDEX IF NOT EXISTS idx_calibration_records_origin
    ON calibration_records (origin);
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
