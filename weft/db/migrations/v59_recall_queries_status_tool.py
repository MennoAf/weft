"""Migration 59: allow 'status' in weft_recall_queries.tool_name.

``weft_status`` now logs each topic ask as a ``weft_recall_queries`` row so the
L1 Resolution Ratchet compounding loop can read its ``was_empty`` signal
(``result_count == 0``) the same way it reads ``weft_recall`` / ``weft_search_all``
asks. The v50 table constrained ``tool_name IN ('recall', 'search_all')``, which
would reject the new ``'status'`` rows — this migration widens the CHECK.

The constraint is dropped by its auto-generated name
(``weft_recall_queries_tool_name_check``) and recreated with an explicit name so
future migrations can target it deterministically. ``IF EXISTS`` keeps the drop
idempotent across partially-migrated environments.
"""

from __future__ import annotations

VERSION = 59
DESCRIPTION = "weft_recall_queries.tool_name: allow 'status' (weft_status ratchet feed)"
SQL = r"""
ALTER TABLE weft_recall_queries
    DROP CONSTRAINT IF EXISTS weft_recall_queries_tool_name_check;

ALTER TABLE weft_recall_queries
    ADD CONSTRAINT weft_recall_queries_tool_name_check
    CHECK (tool_name IN ('recall', 'search_all', 'status'));
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
