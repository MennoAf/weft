"""Migration 9: Add index on agent_id for scoped queries"""

from __future__ import annotations

VERSION = 9
DESCRIPTION = 'Add index on agent_id for scoped queries'
SQL = r"""
CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories (agent_id);
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
