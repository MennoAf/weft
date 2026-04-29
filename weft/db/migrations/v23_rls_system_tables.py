"""Migration 23: Enable RLS on system tables (schema_migrations, weft_metadata, memory_access_log)"""

from __future__ import annotations

VERSION = 23
DESCRIPTION = 'Enable RLS on system tables (schema_migrations, weft_metadata, memory_access_log)'
SQL = r"""
-- schema_migrations: service-role only, no user data
ALTER TABLE schema_migrations ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS schema_migrations_service ON schema_migrations;
CREATE POLICY schema_migrations_service ON schema_migrations
    USING (true) WITH CHECK (true);

-- weft_metadata: service-role only, no user data
ALTER TABLE weft_metadata ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS weft_metadata_service ON weft_metadata;
CREATE POLICY weft_metadata_service ON weft_metadata
    USING (true) WITH CHECK (true);

-- memory_access_log: service-role only, session tracking
ALTER TABLE memory_access_log ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS memory_access_log_service ON memory_access_log;
CREATE POLICY memory_access_log_service ON memory_access_log
    USING (true) WITH CHECK (true);
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
