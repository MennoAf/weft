"""Migration 31: Ensure user_id column exists and is nullable on all user-scoped tables"""

from __future__ import annotations

VERSION = 31
DESCRIPTION = 'Ensure user_id column exists and is nullable on all user-scoped tables'
SQL = r"""
-- Add user_id to behaviors if missing (created in migration 10 but ensure it exists)
ALTER TABLE behaviors ADD COLUMN IF NOT EXISTS user_id TEXT;

-- Add user_id to entities if missing
ALTER TABLE entities ADD COLUMN IF NOT EXISTS user_id TEXT;

-- Add user_id to episodes if missing
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS user_id TEXT;

-- Add user_id to modes if missing (created in migration 20 but ensure it exists)
ALTER TABLE modes ADD COLUMN IF NOT EXISTS user_id TEXT;

-- Add user_id to autonomy_policies if missing (created in migration 24 but ensure it exists)
ALTER TABLE autonomy_policies ADD COLUMN IF NOT EXISTS user_id TEXT;

-- Add user_id to calibration_records if missing (created in migration 29 but ensure it exists)
ALTER TABLE calibration_records ADD COLUMN IF NOT EXISTS user_id TEXT;

-- Add user_id to degradation_policies if missing (created in migration 30 but ensure it exists)
ALTER TABLE degradation_policies ADD COLUMN IF NOT EXISTS user_id TEXT;

-- Create indexes for efficient user_id filtering (idempotent)
CREATE INDEX IF NOT EXISTS idx_behaviors_user ON behaviors (user_id);
CREATE INDEX IF NOT EXISTS idx_entities_user ON entities (user_id);
CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes (user_id);
CREATE INDEX IF NOT EXISTS idx_modes_user ON modes (user_id);
CREATE INDEX IF NOT EXISTS idx_autonomy_policies_user ON autonomy_policies (user_id);
CREATE INDEX IF NOT EXISTS idx_calibration_records_user ON calibration_records (user_id);
CREATE INDEX IF NOT EXISTS idx_degradation_policies_user ON degradation_policies (user_id);
"""

MIGRATION = (VERSION, DESCRIPTION, SQL)
