# Weft Disaster Recovery Runbook

## Overview

Weft stores all memory data in a Supabase PostgreSQL instance (source of truth). Automated backups run every 12 hours via GitHub Actions, producing JSON files with full memory data including embeddings.

**Recovery chain (most to least complete):**

1. **JSON backup** — Full fidelity: memories, embeddings, relationships, metadata
2. **Markdown fallback** (`~/.weft/fallback.md`) — Text-only, no embeddings or relationships
3. **Manual re-ingestion** — `weft ingest` to bootstrap from codebase; conversations lost

## Scenario 1: Total Supabase Data Loss

Complete loss of all data in the Supabase Postgres instance.

### Steps

1. **Get the latest backup** from GitHub Actions artifacts:

```bash
# List recent workflow runs
gh run list --workflow=backup.yml --limit=5

# Download the most recent backup artifact
gh run download <RUN_ID> --name weft-backup-<RUN_ID>
```

Or from the `backups` branch (if commit-to-branch was used):

```bash
git fetch origin backups
git show origin/backups:weft-backup-latest.json > weft-backup-latest.json
```

2. **Verify the backup** before restoring:

```bash
weft restore weft-backup-latest.json --dry-run
```

Expected output shows memory count, relationship count, and 0 errors.

3. **Restore to Supabase** (migrations run automatically):

```bash
weft restore weft-backup-latest.json
```

4. **Verify the restore**:

```bash
# Connect via Fly.io MCP and check
weft_status()  # Should show restored memory counts
weft_recall("test query")  # Should return results
```

### Recovery time

- Backup download: < 1 minute
- Restore (50-100 memories): < 30 seconds
- Restore (1000+ memories): ~2-5 minutes

## Scenario 2: Partial Data Corruption

Some memories are corrupted or missing, but the database is still running.

### Steps

1. **Assess the damage**:

```bash
# Check current state via MCP
weft_status()

# Or connect directly
psql "$DATABASE_URL" -c "SELECT COUNT(*), status FROM memories GROUP BY status;"
```

2. **Restore missing data** (duplicates are skipped by default):

```bash
weft restore weft-backup-latest.json
```

This uses `ON CONFLICT DO NOTHING` — existing memories stay untouched, only missing ones are inserted.

3. **If you need to replace corrupted data**, delete the bad rows first:

```bash
psql "$DATABASE_URL" -c "DELETE FROM memories WHERE id = 'weft-XXXX';"
weft restore weft-backup-latest.json
```

## Scenario 3: Migrate Between Postgres Instances

Moving from one database to another (e.g., Supabase project A to project B, or cloud to local).

### Steps

1. **Backup from source**:

```bash
DATABASE_URL="postgresql://..." weft backup -o migration-export.json
```

2. **Point to the new database** and restore:

```bash
DATABASE_URL="postgresql://new-host/..." weft restore migration-export.json
```

Migrations run automatically before restore. The new instance will have identical data.

3. **Update secrets**:

```bash
# Fly.io
fly secrets set DATABASE_URL="postgresql://new-connection-string"

# GitHub Actions
gh secret set DATABASE_URL --body "postgresql://new-connection-string"

# Local .env
# Edit .env with new DATABASE_URL
```

## Scenario 4: Fly.io MCP Server Down

The Weft MCP server on Fly.io is unreachable but Supabase is fine.

### Steps

1. **Check Fly.io status**:

```bash
fly status -a weft-mcp
fly logs -a weft-mcp
```

2. **Restart or redeploy**:

```bash
fly apps restart weft-mcp
# or
fly deploy
```

3. **Fallback: run locally** while Fly.io recovers:

```bash
DATABASE_URL="postgresql://..." weft mcp
```

Then update `.mcp.json` to point to `http://localhost:8000/mcp`.

## Scenario 5: Paused Supabase Project (free tier)

Supabase free-tier projects auto-pause after ~1 week of inactivity. The first
connection after a pause is refused at the socket layer, so `weft` startup fails
with a raw `OSError`/`ConnectionRefusedError`.

`create_pool` (in `weft/db/connection.py`) intercepts this for Supabase DSNs:

- **No token configured (default):** it does **not** auto-restore — it raises a
  clear error pointing at the dashboard so you can unpause manually.
- **`SUPABASE_ACCESS_TOKEN` set:** it calls the Supabase Management API to
  restore (unpause) the project, polls up to 120s for `ACTIVE_HEALTHY`, then
  retries the connection automatically.

```bash
# Opt in to auto-restore (single-user / self-host only — see the warning):
export SUPABASE_ACCESS_TOKEN="sbp_..."   # https://supabase.com/dashboard/account/tokens
```

> ⚠️ **Security:** `SUPABASE_ACCESS_TOKEN` is an **account-scoped** Management
> API token — it can restore, modify, or delete **any** project on the account,
> not just this one. Only set it for a single-user deployment whose account you
> own. **Do not** set it in a hosted/multi-tenant deployment; leave it unset and
> rely on the manual-unpause path (or Supabase's "pause protection" on a paid
> plan).

To unpause manually instead, open
`https://supabase.com/dashboard/project/<project-ref>` and click *Restore*.

## Last Resort: Markdown Fallback

If no JSON backup is available, Weft can still serve memories from `~/.weft/fallback.md` — a text export without embeddings or relationships.

### Generate a fallback file

```bash
weft export --format markdown -o ~/.weft/fallback.md
```

### How it works

When the database is unreachable, Weft's fallback reader (`weft.fallback`) provides:
- `read_fallback()` — returns the full markdown contents
- `search_fallback(query)` — keyword search over exported sections

This is lossy (no embeddings, no relationships, no metadata like confidence or access counts) but ensures agents still have *some* memory context during outages.

### Rebuild from fallback

To get back to full fidelity from a markdown fallback:

1. Restore database connectivity
2. Run `weft ingest` to re-bootstrap from codebase
3. Manually re-enter critical memories via `weft_remember`

## Automated Backup Verification

The test suite includes a full backup/restore roundtrip test (`tests/test_backup_restore.py`) that verifies:

- Backup produces valid structure with checksums
- Embeddings survive the roundtrip as float arrays
- All memory fields are preserved (type, topic, confidence, pinned, etc.)
- Relationships are restored with correct foreign keys
- Duplicate handling works (skip or fail)
- Dry-run mode reports without modifying data
- JSON serialization/deserialization doesn't lose data
- Memories without embeddings restore cleanly

Run the verification:

```bash
uv run pytest tests/test_backup_restore.py -v
```

## Checklist: After Any Recovery

- [ ] Verify memory count matches expectations (`weft_status`)
- [ ] Test semantic recall works (`weft_recall("known topic")`)
- [ ] Confirm relationships are intact (check related memories)
- [ ] Run a fresh backup immediately after recovery
- [ ] Verify the GitHub Actions backup workflow still has valid secrets
