# Backup and Disaster Recovery

Weft stores memories and related metadata in PostgreSQL. The database is the
source of truth; backups are portable JSON exports that can be restored to
another PostgreSQL instance.

This is generic self-hosted guidance. Scheduled backup automation, cloud
account topology, and provider-specific runbooks are deployment concerns for
the operator of each installation.

## Export a backup

Create a full backup with the CLI:

```bash
weft backup -o weft-backup.json
```

The export contains memory data, embeddings when present, relationships, and
supported metadata. Treat backup files as sensitive memory data. Store them in
an encrypted location with access controls, and do not commit them to Git.

## Validate and restore

Validate a backup without writing to the database:

```bash
weft restore weft-backup.json --dry-run
```

Restore after confirming the target database and migration state:

```bash
weft restore weft-backup.json
```

A restore is designed to be repeatable: existing rows are skipped where the
backup format supports idempotent keys. Always verify counts and representative
recall after a restore.

## Migrate between PostgreSQL instances

Export from the source, point the CLI at the destination, and restore:

```bash
WEFT_DATABASE_URL="postgresql://source-host/db" \
  weft backup -o migration-export.json

WEFT_DATABASE_URL="postgresql://destination-host/db" \
  weft restore migration-export.json
```

Use a database role with the permissions required by the backup/restore
operation. Re-issue application credentials for the destination; credentials
are not part of a memory backup.

## Recovery checklist

1. Confirm the backup file came from a trusted source.
2. Run the dry-run validation and record memory/relationship counts.
3. Confirm the destination schema is at the expected migration version.
4. Restore into the intended database.
5. Verify counts, representative recall, and relationships.
6. Take a fresh backup after successful recovery.
7. Remove temporary copies using the storage system's secure deletion policy.

## Text fallback

A Markdown export can provide a lossy emergency fallback when the database is
unavailable:

```bash
weft export --format markdown -o ~/.weft/fallback.md
```

The Markdown fallback preserves readable text but not embeddings,
relationships, confidence metadata, or access history. Once the database is
available, restore from the JSON backup when possible and manually re-enter any
critical memories that exist only in the fallback.

## Security notes

- Never place backup files, database URLs, API keys, OAuth keys, or access
  tokens in the repository.
- Keep backup storage and restore credentials separate from application
  credentials.
- Do not disable TLS verification as a routine recovery step.
- Test restore procedures against a disposable database before relying on them
  for an incident.
