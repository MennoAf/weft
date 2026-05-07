# CLI Reference

Every Weft CLI command. Run `weft --help` for the most current list.

## Server lifecycle

| Command | Description |
|---------|-------------|
| `weft mcp` | Start the MCP server (stdio transport) |
| `weft up` | Start Postgres + Redis containers, run migrations |
| `weft down` | Stop containers (data persists in named volumes) |
| `weft status` | Show memory statistics |

## Memory operations

| Command | Description |
|---------|-------------|
| `weft recall QUERY` | Semantic search — `--limit`, `--topic` |
| `weft import FILE` | Import a `MEMORY.md` file — `--dry-run`, `--project-id` |
| `weft export` | Export memories — `--format md\|json`, `--type`, `--topic`, `--status`, `--output` |
| `weft consolidate` | Run decay/dedup/contradiction pipeline — `--dry-run` |

## Ingestion

| Command | Description |
|---------|-------------|
| `weft ingest PATH` | Ingest a codebase as memories — `--project-id`, `--depth` |
| `weft obsidian init VAULT_PATH` | Create a Weft-optimized Obsidian vault structure |
| `weft obsidian sync VAULT_PATH` | Sync vault contents into Weft — `--dry-run`, `--hash-store` |

See [obsidian.md](obsidian.md) for the Obsidian integration in detail.

## Backup + restore

| Command | Description |
|---------|-------------|
| `weft backup` | Create a full backup — `--output` |
| `weft restore FILE` | Restore from backup — `--dry-run` |

See [disaster-recovery.md](disaster-recovery.md) for the recovery playbook.

## Configuration

| Command | Description |
|---------|-------------|
| `weft config show` | Display current configuration (defaults + overrides + sources) |
| `weft config set KEY VALUE` | Persist a config value to `~/.weft/config.toml` |

See [configuration.md](configuration.md) for the full key reference.

## Identity + tokens

| Command | Description |
|---------|-------------|
| `weft identity show` | Show resolved local user_id and source |
| `weft identity set USER_ID` | Persist user_id to `~/.weft/user_id.json` |
| `weft tokens issue` | Mint a bearer token — `--user-id`, `--mode supervisor\|agent`, `--label`, `--expires-in` |
| `weft tokens list` | List a user's tokens — `--user-id`, `--include-revoked` |
| `weft tokens revoke HASH` | Revoke a token by full SHA-256 hash |

See [user-identity.md](user-identity.md) for the resolution chain, caller-mode semantics, and the agent-floor / non-escalation guarantee.
