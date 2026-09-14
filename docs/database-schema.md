# Weft database and schema guide

This is the GitHub-readable companion to [`docs/database-schema.json`](database-schema.json), the machine-readable public-schema manifest. It describes the **47 public tables** currently represented by the manifest: **11 exported** tables and **36 excluded** tables. The manifest is the inventory authority; this guide explains the boundaries a maintainer needs to understand without connecting to a database.

The manifest's migration head is **v74** at the time this guide was written. That is execution-time information, not a promise that a future checkout or installation has the same head. Refresh both the manifest and this guide's review when an additive migration lands.

## Table groups

The table names below are grouped by the manifest's `category`. Each row gives the table's purpose, its ownership/RLS boundary, whether it is in the portable logical export, and what the restricted application runtime may do. `scoped CRUD` means the runtime operates only within the authenticated user/project or workspace policy; it does not mean the role owns the table.

### User data (10)

| Table | Purpose | Ownership / RLS | Export | Restricted runtime |
|---|---|---|---|---|
| `behaviors` | Persistent agent rules and strategies. | user/project scope; enabled, scoped CRUD | included | read-write, scoped CRUD |
| `check_ins` | User mood, sleep, and energy check-ins. | user scope; enabled, scoped CRUD | excluded: not in current portable sections | read-write, scoped CRUD |
| `entities` | Entity graph nodes extracted from memory. | user/project scope; enabled, scoped CRUD | included | read-write, scoped CRUD |
| `entity_mentions` | Links entities to source memories. | user scope; enabled, scoped CRUD | included | read-write, scoped CRUD |
| `memories` | Canonical durable memories and embeddings. | user/project scope; enabled, sentinel or user/project policies | included | read-write, scoped CRUD |
| `memory_relationships` | Directed relationships between memories. | user scope; enabled, sentinel or user policies | included | read-write, scoped CRUD |
| `modes` | Named retrieval personas and weight overrides. | user scope; enabled, scoped CRUD | included | read-write, scoped CRUD |
| `trackers` | Open-loop tracker state and nudges. | user/project scope; enabled, scoped CRUD | included | read-write, scoped CRUD |
| `workspace_members` | Workspace membership and identity edges. | workspace membership; enabled, service and member policies | included | read-write, service/member policies |
| `workspaces` | Collaborative workspace definitions. | workspace membership; enabled, service and member policies | included | read-write, service/member policies |

### Continuity and relationship data (7)

| Table | Purpose | Ownership / RLS | Export | Restricted runtime |
|---|---|---|---|---|
| `belief_claims` | Versioned belief claims anchored to evidence turns. | user/project scope; enabled, scoped CRUD | excluded: derived retrieval-view substrate | read-write, scoped CRUD |
| `episode_memories` | Links episodes to memories. | user scope; enabled, scoped CRUD | included | read-write, scoped CRUD |
| `episode_turns` | Append-oriented conversational turn trace. | user/project scope; enabled, scoped CRUD | excluded: operational conversation trace | read-write, scoped CRUD |
| `episodes` | Episodic timeline containers. | user/project scope; enabled, scoped CRUD | included | read-write, scoped CRUD |
| `shuttle_claims` | Current-value observer and synthesis blackboard. | user/project scope; enabled, scoped CRUD | excluded: derived operational blackboard | read-write, scoped CRUD |
| `topic_digests` | Cached topic digest materialization. | user/project scope; enabled, scoped CRUD | excluded: rebuildable cache | read-write, scoped CRUD |
| `topic_resolution_aliases` | Topic-resolution aliases for recall. | user/project scope; enabled, scoped CRUD | excluded: derived recall index state | read-write, scoped CRUD |

### Operational data (11)

| Table | Purpose | Ownership / RLS | Export | Restricted runtime |
|---|---|---|---|---|
| `alert_state` | Alert cooldown, deduplication, and suppression state. | user scope; enabled, scoped CRUD | excluded: ephemeral scheduler state | read-write, scoped CRUD |
| `alerts` | Scheduled proactive push notifications. | user scope; enabled, scoped CRUD | excluded: scheduler state | read-write, scoped CRUD |
| `autonomy_overrides` | Temporary cost-to-autonomy overrides. | user/project scope; enabled, scoped CRUD | excluded: ephemeral operational state | read-write, scoped CRUD |
| `autonomy_policies` | Per-user autonomy and action policy configuration. | user/project scope; enabled, scoped CRUD | excluded: operational policy state | read-write, scoped CRUD |
| `calibration_records` | Agent action outcome calibration records. | user/project scope; enabled, scoped CRUD | excluded: operational calibration history | read-write, scoped CRUD |
| `cost_enforcement_state` | Daily cost enforcement state. | service policy; enabled, service policy | excluded: ephemeral operational state | read-write, service policy |
| `cost_entries` | Provider token and cost accounting entries. | user/project scope; enabled, scoped CRUD | excluded: operational accounting state | read-write, scoped CRUD |
| `degradation_policies` | Provider/cost degradation policy rules. | user/project scope; enabled, scoped CRUD | excluded: operational policy state | read-write, scoped CRUD |
| `policy_calibration_events` | Autonomy policy calibration event history. | user/project scope; enabled, scoped CRUD | excluded: operational calibration history | read-write, scoped CRUD |
| `replay_queue` | Pending episode replay work queue. | user/project scope; enabled, scoped CRUD | excluded: ephemeral queue state | read-write, scoped CRUD |
| `triggers` | Proactive condition-driven rules. | user/project scope; enabled, scoped CRUD | excluded: operational scheduler configuration | read-write, scoped CRUD |

### Telemetry and audit data (11)

| Table | Purpose | Ownership / RLS | Export | Restricted runtime |
|---|---|---|---|---|
| `board_feedback_proposals` | Board feedback-engine proposal log. | user/project scope; enabled, scoped CRUD | excluded: operational correction telemetry | read-write, scoped CRUD |
| `board_triage_events` | Board triage correction-loop event log. | user/project scope; enabled, scoped CRUD | excluded: operational correction telemetry | read-write, scoped CRUD |
| `memory_access_log` | Memory access audit trail. | service policy; enabled, service policy | excluded: audit and ephemeral access state | read-write, service policy |
| `recall_canary` | Retrieval health canary enrollment and probes. | user/project scope; enabled, scoped CRUD | excluded: health-monitoring state | read-write, scoped CRUD |
| `recall_canary_audit` | Per-run canary outcome log. | user/project scope; enabled, scoped CRUD | excluded: health-monitoring telemetry | read-write, scoped CRUD |
| `turn_access_log` | Session and turn read-access audit trail. | service policy; enabled, service policy | excluded: audit and ephemeral access state | read-write, service policy |
| `weft_counters` | Global named telemetry counters. | service policy; enabled, service policy | excluded: aggregate telemetry | write, aggregate service telemetry |
| `weft_recall_queries` | Per-call recall query observation log. | user/project scope; enabled, scoped CRUD | excluded: telemetry, not portable memory data | read-write, scoped CRUD |
| `weft_recovery_attempts` | Bounded retrieval-recovery stage telemetry. | user/project scope; enabled, scoped CRUD | excluded: operational recovery telemetry | read-write, scoped CRUD |
| `weft_tool_usage_coverage` | Tool usage recorder coverage and drain health. | service policy; enabled, service policy | excluded: service telemetry | write, service telemetry |
| `weft_tool_usage_daily` | Daily aggregate MCP tool usage counts. | service policy; enabled, service policy | excluded: aggregate telemetry | write, service telemetry |

### Auth and credential data (5)

| Table | Purpose | Ownership / RLS | Export | Restricted runtime |
|---|---|---|---|---|
| `oauth_access_revocations` | OAuth access-token revocation records. | service policy; enabled, service policy | excluded: credential and ephemeral auth state | read-write, auth path only |
| `oauth_authorization_codes` | Short-lived OAuth authorization codes. | service policy; enabled, service policy | excluded: credential and ephemeral auth state | read-write, auth path only |
| `oauth_clients` | OAuth dynamic-client registrations. | service policy; enabled, service policy | excluded: credential and auth state | read-write, auth path only |
| `oauth_refresh_tokens` | Rotated OAuth refresh tokens. | service policy; enabled, service policy | excluded: credentials | read-write, auth path only |
| `weft_tokens` | Hashed Weft API tokens and caller modes. | service policy; enabled, service policy before user context | excluded: credential and auth state | read-write, auth path only; not portable data |

### Migration and system metadata (3)

| Table | Purpose | Ownership / RLS | Export | Restricted runtime |
|---|---|---|---|---|
| `audit_backfill_user_id` | Audit trail for historical user-id backfill. | migration owner; RLS not enabled by source migration | excluded: migration audit metadata | none; runtime does not use it |
| `schema_migrations` | Append-only applied migration ledger. | migration owner; enabled, service policy | excluded: recreated by migrations | read-only; SELECT only, never ledger writes |
| `weft_metadata` | System-level key/value metadata. | service policy; enabled, service policy | excluded: system metadata | read-only; no migration ownership |

## RLS and runtime roles

The application connects with a restricted application runtime role. It is deliberately not a schema owner and has no `SUPERUSER`, `BYPASSRLS`, public-table ownership, or role memberships. Row-level security (RLS) is enabled on the user-scoped tables and policies bind ordinary reads and writes to the request's user/project or workspace context. Service-policy tables are narrow operational/auth paths, not a way to bypass user isolation.

The database owner/operator is a separate authority. That owner creates extensions, owns tables, grants the runtime role its allowlisted operations, and applies schema changes. The runtime can use scoped CRUD where the table row above says so, read the migration ledger where allowed, and record narrowly scoped service telemetry; it cannot alter DDL, write the migration ledger, acquire ownership, grant privileges, or bypass RLS. In a hosted restricted deployment, `WEFT_MIGRATION_MODE=verify` performs read-only ledger and runtime-invariant checks. `apply` belongs only to an intentional owner-managed migration operation.

These are boundaries, not production connection instructions. Configure the database and credentials through the normal environment/configuration mechanisms described in [`docs/configuration.md`](configuration.md); never paste a credential or a connection value into this guide, an issue, or a commit.

## Owner-managed migrations

Migrations are **append-only** Python modules under `weft/db/migrations/`. Discovery imports only sibling files matching `vNN_*.py`, orders them by ascending numeric version, rejects duplicate versions, and collects each module's `(version, description, sql)` tuple. To add schema, add the next numbered migration; do not edit an applied migration, rewrite the ledger, or change history in place.

The current manifest head is **v74** at execution time. It records the discovered migration descriptions, but it is not a substitute for running the refresh/check workflow against a new checkout. The owner-managed process applies pending DDL under the migration runner's advisory lock and records it in `public.schema_migrations`. The restricted runtime instead verifies that the ledger exactly matches the discovered code set and fails closed when versions are missing or unknown. No production database is changed by this documentation.

A normal startup in a local owner-managed installation may use apply mode. A restricted hosted runtime should use verify mode after the owner has migrated the database. This separation prevents startup code from silently becoming a schema administrator and makes migration ownership auditable.

## pgvector width and codec ordering

The public embedding contract is **`vector(768)`** for embedding columns, including the memory, behavior, entity, episode, and episode-turn vector paths. The local FastEmbed model may natively produce 384 values; Weft pads that output to the configured 768 width. OpenAI's default embedding path uses 768-dimensional Matryoshka output. Providers and configuration must therefore produce vectors that fit the existing 768-dimensional schema; changing width is a migration and re-embedding decision, not a per-request workaround.

The connection layer sets the search path, then registers the pgvector text codec on each pool connection. **Migrations run first; codec registration follows migrations**, because the migration that creates the `vector` extension/type must exist before asyncpg can resolve and register that codec. Future pool connections receive the same initializer. Codec-less backup/restore paths use the pgvector text literal form and explicit casts, so exported vectors remain portable without relying on a live connection codec.

## Export exclusions

The portable JSON backup is logical memory data, not an infrastructure snapshot. The 11 included tables are `behaviors`, `entities`, `entity_mentions`, `episode_memories`, `episodes`, `memories`, `memory_relationships`, `modes`, `trackers`, `workspace_members`, and `workspaces`. They cover durable memories plus the supported relationships, continuity, entity, workspace, tracker, and retrieval-configuration sections.

The other 36 tables are intentionally excluded. Exclusions cover scheduler and queue state, caches and derived indexes, audit/access and health telemetry, accounting/calibration/degradation state, the migration ledger/system metadata, and OAuth/API-token material. In particular, credentials, tokens, ephemeral leases, operational telemetry, and infrastructure state are never part of a portable memory backup. Follow [`docs/disaster-recovery.md`](disaster-recovery.md) for the backup/restore workflow, and treat an export as sensitive memory data even though it excludes credentials.

An excluded table may still be readable or writable by a restricted runtime path when its manifest contract allows that operation. “Excluded” describes portability, not a blanket SQL privilege decision; ownership/RLS and runtime access are the separate columns shown above.

## Dormant-file status

`weft/db/migrations/pending_v51_episode_turns_fts.py` is a tracked **dormant** draft, not an applied migration. Its filename does **not** match `vNN_*.py`, so it is **not discovered** by the runner and is absent from the current v74 migration set. It must remain dormant until the documented episode-turn volume gate is reached (the draft names approximately 25,000 turns as the promotion trigger).

Promotion is an explicit future decision: rename the file to `v51_episode_turns_fts.py`, then let the owner-managed migration process discover and apply it. Do not rename or edit it as part of ordinary runtime startup, and do not edit any already-applied `vNN_*.py` file. Until promotion, episode-turn keyword recall uses the existing inline text-search path; the dormant draft's generated `search_tsv`/GIN index is not an active schema promise.

## Keeping this guide current

When the migration set changes, refresh `docs/database-schema.json` and review this guide together. Check the discovered head, public table inventory, ownership/RLS, runtime boundary, export state, and dormant-file disposition. Keep examples generic and credential-free: public documentation must not contain database URLs, API keys, access tokens, passwords, or private filesystem paths.
