# User Identity

Weft tags every row it writes with a `user_id` so queries can filter by
owner. There are two execution paths that resolve `user_id` differently;
this document explains how they compose and how to bind them to a single
canonical identity.

## The two paths

### Local MCP client

A local Weft install (e.g. `uv run python -m weft`) reads its identity
from `weft.config.user_identity.get_user_id()`. Resolution order:

1. `WEFT_USER_ID` environment variable — wins if set, not persisted.
2. `~/.weft/user_id.json` `user_id` field — explicit persisted identity.
3. Random UUID fallback — generated on first call, persisted to the file.

The local client passes this value as the `user_id` parameter on MCP tool
calls. It never reads it from anywhere else.

### Hosted MCP server

The deployed server resolves identity per-request through the
Authorization header. Phase 2.5 made bearer tokens first-class
credentials — each issued token binds a `user_id` AND a `caller_mode`
('supervisor' or 'agent') at issuance time, persisted as a SHA-256 hash
in `weft_tokens`. Resolution path:

```
Authorization: Bearer <token>
  → weft.credentials.lookup_token (sha256 → row)
  → row.user_id     → current_user_id ContextVar
  → row.caller_mode → current_caller_mode ContextVar
  → weft.db.connection.acquire (SET LOCAL app.user_id)
  → INSERTs use nullif(current_setting('app.user_id', true), '')
```

A token row is the identity. No JWT decode in the hot path.

**JWT fallback.** If the bearer doesn't match a token row and Supabase
OAuth is enabled (`WEFT_OAUTH_ENABLED=1`), the server falls back to
verifying the bearer as a Supabase JWT. The `sub` claim becomes the
user_id and caller_mode defaults to 'supervisor' (until OAuth scope
claims ship). A token row always wins over JWT — same bearer, same
identity, predictable resolution.

**Legacy WEFT_API_KEY.** Existing deployments that ship the server's
bearer via the `WEFT_API_KEY` env var still work: at lifespan startup,
Weft hashes the env value and inserts a matching token row labeled
`legacy-env-key`, bound to `WEFT_DEFAULT_USER_ID` and supervisor mode.
Idempotent across restarts. The credential then resolves through
`lookup_token` like any other bearer; a deprecation warning fires on
each successful resolution. Migrate to per-client issued tokens via
`weft tokens issue` (or the `weft_token_issue` MCP tool) and rotate
before the env-var path is removed.

### Caller modes

Every token is bound at issuance to one of two trust tiers:

- **`supervisor`** — full write authority. The Face (Claude Code with
  the human in the loop) and the operator's CLI. Supervisor tokens may
  *downgrade* themselves to agent for testing by sending
  `X-Weft-Caller-Mode: agent` on the request.
- **`agent`** — agent containers (e.g. Wick) running autonomously.
  Writes flagged as instruction-shaped by Layer 3 land in
  `review_status='pending_review'` and stay invisible to recall until
  the supervisor approves them via `weft_quarantine_review`. Agent
  tokens **cannot escalate** — the row's `caller_mode` is the floor;
  any `X-Weft-Caller-Mode: supervisor` header on an agent token is
  ignored.

The caller mode is a row-level fact. Read by `current_caller_mode` in
`store_memory` (stamped into `memories.write_provenance`) and by
`weft_quarantine_review` / `weft_token_*` (which reject agent-mode
calls outright).

## The binding problem

Without coordination, the same human ends up with two distinct user_ids —
a random local UUID and their JWT `sub`. Data written on one path is
surfaced on the other only via OR-NULL filtering (which matches truly-
global rows, not rows owned by the other identity).

## How to bind them

Set your local `user_id` to your JWT `sub`. Two ways:

```bash
# Option 1 — persist via CLI (writes ~/.weft/user_id.json)
weft identity set <your-jwt-sub>

# Option 2 — ephemeral env override (shell session only)
export WEFT_USER_ID=<your-jwt-sub>
```

Verify:

```bash
weft identity show
# user_id: <your-jwt-sub>
# source:  from /Users/you/.weft/user_id.json
```

Now local MCP calls send the same `user_id` the hosted server derives
from your JWT. Queries from either path see your complete data.

## Finding your JWT sub

If you access Weft through Claude.ai or a Supabase-authenticated client,
your `sub` is the `auth.users.id` UUID. Ways to retrieve it:

- Claude.ai: not directly exposed — use the Supabase dashboard or
  decode a JWT via `python -c "import jwt; print(jwt.decode(<token>, options={'verify_signature': False})['sub'])"`.
- Supabase dashboard → Authentication → Users → your row → `id`.

## Legacy NULL rows

Rows written before user-scoping shipped had `user_id = NULL`. Migration
36 (`Schema v1: SYSTEM_GLOBAL sentinel + NOT NULL user_id + RLS rewrite`)
backfilled every NULL to the `__system_global_zathras__` sentinel and
added `NOT NULL` to every user-scoped table — so going forward, a row
with no owner is structurally impossible. Forgetting to set
`app.user_id` becomes a fail-loud constraint violation rather than a
silent global write.

If you need to reassign legacy rows from the sentinel to your own
identity (e.g. memories you wrote pre-identity-binding that should
become yours rather than system-owned), do it as a one-off SQL
operation with the explicit user_id you want to stamp. There is no
automated tool — the deliberate friction is intentional.

## Why this design

- **No DB alias table.** Our legacy corpus (pre-Phase-1) is almost
  entirely NULL or JWT-sub-owned. No accumulated data under a random
  local UUID to preserve, so a mapping table would be overhead for a
  problem we don't have.
- **Config-driven single identity** scales to the common case (one human
  → one canonical ID) without infrastructure.
- **Explicit overrides on admin operations** (env var, function arg)
  keep prod-touching scripts safe by removing implicit dependence on
  whatever `~/.weft/user_id.json` happens to contain.

If multiple distinct identities accrue for the same human later (e.g. a
second auth provider), revisit with a `user_identities` table that maps
aliases to a canonical ID, and expand the OR-NULL filter to an
`IN (canonical, alias1, alias2, ...)` clause.
