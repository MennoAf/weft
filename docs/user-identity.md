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

The deployed server (`weft-mcp.fly.dev`) extracts `user_id` per-request
from the caller's Supabase JWT. Flow:

```
Authorization: Bearer <jwt>
  → weft.auth.extract_user_id_from_header (decodes the sub claim)
  → current_user_id ContextVar
  → weft.db.connection.set_user_context (SET LOCAL app.user_id)
  → INSERTs use nullif(current_setting('app.user_id', true), '')
```

The JWT `sub` claim is the identity. No config file involved.

**Single-tenant fallback.** While Supabase user auth isn't wired through
the Claude Code MCP client (blocked on upstream OAuth token-exchange
support), the server supports a `WEFT_DEFAULT_USER_ID` env var. When the
API key gate accepts a request and no valid JWT is attached, writes are
stamped with the default UUID. The fallback is coupled to API-key
authentication — anonymous traffic never inherits the default identity.
Clear or unset the variable the moment a second user arrives; JWT sub
already takes precedence when present, so shipping real auth doesn't
require pulling the fallback first.

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
