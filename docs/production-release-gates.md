# Production release gates: MCP transport and Postgres isolation

These checks distinguish repository behavior from deployed topology. Passing local testcontainers and in-process HTTP tests is necessary but not proof that Fly and Supabase use the intended roles/configuration.

## Streamable HTTP prime gate

The release-gating transport is Streamable HTTP, matching Fly. `tests/test_prime_streamable_http.py` uses the production FastMCP singleton and registered tool provider over:

```text
FastMCP Client
  -> StreamableHttpTransport
  -> HTTPX ASGI transport
  -> FastMCP Streamable HTTP session manager
  -> registered tools/list + tools/call
  -> weft_prime
```

It does not call the Python handler directly. The test verifies:

- `weft_prime` appears in `tools/list`;
- the active FastMCP request transport—not ambient process configuration—controls roots behavior;
- Streamable HTTP and SSE never issue the `roots/list` reverse RPC, so omitted `project_id` returns within an outer deadline even when the client advertises a silent roots callback;
- stdio retains bounded roots discovery for local clients, even if a stale HTTP environment value is present;
- explicit `project_id` remains correctly scoped even if the client callback is silent;
- progressive and full disclosure respect a 500-token output budget.

The suite also serves the production FastMCP ASGI app through Uvicorn on a real loopback TCP socket. This catches response-stream and cancellation-lifecycle behavior hidden by HTTPX's in-process `ASGITransport`, and asserts clean server shutdown after a silent-roots call.

On HTTP/SSE, omitted `project_id` is deliberately **user-wide**, not repo-scoped: project filters and project handoff continuity are absent, while authenticated-user/workspace RLS still applies. The response reports `project_resolution.scope="user-wide"` and warns callers to pass an explicit ID. Agents that require repository isolation must provide `project_id`; the liveness fallback must not be described as scoped.

The current public `weft_prime` schema supports `progressive` and `full`. It does **not** support a `minimal` disclosure value; the approved plan's minimal-mode check is therefore an API/spec mismatch, not silently claimed coverage. Adding a new mode requires a separate product decision.

The local harness covers MCP framing, session management, registration, dispatch, TCP, and Uvicorn shutdown, but not TLS, Fly proxy behavior, or real credentials. Before public release, run a redacted staging smoke against the deployed `/mcp` URL. Record commit/image, timestamp, client/version, elapsed time, disclosure, token counts, and status—never the bearer, DSN, raw memories, or user IDs.

## Required Postgres role topology

Use separate roles:

| Role | Purpose | Required attributes |
|---|---|---|
| migration/owner role | migrations, ownership, backups/admin | May own tables; credentials unavailable to ordinary MCP requests |
| application role | Fly MCP connection pool | `LOGIN`, `NOSUPERUSER`, `NOBYPASSRLS`; must not own user-scoped tables |

The owner role applies migrations out of band. Fly must set
`WEFT_MIGRATION_MODE=verify`, which performs only a read-only exact comparison
against `public.schema_migrations` and fails startup if owner-managed migrations
are pending. Do not grant DDL or ownership to the application role to make a
deploy pass.

For the current policy design, `FORCE ROW LEVEL SECURITY` is not required when the application role is not the table owner. If production connects as the owner, ordinary RLS is bypassable and the release gate fails unless FORCE RLS is deliberately enabled and verified. A service/admin role that intentionally bypasses RLS must be isolated from request paths and documented separately.

Local production-equivalent assertions live in `tests/test_rls_pentest.py` and `tests/test_rls_invariants.py`:

- effective app role is login-enabled, non-superuser, and lacks `BYPASSRLS`;
- app role differs from the `memories` owner;
- RLS is enabled and required CRUD policies exist;
- two users are isolated for SELECT/INSERT/UPDATE/DELETE;
- unset identity cannot write;
- an invalid identity cannot write another user's row;
- attempts to disable row security fail.

## Deployed read-only role probe

Run with administrative visibility but redact role/database names in stored artifacts if sensitive:

```sql
SELECT current_user,
       r.rolsuper,
       r.rolbypassrls,
       r.rolcanlogin
FROM pg_roles r
WHERE r.rolname = current_user;

SELECT c.relname,
       c.relrowsecurity,
       c.relforcerowsecurity,
       owner.rolname AS owner_name,
       current_user = owner.rolname AS app_is_owner
FROM pg_class c
JOIN pg_roles owner ON owner.oid = c.relowner
WHERE c.oid = 'public.memories'::regclass;
```

Release requirements for the application connection:

- `rolsuper = false`;
- `rolbypassrls = false`;
- `app_is_owner = false`, or `relforcerowsecurity = true` with reviewed implications;
- `relrowsecurity = true`;
- the same DSN/role setup is used by Fly's production `create_pool()` path.

Then provision isolated synthetic users in staging and run the CRUD matrix through the production connection setup. Do not perform destructive probes against real user rows. Use synthetic/redacted content and delete only the staging fixture.

## Supabase pooler requirements verified during cutover

The restricted pooler login preserves Supabase's tenant suffix while changing
the database role prefix. The suffix and credentials are secrets and must not
be written to logs or artifacts.

Supabase installs pgvector in the `extensions` schema and its pooler resets new
sessions to `"$user", public`. The application role therefore needs `USAGE` on
`extensions`, and `weft.db.connection._pgvector_codec_init` explicitly sets
`search_path TO public, extensions` before registering the codec. A startup
schema check alone is insufficient evidence: the release gate includes a real
vector-backed write through the production MCP boundary.

## Gate status

- Repository Streamable HTTP registered-tool harness: **PASS**.
- Local production-equivalent RLS role/CRUD suite: **PASS**.
- Real Fly Streamable HTTP smoke: **PASS** on 2026-07-20 UTC, main commit `8a0861d`, image `deployment-01KY065WB2XV30E12YF6AHMC0R`. Redacted elapsed times: absent roots 4.170s, empty roots 2.733s, silent advertised roots 2.932s, explicit progressive 2.657s, explicit full 2.856s. All calls respected the 500-token budget; omitted-project responses reported `scope="user-wide"`; the ephemeral five-minute supervisor token was revoked immediately after the matrix.
- Real Supabase effective-role and CRUD verification: **PASS** on 2026-07-20 UTC, Fly v163 / commit `351103b` / image `deployment-01KXZY9DRYNX6SHAHAVZF5ETX4`.
- Real vector-backed production writes: **PASS** (`weft_learn` and `weft_handoff`) after the pooler search-path fix.

The production application connection reports login enabled, `SUPERUSER`,
`CREATEROLE`, `CREATEDB`, and `BYPASSRLS` false, and zero public-object
ownership. Synthetic two-user INSERT/SELECT/UPDATE/DELETE isolation passed;
unset identity failed loudly; fixtures were removed. Stored evidence excludes
DSNs, passwords, tenant suffixes, synthetic user IDs, and memory content.
