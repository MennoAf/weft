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
- omitted `project_id` returns within an outer five-second deadline when roots capability is absent, roots returns empty, or the roots callback remains silent;
- the silent callback exercises the server's two-second reverse-RPC timeout;
- explicit `project_id` bypasses roots even if the client callback is silent;
- progressive and full disclosure respect a 500-token output budget.

The current public `weft_prime` schema supports `progressive` and `full`. It does **not** support a `minimal` disclosure value; the approved plan's minimal-mode check is therefore an API/spec mismatch, not silently claimed coverage. Adding a new mode requires a separate product decision.

The in-process harness covers MCP framing, session management, registration, and dispatch but not TCP, TLS, Fly proxy behavior, or real credentials. Before public release, run a redacted staging smoke against the deployed `/mcp` URL with a staging token and all three roots behaviors where the client supports them. Record commit/image, timestamp, client/version, elapsed time, disclosure, token counts, and status—never the bearer, DSN, raw memories, or user IDs.

## Required Postgres role topology

Use separate roles:

| Role | Purpose | Required attributes |
|---|---|---|
| migration/owner role | migrations, ownership, backups/admin | May own tables; credentials unavailable to ordinary MCP requests |
| application role | Fly MCP connection pool | `LOGIN`, `NOSUPERUSER`, `NOBYPASSRLS`; must not own user-scoped tables |

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

## Gate status

- Repository Streamable HTTP registered-tool harness: **PASS**.
- Local production-equivalent RLS role/CRUD suite: **PASS**.
- Real Fly Streamable HTTP smoke: **EXTERNAL / PENDING ACCESS**.
- Real Supabase effective-role and staging CRUD verification: **EXTERNAL / PENDING ACCESS**.

A deployed role mismatch blocks multi-user public release. It does not block single-user local evaluation.
