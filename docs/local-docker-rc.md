# Local Docker RC

This is the isolated container launch contract for a local Weft release candidate.
It is separate from `docker-compose.weft.yml`, which remains the host-native
`weft up` infrastructure path and is unchanged.

> **EXPERIMENTAL CHECKPOINT — NOT RELEASE READY.** This local RC now separates
> owner-only schema setup from the long-running app connection. The Compose app
> must run as non-owning `weft_app` (`NOSUPERUSER`, `NOBYPASSRLS`, no role
> memberships) with ordinary RLS effective. Use isolated synthetic testing only:
> never use personal data or shared deployments. Independent verification and a
> fresh Docker acceptance receipt are still required; this document is not a
> release claim.

## Contract

`docker-compose.local.yml` starts five services:

- `postgres` (`pgvector/pgvector:pg16`) and `redis` (`redis:7-alpine`) are
  reachable only on the private Compose network. Neither publishes a host port.
- `app` runs the candidate image's declared `CMD` (`python -m weft.mcp`) with
  `WEFT_DATABASE_URL` and `WEFT_REDIS_URL` pointed at Compose service DNS.
- Only the app's HTTP endpoint is published, and only on host loopback:
  `127.0.0.1:${WEFT_LOCAL_PORT:-18000}`.
- Compose health gates app startup on Postgres and Redis health.
- `migrate` is a one-shot owner-only service that runs `weft migrate` against
  the `weft` owner connection. `bootstrap` then runs the local helper
  `weft.local_bootstrap` to provision `weft_app` with an explicit, source-grounded
  table/operation allowlist. User-scoped CRUD and the directly-called workspace,
  calibration, credential, identity, access-log, counter, and tool-usage telemetry
  tables are intentional trusted-runtime exceptions; dormant `oauth_*` tables are
  denied. Identity/serial sequences receive only `USAGE, SELECT` (never `UPDATE`
  or `setval`), and the runtime role receives no schema DDL or ledger writes.
  Provisioning is restricted to the dedicated `weft` database and its `public`
  schema: it derives `current_database()` and `current_user` (the migration
  object creator), resets both global role configuration and the
  `ALTER ROLE ... IN DATABASE ...` state stored in `pg_db_role_setting`, and
  revokes current and default table/sequence privileges from both `weft_app`
  and `PUBLIC` (global and `public`-schema defaults). Verification audits
  `pg_default_acl`, effective PUBLIC ACLs, and creates owner-owned future table /
  sequence probes after reprovisioning. Expected DDL, unknown-relation, and
  `setval` denials explicitly roll back their transactions and prove the same
  connection remains usable; a database with no allowlisted identity/serial
  sequence records an explicit no-sequence skip rather than claiming coverage.
- `app` receives only the `weft_app` DSN and starts with
  `WEFT_MIGRATION_MODE=verify`. It therefore verifies the migration ledger and
  RLS/runtime invariants but cannot apply schema migrations as the app role.
  The production Dockerfile and hosted deployment files are unchanged.
- `WEFT_API_KEY` and `WEFT_DEFAULT_USER_ID` are required runtime interpolation
  values. The API key is inserted into the local database's legacy credential
  row, bound to the supplied owner UUID, and is never baked into an image or
  committed file.
- `WEFT_OUTBOUND_CONNECTOR=none` and
  `WEFT_QUARANTINE_REVIEW_ENABLED=0` keep integration dispatch and the
  periodic Anthropic quarantine-review worker off. The API key is therefore
  used only for MCP bearer authentication, not as a paid-provider key.

## Build and acceptance

The standard `.dockerignore` excludes `.env`, `.env.*`, `.loom`, `.polytoken`,
`artifacts`, tests, and nested RC worktrees. Verify this boundary before a
build, then build from the repository root:

```bash
docker build --tag weft-rc-local:1.0.0rc1 . && uv run python scripts/local_docker_acceptance.py \
  --image weft-rc-local:1.0.0rc1 \
  --compose-file docker-compose.local.yml \
  --receipt evidence/rc-finish-line/mcp-journey/local-docker-acceptance.json
```

The runner generates a unique Compose project, owner UUID, API key, host port,
and synthetic project ID. Compose first migrates as the owner and provisions the
restricted `weft_app` role; the runner then verifies the effective role metadata
inside the app container (including RLS, ownership, memberships, and ledger
write denial). It calls `/healthz`, initializes streamable HTTP MCP, then uses
the real MCP tools `weft_remember`, `weft_recall`, `weft_prime`, and
`weft_handoff`. The journey additionally captures a `preference` with its
validated `preference_metadata`, revises it, reads the real `supersedes`
relationship, requires the active successor while excluding the archived old
version, and hard-deletes the successor before proving it cannot be recalled.
All positive checks require returned IDs, exact content, and explicit project
scope; negative checks inspect actual result entries by ID or content. The
runner also proves wrong-project and distinct-owner negative isolation with
actual result collections. It restarts app/Postgres/Redis, requires a new MCP
session identity, and repeats the ID/content recall and handoff checks.
Finally it hard-deletes only returned synthetic memory IDs and runs
`docker compose ... down --volumes --remove-orphans` for its unique Compose
project, removing that project's containers, volumes, and networks even when
startup fails partway. The candidate image is intentionally retained; the
runner has no image-deletion operation. Any cleanup error changes the receipt to failed and
returns a nonzero exit. The receipt is machine-readable
(`schema_version=2`, `schema=weft.local-docker-acceptance.v2`) and contains no bearer value. Its
`persistence_evidence` section labels MCP/container persistence separately from
`installed_library_persistence: not_run_host_only`.

Before any resource-affecting operation, the runner performs one fresh-output validation of the receipt, provisional checkpoint, completion marker, and optional status targets. It registers a fresh run identity and output paths before invoking Compose. It is the sole normal cleanup owner: after `up` is attempted it performs one scoped `down`, independent of API-key availability. Every Compose, health, and MCP operation is admitted against the remaining aggregate acceptance deadline; per-operation `--timeout` is only a cap, and retry sleeps consume the same ledger. Synthetic-memory deletion runs first only within its reserved bounded window, preserving the Compose-down and publication reserves. Compose teardown receives the configured `--cleanup-kill-grace`; TERM/KILL, drain, and reap overhead are admitted inside the cleanup reserve rather than added afterward. Terminal publication replaces only this run's generation-matching provisional checkpoint, rejects foreign/preexisting terminal targets, and writes checkpoint, receipt, optional `status.txt`, and then a completion marker containing exact SHA-256 digests. The CLI blocks INT/TERM with POSIX `pthread_sigmask` at one freeze event and keeps them blocked through publication and termination; the library workflow does not alter its caller's mask. Consumers must validate the marker, generation, path roles, digests, selected return code, and external wait outcome. The round2 launcher is a thin `exec` adapter and rejects a zero cleanup cap; it does not own cleanup. Historical round1 and private Linux launchers are evidence artifacts, not supported v3 entry points.

The existing `scripts/rc_smoke.py` remains the lower-level provider-free
Postgres/Redis smoke. This acceptance runner does not replace or rerun it.

## Environment and model prerequisites

The default FastEmbed model is `BAAI/bge-small-en-v1.5`, exposed as a padded
768-dimensional vector to match the pgvector schema. FastEmbed is local and
requires no paid API key, but a cold image/model cache may need outbound access
to download the model. A preloaded cache can be used for a no-network follow-up;
this local RC does **not** claim cold-offline operation.

The acceptance runner does not supply Anthropic, OpenAI, Slack, Discord, or
Supabase credentials. It is provider-free apart from model-registry access for
the first FastEmbed download. The synthetic MCP preference/revision/deletion
journey makes no paid-provider calls. Installed-library persistence is a
separate qualification label and is not established by this host-only runner;
this runner proves persistence through the MCP/container path only. The tested
image is multi-architecture capable
only to the extent that the declared `python:3.13-slim`, FastEmbed, and ONNX
artifacts publish usable host architecture variants; this run proves the
current Docker host only. Run separately on Mac arm64 and Linux amd64 before
claiming a matrix.

## Harness evidence scope (overnight recovery)

The acceptance runner's credential-free host harness is an evidence tool, not a
production or platform qualification. Its F4 durable-output contract applies a
finite grammar to every checkpoint/receipt value, HTTP error, cleanup warning,
primary error, verifier detail, and captured stream: common `Authorization:
Bearer ...`, `API_KEY`/`API-KEY`/`X-Api-Key` labels, `WEFT_*KEY`/`PASSWORD`/`TOKEN`
labels, and `DATABASE_URL`/Postgres/Redis DSN forms are redacted. Runtime values
that this runner generates are also registered exactly before use. Arbitrary
unknown secret formats are not promised. Redaction occurs before bounded
serialization; streams and receipt containers are capped, and truncation is
bounded at the documented edges.

F5 checkpoints persist a named `running` phase before each Compose, health,
initialize, HTTP tool/auth, isolation, restart, persistence, and cleanup
operation, then `passed` or `failed` afterward. Failed phases retain sanitized
errors; interrupted active phases remain failed and verifier `probe_details` and
explicit `skips` are retained. These guarantees are tested with fake Compose,
HTTP, and workflow seams only. The test harness does not run Docker, Compose,
Postgres, Redis, Linux amd64, remote services, paid providers, production,
canonical probes, or `tests/test_store.py`; it makes no SIGKILL or escaped-process
containment guarantee. The round2 wrapper and historical launchers remain
separate artifacts and are not silently rewritten by this evidence pass.

## Safety notes

- Never put a real API key, user ID, or existing project ID in the Compose file.
- Use a fresh receipt path for every run; failed receipts are useful evidence.
- Do not change `WEFT_LOCAL_IMAGE` to a production image without reviewing the
  migration and embedding contract.
- Do not publish DB/cache ports to a LAN. The local development DB password is
  intentionally development-only and is overridable via `WEFT_LOCAL_DB_PASSWORD`.
- This contract does not alter Fly/Supabase deployment files or host-native
  `weft up` semantics.
