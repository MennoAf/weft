# Local Docker RC

This is the isolated container launch contract for a local Weft release candidate.
It is separate from `docker-compose.weft.yml`, which remains the host-native
`weft up` infrastructure path and is unchanged.

> **EXPERIMENTAL CHECKPOINT — NOT RELEASE READY.** The current local DB-owner
> RLS bypass exposes other owners' Prime data. Use isolated synthetic testing
> only: never use personal data or shared deployments. The final strengthened
> acceptance is expected to fail its distinct-owner gate; platform verification
> remains pending.

## Contract

`docker-compose.local.yml` starts three services:

- `postgres` (`pgvector/pgvector:pg16`) and `redis` (`redis:7-alpine`) are
  reachable only on the private Compose network. Neither publishes a host port.
- `app` runs the candidate image's declared `CMD` (`python -m weft.mcp`) with
  `WEFT_DATABASE_URL` and `WEFT_REDIS_URL` pointed at Compose service DNS.
- Only the app's HTTP endpoint is published, and only on host loopback:
  `127.0.0.1:${WEFT_LOCAL_PORT:-18000}`.
- Compose health gates app startup on Postgres and Redis health.
- The app owns local schema setup with `WEFT_MIGRATION_MODE=apply`. This is a
  local-only invocation setting; the production Dockerfile still requires an
  explicit migration mode and does not default to `apply`.
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
docker build --tag weft-rc-local:1.0.0rc1 .
uv run python scripts/local_docker_acceptance.py \
  --image weft-rc-local:1.0.0rc1 \
  --compose-file docker-compose.local.yml \
  --receipt artifacts/local-docker-rc-20260912/local-docker-acceptance.json
```

The runner generates a unique Compose project, owner UUID, API key, host port,
and synthetic project ID. It calls `/healthz`, initializes streamable HTTP MCP,
then uses the real MCP tools `weft_remember`, `weft_recall`, `weft_prime`, and
`weft_handoff`. Recall checks require the returned memory ID, exact content, and
explicit project evidence; prime checks require the returned authoritative
handoff ID/content. The runner also proves wrong-project and distinct-owner
negative isolation with actual result collections. It restarts app/Postgres/Redis
and repeats the ID/content recall and handoff checks. Finally it hard-deletes
only returned synthetic memory IDs and runs
`docker compose ... down --volumes --remove-orphans` for its unique project even
when startup fails partway. Any cleanup error changes the receipt to failed and
returns a nonzero exit. The receipt is machine-readable
(`schema=weft.local-docker-acceptance.v1`) and contains no bearer value.

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
the first FastEmbed download. The tested image is multi-architecture capable
only to the extent that the declared `python:3.13-slim`, FastEmbed, and ONNX
artifacts publish usable host architecture variants; this run proves the
current Docker host only. Run separately on Mac arm64 and Linux amd64 before
claiming a matrix.

## Safety notes

- Never put a real API key, user ID, or existing project ID in the Compose file.
- Use a fresh receipt path for every run; failed receipts are useful evidence.
- Do not change `WEFT_LOCAL_IMAGE` to a production image without reviewing the
  migration and embedding contract.
- Do not publish DB/cache ports to a LAN. The local development DB password is
  intentionally development-only and is overridable via `WEFT_LOCAL_DB_PASSWORD`.
- This contract does not alter Fly/Supabase deployment files or host-native
  `weft up` semantics.
