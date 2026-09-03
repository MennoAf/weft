# Offline Handoff — Weft MCP Recovery

**Date:** 2026-09-03
**Purpose:** Preserve the complete recovery context before shutting down/rebooting Polytoken and Codex. This document intentionally contains no API keys, bearer tokens, database passwords, or other credentials.

## Current production state

Weft production is healthy again on Fly.io.

- Fly app: `weft-mcp`
- Hostname: `weft-mcp.fly.dev`
- MCP endpoint: `https://weft-mcp.fly.dev/mcp`
- Health endpoint: `https://weft-mcp.fly.dev/healthz`
- Last verified health response: `HTTP 200`, `{"status":"ok"}`
- Current recovery machine created during deployment: `8027d0a6609328`
- Deployment strategy: blue/green
- Production embedding calls observed after recovery: OpenAI returned HTTP 200
- Production config: Streamable HTTP, `WEFT_MIGRATION_MODE=verify`, OpenAI embedding configuration supplied through production settings/secrets

## Incident root cause

Both Polytoken and Codex lost Weft because the Fly machine entered a crash loop and stopped. The decisive Fly log was:

```text
RuntimeError: Database migration verification failed: unknown applied versions=[72]. Run owner-managed migrations before restarting Weft.
```

The machine exited with code 1 and reached its maximum restart count:

```text
machine has reached its max restart count of 10
```

Meaning:

- The production database already contained migration 72.
- The running application image did not contain migration 72 in its discovered migration set.
- Production uses `WEFT_MIGRATION_MODE=verify`, so startup correctly refuses to run with a database/code migration mismatch.
- This was a server deployment/image mismatch, not a Polytoken or Codex credential/configuration failure.

## Recovery performed

A clean recovery branch was created from the last remote production baseline, not from the stale local branch:

```text
Branch: recovery/v72-production-clean
Commit: eac5821 fix(recovery): align production migration ledger with v72
```

That commit contains exactly one file:

```text
weft/db/migrations/v72_retrieval_recovery_attempts.py
```

The recovery image retained:

- OpenAI as the embedding default/runtime provider
- FastMCP 2.x dependency range
- Root private Fly configuration targeting `weft-mcp`
- Streamable HTTP transport
- Existing production runtime behavior
- Migration v73
- Added migration v72

It did **not** include:

- RC retrieval-recovery implementation
- FastEmbed default changes
- FastMCP 3.x
- LongMemEval generated data or snapshots
- Internal benchmark artifacts
- The dirty development worktree

Deployment command used from `/tmp/weft-production-recovery-clean`:

```bash
fly deploy --app weft-mcp --config fly.toml
```

Deployment completed successfully. The new green machine passed its health check, the old crash-looping machine was stopped/destroyed, and public health returned HTTP 200.

## Important branch/history correction

The first recovery branch was rejected by GitHub because it was based on stale local `main`, whose reachable history contained this 228 MB internal artifact:

```text
benchmarks/longmemeval/snapshots/baseline_v1_local/manifest.json
```

The clean replacement branch was based on remote commit `0a65b76`, and its reachable history contains no blobs over 100 MB and no generated LongMemEval snapshot. It was pushed successfully as:

```text
origin/recovery/v72-production-clean
```

The earlier branch remains local only:

```text
recovery/v72-production-alignment
```

The later remote `origin/main` advanced to include v72 and v74 plus the broader RC history. Do not deploy `origin/main` merely because it is canonical: it has not yet been approved as the production image that replaced the recovery deployment.

## Repository state now

Local `main` was deliberately synchronized with remote `origin/main` after preserving local work:

```text
main == origin/main == 3ce273f
```

The final synchronization check was:

```text
0 0
```

The following local work was preserved and must not be deleted casually:

- Internal history branch: `internal/benchmark-research-20260903`
- Original recovery branch: `recovery/v72-production-alignment`
- Clean recovery branch: `recovery/v72-production-clean`
- Stash: `stash@{0}` with message:
  `preserve internal benchmark and RC work before main alignment 2026-09-03`
- Ignored local worktrees remain on disk:
  - `.ci-rc-worktree/`
  - `.rc-candidate-worktree/`

The private production `fly.toml` was restored locally and is excluded through `.git/info/exclude`. It is not in public Git history. The public repository contains only sanitized deployment examples.

## Deployment guard added

`main` now contains:

```text
scripts/deploy_production_guard.py
scripts/deploy_production.sh
```

Use:

```bash
./scripts/deploy_production.sh
```

The guard requires:

- branch exactly `main`
- local `main` synchronized with `origin/main`
- no ordinary dirty or untracked deployment-ref files
- private local `fly.toml` targeting `weft-mcp`
- production Streamable HTTP and `/healthz` settings
- no tracked generated benchmark data/snapshots/runs/results
- no reachable Git blob over GitHub's 100 MB limit
- Dockerfile does not copy benchmark content into the image

The known ignored local worktree directories are allowed to remain physically present. Other changes block deployment.

**Current limitation:** the guard protects branch cleanliness and artifact boundaries, but `main` currently includes broader RC history. A future improvement should require an explicitly approved production tag or release ref before deploying, rather than treating every synchronized `main` as production-approved.

## Reboot/reconnect test

After rebooting or restarting Polytoken:

1. Open a fresh session in this Weft repository.
2. Run `/prime`.
3. Confirm that Weft tools are listed and `weft_prime` returns.
4. Run one simple recall/read operation.
5. Repeat from Codex if needed.

The expected hosted registration is in the global Claude configuration and uses:

```text
https://weft-mcp.fly.dev/mcp
```

Do not copy credentials into this repository's `.mcp.json`. The project-local `.mcp.json` currently contains Loom only; the hosted Weft registration is global/client-side.

## If reconnect fails after reboot

First check the server before changing client config:

```bash
curl -i https://weft-mcp.fly.dev/healthz
fly status -a weft-mcp
fly logs -a weft-mcp --no-tail
```

Interpretation:

- `HTTP 200` plus `{"status":"ok"}`: Fly startup is healthy; investigate client MCP registration, client session cache, or token handling.
- `HTTP 503`: inspect Fly logs first. Do not change Polytoken config yet.
- `unknown applied versions=[N]`: deployed image is missing a database migration. Do not edit the database ledger manually; deploy code containing that migration.
- `pending versions=[N]`: image has a migration the database lacks. Use the owner-managed migration process, not the restricted production runtime.
- `401` on `/mcp`: token mismatch/rotation issue. Do not paste the token into chat or Git; compare/rotate it through the private credential path.
- `HTTP 200` health but MCP tools unavailable: check that the client is using the hosted HTTP entry, not the project-local Loom-only `.mcp.json`; restart the client session after server recovery.

## Do not repeat these failure modes

- Do not deploy from a dirty root checkout.
- Do not deploy from `.rc-candidate-worktree/`, benchmark worktrees, or recovery branches unless explicitly approved.
- Do not deploy committed local `main` without first synchronizing it to `origin/main`.
- Do not use sanitized `deploy/examples/fly/fly.example.toml` for production.
- Do not remove or manually edit migration ledger rows to hide a mismatch.
- Do not commit LongMemEval data, generated snapshots, benchmark runs, or credentials.
- Do not assume a green GitHub branch is a production-approved image.

## Next planned work after reconnect succeeds

1. Confirm Polytoken and Codex can both call Weft after reboot.
2. Leave production on the known-good recovery image until the RC is reviewed.
3. Add an explicit production release tag/deployment ref requirement to the guard.
4. Validate the RC on the target Linux machine with OpenAI embeddings and the production-like dependency profile.
5. Apply migrations through the owner-managed process before deploying any image whose code migration set is newer than the database.
6. Run a production smoke test covering health, MCP initialization, `weft_prime`, and one recall operation.
7. Rotate the exposed hosted bearer credential through the private credential-management path; do not record the replacement value here.

## Session provenance

Key evidence captured during this incident:

- Fly machine state was `stopped` with warning health after restart exhaustion.
- Fly logs showed FastMCP 3.0.2 startup followed by migration verification failure for unknown database version 72.
- Recovery deploy from `recovery/v72-production-clean` succeeded with 1/1 health checks passing.
- Public `/healthz` returned HTTP 200 after deployment.
- OpenAI embedding requests returned HTTP 200 after deployment.
- `main` and `origin/main` were synchronized after preserving local work.
