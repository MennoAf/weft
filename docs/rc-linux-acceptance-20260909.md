# Updated Linux RC acceptance — 2026-09-09

## Verdict

**Linux installed-wheel runtime, container runtime, regression and scoped secret gates: PASS. Overall release acceptance: HOLD pending a clean final revision and exact-revision CI evidence.**

Final build/container evidence supersedes the initial artifact table where noted in `rc-linux-acceptance-20260909-final-addendum.md`. The rebuilt wheel has the same hash, so installed-wheel runtime evidence remains applicable. Full candidate-history Gitleaks completed: 509 reachable commits, 469 snapshots, three findings independently confirmed as test fixtures; verdict PASS/WITH-CLASSIFIED-FIXTURES. No raw secret values are included. Details remain in `artifacts/final-security-summary.md`. Historical incomplete-gate statements below describe the earlier pass and are superseded by this update.

This is evidence for a deliberately transferred source snapshot, not approval to publish, push, tag, or deploy. No such action was performed. Existing Linux RC services and secrets were outside this workbench test scope.

## Candidate provenance

Base commit: `16c97f9bb780106b264b7aeff065d9a5fffe9034`, branch `rc/linux-rehearsal-20260905`.

Tested source included these uncommitted overlays:

| File | SHA256 |
| --- | --- |
| `weft/cli.py` | `e3c5dbab8456d50ac01ef26e622ecb7d7c322d353ef2013407e56c3a789b7768` |
| `tests/test_cli_recall.py` | `a9957abc8c816be1f63dc38696111e64ae77fe0afa334eb598d5e0eb31f3d7f6` |
| `docs/rc-smoke.md` at build time | `0864608a697a3250a7676dacb92aacd508790fc963f4f8b03bc4c2b0b4cc58e7` |

The build-time documentation incorrectly counted 74 migrations. Runtime discovery and ledger verification established **73 active migrations: versions 1–50 and 52–74**. `pending_v51_episode_turns_fts.py` is intentionally excluded. Final documentation hash is `a2a453d744070e64b71e5c7a7f956fa22d0fbb66a9dabc99d4b2b92b93087c98`.

A subsequent build-context-only `.dockerignore` correction excludes local nested worktrees and `artifacts/`; hash `6df66175c6ba3aac0bf9c28f267c6873332e4646fcff1b7ba41bb09b322f3353`. Neither this correction nor the final documentation wording is part of the artifacts below. Python code and regression-test hashes are unchanged. This evidence document is also post-build. No clean final commit has been created.

## Artifacts

Built on Pop!_OS Linux x86_64 using Python 3.12.3, uv 0.12.8 and rootless Podman 4.9.3, from a clean Git archive plus the explicit overlays. No Mac virtual environments, credentials, or unrelated local artifacts were transferred as source.

| Artifact | SHA256 / identity |
| --- | --- |
| Wheel `weft_memory-1.0.0rc1-py3-none-any.whl` | `4a07bc52f24cfe1eacd410655cef632846dc649609a7f8f0a2f6d15ea00f88de` |
| Source distribution | `182e8cce90f4b653f83f8a3a1099690be1f8acee5096bcb7a698810d6e81c2e8` |
| OCI archive | `0af5b65f4d498db4fb343f85466e367ec0ecd2dad4f399b4699e0da488342b03` |
| Image ID | `352ee715e5b22a7684f71be4447d082edb5434ed05d07b381e5febc23707b322` |

Clean wheel and sdist member listings were captured; no nested worktree, `.venv`, or `.git` material was present in the new sdist. An older local sdist was contaminated by workspace material and is explicitly **not** release evidence. The image built and was exported; a complete image-runtime smoke is not established by the wheel-runtime tests.

## Verified behavior

The CLI repair uses the canonical codec-equipped pool, resolves and binds local owner identity, passes that identity explicitly to vector search, uses the canonical transaction scope, resets context, and closes the pool on failure. Five focused tests pass; 33 related auth/connection tests also pass.

Installed-wheel verification ran outside the source checkout using fresh, uniquely named PostgreSQL/Redis resources:

- Runtime and installed package version both `1.0.0rc1`.
- Empty database applied the exact 73-version set; ledger contained that set; second migration pass returned `[]`.
- Local FastEmbed provider, 768 dimensions, without paid provider credentials.
- Owner-A stdio MCP write and keyword/semantic recall passed.
- Actual installed CLI recall passed for owner A.
- Negative owner-B visibility check passed; assertions inspect results rather than the echoed query heading.
- PostgreSQL restart retained the records and owner-scoped retrieval behavior.
- Fresh-shell check loaded persisted database, Redis and provider configuration without their environment overrides. This is not an independently audited end-to-end public quickstart or a broad claim about all identity setup paths.
- Exact runtime resources were removed afterward.

Migration verification used a raw pool for initial DDL, followed by recreation of the canonical pool for vector operations. The migration runner returns applied version numbers; initial setup assertions must precede any schema-applying MCP startup.

## Full Linux regression

Frozen dependencies with all extras were necessary because tests import optional Anthropic and OpenAI packages at collection time. The run used a separate environment, sanitized provider/owner variables, and a job-owned Podman API socket rather than changing the global service.

Command: `uv run pytest tests/ -q -ra --disable-warnings` (absolute uv executable in SSH; under a 3600-second timeout).

- Collection: **3770 tests**, exit 0.
- Full result: **3766 passed, 4 skipped, 7 warnings**, **3437.91 seconds**, exit **0**.
- Skips covered disabled live model tests, an empty parameter set, and a retrieval-recovery fallback case.
- Warnings remain recorded, not silently waived.
- Cleanup reported exit 0; job socket absent and no test/uv/API processes or Ryuk containers remained according to the final verification.

Earlier collection failures from missing extras and runner failures from logging/path/tag mistakes are harness failures, not product passes. Their logs were retained. No repeated large-run failure is counted as successful acceptance.

## Security and remaining HOLD gates

- Scoped intended-source Gitleaks 8.30.1 scan: no findings, redacted output.
- Bounded recent-history scan: 18 commits, no findings.
- Full reachable-history scan timed out after 180 seconds: **not verified**.
- No exact-base-SHA CI run was found; older successful main runs do not validate this patched candidate.
- No full dependency vulnerability audit was run.
- Docker build-context hygiene correction is local and requires final-context verification/rebuild.
- Final clean commit and artifacts incorporating final docs/context changes remain to be produced.
- Full container-runtime acceptance remains separate from the passing installed-wheel runtime.

These are release-evidence gaps, not failures of the passing wheel tests. Do not declare the overall RC ready until they are closed or explicitly dispositioned by the operator. No publication is authorized by this report.

## Evidence locations

Build job: `/home/jasonbauman/agent-workbench/jobs/linux-rc-20260909T012339Z-41410-rerun`.

Runtime job: `/home/jasonbauman/agent-workbench/jobs/linux-rc-runtime-corrected-20260909T014535Z-42861`.

Local retained runtime evidence: `artifacts/linux-rc-runtime-corrected-20260909T014535Z-42861-evidence/` (migration, MCP, CLI, restart, fresh-shell and cleanup phase logs).

Local retained full regression logs: `artifacts/linux-regression-final-logs/`.

Build artifacts: `artifacts/linux-rc-20260909T012339Z-41410-rerun-evidence/`.

Loom records: `loom-4b3479e9` (CLI scope/cleanup), `loom-60926005` (bounded artifact build), `loom-a95a98cc` (runtime), `loom-40e0a571` (full Linux regression).
