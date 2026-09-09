# RC Linux acceptance final addendum — 2026-09-09

This addendum records the bounded final build and container gate closure for the
candidate described in `docs/rc-linux-acceptance-20260909.md`. The main
acceptance report was not overwritten or edited. No commit, push, publication,
deployment, sudo/admin action, reboot, or service/secret change was performed.

## Exact source inputs

- Base: `16c97f9bb780106b264b7aeff065d9a5fffe9034`
- Source transfer: clean `git archive` of that base, staged as
  `source.tar`; SHA256:
  `0f96e3357637eaeb09608f00181cf35e9db365ca725b5d5994f21ac933ed03f2`
- Exactly four explicit overlays were applied after extraction:

| Overlay | SHA256 |
| --- | --- |
| `weft/cli.py` | `e3c5dbab8456d50ac01ef26e622ecb7d7c322d353ef2013407e56c3a789b7768` |
| `tests/test_cli_recall.py` | `a9957abc8c816be1f63dc38696111e64ae77fe0afa334eb598d5e0eb31f3d7f6` |
| `docs/rc-smoke.md` | `a2a453d744070e64b71e5c7a7f956fa22d0fbb66a9dabc99d4b2b92b93087c98` |
| `.dockerignore` | `6df66175c6ba3aac0bf9c28f267c6873332e4646fcff1b7ba41bb09b322f3353` |

Unique workbench job:
`/home/jasonbauman/agent-workbench/jobs/weft-rc-final-20260909T024226Z-45929`

The workbench was reached with the configured SSH profile. It is Pop!_OS
Linux x86_64 with rootless Podman 4.9.3. Existing `weft-linux-rc` services
and secrets were preserved. The remote build used the documented absolute uv
path `/home/jasonbauman/.local/bin/uv`.

## Clean rebuild evidence

The final source distribution and wheel were built from the extracted source
after applying the four overlays. Member-list checks passed:

- sdist contained no `.git`, `.venv`, nested worktree, or `artifacts/` entries.
- wheel contained no `.git`, `.venv`, nested worktree, or `artifacts/` entries.
- wheel member listing contained `weft/cli.py`, `capability_registry/`, and
  `docker-compose.weft.yml`.
- sdist member listing contained the corrected `docs/rc-smoke.md` and
  `.dockerignore`.

| Artifact | SHA256 |
| --- | --- |
| `weft_memory-1.0.0rc1-py3-none-any.whl` | `4a07bc52f24cfe1eacd410655cef632846dc649609a7f8f0a2f6d15ea00f88de` |
| `weft_memory-1.0.0rc1.tar.gz` | `79bb0bb7ffd14c753e9e9924f316b8c840195101c753c2553ccaaf5fbbb2c9a0` |
| OCI archive `container.oci.tar` | `aebffa397f601e02ee12cddfd17db4047aeaea1ed65cf69285d054905da97f7e` |

Image tag:
`localhost/weft-rc-final-20260909t024226z-45929:1.0.0rc1`

Image ID:
`bf8c3e9ed42990648841a161890920863e246de4842b37c4739b796fe4e4633d`

## Container runtime smoke

The existing `scripts/rc_smoke.py` was not broadly changed. Because the
workbench has no Docker CLI, a job-local executable symlink named `docker`
pointing to `/usr/bin/podman` was placed first on `PATH`; a syntax check and
`docker --version`/Podman compatibility check passed. The existing smoke was
then run unchanged against the final image, with disposable uniquely named
resources and a 300-second outer bound.

Observed output:

```text
phase=init version=1.0.0rc1 migrations=73 write=ok
phase=verify version=1.0.0rc1 migrations=0 recall=ok
RC Docker smoke passed
```

This establishes the requested runtime checks:

- runtime and installed package version `1.0.0rc1`;
- fresh database applies exactly 73 migrations;
- owner-scoped write succeeds;
- PostgreSQL restart occurs before verification;
- second migration pass applies zero migrations;
- keyword recall finds the sentinel after restart;
- cleanup leaves no `rc-postgres-*`, `rc-redis-*`, or `weft-rc-smoke-*`
  disposable resources.

No paid provider credentials were used; the smoke uses its local/provider-free
configuration. Full regression and wheel-only gates were not rerun because the
prior report records the exact `3766 passed, 4 skipped, 7 warnings` regression,
and this final change set is limited to the documented docs/build-context
corrections plus artifact rebuild and container runtime validation.

## Evidence paths

- Build artifacts and member listings:
  `/home/jasonbauman/agent-workbench/jobs/weft-rc-final-20260909T024226Z-45929/artifacts/`
- Build log:
  `/home/jasonbauman/agent-workbench/jobs/weft-rc-final-20260909T024226Z-45929/logs/uv-build.log`
- Image build log:
  `/home/jasonbauman/agent-workbench/jobs/weft-rc-final-20260909T024226Z-45929/logs/podman-build.log`
- Runtime smoke log:
  `/home/jasonbauman/agent-workbench/jobs/weft-rc-final-20260909T024226Z-45929/logs/rc-smoke.log`

This addendum closes the bounded final source-rebuild, artifact-manifest, and
container-runtime gates. Other HOLD items in the main report (for example,
full reachable-history scanning, dependency vulnerability auditing, exact-base
CI, and release authorization) remain outside this bounded task.
