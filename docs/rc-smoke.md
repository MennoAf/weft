# RC smoke rehearsal

This is the bounded local/container rehearsal for a Weft release candidate. It
uses a disposable Docker network and unnamed containers. It does not use the
normal `weft` Compose project, host ports, or persistent named volumes.

## Prerequisites

- Docker Desktop or Docker Engine is running.
- The checkout is the candidate commit being tested.
- `uv` is installed for the wheel-install check.

## Source-checkout smoke

Build the image from the checkout root. The explicit `.` matters: it keeps the
Docker build context aligned with the Dockerfile and packaged Compose resource.

```bash
docker build --tag weft-rc-local:1.0.0rc1 .
python scripts/rc_smoke.py --image weft-rc-local:1.0.0rc1
```

Expected output includes:

```text
phase=init version=1.0.0rc1 migrations=72 write=ok
phase=verify version=1.0.0rc1 migrations=0 recall=ok
RC Docker smoke passed
```

The script always removes its temporary Postgres, Redis, and private-network
resources, including after a failed assertion.

## Clean wheel check

The CI job separately proves the artifact users install rather than only importing a
source checkout. The Docker image smoke below is an additional runtime check of the
container artifact; it is not a substitute for the wheel check:

```bash
uv build
uv venv --python 3.13 /tmp/weft-wheel-smoke
uv pip install --python /tmp/weft-wheel-smoke/bin/python dist/*.whl
/tmp/weft-wheel-smoke/bin/python -c \
  'import importlib.metadata, weft; assert weft.__version__ == importlib.metadata.version("weft-memory") == "1.0.0rc1"'
```

## Linux owner rehearsal

Run the public quickstart on a clean Linux machine separately from this
container smoke. Record the distribution, architecture, Python/uv versions,
commands, elapsed time to first write and recall, restart persistence, and any
undocumented intervention. Redact credentials and personal memory content.

```bash
uv sync
uv run pytest tests/ -q
uv run python -m weft up
uv run python -m weft status
uv run python -m weft config show
```

After a write and recall check, stop and restart the infrastructure, verify the
memory remains searchable, and finish with:

```bash
uv run python -m weft down
```

## Acceptance checklist

- [ ] The image builds from a clean checkout context.
- [ ] Runtime and installed package versions both report `1.0.0rc1`.
- [ ] Fresh Postgres applies exactly 72 migrations.
- [ ] A second migration pass applies zero migrations.
- [ ] An owner-scoped memory is written and recalled by keyword.
- [ ] Recall survives a Postgres restart.
- [ ] Temporary Docker resources are removed.
- [ ] Wheel installation works in a clean environment.
- [ ] Linux owner rehearsal is recorded separately as release evidence.
