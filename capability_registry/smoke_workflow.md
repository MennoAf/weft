# Capability Registry Smoke Workflow

## Prerequisites

- Python 3.11+.
- A running Weft instance and database for write-mode steps.
- `capability_registry` available on `PYTHONPATH`.
- A configured source repository, such as Muttr, with readable Python files.
- A Weft project ID for write-mode steps.

> **Current checkout note:** `ingest.py`, `lookup.py`, and their tests are
> present. The `seeds.py` and `stale_checker.py` modules referenced below are
> companion workflow steps and must land before Steps 2, 3, and 5 are runnable.

## Step 1: Dry-run scan

Run the scanner with the example configuration:

```bash
python -m capability_registry.ingest --config capability_registry/example_config.toml
```

Expected output includes a dry-run report with the entry count, file and symbol
details, topic tags, content previews, and JSON confirming `"mode": "dry_run"`.
No Weft memories are written.

```text
Capability Registry dry run: <entry-count> entries.
FILE: crawl/escalation.py
SYMBOL: LazyEscalationPolicy
KIND: class
TOPICS: repo:muttr, file:crawl/escalation.py, capability:bot-block-hardening
CONTENT_PREVIEW: CAPABILITY: bot-block-hardening ...
```

## Step 2: Seed dry-run

Preview the standard Muttr capability seeds:

```bash
python -m capability_registry.seeds --dry-run --project-id <your-project-id>
```

Expected output includes the Muttr bot-block entry and the Delphi↔Muttr entry,
with their repository, file, and capability topics. Confirm that the command
performs no Weft writes.

```json
{
  "mode": "dry_run",
  "entries": [
    {"repo": "muttr", "topics": ["capability:bot-block-hardening"]},
    {"repo": "muttr", "topics": ["capability:delphi-muttr-reuse"]}
  ]
}
```

## Step 3: Write seeds to Weft

Edit `capability_registry/example_config.toml` for the target project:

```toml
dry_run = false
weft_project_id = '<your-project-id>'
```

Then write the seeds:

```bash
python -m capability_registry.seeds --project-id <your-project-id>
```

Expected output is JSON containing the written memory IDs:

```json
{
  "mode": "write",
  "written_memory_ids": ["<memory-id>", "<memory-id>"]
}
```

## Step 4: Query via MCP tool

From an agent session rooted in the Weft project, call:

```text
weft_capability_lookup(query='bot blocked crawler')
```

Expected output contains a reuse-oriented entry like:

```text
[repo:muttr] crawl/escalation.py :: LazyEscalationPolicy
Capabilities: bot-block-hardening
Reuse by subclassing the lazy escalation policy for bot-blocked crawlers.
```

## Step 5: Stale detection

Compare indexed capability files with the current repository contents:

```bash
python -m capability_registry.stale_checker --project-id <id> --repo-roots '{"muttr": "/path/to/muttr"}'
```

Expected output is a JSON report listing changed and missing capability files:

```json
{
  "changed": ["crawl/escalation.py"],
  "missing": [],
  "count": 1
}
```

## Step 6: Unit tests

Run the capability-registry unit modules:

```bash
python -m tests.test_capability_scanner
python -m tests.test_capability_ingest
python -m tests.test_capability_lookup
```

Expected output reports all tests as `PASS` and exits with code 0.

## Acceptance criteria

- [ ] Dry-run scanning produces entries without Weft writes.
- [ ] Seed dry-run reports the Muttr and Delphi↔Muttr entries with correct topics.
- [ ] Seed write mode stores the entries and returns memory IDs.
- [x] MCP lookup returns formatted capability entries from indexed memories.
- [ ] Stale checker detects changed and missing files.
- [x] Capability lookup tests pass.
