# Local-only personal cases

This directory is **gitignored**. Files placed here are private acceptance-test
cases seeded from the user's real Weft data (people, projects, trackers, Slack
ingest history). They follow the same pytest pattern as the committed cases in
`tests/acceptance/test_*.py`, just with non-synthetic seed data.

## How to add a case

1. Snapshot the relevant project from your live Weft DB to a JSON fixture in
   this directory (snapshot CLI is forthcoming).
2. Write a `test_<descriptive_name>.py` here that loads the snapshot under a
   sandbox `project_id` (use `sandbox_project_id("local_<case_id>")` to keep
   the prefix conventions consistent with the committed cases).
3. Run with `uv run pytest tests/acceptance/cases_local/ -v`.

## What stays in here

* Real names, real project IDs, real Slack channels
* Snapshots of your actual workflow shape — meeting prep against real people,
  weekly summaries against your real calendar, the queries that matter to you

## What does NOT belong here

Anything you'd consider committing. If a case generalizes — same shape, no
identifying data — promote it to `tests/acceptance/test_*.py` with the
synthetic persona convention (`Jim Boblaw`, `project A/B`, etc.).
