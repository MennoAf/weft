# RC-0 Evidence — Candidate Hygiene and Baseline

**Status:** CLOSED WITH FOLLOW-UPS  
**Date:** 2026-08-28  
**Plan:** `docs/release-candidate-plan.md` (RC-0)  
**Candidate branch:** `rc-candidate-20260827`  
**Candidate baseline commit:** `d310d111312f227064df16924ad9b8b20f345461`

## Scope and safety boundary

This pass performed containment and evidence capture only. The dirty `main`
worktree was not reset, cleaned, stashed, or otherwise altered. No push, tag,
deploy, credential value capture, or history rewrite was performed.

## Evidence

### P0 credential action

- The owner-confirmed Claude and Codex credential rotation remains complete.
- No credential values are copied into this record or release evidence.

### Dirty-work preservation and classification

The main worktree remains intentionally dirty so active work is not discarded.
The status snapshot contained 56 modified or untracked paths, classified as:

- 23 benchmark/recovery paths, including local fixtures, datasets, reports, and
  pilot harness work;
- 18 test paths;
- 10 `weft/` implementation or migration paths;
- 3 documentation paths;
- 2 other local/candidate paths.

The candidate is isolated from that work in its own worktree and branch.

### Reproducible candidate baseline

The candidate worktree was clean at inspection and is based on:

```text
branch: rc-candidate-20260827
commit: d310d111312f227064df16924ad9b8b20f345461
ahead of origin/robustness-hardening: 3 commits
```

This evidence file is the only intended addition from this RC-0 pass; the
candidate must be rechecked clean after its commit.

### Repository inventory

Inventory captured during this pass:

- 37 local branches;
- 41 remote-tracking branches;
- 3 tags: `pre-oauth-integration`, `v0.1.0`, `v1.0.0`;
- 1 tracked workflow: `.github/workflows/backup.yml`;
- 0 tracked release/checksum artifact files;
- sanitized public deployment examples remain under `deploy/examples/fly/`.

### Ignore and boundary checks

The candidate ignore rules explicitly cover:

- local benchmark datasets and generated run directories;
- local retrieval-recovery pilot runs;
- nested/local worktrees;
- `/tls-release/`;
- local credentials and environment files;
- generated build, cache, and virtual-environment content.

The candidate tracked-file scan found no tracked `docs/internal/`, `.env`,
backup data, recovery runbook, nested worktree, or `tls-release/` path. The
tracked operational surface observed in the candidate is limited to sanitized
Fly examples and source-level backup/credential support that still require the
RC-1 public/private policy review.

### History screen

A non-destructive history/path screen was run over reachable candidate history.
It found historical personal absolute-path references in older commits,
including the already-sanitized release-history commits and older benchmark,
hook, and development commits. These are accepted as low-sensitivity
personally identifiable development metadata: the owner's public name and
email are not release blockers, and the local directory names expose no useful
credential or account secret. No credential-shaped values were captured in
this record. A formal Gitleaks scan was subsequently run against the candidate
working tree and all reachable Git history/refs; the redacted findings and
classification are recorded below.

### Dedicated secret scan

Gitleaks `8.30.1` was installed via Homebrew and run against the isolated
candidate. Scan reports were written to temporary paths outside the repository;
all report output used `--redact`, and no secret values were copied into release
evidence.

- Current candidate tree: 3 findings, all in tests.
- Reachable candidate history: 4 findings across 494 commits.
- The history findings are the same three current test fixtures plus one
  historical OAuth negative-test fixture in a file removed from the candidate.
- Classifications: the Discord snowflake is a documented numeric test ID; the
  identity result is generated keypair test data; the Slack result is explicitly
  named `FAKE_SECRET`; and the removed OAuth result is a literal dummy
  refresh-token value used to test rejection behavior.
- No live production credential, private key, database credential, provider
  token, webhook secret, or account secret was identified.
- Scan exit status was non-zero only because Gitleaks reports these intentional
  test fixtures; no finding was assessed as a release-blocking secret.

This closes the RC-0 dedicated-secret-scan follow-up for the scanned candidate
refs. Future changes should rerun the scan before publication, and any newly
introduced finding must be reviewed rather than blanket-ignored.

## RC-0 exit assessment

| Exit criterion | Assessment | Evidence / follow-up |
| --- | --- | --- |
| Credential rotation complete | **MET** | Owner-confirmed; values intentionally omitted. |
| Release baseline clean and reproducible | **MET** | Isolated candidate branch and recorded SHA. |
| No active user work discarded | **MET** | Dirty main preserved; no destructive cleanup. |
| Exact candidate commit recorded | **MET** | Baseline and evidence commit SHAs are recorded above. |
| Workspace/modified-path inventory complete | **MET FOR THIS PASS** | 56-path classification recorded above. |
| Branch/tag/workflow/release-asset inventory complete | **MET FOR THIS PASS** | Counts and names recorded above. |
| Historical absolute-path finding | **ACCEPTED / NON-BLOCKING** | Low-sensitivity development metadata; no history rewrite planned. |
| Formal publishable-history secret scan | **MET FOR SCANNED REFS** | Gitleaks 8.30.1 found only classified test fixtures; rerun after candidate changes. |

## Remaining work carried forward

These are explicit follow-ups, not RC-0 claims:

1. No history rewrite is planned for the accepted low-sensitivity absolute
   paths; retain the full history and document this decision under G1.
2. Rerun Gitleaks over the exact final publishable history after RC-1 through
   RC-3 changes; record results without storing secret values.
3. Complete RC-1 public/private review of source-level backup and credential
   support, all branches/tags, and any release artifacts before publication.
4. Add RC-2/RC-3 license, package, CI, release-artifact, and independent Linux
   rehearsal evidence. The candidate currently has no root `LICENSE` file and
   no CI/release workflow.
5. Keep RC-0 closed as containment/baseline only; do not treat this record as
   authorization to tag, publish, deploy, or claim G1–G6 passed.
