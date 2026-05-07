---
description: Capture session learnings and handoff context for the next session via Weft
allowed-tools: [mcp__weft__weft_learn, mcp__weft__weft_handoff, mcp__weft__weft_remember, mcp__weft__weft_behavior_add]
---

# Session Handoff

Run before ending a non-trivial session. All three steps unless explicitly skipping one.

## Step 1: Learn

Call `weft_learn` with a summary of what was learned this session. Skip only if the session was truly trivial (a single quick question, no code changes).

Include:

- Gotchas, edge cases, or surprises encountered
- Patterns or conventions confirmed
- Debugging insights or fixes that future sessions should know about
- Corrections to previous assumptions
- Behavioral rules discovered (e.g., "when writing tests, always use X" — these get auto-extracted as behaviors)

If the session completed a tracked task (Loom, JIRA, GitHub issue, etc.), pass the identifier as `task_id` — that creates a linked milestone in the next session's prime.

## Step 2: Behaviors

Review the session for any recurring patterns, user corrections, or confirmed workflows that should become persistent rules. Call `weft_behavior_add` for any that weren't auto-extracted in Step 1. Good candidates:

- User corrections ("don't do X, do Y instead")
- Confirmed workflows ("always run tests before committing")
- Project-specific rules ("when touching the primer, run the equivalence tests")

Skip if nothing qualifies.

## Step 3: Handoff

Call `weft_handoff` with structured context:

- **summary** — what was accomplished (be specific: commits, files changed, features shipped)
- **in_progress** — anything partially done that needs follow-up
- **next_steps** — recommended next actions with reasoning
- **open_questions** — unresolved decisions or things to investigate

The next session's `weft_prime` surfaces the most recent handoff in tier-1 — it's the load-bearing bridge between sessions.

## Output

Confirm to the user:

- What was captured in `weft_learn` (including any auto-extracted behaviors)
- Any behaviors added manually in Step 2
- What was captured in `weft_handoff`
- Recommended next steps for the next session
