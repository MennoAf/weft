#!/usr/bin/env bash
# weft-precompact.sh — Claude Code PreCompact hook.
#
# Fires immediately before Claude Code compacts context. Injects a
# system-reminder telling the agent to write a comprehensive handoff to
# Weft *now* — so the load-bearing bridge survives the compaction
# boundary and the next session's weft_prime can recover working state.
#
# Contract: stdout must be a single JSON object with hookSpecificOutput.
# Exit code is informational only; it cannot block compaction.

set -euo pipefail

cat <<'JSON'
{
  "hookSpecificOutput": {
    "hookEventName": "PreCompact",
    "additionalContext": "<system-reminder>\nContext compaction is imminent. Before continuing your response, call weft_handoff to capture the working state of this session. The handoff is the load-bearing bridge across compaction — without it, post-compaction context will be thin and the next prime will surface stale state.\n\nWrite a handoff that covers:\n- summary: what this session accomplished or is mid-flight on\n- in_progress: anything actively underway and where it stands\n- next_steps: the concrete next actions a future session should pick up\n- open_questions: unresolved decisions or ambiguities worth flagging\n\nBe specific. Name files, commit hashes, test counts, decision rationale. The handoff is consumed by your future self with no other context.\n\nIf Weft tools are unreachable, fall back to a brief plain-text summary in your response — but try Weft first.\n</system-reminder>"
  }
}
JSON
