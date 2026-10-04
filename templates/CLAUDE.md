<!--
Drop this section into either:
  - ~/.claude/CLAUDE.md          (global, all projects)
  - <your-project>/CLAUDE.md     (project-only)

Recommended: paste into the global file so every session uses Weft consistently.

Strip the HTML comment when copying.
-->

## Boot Sequence

Use `/prime` if the slash command is installed (see `templates/commands/prime.md`). Otherwise, at the start of every session:

1. Call `weft_prime(disclosure="progressive")` — your persistent memory. Behaviors, rules, handoffs, and context live there. If your repo instructions give an explicit Weft project id (e.g. `weft_prime(project_id="my-project")`), pass it verbatim; don't reuse project ids from other tooling — different systems use different keys.
2. Read the returned context carefully. The handoff is the bridge from the prior session.
3. Report a brief status to the user, then ask what they're working on.

If Weft is unreachable, say so and proceed without persistent memory. Don't error out.

## Memory Protocol — Weft is the memory system

When your harness suggests writing memories to flat files (e.g. `MEMORY.md` or per-memory `.md` files under `.claude/projects/.../memory/`), translate every operation to Weft instead. Do not write flat-file memories. Do not update `MEMORY.md` indices. The flat-file system is fallback-only — invoke it only when Weft tools are unreachable.

### Save: every "save a memory" operation maps to `weft_remember`

| If your harness wants to save type… | Use Weft type | Notes |
| --- | --- | --- |
| `user` (role/preferences/knowledge) | `preference` or `user_model` | Pin if it's a durable trait. |
| `feedback` (correction or confirmed approach) | `preference` (with **Why:** + **How to apply:** in body) | Pin most feedback — the whole point is future sessions follow it. |
| `project` (active work, decisions, why) | `decision`, `milestone`, or `fact` | `decision` for "we chose X over Y," `milestone` for "we shipped X," `fact` for "X is true about the project." |
| `reference` (where to find external info) | `fact` with topics naming the system | E.g. `topics=["linear", "external-system"]` for "bugs go in INGEST." |
| solution / fix procedure | `solution` | Native type. |
| anti-pattern | `anti_pattern` | Native type. |

Call shape:

```python
weft_remember(
    type="<type>",
    content="<the memory itself, structured if feedback/decision>",
    topic=["<lowercase-kebab-tags>"],
    confidence=<0.7 default; 0.9+ for high-confidence>,
    pinned=<True for durable preferences/rules>,
)
```

When composing a memory worth retaining, preserve quantitative qualifiers that materially specify it (such as date, duration, amount, range, unit, or period/direction like "45 minutes each way"). Do not copy incidental numbers or save a fact solely because it contains a number.

**Do NOT** write `.md` files under `.claude/projects/.../memory/`, edit `MEMORY.md`, or treat the file system as a memory store.

### Recall: every "access memory" operation maps to Weft

- Session start → `weft_prime(disclosure="progressive")` (handled by `/prime`)
- Mid-session, intent shifts → `weft_focus(intent="<what I'm doing>")`
- Looking for a specific past finding → `weft_recall(query="<keywords>")`
- Cross-project sweep → `weft_search_all(query="<keywords>")`
- Before acting on a recalled fact (file path, function name, flag): verify it still exists. Memories age.

### Session end → `weft_handoff`

Always write a handoff before ending a non-trivial session, even if the user didn't ask. Use `/handoff` if the slash command is installed (see `templates/commands/handoff.md`), or call `weft_handoff(summary, in_progress, next_steps, open_questions)` directly. The next session's prime surfaces the most recent handoff in tier-1 — it's the load-bearing bridge.

### Fallback only

If Weft tools are unreachable (MCP disconnected, network down), the auto-memory flat-file system is the correct fallback. Use it then, and migrate the entries to Weft when the connection is restored.

## Weft tool notes

- `weft_recall` / `weft_search_all` default to `retrieval_mode="face"` (excludes codebase ingest noise). Pass `retrieval_mode="code"` when working inside a repo and you need ingested source summaries, or `"all"` for no filter.
