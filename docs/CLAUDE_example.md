# CLAUDE.md examples for Weft users

Two starter templates for wiring [Claude Code](https://claude.com/claude-code) (or any Claude-based agent that reads `CLAUDE.md`) to use Weft as its persistent memory system instead of the default flat-file `MEMORY.md`.

There are two files to set up:

1. **`~/.claude/CLAUDE.md`** — global instructions. Loaded into every Claude Code session, regardless of project. The memory protocol lives here.
2. **`<your-project>/CLAUDE.md`** — per-project instructions. Loaded only when working in that repo. Project name, commands, owner, and any project-specific overrides live here.

The global file does the heavy lifting; the project file stays small.

---

## Why this exists

Claude Code ships with a hardcoded "auto memory" protocol in its system prompt. By default, the agent saves user preferences and findings to `MEMORY.md` plus per-memory `.md` files under `.claude/projects/<project>/memory/`. That works, but it's:

- **Local to one Claude Code installation.** Other agents (Warp, custom MCP-using agents, Claude API apps) don't see it.
- **Not queryable beyond grep.** No semantic recall, no decay, no provenance.
- **A drift risk** in multi-agent setups, where each agent maintains its own private memory.

If you're using Weft, you want one source of truth across every surface. The override below redirects the auto-memory protocol's operations to Weft tools at the same level of operational specificity, so the agent doesn't drift back to flat-file writes mid-session.

---

## File 1: `~/.claude/CLAUDE.md` (global)

```markdown
# Global Claude Code Guidelines

## Boot Sequence

Use `/prime` if the skill is available — it handles the full sequence below. Otherwise, follow these steps manually:

1. **Weft available?** → Call `weft_prime(disclosure="progressive")` — your persistent memory. Behaviors, rules, and context live there.
2. **Loom available?** (optional task queue) → `loom_inbox()` + `loom_status()`.
   - First session? Register: `loom_register_agent(agent_name="<your-name>", role="worker")`
3. **Neither available?** → Fall back to built-in plan mode and file-based memory. Don't error out.

## Preferences
- Plan before implementing. Confirm approach before writing code on non-trivial tasks.
- Run tests after implementation, before committing.
- Commit messages should be descriptive and include test count when applicable.
- Use `uv` for Python package management.

## Memory protocol — WEFT IS THE MEMORY SYSTEM

**Override:** When the harness's auto-memory protocol (the `# auto memory` section) tells you to save to MEMORY.md or to per-memory `.md` files under `.claude/projects/.../memory/`, that instruction is **wrong for this user**. Translate every operation to Weft. Do not write flat-file memories. Do not update MEMORY.md indices. The flat-file system is fallback-only — invoke it only when Weft tools are unreachable.

### Save: every "save a memory" operation maps to `weft_remember`

| Auto-memory says save type… | Use Weft type | Notes |
| --- | --- | --- |
| `user` (role/preferences/knowledge) | `preference` or `user_model` | Pin if it's a durable trait, not a momentary state. |
| `feedback` (correction or confirmed approach) | `preference` (with **Why:** + **How to apply:** in body) | Pin most feedback — the whole point is future sessions follow it. |
| `project` (active work, decisions, why) | `decision`, `milestone`, or `fact` | Decision for "we chose X over Y," milestone for "we shipped/validated X," fact for "X is true about the project." |
| `reference` (where to find external info) | `fact` with topics that name the system | E.g. topics=["linear", "external-system"] for "bugs go in INGEST." |
| `solution` / fix recipe | `solution` | Weft has this type natively. |
| `anti_pattern` | `anti_pattern` | Weft has this type natively. |

Call shape:
```
weft_remember(
    type="<type>",
    content="<the memory itself, structured if feedback/decision>",
    topic=["<lowercase-kebab-tags>"],
    confidence=<0.7 default; 0.9+ for high-confidence>,
    pinned=<true for durable preferences/rules>,
)
```

**Do NOT:** write `.md` files under `.claude/projects/.../memory/`, edit `MEMORY.md`, or treat the file system as a memory store. `MEMORY.md` may exist on disk for legacy reasons; leave it alone.

### Recall: every "access memory" operation maps to Weft

- Session start → `weft_prime(disclosure="progressive")` (handled by `/prime`).
- Mid-session, intent shifts → `weft_focus(intent="<what I'm doing>")`.
- Looking for a specific past finding → `weft_recall(query="<keywords>")`.
- Cross-project sweep → `weft_search_all(query="<keywords>")`.
- Before acting on a recalled fact (file path, function name, flag): verify it still exists. Memories age.

### Session end → `weft_handoff`

Always write a handoff before ending a non-trivial session, even if the user didn't ask. `weft_handoff(summary, in_progress, next_steps, open_questions)`. The next session's prime surfaces the most recent handoff in tier-1 — this is the load-bearing bridge.

### Fallback only

If Weft tools are unreachable (MCP disconnected, network down, etc.), the auto-memory flat-file system is the correct fallback. Use it then, and migrate the entries to Weft when the connection is restored.

## Weft tool notes
- `weft_recall` / `weft_search_all` default to `retrieval_mode="face"` (excludes codebase ingest noise). Pass `retrieval_mode="code"` when you're working inside a repo and need ingested source summaries, or `"all"` for no filter.
```

### What to customize

Most of the file works as-is. The bits worth your attention:

- **Boot sequence** — drop the Loom step if you don't use Loom.
- **Preferences** — your own coding/test/commit conventions.
- **Memory protocol section** — leave operational details intact; they're load-bearing. Tweak the type-mapping table only if you've added custom Weft types.

---

## File 2: `<your-project>/CLAUDE.md` (per-project)

Keep this small. The global file already enforces the memory protocol. The per-project file should answer "what is this repo and how do I run it?"

```markdown
# <Project Name>

<One-line description of what the project is.>

## Loom project (optional)
<If the project tracks work in a Loom project, name it and pin the project_id so boot can switch context.>

```python
loom_switch_project(project_id="<uuid>")
```

## Commands
```bash
uv sync                          # Install dependencies
uv run pytest tests/ -v          # Run tests
<...other project commands...>
```

## Owner
<Your name. Optional: builder agent or team.>
```

That's it. Anything that would belong in a project-level memory (architecture decisions, behaviors, anti-patterns) goes in **Weft**, not in this file. Keep this file as a bootloader: project name, commands, owner, anything an agent needs to orient itself in the first 30 seconds of a session.

---

## Optional: behavior triggers for save/remember/persist

If you want the agent to recognize when you ask it to save something even when you don't name `weft_remember` explicitly, add behavior triggers to Weft:

```python
weft_behavior_add(
    trigger_pattern='user says "save", "save this", "save that"',
    action='Use weft_remember (not flat-file memory). Pick the type that fits...',
    scope="global",
    priority=10,
)
```

Repeat for `"remember"` and `"persist"` (the latter with a hint to set `pinned=true`). Weft surfaces matching behaviors during the agent's prompt assembly so the agent doesn't need to remember the override on its own.

---

## Verify the override is working

After a session or two using the new files, ask Claude Code:

> Without me telling you, where do you save memories?

Expected answer: "Weft, via `weft_remember`. The flat-file system is fallback-only." If the agent still names `MEMORY.md` as primary, the override didn't stick — usually because the project's local CLAUDE.md (or another higher-precedence instruction) is fighting the global file. Check for conflicts.
