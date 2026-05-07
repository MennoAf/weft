# Wiring your agent to Weft

How to make [Claude Code](https://claude.com/claude-code) (or any Claude-based agent that reads `CLAUDE.md` and supports MCP slash commands) use Weft as its persistent memory system instead of the default flat-file `MEMORY.md`.

## Why this exists

Claude Code ships with a hardcoded "auto memory" protocol in its system prompt. By default, the agent saves user preferences and findings to `MEMORY.md` plus per-memory `.md` files under `.claude/projects/<project>/memory/`. That works, but it's:

- **Local to one Claude Code installation.** Other agents (Warp, custom MCP-using agents, Claude API apps) don't see it.
- **Not queryable beyond grep.** No semantic recall, no decay, no provenance.
- **A drift risk** in multi-agent setups, where each agent maintains its own private memory.

If you're using Weft, you want one source of truth across every surface. The override below redirects the auto-memory protocol's operations to Weft tools at the same level of operational specificity, so the agent doesn't drift back to flat-file writes mid-session.

## The setup

Three files, three places. The artifacts live in [`templates/`](../templates/) — copy-paste-ready.

### 1. Memory protocol → `~/.claude/CLAUDE.md`

Append [`templates/CLAUDE.md`](../templates/CLAUDE.md) to your global `~/.claude/CLAUDE.md` (create the file if it doesn't exist).

This is the load-bearing piece. It tells the agent:

- Use `weft_remember` / `weft_recall` / `weft_handoff` instead of writing flat-file memories
- Map the harness's auto-memory types (`user`, `feedback`, `project`, `reference`) to Weft types (`preference`, `decision`, `milestone`, `fact`, etc.)
- Run `/prime` at session start, `/handoff` at session end
- Treat the flat-file system as fallback-only — used only when MCP is unreachable

### 2. Slash commands → `~/.claude/commands/`

```bash
mkdir -p ~/.claude/commands
cp templates/commands/prime.md ~/.claude/commands/
cp templates/commands/handoff.md ~/.claude/commands/
```

`/prime` calls `weft_prime` and reports the prior session's handoff + open issues. `/handoff` calls `weft_learn` then `weft_handoff` so the next session has continuity.

### 3. Per-project `CLAUDE.md`

Keep this small. The global file enforces the memory protocol — your per-project file should answer "what is this repo and how do I run it?":

```markdown
# <Project Name>

<One-line description.>

## Commands
\`\`\`bash
uv sync                          # Install dependencies
uv run pytest tests/ -v          # Run tests
\`\`\`

## Owner
<Your name. Optional: builder agent or team.>
```

Anything that would belong in a project-level memory (architecture decisions, behaviors, anti-patterns) goes in **Weft**, not in this file. Keep it as a bootloader — name, commands, owner, anything an agent needs to orient itself in the first 30 seconds.

## What to customize

**Load-bearing (don't change):**

- The `weft_prime` / `weft_remember` / `weft_handoff` call shape in `templates/CLAUDE.md`
- The "Do NOT write flat-file memories" override
- The recall mapping table

**Customize freely:**

- Add your own coding/test/commit preferences alongside the Memory Protocol section
- Add behavior triggers via `weft_behavior_add` for personal conventions
- Drop the boot-sequence Loom mention if you don't use Loom
- Tweak the type-mapping table if you've added custom Weft types

## Optional: behavior triggers for save/remember/persist

If you want the agent to recognize when you ask it to save something even when you don't name `weft_remember` explicitly:

```python
weft_behavior_add(
    trigger_pattern='user says "save", "save this", "save that"',
    action='Use weft_remember (not flat-file memory). Pick the type that fits...',
    scope="global",
    priority=10,
)
```

Repeat for `"remember"` and `"persist"` (the latter with a hint to set `pinned=True`). Weft surfaces matching behaviors during the agent's prompt assembly so the agent doesn't need to remember the override on its own.

## Verify the override is working

After a session or two using the new files, ask Claude Code:

> Without me telling you, where do you save memories?

Expected answer: "Weft, via `weft_remember`. The flat-file system is fallback-only." If the agent still names `MEMORY.md` as primary, the override didn't stick — usually because a project-local `CLAUDE.md` (or another higher-precedence instruction) is fighting the global file. Check for conflicts.
