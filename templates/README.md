# Templates

Drop-in artifacts for wiring [Claude Code](https://claude.com/claude-code) (or any Claude-based agent that reads `CLAUDE.md` and supports MCP slash commands) to Weft as its persistent memory system.

Three files, three places they go:

| File | Where it goes | What it does |
| --- | --- | --- |
| `CLAUDE.md` | Append to `~/.claude/CLAUDE.md` (or your project's `CLAUDE.md`) | Tells the agent to use Weft for memory instead of flat files |
| `commands/prime.md` | `~/.claude/commands/prime.md` | Adds `/prime` to load Weft context at session start |
| `commands/handoff.md` | `~/.claude/commands/handoff.md` | Adds `/handoff` to capture learnings + handoff before ending a session |

## Install (3 steps)

### 1. Memory protocol

Open `~/.claude/CLAUDE.md` (create it if it doesn't exist), paste the contents of `templates/CLAUDE.md`, save. Done — every Claude Code session in any project now knows to use Weft for memory.

### 2. Slash commands

```bash
mkdir -p ~/.claude/commands
cp templates/commands/prime.md ~/.claude/commands/prime.md
cp templates/commands/handoff.md ~/.claude/commands/handoff.md
```

If you already have `/prime` or `/handoff` defined, rename these to `weft-prime` / `weft-handoff` instead.

### 3. Verify

In a fresh Claude Code session inside any project that has Weft registered as an MCP server, type `/prime`. The agent should call `weft_prime`, summarize what it found, and ask what you're working on.

Then, before ending the session, type `/handoff`. The agent should call `weft_learn` and `weft_handoff`. The next session's `/prime` will surface that handoff at the top of its report.

## What's customizable vs. load-bearing

**Load-bearing (don't change):**

- The `weft_prime` / `weft_remember` / `weft_handoff` call shape in `CLAUDE.md`
- The "Do NOT write flat-file memories" override
- The recall mapping table

**Customize freely:**

- Add your own coding/test/commit preferences to your `CLAUDE.md` outside the Memory Protocol section
- Add behavior triggers via `weft_behavior_add` for your own conventions
- Drop the boot-sequence Loom reference if you don't use Loom
- Tweak the type-mapping table if you've added custom Weft types

## Deeper dive

For the full background on why this exists, the per-project vs. global file split, and how to verify the override is sticking, see [`docs/CLAUDE_example.md`](../docs/CLAUDE_example.md).
