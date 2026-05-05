# Claude Code hooks for Weft

Hook scripts that make Weft feel native to [Claude Code](https://claude.com/claude-code) — automatic, not something the agent has to remember.

## `weft-precompact.sh` — preserve context across compaction

Claude Code compacts the conversation when context runs low. By default, anything not already in a tool-result or persisted memory is summarized away by the harness's compaction model — and a lot of working state (in-flight decisions, half-formed plans, file paths just discovered) gets thinned to bullet points.

This hook fires on the `PreCompact` event and injects a system-reminder telling Claude to call `weft_handoff` *before* compaction runs. The handoff lands in Weft, the next session's `weft_prime` surfaces it in tier-1, and the agent picks up where the prior session left off without paying the compaction tax.

### Install

1. Copy the script somewhere stable on your machine. A common spot:

   ```bash
   mkdir -p ~/.claude/hooks
   cp weft-precompact.sh ~/.claude/hooks/
   chmod +x ~/.claude/hooks/weft-precompact.sh
   ```

2. Register it in `~/.claude/settings.json` under `hooks.PreCompact`:

   ```json
   {
     "hooks": {
       "PreCompact": [
         {
           "matcher": "*",
           "hooks": [
             {
               "type": "command",
               "command": "/Users/YOU/.claude/hooks/weft-precompact.sh",
               "timeout": 10
             }
           ]
         }
       ]
     }
   }
   ```

   Replace `/Users/YOU` with your actual home path. Claude Code does not expand `~` inside the `command` field — use absolute paths.

3. Restart your Claude Code session (or start a new one) so the hook registers.

### Verify

Run the hook directly — it should print a single JSON object:

```bash
~/.claude/hooks/weft-precompact.sh | python3 -m json.tool
```

You should see a `hookSpecificOutput` block with `hookEventName: "PreCompact"` and an `additionalContext` field containing the system-reminder text.

### What the hook does *not* do

- **It does not call `weft_handoff` itself.** The hook is a shell script; it has no MCP access. It nudges Claude (which does have MCP access) to make the call.
- **It cannot block compaction.** PreCompact hooks are advisory. If Claude ignores the reminder or the model itself decides not to call `weft_handoff`, compaction proceeds anyway. In practice the reminder is reliable because it lands as a high-salience system-reminder right before the next assistant turn.
- **It does not replace `/handoff` or end-of-session handoffs.** Treat this as a safety net for sessions that bump the context limit unexpectedly. For deliberate session boundaries, write the handoff yourself.

### Customizing

The text inside `additionalContext` is the entire interface between the hook and the agent. If you want different sections in the handoff (e.g. you don't track `open_questions`, or you want to include `decisions_made` explicitly), edit the script. Keep the `<system-reminder>` wrapper — it's what makes Claude Code treat the text as a directive rather than a normal message.

### Related

- [`docs/CLAUDE_example.md`](../CLAUDE_example.md) — the global `CLAUDE.md` template that points the auto-memory protocol at Weft.
- [`docs/wick-handoff.md`](../wick-handoff.md) — context on how handoffs are consumed downstream.
