# Claude-specific setup

The general Weft memory protocol is in [AGENTS.md](AGENTS.md).

## Claude Desktop

Add Weft to Claude Desktop's `claude_desktop_config.json` MCP server configuration:

```json
{
  "mcpServers": {
    "weft": {
      "command": "weft",
      "args": ["mcp"]
    }
  }
}
```

## Claude Code

For the Claude Code memory protocol, copy the content of [`templates/CLAUDE.md`](templates/CLAUDE.md) into your global `~/.claude/CLAUDE.md` or project `CLAUDE.md`. Optional `/prime` and `/handoff` slash commands are in [`templates/commands/`](templates/commands/); install them under `~/.claude/commands/` to use those commands in Claude Code.

Claude Code's optional PreCompact hook is documented in [Claude Code hooks for Weft](docs/claude-code-hooks/README.md). It is specific to Claude Code and is not required to connect other MCP-capable agent harnesses to Weft.
