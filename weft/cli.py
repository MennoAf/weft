"""Weft CLI — memory management commands."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import click

from weft.config import CONFIG_PATH, load_config, load_config_file, save_config_value

COMPOSE_FILE = Path(__file__).parent.parent / "docker-compose.weft.yml"

# MCP server entry for Claude Code
_MCP_ENTRY = {
    "command": "weft",
    "args": ["mcp"],
}


@click.group()
def cli():
    """Weft — Persistent agent memory system."""
    pass


@cli.command(name="mcp")
def mcp_server():
    """Start the Weft MCP server (stdio transport)."""
    from weft.mcp import mcp
    mcp.run()


def _register_mcp(project_dir: Path | None = None) -> None:
    """Register Weft as an MCP server in Claude Code configuration.

    If *project_dir* is given, writes to ``<project_dir>/.mcp.json``.
    Otherwise writes to ``~/.claude.json`` (global scope).
    """
    if project_dir is not None:
        mcp_path = project_dir / ".mcp.json"
        data = json.loads(mcp_path.read_text()) if mcp_path.exists() else {}
        servers = data.setdefault("mcpServers", {})
        if servers.get("weft") == _MCP_ENTRY:
            return  # already registered
        servers["weft"] = _MCP_ENTRY
        mcp_path.write_text(json.dumps(data, indent=2) + "\n")
        click.echo(f"Registered Weft MCP server in {mcp_path}")
    else:
        claude_json = Path.home() / ".claude.json"
        data = json.loads(claude_json.read_text()) if claude_json.exists() else {}
        servers = data.setdefault("mcpServers", {})
        if servers.get("weft") == _MCP_ENTRY:
            return
        servers["weft"] = _MCP_ENTRY
        claude_json.write_text(json.dumps(data, indent=2) + "\n")
        click.echo(f"Registered Weft MCP server globally in {claude_json}")


@cli.command()
@click.option("--global", "global_", is_flag=True, help="Register MCP server globally instead of per-project")
def up(global_: bool):
    """Start Weft infrastructure (Postgres + Redis) and run migrations."""
    click.echo("Starting Weft containers...")
    result = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "-p", "weft", "up", "-d"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        click.echo(f"Error: {result.stderr}", err=True)
        sys.exit(1)
    click.echo("Weft containers started.")

    # Run migrations
    async def _migrate():
        import asyncpg
        from weft.db.migrations import run_migrations

        config = load_config()
        pool = await asyncpg.create_pool(config.database.url, min_size=1, max_size=2)
        applied = await run_migrations(pool)
        await pool.close()
        return applied

    applied = asyncio.run(_migrate())
    if applied:
        click.echo(f"Applied {len(applied)} migration(s).")
    else:
        click.echo("Migrations up to date.")

    # Register MCP server
    if global_:
        _register_mcp()
    else:
        _register_mcp(Path.cwd())


@cli.command()
def down():
    """Stop Weft infrastructure."""
    click.echo("Stopping Weft containers...")
    result = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "-p", "weft", "down"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        click.echo(f"Error: {result.stderr}", err=True)
        sys.exit(1)
    click.echo("Weft containers stopped.")


@cli.command()
def status():
    """Show memory statistics overview."""
    from rich.console import Console
    from rich.table import Table

    async def _status():
        import asyncpg
        from weft.store import get_stats

        config = load_config()
        pool = await asyncpg.create_pool(config.database.url, min_size=1, max_size=2)
        stats = await get_stats(pool)
        await pool.close()
        return stats

    stats = asyncio.run(_status())
    console = Console()

    console.print(f"\n[bold]Weft Memory Status[/bold]")
    console.print(f"Total memories: {stats['total']}\n")

    if stats["by_status"]:
        t = Table(title="By Status")
        t.add_column("Status")
        t.add_column("Count", justify="right")
        for s, c in stats["by_status"].items():
            t.add_row(s, str(c))
        console.print(t)

    if stats["by_type"]:
        t = Table(title="By Type")
        t.add_column("Type")
        t.add_column("Count", justify="right")
        for tp, c in stats["by_type"].items():
            t.add_row(tp, str(c))
        console.print(t)

    if stats["top_topics"]:
        t = Table(title="Top Topics")
        t.add_column("Topic")
        t.add_column("Count", justify="right")
        for topic, c in stats["top_topics"].items():
            t.add_row(topic, str(c))
        console.print(t)


@cli.command()
@click.option("--dry-run", is_flag=True, help="Show what would happen without making changes")
def consolidate(dry_run: bool):
    """Run memory consolidation: decay, deduplicate, flag contradictions."""
    from rich.console import Console

    async def _consolidate():
        import asyncpg
        from weft.consolidation import consolidate as run_consolidation

        config = load_config()
        pool = await asyncpg.create_pool(config.database.url, min_size=1, max_size=2)
        report = await run_consolidation(pool, dry_run=dry_run)
        await pool.close()
        return report

    report = asyncio.run(_consolidate())
    console = Console()

    prefix = "[dim](dry run)[/dim] " if dry_run else ""
    console.print(f"\n[bold]{prefix}Consolidation Report[/bold]\n")
    console.print(f"Decayed: {len(report.decayed)}")
    console.print(f"Duplicates merged: {len(report.duplicates_merged)}")
    console.print(f"Contradictions flagged: {len(report.contradictions_flagged)}")

    if report.errors:
        console.print(f"\n[red]Errors: {len(report.errors)}[/red]")
        for err in report.errors:
            console.print(f"  - {err}")


@cli.command(name="export")
@click.option("--format", "-f", "fmt", type=click.Choice(["md", "json"]), default="md", help="Output format")
@click.option("--type", "-t", "memory_type", default=None, help="Filter by memory type")
@click.option("--topic", default=None, help="Filter by topic")
@click.option("--status", "-s", default="active", help="Filter by status")
@click.option("--output", "-o", "output_file", default=None, type=click.Path(), help="Write to file instead of stdout")
def export_cmd(fmt: str, memory_type: str | None, topic: str | None, status: str, output_file: str | None):
    """Export memories as markdown or JSON."""

    async def _export():
        import asyncpg
        from weft.exporter import export_memories

        config = load_config()
        pool = await asyncpg.create_pool(config.database.url, min_size=1, max_size=2)
        result = await export_memories(
            pool,
            format=fmt,
            memory_type=memory_type,
            topic=topic,
            status=status,
        )
        await pool.close()
        return result

    result = asyncio.run(_export())

    if output_file:
        Path(output_file).write_text(result, encoding="utf-8")
        click.echo(f"Exported to {output_file}")
    else:
        click.echo(result)


@cli.command(name="import")
@click.argument("file", type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True, help="Show what would be imported without storing")
@click.option("--project-id", default=None, help="Assign project ID to imported memories")
def import_cmd(file: str, dry_run: bool, project_id: str | None):
    """Import memories from a MEMORY.md file."""
    from weft.importer import import_memories, parse_memory_md

    # Parse the file first
    parsed = parse_memory_md(file)
    click.echo(f"Parsed {len(parsed.memories)} memories from {file}")
    if parsed.skipped:
        click.echo(f"Skipped {parsed.skipped} empty sections")

    if not parsed.memories:
        click.echo("Nothing to import.")
        return

    async def _import():
        import asyncpg

        from weft.embeddings import get_provider

        config = load_config()
        pool = await asyncpg.create_pool(config.database.url, min_size=1, max_size=2)
        provider = get_provider(config.embedding.provider)
        report = await import_memories(
            pool,
            provider,
            parsed.memories,
            project_id=project_id,
            dry_run=dry_run,
        )
        await pool.close()
        return report

    report = asyncio.run(_import())

    prefix = "(dry run) " if dry_run else ""
    click.echo(f"\n{prefix}Import Report:")
    click.echo(f"  Stored: {report.stored}")
    click.echo(f"  Duplicates skipped: {report.skipped_duplicate}")
    if report.skipped_empty:
        click.echo(f"  Empty skipped: {report.skipped_empty}")
    if report.errors:
        click.echo(f"  Errors: {len(report.errors)}")
        for err in report.errors:
            click.echo(f"    - {err}")


@cli.command()
@click.argument("query")
@click.option("--limit", "-n", default=5, help="Number of results")
@click.option("--topic", "-t", default=None, help="Filter by topic")
def recall(query: str, limit: int, topic: str | None):
    """Search memories by semantic query."""
    from rich.console import Console

    async def _recall():
        import asyncpg
        from weft.embeddings import get_provider
        from weft.store import search_by_vector

        config = load_config()
        pool = await asyncpg.create_pool(config.database.url, min_size=1, max_size=2)
        provider = get_provider(config.embedding.provider)
        embedding = await provider.embed(query)
        results = await search_by_vector(pool, embedding, limit=limit, topic=topic)
        await pool.close()
        return results

    results = asyncio.run(_recall())
    console = Console()

    if not results:
        console.print("[dim]No matching memories found.[/dim]")
        return

    console.print(f"\n[bold]Results for:[/bold] {query}\n")
    for i, r in enumerate(results, 1):
        m = r.memory
        sim = f"{r.similarity:.3f}"
        topics = ", ".join(m.topic) if m.topic else "-"
        console.print(f"[bold]{i}.[/bold] [{m.type.value}] (sim={sim}, conf={m.confidence})")
        console.print(f"   {m.content[:120]}")
        console.print(f"   [dim]topics: {topics} | id: {m.id}[/dim]\n")


@cli.command()
@click.option("--force", is_flag=True, help="Seed even if memories already exist")
def seed(force: bool):
    """Load starter memories into the store."""

    async def _seed():
        import asyncpg

        from weft.embeddings import get_provider
        from weft.seed import seed_memories

        config = load_config()
        pool = await asyncpg.create_pool(config.database.url, min_size=1, max_size=2)
        provider = get_provider(config.embedding.provider, model_name=config.embedding.model)
        count = await seed_memories(pool, provider, force=force)
        await pool.close()
        return count

    count = asyncio.run(_seed())
    if count:
        click.echo(f"Seeded {count} memories.")
    else:
        click.echo("No memories seeded (store already populated or no seeds found).")


@cli.group()
def config():
    """View and modify Weft configuration."""
    pass


@config.command()
def show():
    """Display current configuration as a table."""
    from rich.console import Console
    from rich.table import Table

    cfg = load_config()
    console = Console()

    t = Table(title="Weft Configuration")
    t.add_column("Key", style="bold")
    t.add_column("Value")
    t.add_column("Source", style="dim")

    # Determine which keys are set in the TOML file vs env vars
    toml_data = load_config_file()
    toml_flat: set[str] = set()
    for k, v in toml_data.items():
        if isinstance(v, dict):
            for sk in v:
                toml_flat.add(f"{k}.{sk}")
        else:
            toml_flat.add(k)

    env_keys: dict[str, str] = {
        "database.url": "WEFT_DATABASE_URL",
        "redis.url": "WEFT_REDIS_URL",
        "embedding.provider": "WEFT_EMBEDDING_PROVIDER",
        "embedding.model": "WEFT_EMBEDDING_MODEL",
        "log_level": "WEFT_LOG_LEVEL",
    }

    def _source(key: str) -> str:
        env_var = env_keys.get(key)
        if env_var and os.environ.get(env_var):
            return f"env ({env_var})"
        if key in toml_flat:
            return f"toml ({CONFIG_PATH})"
        return "default"

    rows = [
        ("project_name", cfg.project_name),
        ("log_level", cfg.log_level),
        ("database.url", cfg.database.url),
        ("database.pool_min_size", str(cfg.database.pool_min_size)),
        ("database.pool_max_size", str(cfg.database.pool_max_size)),
        ("redis.url", cfg.redis.url),
        ("embedding.provider", cfg.embedding.provider),
        ("embedding.model", cfg.embedding.model),
        ("embedding.dimensions", str(cfg.embedding.dimensions)),
        ("embedding.batch_size", str(cfg.embedding.batch_size)),
        ("retrieval.default_top_k", str(cfg.retrieval.default_top_k)),
        ("retrieval.similarity_threshold", str(cfg.retrieval.similarity_threshold)),
        ("retrieval.context_budget_tokens", str(cfg.retrieval.context_budget_tokens)),
        ("decay.enabled", str(cfg.decay.enabled)),
        ("decay.half_life_days", str(cfg.decay.half_life_days)),
        ("decay.floor_score", str(cfg.decay.floor_score)),
    ]

    for key, value in rows:
        t.add_row(key, value, _source(key))

    console.print(t)


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str):
    """Persist a configuration value to ~/.weft/config.toml."""
    from weft.config import _KEY_MAP

    if key not in _KEY_MAP:
        valid = ", ".join(sorted(_KEY_MAP.keys()))
        click.echo(f"Error: Unknown config key '{key}'.\nValid keys: {valid}", err=True)
        sys.exit(1)

    save_config_value(key, value)
    click.echo(f"Saved {key} = {value} to {CONFIG_PATH}")
