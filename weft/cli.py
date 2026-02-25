"""Weft CLI — memory management commands."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import click

from weft.config import load_config

COMPOSE_FILE = Path(__file__).parent.parent / "docker-compose.weft.yml"


@click.group()
def cli():
    """Weft — Persistent agent memory system."""
    pass


@cli.command()
def up():
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
