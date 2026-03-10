"""Weft CLI — memory management commands."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
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
        provider = get_provider(config.embedding.provider, model_name=config.embedding.model, dimensions=config.embedding.dimensions)
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
        provider = get_provider(config.embedding.provider, model_name=config.embedding.model, dimensions=config.embedding.dimensions)
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
        provider = get_provider(config.embedding.provider, model_name=config.embedding.model, dimensions=config.embedding.dimensions)
        count = await seed_memories(pool, provider, force=force)
        await pool.close()
        return count

    count = asyncio.run(_seed())
    if count:
        click.echo(f"Seeded {count} memories.")
    else:
        click.echo("No memories seeded (store already populated or no seeds found).")


@cli.command()
@click.argument("path", default=None, required=False, type=click.Path())
@click.option("--project-id", required=True, help="Project ID to associate ingested memories with")
@click.option("--depth", default="full", type=click.Choice(["architecture", "full"]), help="Ingestion depth")
def ingest(path: str | None, project_id: str, depth: str):
    """Ingest a codebase directory into Weft as memories."""
    target = Path(path) if path else Path.cwd()
    if not target.is_dir():
        click.echo(f"Error: {target} is not a directory.", err=True)
        sys.exit(1)

    async def _ingest():
        import asyncpg
        from anthropic import AsyncAnthropic

        from weft.embeddings import get_provider
        from weft.ingest import run_ingest

        config = load_config()
        pool = await asyncpg.create_pool(config.database.url, min_size=1, max_size=2)
        # Use WEFT_API_KEY / config.api_key if ANTHROPIC_API_KEY isn't set
        api_key = os.environ.get("ANTHROPIC_API_KEY") or config.api_key
        if not api_key:
            click.echo("Error: No API key found. Set ANTHROPIC_API_KEY or WEFT_API_KEY.", err=True)
            sys.exit(1)
        client = AsyncAnthropic(api_key=api_key)
        provider = get_provider(config.embedding.provider, model_name=config.embedding.model, dimensions=config.embedding.dimensions)
        result = await run_ingest(
            target, project_id, depth=depth, pool=pool, client=client,
            embedding_provider=provider,
        )
        await pool.close()
        return result

    result = asyncio.run(_ingest())
    click.echo(f"\nIngest complete for {target}")
    click.echo(f"  Project: {project_id}")
    click.echo(f"  Depth: {depth}")
    click.echo(f"  Files discovered: {result['files_discovered']}")
    click.echo(f"  Files summarized: {result['files_summarized']}")
    click.echo(f"  Architecture stored: {result['architecture_stored']}")


@cli.command()
@click.option("--output", "-o", "output_file", default=None, type=click.Path(), help="Write to file (default: weft-backup-<timestamp>.json)")
def backup(output_file: str | None):
    """Create a full backup of all memories, relationships, and embeddings."""
    from weft.backup import backup_all, verify_backup

    async def _backup():
        from weft.db.connection import create_pool

        config = load_config()
        pool = await create_pool(config)
        data = await backup_all(pool)
        await pool.close()
        return data

    data = asyncio.run(_backup())

    # Verify before writing
    report = verify_backup(data)
    if not report["valid"]:
        click.echo("WARNING: Backup verification found issues:", err=True)
        for issue in report["issues"]:
            click.echo(f"  - {issue}", err=True)

    if not output_file:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_file = f"weft-backup-{ts}.json"

    Path(output_file).write_text(json.dumps(data, indent=2), encoding="utf-8")

    click.echo(f"Backup written to {output_file}")
    click.echo(f"  Memories: {data['memory_count']}")
    click.echo(f"  Relationships: {data['relationship_count']}")
    click.echo(f"  With embeddings: {report['memories_with_embeddings']}")
    click.echo(f"  Checksum: {data['checksum'][:16]}...")


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True, help="Show what would be restored without writing")
@click.option("--no-skip-duplicates", is_flag=True, help="Fail on duplicate IDs instead of skipping")
def restore(file: str, dry_run: bool, no_skip_duplicates: bool):
    """Restore memories and relationships from a backup file."""
    from weft.backup import restore_all, verify_backup

    # Load and verify
    raw = Path(file).read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        click.echo(f"Error: Invalid JSON in {file}: {e}", err=True)
        sys.exit(1)

    report = verify_backup(data)
    if not report["valid"]:
        click.echo("Backup verification failed:", err=True)
        for issue in report["issues"]:
            click.echo(f"  - {issue}", err=True)
        sys.exit(1)

    click.echo(f"Backup file: {file}")
    click.echo(f"  Version: {data.get('version')}")
    click.echo(f"  Exported: {data.get('exported_at')}")
    click.echo(f"  Memories: {report['memory_count']}")
    click.echo(f"  Relationships: {report['relationship_count']}")
    click.echo(f"  With embeddings: {report['memories_with_embeddings']}")

    async def _restore():
        from weft.db.connection import create_pool
        from weft.db.migrations import run_migrations

        config = load_config()
        pool = await create_pool(config)
        await run_migrations(pool)
        result = await restore_all(
            pool, data,
            dry_run=dry_run,
            skip_duplicates=not no_skip_duplicates,
        )
        await pool.close()
        return result

    result = asyncio.run(_restore())

    prefix = "(dry run) " if dry_run else ""
    click.echo(f"\n{prefix}Restore Report:")
    click.echo(f"  Memories restored: {result['memories_restored']}")
    click.echo(f"  Memories skipped: {result['memories_skipped']}")
    click.echo(f"  Relationships restored: {result['relationships_restored']}")
    click.echo(f"  Relationships skipped: {result['relationships_skipped']}")
    if result["errors"]:
        click.echo(f"  Errors: {len(result['errors'])}")
        for err in result["errors"]:
            click.echo(f"    - {err}")


@cli.group()
def obsidian():
    """Obsidian vault sync commands."""
    pass


@obsidian.command(name="init")
@click.argument("vault_path", type=click.Path())
def obsidian_init(vault_path: str):
    """Create vault folder structure and frontmatter templates."""
    from weft.obsidian.vault_init import init_vault

    result = init_vault(Path(vault_path))
    click.echo(f"Initialized vault at {result['vault_path']}")
    click.echo(f"  Directories created: {result['dirs_created']}/{result['total_dirs']}")
    click.echo(f"  Templates created: {result['templates_created']}/{result['total_templates']}")
    click.echo("\nOpen this folder in Obsidian to start using it.")
    click.echo("Templates are in the templates/ folder — configure Obsidian to use them.")


@obsidian.command(name="sync")
@click.argument("vault_path", type=click.Path(exists=True))
@click.option("--hash-store", "hash_store_path", default=None, type=click.Path(), help="Path to hash store JSON file")
@click.option("--dry-run", is_flag=True, help="Show what would be synced without storing")
def obsidian_sync(vault_path: str, hash_store_path: str | None, dry_run: bool):
    """Sync an Obsidian vault into Weft memories."""
    from weft.obsidian.hash_store import HashStore
    from weft.obsidian.sync import discover_vault_files, sync_vault

    target = Path(vault_path)

    if dry_run:
        files = discover_vault_files(target)
        click.echo(f"Found {len(files)} markdown files in {target}")
        for f in files:
            click.echo(f"  {f}")
        return

    async def _sync():
        import asyncpg

        from weft.embeddings import get_provider

        config = load_config()
        pool = await asyncpg.create_pool(config.database.url, min_size=1, max_size=2)
        provider = get_provider(config.embedding.provider, model_name=config.embedding.model, dimensions=config.embedding.dimensions)

        hs = None
        if hash_store_path:
            hs = HashStore(Path(hash_store_path))

        result = await sync_vault(target, pool, provider, hash_store=hs)
        await pool.close()
        return result

    result = asyncio.run(_sync())
    click.echo(f"\nSync complete for {target}")
    click.echo(f"  Files found: {result.files_found}")
    click.echo(f"  Files synced: {result.files_synced}")
    click.echo(f"  Files skipped (unchanged): {result.files_skipped}")
    click.echo(f"  Files errored: {result.files_errored}")
    click.echo(f"  Memories created: {result.memories_created}")
    click.echo(f"  Memories archived: {result.memories_archived}")


@cli.group()
def slack():
    """Slack channel sync commands."""
    pass


@slack.command(name="sync")
@click.option("--token", "bot_token", envvar="SLACK_BOT_TOKEN", required=True, help="Slack bot token (or set SLACK_BOT_TOKEN)")
@click.option("--limit", "limit_per_channel", default=200, help="Max messages per channel")
@click.option("--state-file", "state_path", default=None, type=click.Path(), help="Path to sync state JSON file")
def slack_sync(bot_token: str, limit_per_channel: int, state_path: str | None):
    """Sync Slack channel history into Weft memories."""
    from weft.slack.hash_store import SlackSyncState
    from weft.slack.sync import sync_slack_sdk

    async def _sync():
        import asyncpg

        from weft.embeddings import get_provider

        cfg = load_config()
        pool = await asyncpg.create_pool(cfg.database.url, min_size=1, max_size=2)
        provider = get_provider(cfg.embedding.provider, model_name=cfg.embedding.model, dimensions=cfg.embedding.dimensions)

        ss = None
        if state_path:
            ss = SlackSyncState(Path(state_path))

        result = await sync_slack_sdk(
            pool,
            bot_token,
            provider,
            sync_state=ss,
            limit_per_channel=limit_per_channel,
        )
        await pool.close()
        return result

    result = asyncio.run(_sync())
    click.echo("\nSlack sync complete")
    click.echo(f"  Channels synced: {result.channels_synced}")
    click.echo(f"  Messages found: {result.messages_found}")
    click.echo(f"  Messages synced: {result.messages_synced}")
    click.echo(f"  Messages skipped (unchanged): {result.messages_skipped}")
    click.echo(f"  Messages updated: {result.messages_updated}")
    click.echo(f"  Messages errored: {result.messages_errored}")
    click.echo(f"  Memories created: {result.memories_created}")
    click.echo(f"  Memories archived: {result.memories_archived}")


@cli.command(name="re-embed")
@click.option("--batch-size", default=64, help="Number of memories to embed per batch")
@click.option("--dry-run", is_flag=True, help="Show counts without re-embedding")
@click.option("--table", "tables", multiple=True, default=("memories", "behaviors", "entities"),
              help="Tables to re-embed (default: all three)")
def re_embed(batch_size: int, dry_run: bool, tables: tuple[str, ...]):
    """Re-embed all memories/behaviors/entities with the current embedding provider.

    Use after switching embedding providers or dimensions.
    """
    from rich.console import Console

    console = Console()

    async def _re_embed():
        import asyncpg

        from weft.embeddings import get_provider
        from weft.store import _vec_to_pgvector

        from weft.db.connection import create_pool

        config = load_config()
        pool = await create_pool(config)
        provider = get_provider(
            config.embedding.provider,
            model_name=config.embedding.model,
            dimensions=config.embedding.dimensions,
        )

        # Run migrations first to ensure schema is up to date
        from weft.db.migrations import run_migrations
        applied = await run_migrations(pool)
        if applied:
            console.print(f"Applied {len(applied)} pending migration(s)")

        console.print(f"Provider: [bold]{provider.provider_name}[/bold] ({provider.dimensions} dims)")

        # Table → content column mapping
        table_content_col = {
            "memories": "content",
            "behaviors": "action",
            "entities": "description",
        }

        total_updated = 0

        for table in tables:
            content_col = table_content_col.get(table)
            if not content_col:
                console.print(f"[red]Unknown table: {table}[/red]")
                continue

            # Check table exists
            exists = await pool.fetchval(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = $1)",
                table,
            )
            if not exists:
                console.print(f"\n[dim]{table}[/dim]: table does not exist, skipping")
                continue

            # Count rows
            count = await pool.fetchval(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
            console.print(f"\n[bold]{table}[/bold]: {count} rows")

            if count == 0 or dry_run:
                continue

            # Fetch all rows with content
            if table == "entities":
                # entities may have NULL description
                rows = await pool.fetch(
                    f"SELECT id, {content_col} FROM {table} WHERE {content_col} IS NOT NULL"  # noqa: S608
                )
            else:
                rows = await pool.fetch(f"SELECT id, {content_col} FROM {table}")  # noqa: S608

            # Process in batches
            updated = 0
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                texts = [r[content_col] for r in batch]
                ids = [r["id"] for r in batch]

                embeddings = await provider.embed_batch(texts)

                async with pool.acquire() as conn:
                    async with conn.transaction():
                        for row_id, emb in zip(ids, embeddings):
                            await conn.execute(
                                f"UPDATE {table} SET embedding = $1 WHERE id = $2",  # noqa: S608
                                _vec_to_pgvector(emb),
                                row_id,
                            )
                updated += len(batch)
                console.print(f"  {updated}/{len(rows)} rows updated", end="\r")

            console.print(f"  {updated}/{len(rows)} rows updated")
            total_updated += updated

        await pool.close()
        return total_updated

    total = asyncio.run(_re_embed())
    if dry_run:
        console.print("\n[dim](dry run — no changes made)[/dim]")
    else:
        console.print(f"\n[bold]Re-embedded {total} rows.[/bold]")


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
