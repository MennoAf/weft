"""Weft CLI — memory management commands."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import click

from weft.config import CONFIG_PATH, load_config, load_config_file, save_config_value

COMPOSE_FILE = Path(__file__).parent.parent / "docker-compose.weft.yml"
_COMPOSE_PROJECT_ENV = "WEFT_COMPOSE_PROJECT"


def _compose_project_name() -> str:
    """Return the Docker Compose project name for local infrastructure."""
    return os.environ.get(_COMPOSE_PROJECT_ENV, "weft")


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
        ["docker", "compose", "-f", str(COMPOSE_FILE), "-p", _compose_project_name(), "up", "-d"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        click.echo(f"Error: {result.stderr}", err=True)
        sys.exit(1)
    click.echo("Weft containers started.")

    # Run migrations through the same owner-safe path used by deployments.
    applied = asyncio.run(_run_owner_migrations(load_config()))
    if applied:
        click.echo(f"Applied {len(applied)} migration(s).")
    else:
        click.echo("Migrations up to date.")

    # Register MCP server
    if global_:
        _register_mcp()
    else:
        _register_mcp(Path.cwd())


def _migration_config(ca_cert_file: str | None, database_url: str | None):
    """Load an owner-migration config from explicit owner-only inputs."""
    owner_url = database_url or os.environ.get("WEFT_OWNER_DATABASE_URL")
    if not owner_url:
        raise click.ClickException(
            "Owner migration requires WEFT_OWNER_DATABASE_URL or --database-url; "
            "the generic DATABASE_URL may point at restricted weft_app."
        )
    if ca_cert_file:
        os.environ["WEFT_DATABASE_CA_CERT_FILE"] = ca_cert_file
    os.environ["DATABASE_URL"] = owner_url
    os.environ["WEFT_MIGRATION_MODE"] = "apply"
    return load_config()


async def _run_owner_migrations(config):
    from weft.db.connection import create_pool
    from weft.db.migrations import run_migrations

    pool = await create_pool(config)
    try:
        role = await pool.fetchval("SELECT current_user")
        if role == "weft_app":
            raise RuntimeError(
                "refusing owner migration as restricted runtime role weft_app; "
                "use the Supabase owner/migration connection"
            )
        return await run_migrations(pool)
    finally:
        await pool.close()


@cli.command(name="migrate")
@click.option(
    "--ca-cert-file",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    default=None,
    help="Path to the trusted database CA PEM file.",
)
@click.option(
    "--database-url",
    default=None,
    help="Owner connection URL; prefer WEFT_OWNER_DATABASE_URL.",
)
def migrate_cmd(ca_cert_file: Path | None, database_url: str | None):
    """Apply pending owner-managed database migrations safely.

    This command is for the migration/owner role only. Fly production uses
    WEFT_MIGRATION_MODE=verify and the restricted weft_app role.
    """
    config = _migration_config(
        str(ca_cert_file) if ca_cert_file else None,
        database_url,
    )

    try:
        applied = asyncio.run(_run_owner_migrations(config))
    except Exception as exc:
        raise click.ClickException(f"Owner migration failed: {exc}") from exc

    if applied:
        click.echo(f"Applied migration(s): {', '.join(map(str, applied))}")
    else:
        click.echo("Migrations up to date.")


@cli.command(name="deploy")
@click.option("--app", default="weft-mcp", show_default=True, help="Fly app name.")
@click.option("--wait-timeout", default="10m", show_default=True, help="Fly machine health wait timeout.")
@click.option(
    "--ca-cert-file",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    default=None,
    help="Path to the owner database CA PEM file for preflight.",
)
@click.option("--skip-preflight", is_flag=True, help="Skip the read-only migration preflight (not recommended).")
def deploy_cmd(app: str, wait_timeout: str, ca_cert_file: Path | None, skip_preflight: bool):
    """Preflight and deploy the current Weft release to Fly.io."""
    if not skip_preflight:
        if not os.environ.get("WEFT_OWNER_DATABASE_URL"):
            raise click.ClickException(
                "Deploy preflight needs WEFT_OWNER_DATABASE_URL; set the owner connection "
                "in the environment or use --skip-preflight only after a separate check."
            )
        os.environ["DATABASE_URL"] = os.environ["WEFT_OWNER_DATABASE_URL"]
        if ca_cert_file:
            os.environ["WEFT_DATABASE_CA_CERT_FILE"] = str(ca_cert_file)
        config = load_config()

        async def _preflight():
            from weft.db.connection import create_pool
            from weft.db.migrations._runner import verify_migration_ledger

            pool = await create_pool(config)
            try:
                await verify_migration_ledger(pool)
            finally:
                await pool.close()

        try:
            asyncio.run(_preflight())
        except Exception as exc:
            raise click.ClickException(
                "Deployment preflight failed. Apply owner migrations with "
                "'weft migrate' before deploying: " + str(exc)
            ) from exc
        click.echo("Migration preflight passed.")

    result = subprocess.run(
        ["fly", "deploy", "-a", app, "--wait-timeout", wait_timeout],
        check=False,
    )
    if result.returncode:
        raise click.ClickException(f"Fly deploy failed with exit code {result.returncode}.")


@cli.command()
def down():
    """Stop Weft infrastructure."""
    click.echo("Stopping Weft containers...")
    result = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "-p", _compose_project_name(), "down"],
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

    console.print("\n[bold]Weft Memory Status[/bold]")
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


@cli.command(name="auto-consolidate")
@click.option("--dry-run", is_flag=True, help="Check if due without running")
@click.option("--force", is_flag=True, help="Run even if not due")
def auto_consolidate(dry_run: bool, force: bool):
    """Run consolidation if due (for cron/scheduled jobs)."""
    import json as json_mod

    async def _run():
        import asyncpg
        from weft.consolidation import consolidate_if_due, should_consolidate, consolidate, record_consolidation_run

        config = load_config()
        pool = await asyncpg.create_pool(config.database.url, min_size=1, max_size=2)
        try:
            if dry_run:
                due = await should_consolidate(pool)
                return {"ran": False, "due": due, "dry_run": True}

            if force:
                await record_consolidation_run(pool, status="running")
                report = await consolidate(pool)
                processed = len(report.decayed) + len(report.duplicates_merged) + len(report.contradictions_flagged)
                await record_consolidation_run(pool, memories_processed=processed, status="completed")
                return {
                    "ran": True, "forced": True,
                    "decayed": len(report.decayed),
                    "duplicates_merged": len(report.duplicates_merged),
                    "contradictions_flagged": len(report.contradictions_flagged),
                }

            return await consolidate_if_due(pool)
        finally:
            await pool.close()

    result = asyncio.run(_run())
    click.echo(json_mod.dumps(result, indent=2))
    raise SystemExit(0 if result.get("ran", False) or not result.get("due", True) else 0)


@cli.command(name="export")
@click.option("--format", "-f", "fmt", type=click.Choice(["md", "json"]), default="md", help="Output format")
@click.option("--type", "-t", "memory_type", default=None, help="Filter by memory type")
@click.option("--topic", default=None, help="Filter by topic")
@click.option("--status", "-s", default="active", help="Filter by status")
@click.option("--output", "-o", "output_file", default=None, type=click.Path(), help="Write to file instead of stdout")
@click.option("--all", "export_all", is_flag=True, help="Export all users' memories (operator-only)")
def export_cmd(fmt: str, memory_type: str | None, topic: str | None, status: str, output_file: str | None, export_all: bool):
    """Export memories as markdown or JSON."""

    async def _export():
        import asyncpg
        from weft.auth import resolve_caller_user_id
        from weft.exporter import export_memories

        config = load_config()
        pool = await asyncpg.create_pool(config.database.url, min_size=1, max_size=2)
        result = await export_memories(
            pool,
            format=fmt,
            memory_type=memory_type,
            topic=topic,
            status=status,
            user_id=None if export_all else resolve_caller_user_id(),
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
        from weft.auth import current_user_id
        from weft.config.user_identity import get_user_id
        from weft.db.connection import acquire, create_pool
        from weft.embeddings import get_provider
        from weft.store import search_by_vector

        config = load_config()
        pool = await create_pool(config)
        identity_token = None
        try:
            caller_uid = get_user_id()
            identity_token = current_user_id.set(caller_uid)
            provider = get_provider(
                config.embedding.provider,
                model_name=config.embedding.model,
                dimensions=config.embedding.dimensions,
            )
            embedding = await provider.embed(query)
            async with acquire(pool):
                return await search_by_vector(
                    pool,
                    embedding,
                    limit=limit,
                    topic=topic,
                    user_id=caller_uid,
                )
        finally:
            if identity_token is not None:
                current_user_id.reset(identity_token)
            await pool.close()

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
@click.argument("intent")
@click.option("--project", "-p", default=None, help="Project ID")
@click.option("--exclude", "-e", multiple=True, help="Memory IDs to exclude (repeatable)")
@click.option("--budget", "-b", default=1200, help="Token budget (default 1200)")
def focus(intent: str, project: str | None, exclude: tuple[str, ...], budget: int):
    """Post-intent re-prime — surface memories the primer missed."""
    from rich.console import Console

    async def _focus():
        import asyncpg
        from weft.db.connection import _pgvector_codec_init
        from weft.embeddings import get_provider
        from weft.focus import build_focus

        config = load_config()
        pool = await asyncpg.create_pool(
            config.database.url, min_size=1, max_size=2, init=_pgvector_codec_init,
        )
        provider = get_provider(
            config.embedding.provider,
            model_name=config.embedding.model,
            dimensions=config.embedding.dimensions,
        )
        result = await build_focus(
            pool,
            intent=intent,
            embedding_fn=provider.embed,
            project_id=project,
            exclude_memory_ids=list(exclude),
            budget_tokens=budget,
        )
        await pool.close()
        return result

    result = asyncio.run(_focus())
    console = Console()
    console.print(result.format())


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
    # v1.2 sections — only show non-zero counts to keep the output tight.
    for key in (
        "behaviors_count",
        "entities_count",
        "entity_mentions_count",
        "episodes_count",
        "episode_memories_count",
        "modes_count",
        "trackers_count",
        "workspaces_count",
        "workspace_members_count",
    ):
        n = report.get(key, 0)
        if n:
            label = key.removesuffix("_count").replace("_", " ").title()
            click.echo(f"  {label}: {n}")
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
    for section in (
        "memories",
        "relationships",
        "behaviors",
        "entities",
        "entity_mentions",
        "episodes",
        "episode_memories",
        "modes",
        "trackers",
        "workspaces",
        "workspace_members",
    ):
        restored = result.get(f"{section}_restored", 0)
        skipped = result.get(f"{section}_skipped", 0)
        if restored or skipped:
            label = section.replace("_", " ").title()
            click.echo(f"  {label}: {restored} restored, {skipped} skipped")
    if result["errors"]:
        click.echo(f"  Errors: {len(result['errors'])}")
        for err in result["errors"]:
            click.echo(f"    - {err}")


@cli.group()
def quarantine():
    """Layer 3 / 3.5 quarantine review commands for agent-provenance writes."""
    pass


@quarantine.command(name="review-llm")
@click.option(
    "--limit", default=50, type=int,
    help="Max memories to classify per run.",
)
@click.option(
    "--concurrency", default=4, type=int,
    help="Max concurrent LLM calls.",
)
@click.option(
    "--since", "since_iso", default=None,
    help=(
        "Override the persistent watermark with an ISO-8601 timestamp. "
        "Useful for ad-hoc backfills."
    ),
)
@click.option(
    "--no-watermark", is_flag=True,
    help="Do not advance the persisted watermark — useful for dry diagnosis.",
)
@click.option(
    "--model", default="claude-haiku-4-5-20251001",
    help="Anthropic model to use (Haiku by default for cost).",
)
def quarantine_review_llm(
    limit: int,
    concurrency: int,
    since_iso: str | None,
    no_watermark: bool,
    model: str,
):
    """Run an LLM-review pass on agent-provenance memories that bypassed Layer 3.

    Reads memories with ``write_provenance='agent'`` and
    ``review_status='active'`` newer than the stored watermark, asks Haiku
    to classify each as FACT or INSTRUCTION, and flips the
    INSTRUCTION-verdict rows to ``pending_review`` for supervisor review.
    """
    async def _run():
        import asyncpg
        from anthropic import AsyncAnthropic

        from weft.quarantine_review import llm_review_pending

        config = load_config()
        api_key = os.environ.get("ANTHROPIC_API_KEY") or config.api_key
        if not api_key:
            click.echo(
                "Error: No API key found. Set ANTHROPIC_API_KEY or WEFT_API_KEY.",
                err=True,
            )
            sys.exit(1)
        client = AsyncAnthropic(api_key=api_key)
        pool = await asyncpg.create_pool(
            config.database.url, min_size=1, max_size=2,
        )
        since = (
            datetime.fromisoformat(since_iso) if since_iso else None
        )
        try:
            return await llm_review_pending(
                pool, client,
                since=since,
                limit=limit,
                concurrency=concurrency,
                model=model,
                advance_watermark=not no_watermark,
            )
        finally:
            await pool.close()

    report = asyncio.run(_run())
    click.echo("Quarantine LLM review:")
    click.echo(f"  Checked:    {report.checked}")
    click.echo(f"  Flagged:    {report.flagged}")
    click.echo(f"  Ambiguous:  {report.ambiguous}")
    click.echo(f"  Errors:     {len(report.errors)}")
    if report.flagged_ids:
        click.echo("  Flagged IDs:")
        for mid in report.flagged_ids:
            click.echo(f"    - {mid}")
    if report.errors:
        click.echo("  Error details:")
        for err in report.errors:
            click.echo(f"    - {err}")
    click.echo(f"  Watermark before: {report.watermark_before}")
    click.echo(f"  Watermark after:  {report.watermark_after}")


@cli.command()
def fsck():
    """List orphan memories: active memories reachable ONLY by vector cosine.

    Orphans are memories with NO topic tags AND NO entity mentions AND NO
    episode membership. They hide in the vector index but cannot be recalled
    through tag/entity/episode navigators — a leading indicator of future
    recall misses.
    """
    async def _run():
        import asyncpg
        from weft.fsck import list_orphan_memories
        from weft.config.user_identity import get_user_id

        config = load_config()
        pool = await asyncpg.create_pool(
            config.database.url, min_size=1, max_size=2,
        )
        try:
            caller_uid = get_user_id()
            orphans = await list_orphan_memories(pool, user_id=caller_uid)
            return orphans
        finally:
            await pool.close()

    orphans = asyncio.run(_run())
    click.echo(f"Found {len(orphans)} orphan memory(ies) in vector index only:")
    if orphans:
        for orphan in orphans:
            click.echo(f"  - {orphan['memory_id']}: {orphan['reason']}")
    else:
        click.echo("  (None)")


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
        from weft.db.connection import create_pool
        from weft.db.reembed import reembed_table
        from weft.embeddings import get_provider

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

        if dry_run:
            for table in tables:
                count = await pool.fetchval(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
                console.print(f"\n[bold]{table}[/bold]: {count} rows")
            await pool.close()
            return 0

        total_updated = 0
        for table in tables:
            count = await pool.fetchval(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
            console.print(f"\n[bold]{table}[/bold]: {count} rows")
            if count == 0:
                continue
            updated = await reembed_table(pool, table, provider, batch_size, force=True)
            console.print(f"  {updated} rows re-embedded")
            total_updated += updated

        await pool.close()
        return total_updated

    total = asyncio.run(_re_embed())
    if dry_run:
        console.print("\n[dim](dry run — no changes made)[/dim]")
    else:
        console.print(f"\n[bold]Re-embedded {total} rows.[/bold]")


@cli.command(name="calendar-auth")
@click.option(
    "--client-secrets",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Path to Google OAuth client secrets JSON file.",
)
def calendar_auth(client_secrets: str):
    """Authenticate with Google Calendar (one-time setup)."""
    from weft.google_calendar import run_oauth_flow, CREDENTIALS_PATH

    try:
        run_oauth_flow(client_secrets)
        click.echo(f"Credentials saved to {CREDENTIALS_PATH}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


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
        "database.ca_cert_file": "WEFT_DATABASE_CA_CERT_FILE",
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
        ("database.ca_cert_file", str(cfg.database.ca_cert_file or "")),
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


# ---------------------------------------------------------------------------
# Identity — manage the canonical user_id for this installation.
# ---------------------------------------------------------------------------


@cli.group()
def identity():
    """Manage the canonical user_id for this Weft installation.

    Precedence: WEFT_USER_ID env var → ~/.weft/user_id.json → random UUID.
    Set your identity explicitly to bind local data to your hosted JWT sub
    (or any other canonical ID) so both paths agree on who owns what.
    """
    pass


@identity.command("show")
def identity_show():
    """Print the current user_id and how it was resolved."""
    from weft.config.user_identity import describe_user_id

    info = describe_user_id()
    source_note = {
        "env": f"from ${info['env_var']} environment variable",
        "config": f"from {info['config_path']}",
        "unset": "not yet set — next call to get_user_id() will generate a random UUID",
    }.get(info["source"], info["source"])

    if info["user_id"]:
        click.echo(f"user_id: {info['user_id']}")
        click.echo(f"source:  {source_note}")
    else:
        click.echo("user_id: (unset)")
        click.echo(f"source:  {source_note}")
        click.echo(
            "\nTo bind this install to a specific identity (e.g. your hosted "
            "JWT sub), run:\n  weft identity set <user-id>"
        )


@identity.command("set")
@click.argument("user_id")
def identity_set(user_id: str):
    """Persist a canonical user_id to ~/.weft/user_id.json.

    Overwrites any existing value. The WEFT_USER_ID env var still takes
    precedence at read time if set — unset it in your shell if you want
    the persisted value to be authoritative.
    """
    from weft.config.user_identity import describe_user_id, set_user_id

    set_user_id(user_id)
    info = describe_user_id()
    click.echo(f"Saved user_id={user_id} to {info['config_path']}")
    if info["source"] == "env":
        click.echo(
            f"\nNote: ${info['env_var']} is currently set and overrides the "
            f"config file. Unset it to make the persisted value authoritative."
        )


@cli.group()
def tokens():
    """Manage Weft bearer tokens (Phase 2.5 credential-bound auth).

    Tokens are first-class auth credentials: each one binds a user to a
    caller mode (supervisor or agent) at issuance time. The middleware
    resolves Authorization headers through these rows, so an agent
    holding a valid token cannot forge ``X-Weft-Caller-Mode: supervisor``.

    Bootstrap flow for a new install:
      weft tokens issue --user-id <UUID> --mode supervisor --label face
      → copy the printed token, set as Authorization header on the client.
    """
    pass


def _parse_expires_in(spec: str | None) -> "timedelta | None":
    """Accept '30d', '12h', '60m'. None means no expiry."""
    from datetime import timedelta

    if not spec:
        return None
    spec = spec.strip().lower()
    units = {"d": "days", "h": "hours", "m": "minutes"}
    if spec[-1] not in units or not spec[:-1].isdigit():
        raise click.BadParameter(
            f"--expires-in must be N(d|h|m), got {spec!r}"
        )
    return timedelta(**{units[spec[-1]]: int(spec[:-1])})


def _format_token_status(row) -> str:
    if row.revoked_at is not None:
        return "revoked"
    if row.expires_at is not None and not row.is_active():
        return "expired"
    return "active"


def _format_dt(dt) -> str:
    if dt is None:
        return "-"
    return dt.strftime("%Y-%m-%d %H:%M")


def _short_hash(token_hash: str) -> str:
    """First 12 chars of the SHA-256 hex — enough to disambiguate
    in the listing without leaking enough material to be useful if
    the table output ends up in a bug report."""
    return token_hash[:12]


@tokens.command("issue")
@click.option("--user-id", required=True, help="User this token belongs to.")
@click.option(
    "--mode",
    type=click.Choice(["supervisor", "agent"]),
    required=True,
    help="Caller mode the token is bound to.",
)
@click.option("--label", default=None, help="Operator-supplied free text.")
@click.option(
    "--expires-in",
    default=None,
    help="Optional expiry: N(d|h|m), e.g. 30d. Default: never expires.",
)
def tokens_issue(user_id: str, mode: str, label: str | None, expires_in: str | None):
    """Mint a bearer token. Plaintext is printed ONCE — store it now."""
    expires_delta = _parse_expires_in(expires_in)

    async def _issue():
        import asyncpg
        from weft.credentials import issue_token

        config = load_config()
        pool = await asyncpg.create_pool(config.database.url, min_size=1, max_size=2)
        try:
            return await issue_token(
                pool,
                user_id=user_id,
                caller_mode=mode,
                label=label,
                expires_in=expires_delta,
            )
        finally:
            await pool.close()

    plaintext, row = asyncio.run(_issue())
    click.echo(f"Token: {plaintext}")
    click.echo(f"Hash:  {row.token_hash}")
    click.echo(f"User:  {row.user_id}")
    click.echo(f"Mode:  {row.caller_mode}")
    if row.label:
        click.echo(f"Label: {row.label}")
    if row.expires_at:
        click.echo(f"Expires: {_format_dt(row.expires_at)}")
    click.echo("")
    click.echo(
        "Store this token now — it will not be shown again. "
        "Use the hash to revoke later."
    )


@tokens.command("list")
@click.option(
    "--user-id",
    required=True,
    help="User whose tokens to list.",
)
@click.option(
    "--include-revoked",
    is_flag=True,
    default=False,
    help="Include revoked rows alongside live ones.",
)
def tokens_list(user_id: str, include_revoked: bool):
    """List a user's tokens, newest first."""
    async def _list():
        import asyncpg
        from weft.credentials import list_tokens

        config = load_config()
        pool = await asyncpg.create_pool(config.database.url, min_size=1, max_size=2)
        try:
            return await list_tokens(
                pool, user_id, include_revoked=include_revoked,
            )
        finally:
            await pool.close()

    rows = asyncio.run(_list())
    if not rows:
        click.echo("No tokens found.")
        return

    header = (
        f"{'HASH':<14}{'MODE':<11}{'LABEL':<22}"
        f"{'CREATED':<18}{'LAST USED':<18}{'STATUS':<9}"
    )
    click.echo(header)
    click.echo("-" * len(header))
    for row in rows:
        click.echo(
            f"{_short_hash(row.token_hash):<14}"
            f"{row.caller_mode:<11}"
            f"{(row.label or '-'):<22}"
            f"{_format_dt(row.created_at):<18}"
            f"{_format_dt(row.last_used_at):<18}"
            f"{_format_token_status(row):<9}"
        )


@tokens.command("revoke")
@click.argument("token_hash")
def tokens_revoke(token_hash: str):
    """Revoke a token by its full SHA-256 hash.

    Get the hash from `weft tokens list` (the HASH column shows the
    first 12 chars; pass the full 64-char value here to disambiguate
    intent — partial hashes are deliberately not accepted)."""
    if len(token_hash) != 64:
        raise click.BadParameter(
            f"token_hash must be the full 64-char SHA-256 hex, got {len(token_hash)} chars"
        )

    async def _revoke():
        import asyncpg
        from weft.credentials import revoke_token

        config = load_config()
        pool = await asyncpg.create_pool(config.database.url, min_size=1, max_size=2)
        try:
            return await revoke_token(pool, token_hash)
        finally:
            await pool.close()

    flipped = asyncio.run(_revoke())
    if flipped:
        click.echo(f"Revoked {_short_hash(token_hash)}")
    else:
        click.echo(
            f"No live token matched {_short_hash(token_hash)} "
            "(unknown hash or already revoked).",
            err=True,
        )

