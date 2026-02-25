"""Weft CLI — memory management commands."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click

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
    # TODO: run migrations once db module is built


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
    # TODO: implement once store module is built
    click.echo("Weft status: not yet implemented (pending store.py)")


@cli.command()
@click.argument("query")
def recall(query: str):
    """Search memories by semantic query."""
    # TODO: implement once store + embeddings modules are built
    click.echo(f"Recall for '{query}': not yet implemented (pending store.py + embeddings)")
