#!/usr/bin/env python3
"""Run a bounded, provider-free RC smoke against disposable Docker services.

The script starts only Postgres/pgvector and Redis on a private Docker network.
It never uses the default ``weft`` Compose project, host ports, or named volumes.
The application probe runs inside the supplied RC image, so the smoke validates
the installed artifact rather than the source checkout.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import uuid


def run(command: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run a command and raise with its captured output on failure."""
    result = subprocess.run(command, text=True, capture_output=True, env=env)
    if result.returncode:
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        raise SystemExit(result.returncode)
    return result


def wait_for(container: str, command: list[str], timeout: float) -> None:
    """Wait until a command executed in *container* succeeds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["docker", "exec", container, *command],
            text=True,
            capture_output=True,
        )
        if result.returncode == 0:
            return
        time.sleep(2)
    raise RuntimeError(f"Timed out waiting for {container}: {' '.join(command)}")


def probe(image: str, network: str, dsn: str, phase: str, sentinel: str) -> str:
    """Run one application probe inside the RC image on the private network."""
    code = r'''
import asyncio
import importlib.metadata
import sys

import asyncpg
import weft
from weft.auth import current_user_id
from weft.db.connection import acquire
from weft.db.migrations import run_migrations
from weft.models import MemoryCreate, MemorySource, MemoryType
from weft.store import search_by_keyword, store_memory

DSN = sys.argv[1]
PHASE = sys.argv[2]
SENTINEL = sys.argv[3]
UID = "rc-smoke-user"

async def main():
    runtime = weft.__version__
    package = importlib.metadata.version("weft-memory")
    assert runtime == package == "1.0.0rc1", (runtime, package)
    pool = await asyncpg.create_pool(DSN, min_size=1, max_size=2)
    applied = await run_migrations(pool)
    if PHASE == "init":
        assert len(applied) == 72, len(applied)
        current_user_id.set(UID)
        async with acquire(pool):
            await store_memory(
                pool,
                MemoryCreate(
                    type=MemoryType.fact,
                    content=SENTINEL,
                    topic=["rc-smoke"],
                    source=MemorySource.conversation,
                    project_id="rc-smoke",
                ),
                embedding=None,
            )
        print(f"phase=init version={runtime} migrations={len(applied)} write=ok")
    else:
        assert len(applied) == 0, len(applied)
        current_user_id.set(UID)
        async with acquire(pool):
            results = await search_by_keyword(
                pool,
                SENTINEL,
                limit=5,
                project_id="rc-smoke",
                user_id=UID,
            )
        assert results, "sentinel was not recalled after restart"
        assert any(row.memory.content == SENTINEL for row in results)
        print(f"phase=verify version={runtime} migrations={len(applied)} recall=ok")
    await pool.close()

asyncio.run(main())
'''
    result = run(
        [
            "docker", "run", "--rm", "--network", network,
            "-e", f"WEFT_DATABASE_URL={dsn}",
            "-e", "WEFT_REDIS_URL=redis://rc-redis:6379",
            "-e", "WEFT_TESTING=1",
            image, "/app/.venv/bin/python", "-c", code, dsn, phase, sentinel,
        ]
    )
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="RC image containing the installed Weft artifact")
    parser.add_argument("--timeout", type=float, default=90.0, help="Readiness timeout in seconds")
    args = parser.parse_args()

    suffix = uuid.uuid4().hex[:10]
    network = f"weft-rc-smoke-{suffix}"
    postgres = f"rc-postgres-{suffix}"
    redis = f"rc-redis-{suffix}"
    dsn = f"postgresql://weft:weft_local@{postgres}:5432/weft"
    sentinel = f"Weft RC smoke sentinel {suffix}"
    resources = [postgres, redis]
    try:
        run(["docker", "network", "create", network])
        run(
            [
                "docker", "run", "-d", "--name", postgres, "--network", network,
                "-e", "POSTGRES_DB=weft", "-e", "POSTGRES_USER=weft",
                "-e", "POSTGRES_PASSWORD=weft_local", "pgvector/pgvector:pg16",
            ]
        )
        run(
            [
                "docker", "run", "-d", "--name", redis, "--network", network,
                "--network-alias", "rc-redis", "redis:7-alpine",
                "redis-server", "--appendonly", "yes",
            ]
        )
        wait_for(postgres, ["pg_isready", "-U", "weft", "-d", "weft"], args.timeout)
        wait_for(redis, ["redis-cli", "ping"], args.timeout)
        print(probe(args.image, network, dsn, "init", sentinel), end="")
        run(["docker", "restart", postgres])
        wait_for(postgres, ["pg_isready", "-U", "weft", "-d", "weft"], args.timeout)
        print(probe(args.image, network, dsn, "verify", sentinel), end="")
        print("RC Docker smoke passed")
        return 0
    finally:
        for container in resources:
            subprocess.run(["docker", "rm", "-f", container], capture_output=True)
        subprocess.run(["docker", "network", "rm", network], capture_output=True)


if __name__ == "__main__":
    raise SystemExit(main())
