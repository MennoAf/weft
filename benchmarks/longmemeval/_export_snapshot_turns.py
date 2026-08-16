import asyncio
from pathlib import Path

from benchmarks.longmemeval.adapter import _make_pool
from benchmarks.longmemeval.snapshot import (
    EPISODE_TURN_SNAPSHOT_COLUMNS,
    _copy_out_filtered,
)


async def main() -> None:
    pool = await _make_pool()
    try:
        await _copy_out_filtered(
            pool,
            Path("benchmarks/longmemeval/snapshots/baseline_v1_local/episode_turns.csv.gz"),
            "SELECT " + ", ".join(
                f"t.{column}" for column in EPISODE_TURN_SNAPSHOT_COLUMNS
            ) + " FROM episode_turns t "
            "JOIN episodes e ON t.episode_id=e.id "
            "WHERE e.project_id LIKE 'lme_%'",
        )
    finally:
        await pool.close()
    Path("benchmarks/longmemeval/runs/export_turns_baseline_v1.done").write_text("ok")


if __name__ == "__main__":
    asyncio.run(main())
