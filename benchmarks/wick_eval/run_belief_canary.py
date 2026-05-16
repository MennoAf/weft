"""Belief-view detector canary harness.

Loads a fixture of hand-typed EpisodeTurn entries, runs each through the
belief detector, and measures over-extraction rate (claims emitted that are
NOT in the expected_claims set).

Usage:
  uv run python benchmarks/wick_eval/run_belief_canary.py --fixture <path>

Output:
  One JSON line to stdout with keys:
    - overall_over_extraction_rate (float)
    - total_turns (int)
    - total_emitted_claims (int)
    - total_extra_claims (int)
    - total_missed_claims (int)
    - false_positive_turns (list[str])
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
from dotenv import load_dotenv

from weft.models import EpisodeTurn, TurnRole
from weft.views.belief_detector import detect_belief_updates

logger = logging.getLogger(__name__)


def _load_fixture(fixture_path: Path) -> list[dict[str, Any]]:
    """Load a canary fixture JSON file."""
    with fixture_path.open(encoding="utf-8") as f:
        return json.load(f)


async def _run_canary(fixture: list[dict[str, Any]]) -> dict[str, Any]:
    """Process each fixture entry and compute metrics.

    Treats [] and [single ClaimUpdate(confidence=0.0, attribute=None)] as
    "no claim emitted".
    """
    total_turns = 0
    total_emitted_claims = 0
    total_extra_claims = 0
    total_missed_claims = 0
    false_positive_turns: list[str] = []

    for entry in fixture:
        turn_id = entry.get("turn_id", "unknown")
        episode_id = entry.get("episode_id", "unknown")
        turn_index = entry.get("turn_index", 0)
        role_str = entry.get("role", "").lower()
        content = entry.get("content", "")
        occurred_at_str = entry.get("occurred_at", datetime.now(UTC).isoformat().replace("+00:00", "Z"))
        expected_claims = set(entry.get("expected_claims", []))

        # Validate role
        valid_roles = {"user", "assistant", "tool", "system"}
        if role_str not in valid_roles:
            logger.warning(
                "canary.invalid_role: turn_id=%s role=%s (skipping turn)",
                turn_id,
                role_str,
            )
            continue

        # Parse ISO-8601 timestamp
        try:
            if occurred_at_str.endswith("Z"):
                occurred_at_str = occurred_at_str[:-1] + "+00:00"
            occurred_at = datetime.fromisoformat(occurred_at_str)
        except ValueError:
            logger.warning(
                "canary.invalid_timestamp: turn_id=%s timestamp=%s (using now)",
                turn_id,
                occurred_at_str,
            )
            occurred_at = datetime.now(UTC)

        # Construct EpisodeTurn
        try:
            turn = EpisodeTurn(
                id=turn_id,
                episode_id=episode_id,
                turn_index=turn_index,
                role=TurnRole(role_str),
                content=content,
                occurred_at=occurred_at,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "canary.turn_construction_error: turn_id=%s error=%s",
                turn_id,
                exc,
            )
            continue

        # Run detector
        try:
            claims = await detect_belief_updates(turn)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "canary.detector_error: turn_id=%s error=%s",
                turn_id,
                exc,
            )
            continue

        # Extract attributes from emitted claims, treating abstention records
        # (confidence=0.0, attribute=None) as "no claim emitted"
        emitted_attributes: set[str] = set()
        for claim in claims:
            if claim.attribute is not None and claim.confidence > 0.0:
                emitted_attributes.add(claim.attribute)

        # Compute per-turn metrics
        over_extraction = emitted_attributes - expected_claims
        missed = expected_claims - emitted_attributes

        total_turns += 1
        total_emitted_claims += len(emitted_attributes)
        total_extra_claims += len(over_extraction)
        total_missed_claims += len(missed)

        if over_extraction:
            false_positive_turns.append(turn_id)

        logger.debug(
            "canary.turn_processed: turn_id=%s emitted=%r expected=%r "
            "over=%r missed=%r",
            turn_id,
            emitted_attributes,
            expected_claims,
            over_extraction,
            missed,
        )

    # Compute overall over-extraction rate
    overall_over_extraction_rate = (
        total_extra_claims / max(1, total_emitted_claims)
        if total_emitted_claims > 0
        else 0.0
    )

    return {
        "overall_over_extraction_rate": round(overall_over_extraction_rate, 4),
        "total_turns": total_turns,
        "total_emitted_claims": total_emitted_claims,
        "total_extra_claims": total_extra_claims,
        "total_missed_claims": total_missed_claims,
        "false_positive_turns": false_positive_turns,
    }


@click.command()
@click.option(
    "--fixture",
    "fixture_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to the canary fixture JSON file.",
)
@click.option(
    "--log-level",
    default="WARNING",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    help="Logging level.",
)
def cli(fixture_path: Path | None, log_level: str) -> None:
    """Run belief-detector canary on a fixture."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load .env from standard locations
    load_dotenv()
    load_dotenv(Path.home() / ".weft" / ".env")

    # Resolve fixture path
    if fixture_path is None:
        fixture_path = Path(__file__).parent / "canary_belief_detector.json"

    if not fixture_path.exists():
        click.echo(f"error: fixture not found: {fixture_path}", err=True)
        sys.exit(1)

    # Check for ANTHROPIC_API_KEY
    if not os.environ.get("ANTHROPIC_API_KEY"):
        click.echo(
            "error: ANTHROPIC_API_KEY is not set — required for the belief detector",
            err=True,
        )
        sys.exit(1)

    # Load fixture
    try:
        fixture = _load_fixture(fixture_path)
    except (json.JSONDecodeError, FileNotFoundError) as exc:
        click.echo(f"error: failed to load fixture: {exc}", err=True)
        sys.exit(1)

    # Run canary
    try:
        metrics = asyncio.run(_run_canary(fixture))
    except Exception as exc:  # noqa: BLE001
        click.echo(f"error: canary run failed: {exc}", err=True)
        sys.exit(1)

    # Output exactly one JSON line
    click.echo(json.dumps(metrics))


if __name__ == "__main__":
    cli()
