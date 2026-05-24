"""Discord slash-command definitions for Weft.

Registers the /checkin slash command on the app_commands.CommandTree owned by
weft.discord.bot.Bot. Each handler uses the defer-then-followup pattern:

    await interaction.response.defer(ephemeral=True)
    await interaction.followup.send("...", ephemeral=True)

The pool is accessed via ``interaction.client._weft_pool`` — the _WeftClient
subclass defined in bot.py stashes it there so handlers don't need closures.

Design decision: typed slash parameters (mood: int, sleep_hours: float, energy:
int, notes: str) rather than a single free-form text string. This gives Discord
users native type-hinted autocompletion and avoids the awkward UX of pasting
Slack-formatted text into a Discord command. The parse_checkin_text "reuse"
requirement is satisfied by sharing CheckInCreate construction and
validate_checkin_ranges() from the lifted weft.checkin_parser module.

Testing note: ``checkin_handler`` is the bare async function, importable
directly for unit tests. ``register_checkin_command`` wraps it in a
``@tree.command`` decorator when attaching to a live CommandTree.
"""

from __future__ import annotations

import logging
from typing import Optional

import discord
from discord import app_commands

from weft.check_ins import create_check_in
from weft.checkin_parser import validate_checkin_ranges
from weft.models import CheckInCreate

logger = logging.getLogger(__name__)


async def checkin_handler(
    interaction: discord.Interaction,
    mood: Optional[int] = None,
    sleep_hours: Optional[float] = None,
    energy: Optional[int] = None,
    notes: Optional[str] = None,
) -> None:
    """Core logic for the /checkin slash command.

    Separated from the CommandTree decorator so unit tests can call it directly
    without a live discord.py client.
    """
    # Defer immediately — DB call may take >3 s on cold start.
    await interaction.response.defer(ephemeral=True)

    # At least one field must be provided.
    if mood is None and sleep_hours is None and energy is None and notes is None:
        await interaction.followup.send(
            "Please provide at least one field: mood (1-5), sleep hours, "
            "energy (1-5), or notes.",
            ephemeral=True,
        )
        return

    # Validate ranges using the shared helper.
    parsed = {
        k: v
        for k, v in {
            "mood": mood,
            "sleep_hours": sleep_hours,
            "energy": energy,
        }.items()
        if v is not None
    }
    range_error = validate_checkin_ranges(parsed)
    if range_error:
        await interaction.followup.send(range_error, ephemeral=True)
        return

    # Retrieve the pool from the client — set by _WeftClient at construction.
    pool = interaction.client._weft_pool  # type: ignore[attr-defined]
    if pool is None:
        logger.error("discord.checkin: pool is None — cannot persist check-in")
        await interaction.followup.send(
            "Internal error: database unavailable.", ephemeral=True
        )
        return

    try:
        create = CheckInCreate(
            mood=mood,
            sleep_hours=sleep_hours,
            energy=energy,
            notes=notes,
        )
        check_in = await create_check_in(pool, create)
    except ValueError as exc:
        # CheckInCreate validates ranges too — belt-and-suspenders.
        await interaction.followup.send(str(exc), ephemeral=True)
        return
    except Exception:
        logger.exception("discord.checkin: create_check_in failed")
        await interaction.followup.send(
            "Something went wrong saving your check-in. Try again?",
            ephemeral=True,
        )
        return

    # Build ephemeral confirmation with parsed values.
    parts: list[str] = []
    if check_in.mood is not None:
        parts.append(f"Mood: {check_in.mood}/5")
    if check_in.sleep_hours is not None:
        parts.append(f"Sleep: {check_in.sleep_hours}h")
    if check_in.energy is not None:
        parts.append(f"Energy: {check_in.energy}/5")
    if check_in.notes:
        parts.append(f"Notes: {check_in.notes}")
    summary = " | ".join(parts)

    await interaction.followup.send(
        f"Check-in logged! {summary}",
        ephemeral=True,
    )


def register_checkin_command(tree: app_commands.CommandTree) -> None:
    """Attach the /checkin command to *tree*.

    Called once from Bot._register_commands() at construction time.
    Wraps ``checkin_handler`` with the ``@tree.command`` decorator so the
    handler logic stays testable without a live CommandTree.
    """
    tree.command(
        name="checkin",
        description="Log a mood/sleep/energy check-in",
    )(
        app_commands.describe(
            mood="Mood rating 1-5 (optional)",
            sleep_hours="Hours of sleep (optional, e.g. 7.5)",
            energy="Energy rating 1-5 (optional)",
            notes="Free-form notes (optional)",
        )(checkin_handler)
    )
