"""Unit tests for the Discord /checkin slash command handler.

All discord.py objects are mocked — no network connections, no Discord gateway.
Tests cover:
- Happy path: valid inputs produce ephemeral confirmation with parsed values.
- Range validation errors: mood/energy out of 1-5, sleep_hours out of 0-24.
- Empty params: no fields provided → ephemeral error.
- Ephemeral flag: every response path uses ephemeral=True.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Helpers ─────────────────────────────────────────────────────────


def _make_interaction(pool=None) -> MagicMock:
    """Build a minimal mock discord.Interaction."""
    interaction = MagicMock()
    interaction.response = MagicMock()
    interaction.response.defer = AsyncMock()
    interaction.followup = MagicMock()
    interaction.followup.send = AsyncMock()
    # Pool is accessed via interaction.client._weft_pool
    interaction.client = MagicMock()
    interaction.client._weft_pool = pool
    return interaction


def _make_pool_with_checkin(mood=None, sleep_hours=None, energy=None, notes=None):
    """Return a mock pool whose create_check_in returns a fake CheckIn."""
    from weft.models import CheckIn

    fake_check_in = CheckIn(
        mood=mood,
        sleep_hours=sleep_hours,
        energy=energy,
        notes=notes,
    )

    pool = MagicMock()
    # We patch create_check_in directly in tests, but keep pool available
    # so the command can retrieve it via interaction.client._weft_pool.
    return pool, fake_check_in


# ── Happy-path tests ─────────────────────────────────────────────────


class TestDiscordCheckinHappyPath:
    @pytest.mark.asyncio
    async def test_full_check_in_sends_ephemeral_confirmation(self):
        from weft.discord.commands import checkin_handler as checkin
        from weft.models import CheckIn

        pool = MagicMock()
        fake_check_in = CheckIn(mood=3, sleep_hours=7.0, energy=4, notes="feeling ok")
        interaction = _make_interaction(pool=pool)

        with patch(
            "weft.discord.commands.create_check_in", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = fake_check_in
            await checkin(interaction, mood=3, sleep_hours=7.0, energy=4, notes="feeling ok")

        interaction.response.defer.assert_awaited_once_with(ephemeral=True)
        interaction.followup.send.assert_awaited_once()
        call_kwargs = interaction.followup.send.call_args
        text = call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs.get("content", "")
        assert "Check-in logged" in text
        assert "Mood: 3/5" in text
        assert "Sleep: 7.0h" in text
        assert "Energy: 4/5" in text
        assert "feeling ok" in text
        # Must be ephemeral
        assert call_kwargs.kwargs.get("ephemeral") is True

    @pytest.mark.asyncio
    async def test_mood_only(self):
        from weft.discord.commands import checkin_handler as checkin
        from weft.models import CheckIn

        pool = MagicMock()
        fake_check_in = CheckIn(mood=5)
        interaction = _make_interaction(pool=pool)

        with patch(
            "weft.discord.commands.create_check_in", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = fake_check_in
            await checkin(interaction, mood=5, sleep_hours=None, energy=None, notes=None)

        call_kwargs = interaction.followup.send.call_args
        text = call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs.get("content", "")
        assert "Mood: 5/5" in text
        assert call_kwargs.kwargs.get("ephemeral") is True

    @pytest.mark.asyncio
    async def test_notes_only(self):
        from weft.discord.commands import checkin_handler as checkin
        from weft.models import CheckIn

        pool = MagicMock()
        fake_check_in = CheckIn(notes="just a note")
        interaction = _make_interaction(pool=pool)

        with patch(
            "weft.discord.commands.create_check_in", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = fake_check_in
            await checkin(
                interaction, mood=None, sleep_hours=None, energy=None, notes="just a note"
            )

        call_kwargs = interaction.followup.send.call_args
        text = call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs.get("content", "")
        assert "just a note" in text
        assert call_kwargs.kwargs.get("ephemeral") is True


# ── Empty-params tests ───────────────────────────────────────────────


class TestDiscordCheckinEmptyParams:
    @pytest.mark.asyncio
    async def test_all_none_returns_error(self):
        from weft.discord.commands import checkin_handler as checkin

        interaction = _make_interaction(pool=MagicMock())

        await checkin(
            interaction, mood=None, sleep_hours=None, energy=None, notes=None
        )

        interaction.response.defer.assert_awaited_once_with(ephemeral=True)
        interaction.followup.send.assert_awaited_once()
        call_kwargs = interaction.followup.send.call_args
        text = call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs.get("content", "")
        assert "at least one field" in text.lower() or "provide" in text.lower()
        assert call_kwargs.kwargs.get("ephemeral") is True


# ── Range-validation tests ────────────────────────────────────────────


class TestDiscordCheckinRangeValidation:
    @pytest.mark.asyncio
    async def test_mood_out_of_range_low(self):
        from weft.discord.commands import checkin_handler as checkin

        interaction = _make_interaction(pool=MagicMock())

        await checkin(
            interaction, mood=0, sleep_hours=None, energy=None, notes=None
        )

        call_kwargs = interaction.followup.send.call_args
        text = call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs.get("content", "")
        assert "1-5" in text or "Mood" in text
        assert call_kwargs.kwargs.get("ephemeral") is True

    @pytest.mark.asyncio
    async def test_mood_out_of_range_high(self):
        from weft.discord.commands import checkin_handler as checkin

        interaction = _make_interaction(pool=MagicMock())

        await checkin(
            interaction, mood=6, sleep_hours=None, energy=None, notes=None
        )

        call_kwargs = interaction.followup.send.call_args
        text = call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs.get("content", "")
        assert "1-5" in text or "Mood" in text
        assert call_kwargs.kwargs.get("ephemeral") is True

    @pytest.mark.asyncio
    async def test_energy_out_of_range(self):
        from weft.discord.commands import checkin_handler as checkin

        interaction = _make_interaction(pool=MagicMock())

        await checkin(
            interaction, mood=None, sleep_hours=None, energy=7, notes=None
        )

        call_kwargs = interaction.followup.send.call_args
        text = call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs.get("content", "")
        assert "1-5" in text or "Energy" in text
        assert call_kwargs.kwargs.get("ephemeral") is True

    @pytest.mark.asyncio
    async def test_sleep_hours_out_of_range(self):
        from weft.discord.commands import checkin_handler as checkin

        interaction = _make_interaction(pool=MagicMock())

        await checkin(
            interaction, mood=None, sleep_hours=25.0, energy=None, notes=None
        )

        call_kwargs = interaction.followup.send.call_args
        text = call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs.get("content", "")
        assert "0-24" in text or "Sleep" in text
        assert call_kwargs.kwargs.get("ephemeral") is True

    @pytest.mark.asyncio
    async def test_valid_boundary_values_pass(self):
        """Boundary values 1, 5 for mood/energy and 0, 24 for sleep must pass."""
        from weft.discord.commands import checkin_handler as checkin
        from weft.models import CheckIn

        pool = MagicMock()
        fake_check_in = CheckIn(mood=1, sleep_hours=0.0, energy=5)
        interaction = _make_interaction(pool=pool)

        with patch(
            "weft.discord.commands.create_check_in", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = fake_check_in
            await checkin(
                interaction, mood=1, sleep_hours=0.0, energy=5, notes=None
            )

        call_kwargs = interaction.followup.send.call_args
        text = call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs.get("content", "")
        assert "Check-in logged" in text


# ── Ephemeral-flag tests ──────────────────────────────────────────────


class TestDiscordCheckinEphemeralFlag:
    """Every response path must use ephemeral=True."""

    @pytest.mark.asyncio
    async def test_defer_is_ephemeral(self):
        from weft.discord.commands import checkin_handler as checkin
        from weft.models import CheckIn

        pool = MagicMock()
        fake_check_in = CheckIn(mood=3)
        interaction = _make_interaction(pool=pool)

        with patch(
            "weft.discord.commands.create_check_in", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = fake_check_in
            await checkin(
                interaction, mood=3, sleep_hours=None, energy=None, notes=None
            )

        # The very first thing the handler does is defer(ephemeral=True).
        interaction.response.defer.assert_awaited_once_with(ephemeral=True)
