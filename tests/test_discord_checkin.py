"""Unit tests for the Discord /checkin slash command handler.

All discord.py objects are mocked — no network connections, no Discord gateway.
Tests cover:
- Happy path: valid inputs produce ephemeral confirmation with parsed values.
- Range validation errors: mood/energy out of 1-5, sleep_hours out of 0-24.
- Empty params: no fields provided → ephemeral error.
- Ephemeral flag: every response path uses ephemeral=True.
- User-identity / RLS binding: matched owner → RLS bound; mismatch → ephemeral
  error; missing config → configuration error (not a crash).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Canonical snowflake and Weft UUID used across owner-mapping tests.
OWNER_DISCORD_ID = "123456789012345678"
OWNER_WEFT_USER_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
OTHER_DISCORD_ID = "999999999999999999"


# ── Helpers ─────────────────────────────────────────────────────────


def _make_interaction(pool=None, discord_user_id: str = OWNER_DISCORD_ID) -> MagicMock:
    """Build a minimal mock discord.Interaction.

    ``discord_user_id`` sets interaction.user.id, which the handler reads for
    owner-mapping resolution. Defaults to the canonical owner snowflake so
    pre-existing tests remain green without change.
    """
    interaction = MagicMock()
    interaction.response = MagicMock()
    interaction.response.defer = AsyncMock()
    interaction.followup = MagicMock()
    interaction.followup.send = AsyncMock()
    # Pool is accessed via interaction.client._weft_pool
    interaction.client = MagicMock()
    interaction.client._weft_pool = pool
    # user.id is the Discord snowflake resolved for RLS binding.
    interaction.user = MagicMock()
    interaction.user.id = discord_user_id
    return interaction


def _owner_env(monkeypatch) -> None:
    """Set the two owner-mapping env vars to canonical test values."""
    monkeypatch.setenv("WEFT_DISCORD_OWNER_DISCORD_ID", OWNER_DISCORD_ID)
    monkeypatch.setenv("WEFT_DISCORD_OWNER_WEFT_USER_ID", OWNER_WEFT_USER_ID)
    # Suppress WEFT_TESTING so load_dotenv is skipped but env vars are used.
    monkeypatch.setenv("WEFT_TESTING", "1")


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
    async def test_full_check_in_sends_ephemeral_confirmation(self, monkeypatch):
        _owner_env(monkeypatch)
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
    async def test_mood_only(self, monkeypatch):
        _owner_env(monkeypatch)
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
    async def test_notes_only(self, monkeypatch):
        _owner_env(monkeypatch)
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
    async def test_all_none_returns_error(self, monkeypatch):
        _owner_env(monkeypatch)
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
    async def test_mood_out_of_range_low(self, monkeypatch):
        _owner_env(monkeypatch)
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
    async def test_mood_out_of_range_high(self, monkeypatch):
        _owner_env(monkeypatch)
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
    async def test_energy_out_of_range(self, monkeypatch):
        _owner_env(monkeypatch)
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
    async def test_sleep_hours_out_of_range(self, monkeypatch):
        _owner_env(monkeypatch)
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
    async def test_valid_boundary_values_pass(self, monkeypatch):
        """Boundary values 1, 5 for mood/energy and 0, 24 for sleep must pass."""
        _owner_env(monkeypatch)
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
    async def test_defer_is_ephemeral(self, monkeypatch):
        _owner_env(monkeypatch)
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


# ── Owner-mapping / RLS-binding tests ────────────────────────────────


class TestDiscordCheckinOwnerMapping:
    """Tests for Discord user_id → Weft user_id resolution and RLS binding."""

    @pytest.mark.asyncio
    async def test_matched_owner_binds_rls_user_id(self, monkeypatch):
        """Happy path: matching Discord snowflake → Weft UUID injected into contextvar."""
        _owner_env(monkeypatch)
        from weft.auth import current_user_id
        from weft.discord.commands import checkin_handler as checkin
        from weft.models import CheckIn

        pool = MagicMock()
        fake_check_in = CheckIn(mood=3)
        interaction = _make_interaction(pool=pool, discord_user_id=OWNER_DISCORD_ID)

        captured_user_id: list[str | None] = []

        async def _capture_create(pool, create):
            captured_user_id.append(current_user_id.get())
            return fake_check_in

        with patch("weft.discord.commands.create_check_in", side_effect=_capture_create):
            await checkin(interaction, mood=3, sleep_hours=None, energy=None, notes=None)

        # The contextvar must have held the Weft UUID during the DB call.
        assert captured_user_id == [OWNER_WEFT_USER_ID]
        # Confirmation message sent.
        call_kwargs = interaction.followup.send.call_args
        text = call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs.get("content", "")
        assert "Check-in logged" in text

    @pytest.mark.asyncio
    async def test_mismatched_discord_id_returns_ephemeral_error(self, monkeypatch):
        """User whose Discord snowflake is NOT the configured owner gets rejected."""
        _owner_env(monkeypatch)
        from weft.discord.commands import checkin_handler as checkin

        interaction = _make_interaction(pool=MagicMock(), discord_user_id=OTHER_DISCORD_ID)

        with patch(
            "weft.discord.commands.create_check_in", new_callable=AsyncMock
        ) as mock_create:
            await checkin(interaction, mood=3, sleep_hours=None, energy=None, notes=None)

        # create_check_in must NOT have been called — rejected before DB.
        mock_create.assert_not_awaited()
        interaction.followup.send.assert_awaited_once()
        call_kwargs = interaction.followup.send.call_args
        text = call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs.get("content", "")
        assert "not configured for your account" in text.lower()
        assert call_kwargs.kwargs.get("ephemeral") is True

    @pytest.mark.asyncio
    async def test_missing_config_returns_configuration_error_not_crash(self, monkeypatch):
        """When env vars are absent the handler sends a config error, not a traceback."""
        # Ensure both owner vars are unset.
        monkeypatch.delenv("WEFT_DISCORD_OWNER_DISCORD_ID", raising=False)
        monkeypatch.delenv("WEFT_DISCORD_OWNER_WEFT_USER_ID", raising=False)
        monkeypatch.setenv("WEFT_TESTING", "1")
        from weft.discord.commands import checkin_handler as checkin

        interaction = _make_interaction(pool=MagicMock(), discord_user_id=OWNER_DISCORD_ID)

        with patch(
            "weft.discord.commands.create_check_in", new_callable=AsyncMock
        ) as mock_create:
            # Must not raise — must send an ephemeral error instead.
            await checkin(interaction, mood=3, sleep_hours=None, energy=None, notes=None)

        mock_create.assert_not_awaited()
        interaction.followup.send.assert_awaited_once()
        call_kwargs = interaction.followup.send.call_args
        text = call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs.get("content", "")
        assert "not configured" in text.lower()
        assert call_kwargs.kwargs.get("ephemeral") is True

    @pytest.mark.asyncio
    async def test_rls_contextvar_reset_after_handler(self, monkeypatch):
        """The current_user_id contextvar is cleaned up after the handler returns."""
        _owner_env(monkeypatch)
        from weft.auth import current_user_id
        from weft.discord.commands import checkin_handler as checkin
        from weft.models import CheckIn

        pool = MagicMock()
        fake_check_in = CheckIn(mood=2)
        interaction = _make_interaction(pool=pool, discord_user_id=OWNER_DISCORD_ID)

        # Pre-condition: contextvar is unset (None).
        assert current_user_id.get() is None

        with patch(
            "weft.discord.commands.create_check_in", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = fake_check_in
            await checkin(interaction, mood=2, sleep_hours=None, energy=None, notes=None)

        # Post-condition: contextvar is restored to None.
        assert current_user_id.get() is None
