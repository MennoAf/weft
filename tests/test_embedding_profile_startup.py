"""Startup embedding-profile guard tests.

Covers the production outage class where the deployed image could not honor
embedding provider settings: the unknown-provider error must stay loud, the
configured provider must be recorded as a real profile (moving the active
pointer off the v75 ``legacy`` placeholder exactly once), and booting without
an explicit provider while the placeholder is still active must warn.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from weft.db.reembed import ReembedProfile
from weft.embeddings import get_provider
from weft.mcp.server import (
    _ensure_embedding_profile,
    _unconfigured_embedding_provider_warning,
)


def _provider(
    provider_name: str = "openai",
    model: str = "text-embedding-3-small",
    dimensions: int = 768,
):
    return SimpleNamespace(
        provider_name=provider_name, model_name=model, dimensions=dimensions
    )


class _FakePool:
    """Serves the persisted active pointer and records every write."""

    def __init__(self, active_profile_id: str | None = "legacy"):
        self.active_profile_id = active_profile_id
        self.writes: list[tuple[str, tuple]] = []

    async def fetchval(self, query, *args):
        assert "embedding_profile_state" in query
        return self.active_profile_id in (None, "legacy")

    async def execute(self, query, *args):
        normalized = " ".join(query.split())
        self.writes.append((normalized, args))
        if normalized.startswith("UPDATE embedding_profile_state"):
            if self.active_profile_id in (None, "legacy"):
                self.active_profile_id = args[0]
                return "UPDATE 1"
            return "UPDATE 0"
        return "INSERT 0 1"


def test_get_provider_unknown_provider_stays_loud():
    # Regression pin: the deployed image crashed here at startup when the
    # openai extra was missing. The error must stay loud and name the fix.
    with pytest.raises(ValueError, match="unavailable or unknown") as excinfo:
        get_provider("does-not-exist")
    assert "openai" in str(excinfo.value)


def test_profile_identity_is_deterministic_and_config_sensitive():
    first = ReembedProfile.from_provider(_provider())
    second = ReembedProfile.from_provider(_provider())
    other_dims = ReembedProfile.from_provider(_provider(dimensions=1536))
    assert first.profile_id == second.profile_id
    assert first.profile_id.startswith("emb-")
    assert first.profile_id != other_dims.profile_id


@pytest.mark.asyncio
async def test_seed_creates_real_profile_and_flips_off_legacy():
    pool = _FakePool(active_profile_id="legacy")
    embedding = _provider()
    expected = ReembedProfile.from_provider(embedding)

    was_legacy = await _ensure_embedding_profile(pool, embedding)

    assert was_legacy is True
    inserts = [
        (query, args)
        for query, args in pool.writes
        if query.startswith("INSERT INTO embedding_profiles")
    ]
    assert len(inserts) == 1
    assert inserts[0][1][0] == expected.profile_id
    assert inserts[0][1][1] == "openai"
    assert pool.active_profile_id == expected.profile_id
    activations = [
        query for query, _ in pool.writes if "SET state = 'active'" in query
    ]
    assert len(activations) == 1
    # The 'legacy' placeholder row is never modified: every profile-targeting
    # write addresses the new profile id only.
    assert all(args[0] != "legacy" for _, args in pool.writes)


@pytest.mark.asyncio
async def test_seed_is_idempotent():
    pool = _FakePool(active_profile_id="legacy")
    embedding = _provider()

    assert await _ensure_embedding_profile(pool, embedding) is True
    first_activations = [
        query for query, _ in pool.writes if "SET state = 'active'" in query
    ]
    assert len(first_activations) == 1

    assert await _ensure_embedding_profile(pool, embedding) is False
    assert [
        query for query, _ in pool.writes if "SET state = 'active'" in query
    ] == first_activations
    assert pool.active_profile_id == ReembedProfile.from_provider(
        embedding
    ).profile_id


@pytest.mark.asyncio
async def test_seed_handles_null_active_pointer():
    pool = _FakePool(active_profile_id=None)
    embedding = _provider()

    assert await _ensure_embedding_profile(pool, embedding) is True
    assert pool.active_profile_id == ReembedProfile.from_provider(
        embedding
    ).profile_id


def test_warning_fires_for_legacy_active_with_default_provider():
    message = _unconfigured_embedding_provider_warning(
        provider_env_set=False,
        config_provider="fastembed",
        legacy_active=True,
    )
    assert message is not None
    assert "not explicitly configured" in message
    assert "WEFT_EMBEDDING_PROVIDER" in message
    assert "MODEL/DIMENSIONS" in message


def test_warning_silent_when_provider_env_is_explicit():
    assert _unconfigured_embedding_provider_warning(
        provider_env_set=True,
        config_provider="fastembed",
        legacy_active=True,
    ) is None


def test_warning_silent_when_active_profile_is_real():
    assert _unconfigured_embedding_provider_warning(
        provider_env_set=False,
        config_provider="fastembed",
        legacy_active=False,
    ) is None


def test_warning_silent_for_nondefault_provider_without_env():
    assert _unconfigured_embedding_provider_warning(
        provider_env_set=False,
        config_provider="openai",
        legacy_active=True,
    ) is None
