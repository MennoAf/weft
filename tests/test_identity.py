"""Tests for install identity (ed25519 keypair)."""

from __future__ import annotations

import pytest
from cryptography.exceptions import InvalidSignature

from weft.identity import get_install_id, get_install_identity


@pytest.mark.asyncio
async def test_first_call_generates_and_persists(pool):
    identity = await get_install_identity(pool)
    assert identity.public_key_b64
    assert identity.private_key_b64
    assert identity.install_id == identity.public_key_b64


@pytest.mark.asyncio
async def test_second_call_returns_same_identity(pool):
    a = await get_install_identity(pool)
    b = await get_install_identity(pool)
    assert a.public_key_b64 == b.public_key_b64
    assert a.private_key_b64 == b.private_key_b64


@pytest.mark.asyncio
async def test_keypair_signs_and_verifies(pool):
    identity = await get_install_identity(pool)
    message = b"workspace_acme:read:expires=2026-12-31"
    signature = identity.private_key().sign(message)
    # Round-trip verification proves the stored bytes really are a usable
    # keypair, not just opaque base64.
    identity.public_key().verify(signature, message)
    with pytest.raises(InvalidSignature):
        identity.public_key().verify(signature, b"different message")


@pytest.mark.asyncio
async def test_get_install_id_returns_pubkey(pool):
    identity = await get_install_identity(pool)
    install_id = await get_install_id(pool)
    assert install_id == identity.public_key_b64
