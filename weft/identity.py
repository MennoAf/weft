"""Install identity — ed25519 keypair that addresses this Weft install.

This is the foundational federation primitive: every Weft install has a
long-lived identity it can sign with. Workspace capability tokens (when
those ship) will be signed by the workspace owner's install key and
verified by the recipient install. For v1 we just generate, persist, and
expose the public key as the install ID — no signing yet.

Threat model: ``weft_metadata`` is service-role only. Anyone with read
access to that table already has god-mode on the install, so the private
key is stored in plaintext. Encrypting it under a key that lives in the
same DB would be theatre. If you want hardware-backed keys later, swap
the storage backend; the interface stays the same.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass

import asyncpg
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

logger = logging.getLogger(__name__)

_METADATA_KEY = "install_identity"


@dataclass(frozen=True)
class InstallIdentity:
    """Public-facing handle for the install's signing keypair."""

    public_key_b64: str
    private_key_b64: str

    @property
    def install_id(self) -> str:
        """Stable install identifier — the b64 pubkey itself is the ID.

        Federation can derive shorter fingerprints (e.g. first 8 chars of
        a SHA-256 of the raw key) without changing storage.
        """
        return self.public_key_b64

    def public_key(self) -> Ed25519PublicKey:
        raw = base64.b64decode(self.public_key_b64)
        return Ed25519PublicKey.from_public_bytes(raw)

    def private_key(self) -> Ed25519PrivateKey:
        raw = base64.b64decode(self.private_key_b64)
        return Ed25519PrivateKey.from_private_bytes(raw)


def _generate() -> InstallIdentity:
    sk = Ed25519PrivateKey.generate()
    pk = sk.public_key()
    sk_bytes = sk.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pk_bytes = pk.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return InstallIdentity(
        public_key_b64=base64.b64encode(pk_bytes).decode("ascii"),
        private_key_b64=base64.b64encode(sk_bytes).decode("ascii"),
    )


async def get_install_identity(pool: asyncpg.Pool) -> InstallIdentity:
    """Return the install's keypair, generating + persisting on first call.

    Idempotent: concurrent first calls will both generate, but only one
    write wins (ON CONFLICT DO NOTHING) and the next read returns the
    persisted value.
    """
    row = await pool.fetchrow(
        "SELECT value FROM weft_metadata WHERE key = $1", _METADATA_KEY
    )
    if row is not None:
        payload = row["value"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return InstallIdentity(
            public_key_b64=payload["public_key_b64"],
            private_key_b64=payload["private_key_b64"],
        )

    identity = _generate()
    payload = {
        "public_key_b64": identity.public_key_b64,
        "private_key_b64": identity.private_key_b64,
    }
    await pool.execute(
        """
        INSERT INTO weft_metadata (key, value, updated_at)
        VALUES ($1, $2::jsonb, now())
        ON CONFLICT (key) DO NOTHING
        """,
        _METADATA_KEY,
        json.dumps(payload),
    )

    # Re-read in case another concurrent call won the race.
    row = await pool.fetchrow(
        "SELECT value FROM weft_metadata WHERE key = $1", _METADATA_KEY
    )
    if row is None:
        # Should be impossible after an INSERT ... ON CONFLICT DO NOTHING
        # against a key we just wrote, but log loud rather than panic.
        logger.error("install_identity disappeared after write")
        return identity
    payload = row["value"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    return InstallIdentity(
        public_key_b64=payload["public_key_b64"],
        private_key_b64=payload["private_key_b64"],
    )


async def get_install_id(pool: asyncpg.Pool) -> str:
    """Convenience: just the public key (the install's address)."""
    identity = await get_install_identity(pool)
    return identity.install_id
