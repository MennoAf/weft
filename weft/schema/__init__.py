"""Federated-future schema versioning.

Provides the upgrade-chain pattern from Wick INTERCHANGE: rows on disk may
be at any supported version; readers always upgrade to CURRENT_VERSION
before consuming; writers always write at CURRENT_VERSION. Old versions
get rewritten by a batch migration tool (off-peak) so they can be safely
dropped from SUPPORTED_VERSIONS.

See 2026-04-26-weft-federated-future-schema-v1.md for the design.
"""

from weft.schema.versioning import (
    CURRENT_VERSION,
    SUPPORTED_VERSIONS,
    SYSTEM_GLOBAL_USER_ID,
    upgrade_to_current,
)

__all__ = [
    "CURRENT_VERSION",
    "SUPPORTED_VERSIONS",
    "SYSTEM_GLOBAL_USER_ID",
    "upgrade_to_current",
]
