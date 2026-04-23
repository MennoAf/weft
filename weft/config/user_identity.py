"""User identity for Weft.

Resolves the canonical user_id that tags every row this installation writes.
Precedence, highest priority first:

1. ``WEFT_USER_ID`` env var — explicit override, not persisted. Useful for
   admin scripts and one-shot operations against prod.
2. ``~/.weft/user_id.json`` with an explicit ``user_id`` field — set via
   ``weft identity set <id>`` or ``set_user_id()``. This is how a local
   installation declares its canonical identity (e.g. the JWT ``sub`` used
   by the hosted Weft server) so local and hosted data share one owner.
3. Random UUID fallback — generated and persisted on first call. Preserves
   the original zero-config behavior for fresh installs.

The file format stays stable: ``{"user_id": "..."}``. A newly-persisted
random UUID and a user-set identity are indistinguishable on disk —
``describe_user_id()`` reports the source chain so callers can tell them
apart at runtime.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Literal

_ENV_VAR = "WEFT_USER_ID"


def _config_path() -> Path:
    return Path.home() / ".weft" / "user_id.json"


def _read_config() -> dict:
    path = _config_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _write_config(data: dict) -> None:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def get_user_id() -> str:
    """Get the canonical user ID for this installation.

    Resolves via precedence: env var → config-file explicit → random UUID
    fallback (generated once, persisted). Idempotent across calls.
    """
    env = os.environ.get(_ENV_VAR)
    if env:
        return env

    cfg = _read_config()
    if cfg.get("user_id"):
        return cfg["user_id"]

    # Fallback: generate, persist, return.
    new_id = uuid.uuid4().hex
    _write_config({"user_id": new_id})
    return new_id


def set_user_id(user_id: str) -> None:
    """Persist an explicit canonical user_id to ``~/.weft/user_id.json``.

    Overwrites any existing value. The env var still wins at read time if
    set — callers who rely on this should ensure ``WEFT_USER_ID`` is unset.
    """
    if not user_id or not isinstance(user_id, str):
        raise ValueError(f"user_id must be a non-empty string, got {user_id!r}")
    _write_config({"user_id": user_id})


_Source = Literal["env", "config", "generated", "unset"]


def describe_user_id() -> dict:
    """Report the current user_id and how it was resolved.

    Returns a dict ``{"user_id": str | None, "source": _Source,
    "config_path": str, "env_var": str}`` — does NOT generate a new UUID.
    Safe to call for diagnostics without side effects.
    """
    path = _config_path()
    env = os.environ.get(_ENV_VAR)
    if env:
        return {
            "user_id": env,
            "source": "env",
            "config_path": str(path),
            "env_var": _ENV_VAR,
        }

    cfg = _read_config()
    if cfg.get("user_id"):
        # Config file with explicit user_id — may be user-set or a previously
        # generated fallback. We don't track which (stable file format).
        return {
            "user_id": cfg["user_id"],
            "source": "config",
            "config_path": str(path),
            "env_var": _ENV_VAR,
        }

    return {
        "user_id": None,
        "source": "unset",
        "config_path": str(path),
        "env_var": _ENV_VAR,
    }
