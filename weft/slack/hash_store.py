"""Timestamp-based change detection for Slack message sync."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path.home() / ".weft" / "slack_sync_state.json"


class SlackSyncState:
    """Tracks per-channel sync cursors and per-message memory IDs."""

    def __init__(self, path: Path = DEFAULT_PATH):
        self.path = path
        self._data: dict = {}
        self._load()

    def _load(self):
        if not self.path.exists():
            self._data = {"channels": {}, "messages": {}}
            return
        try:
            raw = self.path.read_text(encoding="utf-8")
            self._data = json.loads(raw)
            self._data.setdefault("channels", {})
            self._data.setdefault("messages", {})
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Slack sync state corrupted, starting fresh: %s", exc)
            self._data = {"channels": {}, "messages": {}}

    def save(self):
        """Persist sync state to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, indent=2), encoding="utf-8"
        )

    def get_last_sync_ts(self, channel_id: str) -> str | None:
        """Get the oldest timestamp to resume from for a channel."""
        return self._data["channels"].get(channel_id, {}).get("last_sync_ts")

    def set_last_sync_ts(self, channel_id: str, ts: str):
        """Update the sync cursor for a channel."""
        ch = self._data["channels"].setdefault(channel_id, {})
        ch["last_sync_ts"] = ts
        ch["synced_at"] = datetime.now(timezone.utc).isoformat()

    def _msg_key(self, channel_id: str, message_ts: str) -> str:
        return f"{channel_id}:{message_ts}"

    def get_memory_ids(self, channel_id: str, message_ts: str) -> list[str]:
        """Get memory IDs for a previously synced message."""
        key = self._msg_key(channel_id, message_ts)
        entry = self._data["messages"].get(key)
        return entry.get("memory_ids", []) if entry else []

    def get_edited_ts(self, channel_id: str, message_ts: str) -> str | None:
        """Get the stored edited timestamp for a message."""
        key = self._msg_key(channel_id, message_ts)
        entry = self._data["messages"].get(key)
        return entry.get("edited_ts") if entry else None

    def update_message(
        self,
        channel_id: str,
        message_ts: str,
        memory_ids: list[str],
        edited_ts: str | None = None,
    ):
        """Record a synced message and its memory IDs."""
        key = self._msg_key(channel_id, message_ts)
        self._data["messages"][key] = {
            "memory_ids": memory_ids,
            "edited_ts": edited_ts,
            "synced_at": datetime.now(timezone.utc).isoformat(),
        }

    def remove_message(self, channel_id: str, message_ts: str):
        """Remove a message from sync state."""
        key = self._msg_key(channel_id, message_ts)
        self._data["messages"].pop(key, None)

    @property
    def message_count(self) -> int:
        return len(self._data.get("messages", {}))

    @property
    def channel_count(self) -> int:
        return len(self._data.get("channels", {}))

    def all_message_keys(self, channel_id: str | None = None) -> set[str]:
        """Get all tracked message keys, optionally filtered by channel."""
        keys = set(self._data.get("messages", {}).keys())
        if channel_id:
            prefix = f"{channel_id}:"
            keys = {k for k in keys if k.startswith(prefix)}
        return keys
