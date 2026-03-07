"""Transform Slack message data into memory content."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class SlackMessage:
    """A parsed Slack message with extracted metadata."""

    text: str
    user: str | None = None
    ts: str = ""
    thread_ts: str | None = None
    edited_ts: str | None = None
    reactions: list[dict] = field(default_factory=list)
    files: list[dict] = field(default_factory=list)
    reply_count: int = 0
    replies: list[SlackMessage] = field(default_factory=list)

    @property
    def is_thread_parent(self) -> bool:
        return self.reply_count > 0 or bool(self.replies)

    @property
    def is_thread_reply(self) -> bool:
        return self.thread_ts is not None and self.thread_ts != self.ts

    @property
    def timestamp(self) -> datetime:
        return datetime.fromtimestamp(float(self.ts), tz=timezone.utc)


# Patterns for extracting structure from Slack messages
_URL_RE = re.compile(r"<(https?://[^|>]+)(?:\|([^>]+))?>")
_USER_MENTION_RE = re.compile(r"<@(U[A-Z0-9]+)>")
_CHANNEL_MENTION_RE = re.compile(r"<#(C[A-Z0-9]+)\|([^>]+)>")
_EMOJI_RE = re.compile(r":([a-z0-9_+-]+):")


def parse_message(raw: dict) -> SlackMessage:
    """Parse a raw Slack API message dict into a SlackMessage."""
    edited = raw.get("edited", {})
    return SlackMessage(
        text=raw.get("text", ""),
        user=raw.get("user"),
        ts=raw.get("ts", ""),
        thread_ts=raw.get("thread_ts"),
        edited_ts=edited.get("ts") if edited else None,
        reactions=raw.get("reactions", []),
        files=[
            {"name": f.get("name", ""), "url": f.get("url_private", "")}
            for f in raw.get("files", [])
        ],
        reply_count=raw.get("reply_count", 0),
    )


def extract_urls(text: str) -> list[dict]:
    """Extract URLs and their display text from Slack-formatted text."""
    urls = []
    for match in _URL_RE.finditer(text):
        urls.append({"url": match.group(1), "label": match.group(2) or match.group(1)})
    return urls


def extract_mentions(text: str) -> list[str]:
    """Extract user IDs from @mentions."""
    return _USER_MENTION_RE.findall(text)


def clean_slack_text(text: str, user_names: dict[str, str] | None = None) -> str:
    """Convert Slack markup to readable plain text.

    - Replaces <url|label> with label (url)
    - Replaces <@U123> with @username if user_names provided
    - Replaces <#C123|channel> with #channel
    """
    if user_names is None:
        user_names = {}

    # URLs: <url|label> → label (url), <url> → url
    def _url_repl(m):
        url, label = m.group(1), m.group(2)
        if label:
            return f"{label} ({url})"
        return url

    result = _URL_RE.sub(_url_repl, text)

    # User mentions
    def _user_repl(m):
        uid = m.group(1)
        name = user_names.get(uid, uid)
        return f"@{name}"

    result = _USER_MENTION_RE.sub(_user_repl, result)

    # Channel mentions
    result = _CHANNEL_MENTION_RE.sub(r"#\2", result)

    return result


def build_memory_content(
    message: SlackMessage,
    channel_name: str,
    user_names: dict[str, str] | None = None,
) -> str:
    """Build a memory content string from a message (with optional thread replies)."""
    if user_names is None:
        user_names = {}

    parts: list[str] = []

    # Header
    author = user_names.get(message.user, message.user) if message.user else "unknown"
    date_str = message.timestamp.strftime("%Y-%m-%d %H:%M UTC")
    parts.append(f"# Slack: #{channel_name} — {date_str}")
    parts.append(f"From: @{author}")
    parts.append("")

    # Main message
    cleaned = clean_slack_text(message.text, user_names)
    parts.append(cleaned)

    # Thread replies
    if message.replies:
        parts.append("")
        parts.append("---")
        parts.append(f"**Thread ({len(message.replies)} replies):**")
        for reply in message.replies:
            reply_author = (
                user_names.get(reply.user, reply.user) if reply.user else "unknown"
            )
            reply_time = reply.timestamp.strftime("%H:%M")
            reply_text = clean_slack_text(reply.text, user_names)
            parts.append(f"\n**@{reply_author}** ({reply_time}):")
            parts.append(reply_text)

    # Reactions
    if message.reactions:
        reaction_strs = []
        for r in message.reactions:
            count = r.get("count", len(r.get("users", [])))
            reaction_strs.append(f":{r['name']}: ({count})")
        parts.append("")
        parts.append(f"Reactions: {' '.join(reaction_strs)}")

    # Files
    if message.files:
        parts.append("")
        parts.append("Attachments:")
        for f in message.files:
            if f["name"]:
                parts.append(f"- {f['name']}")

    # URLs
    urls = extract_urls(message.text)
    for reply in message.replies:
        urls.extend(extract_urls(reply.text))
    if urls:
        seen = set()
        unique_urls = []
        for u in urls:
            if u["url"] not in seen:
                seen.add(u["url"])
                unique_urls.append(u)
        parts.append("")
        parts.append("Links:")
        for u in unique_urls:
            parts.append(f"- {u['label']}: {u['url']}")

    return "\n".join(parts)
