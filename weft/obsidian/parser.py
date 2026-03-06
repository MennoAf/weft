"""Obsidian markdown parser — frontmatter, wikilinks, tags, heading splitting."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import frontmatter
import yaml
from dateutil.parser import parse as parse_date


@dataclass
class ParsedNote:
    """Result of parsing an Obsidian markdown note."""

    body: str
    frontmatter: dict
    title: str | None = None
    tags: list[str] = field(default_factory=list)
    frontmatter_tags: list[str] = field(default_factory=list)
    inline_tags: list[str] = field(default_factory=list)
    wikilinks: list[dict] = field(default_factory=list)
    embeds: list[dict] = field(default_factory=list)
    callout_types: list[str] = field(default_factory=list)
    date: str | None = None
    aliases: list[str] = field(default_factory=list)


@dataclass
class Section:
    """A section of a note split by headings."""

    heading: str | None
    content: str
    index: int
    total: int


# Regex patterns
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
_EMBED_RE = re.compile(r"!\[\[([^\]]+)\]\]")
_INLINE_TAG_RE = re.compile(r"(?:^|\s)#([a-zA-Z][\w/-]*)", re.MULTILINE)
_CALLOUT_RE = re.compile(r"^>\s*\[!(\w+)\]", re.MULTILINE)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

_DATE_KEYS = ("date", "created", "updated", "modified")


def normalize_tag(tag: str) -> str:
    """Strip #, lowercase, replace spaces with hyphens."""
    return tag.lstrip("#").lower().replace(" ", "-")


def extract_date(fm: dict) -> str | None:
    """Extract and normalize a date from frontmatter."""
    for key in _DATE_KEYS:
        val = fm.get(key)
        if val is None:
            continue
        if isinstance(val, datetime):
            if val.tzinfo is None:
                val = val.replace(tzinfo=timezone.utc)
            return val.isoformat()
        if isinstance(val, str):
            try:
                dt = parse_date(val)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.isoformat()
            except (ValueError, OverflowError):
                continue
        # date objects (datetime.date)
        if hasattr(val, "isoformat"):
            return (
                datetime.combine(val, datetime.min.time(), tzinfo=timezone.utc)
                .isoformat()
            )
    return None


def extract_wikilinks(text: str) -> tuple[str, list[dict]]:
    """Extract [[wikilinks]], replace with display text in body."""
    links: list[dict] = []

    def _replace(m: re.Match) -> str:
        target = m.group(1).strip()
        alias = m.group(2).strip() if m.group(2) else None
        links.append({"target": target, "alias": alias})
        return alias or target

    cleaned = _WIKILINK_RE.sub(_replace, text)
    return cleaned, links


def extract_embeds(text: str) -> tuple[str, list[dict]]:
    """Extract ![[embeds]], remove from body."""
    embeds: list[dict] = []

    def _replace(m: re.Match) -> str:
        embeds.append({"target": m.group(1).strip()})
        return ""

    cleaned = _EMBED_RE.sub(_replace, text)
    return cleaned, embeds


def extract_inline_tags(text: str) -> list[str]:
    """Extract #tags from body text."""
    return sorted(set(normalize_tag(t) for t in _INLINE_TAG_RE.findall(text)))


def extract_callout_types(text: str) -> list[str]:
    """Extract callout types from > [!TYPE] blocks."""
    return sorted(set(m.lower() for m in _CALLOUT_RE.findall(text)))


def parse_note(content: str) -> ParsedNote:
    """Parse an Obsidian markdown note into structured data."""
    try:
        post = frontmatter.loads(content)
        fm = dict(post.metadata)
        body = post.content
    except yaml.YAMLError:
        fm = {}
        if content.startswith("---"):
            end = content.find("---", 3)
            body = content[end + 3 :].strip() if end != -1 else content
        else:
            body = content

    title = fm.get("title")
    aliases = fm.get("aliases", []) or []
    if isinstance(aliases, str):
        aliases = [aliases]

    fm_tags_raw = fm.get("tags", []) or []
    if isinstance(fm_tags_raw, str):
        fm_tags_raw = [fm_tags_raw]
    fm_tags = [normalize_tag(t) for t in fm_tags_raw]

    date = extract_date(fm)

    # Extract embeds before wikilinks since ![[x]] contains [[x]]
    body, embeds = extract_embeds(body)
    body, wikilinks = extract_wikilinks(body)
    inline_tags = extract_inline_tags(body)
    callout_types = extract_callout_types(body)

    all_tags = sorted(set(fm_tags + inline_tags))

    return ParsedNote(
        body=body.strip(),
        frontmatter=fm,
        title=title,
        tags=all_tags,
        frontmatter_tags=fm_tags,
        inline_tags=inline_tags,
        wikilinks=wikilinks,
        embeds=embeds,
        callout_types=callout_types,
        date=date,
        aliases=aliases,
    )


def split_by_headings(
    body: str, threshold_bytes: int = 8192
) -> list[Section]:
    """Split body by headings if it exceeds the byte threshold."""
    if len(body.encode("utf-8")) <= threshold_bytes:
        return [Section(heading=None, content=body, index=0, total=1)]

    matches = list(_HEADING_RE.finditer(body))
    if not matches:
        return _split_by_paragraphs(body, threshold_bytes)

    sections: list[tuple[str | None, str]] = []

    pre_heading = body[: matches[0].start()].strip()
    if pre_heading:
        sections.append((None, pre_heading))

    for i, m in enumerate(matches):
        heading = m.group(2).strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections.append((heading, body[start:end].strip()))

    total = len(sections)
    return [
        Section(heading=h, content=c, index=i, total=total)
        for i, (h, c) in enumerate(sections)
    ]


def _split_by_paragraphs(body: str, threshold_bytes: int) -> list[Section]:
    """Split by double newline, grouping paragraphs up to threshold."""
    paragraphs = re.split(r"\n\n+", body)
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0

    for para in paragraphs:
        para_size = len(para.encode("utf-8"))
        if current and current_size + para_size > threshold_bytes:
            chunks.append("\n\n".join(current))
            current = [para]
            current_size = para_size
        else:
            current.append(para)
            current_size += para_size

    if current:
        chunks.append("\n\n".join(current))

    total = len(chunks)
    return [
        Section(heading=None, content=c, index=i, total=total)
        for i, c in enumerate(chunks)
    ]
