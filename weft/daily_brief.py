"""Daily brief assembly — morning digest from Weft memories, check-ins, alerts, and Loom tasks.

Assembles data from multiple sources into a structured brief with both
markdown (for CLI/MCP) and Slack Block Kit (for scheduled delivery) formats.
Each data source is queried independently — a failure in one never blocks others.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import asyncpg

from weft.config import DailyBriefConfig

logger = logging.getLogger(__name__)

BRIEF_MAX_ITEMS_PER_SECTION = 10
_SLACK_TEXT_LIMIT = 2800  # leave headroom under Slack's 3000-char limit


@dataclass
class BriefResult:
    markdown: str = ""
    slack_blocks: list[dict] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# --- Section names (insertion order = display order) ---
SECTION_META = {
    "review_queue": ("📋", "Review Queue"),
    "handoffs": ("🔄", "Recent Handoffs"),
    "checkin_trends": ("💤", "Check-in Trends"),
    "ready_tasks": ("🚀", "Ready Tasks"),
    "alerts": ("🔔", "Alerts Due Today"),
}


# --- Trend computation ---


def compute_trend(recent: list[float], prior: list[float]) -> str:
    """Compare two windows of values and return a trend string.

    Returns 'improving ↑', 'declining ↓', 'stable →', or 'insufficient data'.
    """
    if len(recent) < 2 or len(prior) < 2:
        return "insufficient data"
    recent_avg = sum(recent) / len(recent)
    prior_avg = sum(prior) / len(prior)
    if recent_avg > prior_avg + 0.3:
        return "improving ↑"
    if recent_avg < prior_avg - 0.3:
        return "declining ↓"
    return "stable →"


# --- Data source queries ---


async def _query_review_queue(pool: asyncpg.Pool, as_of: datetime) -> list[str]:
    """Memories with review_after <= today."""
    try:
        from weft.db.connection import get_db

        rows = await get_db(pool).fetch(
            """
            SELECT id, type, content, review_after
            FROM memories
            WHERE status = 'active'
              AND review_after IS NOT NULL
              AND review_after <= $1
            ORDER BY review_after ASC
            LIMIT $2
            """,
            as_of,
            BRIEF_MAX_ITEMS_PER_SECTION,
        )
        items = []
        for r in rows:
            preview = r["content"][:80].replace("\n", " ")
            items.append(f"[{r['type']}] {preview}… (review due {r['review_after'].strftime('%Y-%m-%d')})")
        return items
    except Exception:
        logger.exception("daily_brief.review_queue_error")
        return []


async def _query_handoffs(pool: asyncpg.Pool, as_of: datetime) -> list[str]:
    """Recent handoff memories (last 24h)."""
    try:
        from weft.db.connection import get_db

        cutoff = as_of - timedelta(hours=24)
        rows = await get_db(pool).fetch(
            """
            SELECT id, content, created_at
            FROM memories
            WHERE status = 'active'
              AND type = 'handoff'
              AND created_at >= $1
            ORDER BY created_at DESC
            LIMIT 3
            """,
            cutoff,
        )
        items = []
        for r in rows:
            preview = r["content"][:120].replace("\n", " ")
            items.append(preview)
        return items
    except Exception:
        logger.exception("daily_brief.handoffs_error")
        return []


async def _query_checkin_trends(pool: asyncpg.Pool) -> list[str]:
    """Check-in trends over last 7 days."""
    try:
        from weft.check_ins import list_check_ins

        check_ins = await list_check_ins(pool, limit=30)

        # Split into recent (0-3 days) and prior (4-7 days)
        now = datetime.now(timezone.utc)
        recent_cutoff = now - timedelta(days=3)
        prior_cutoff = now - timedelta(days=7)

        recent = [c for c in check_ins if c.logged_at >= recent_cutoff]
        prior = [c for c in check_ins if prior_cutoff <= c.logged_at < recent_cutoff]

        items = []
        for label, attr in [("Mood", "mood"), ("Sleep", "sleep_hours"), ("Energy", "energy")]:
            recent_vals = [getattr(c, attr) for c in recent if getattr(c, attr) is not None]
            prior_vals = [getattr(c, attr) for c in prior if getattr(c, attr) is not None]
            trend = compute_trend(recent_vals, prior_vals)
            if recent_vals:
                avg = sum(recent_vals) / len(recent_vals)
                unit = "h" if attr == "sleep_hours" else "/5"
                items.append(f"{label}: {avg:.1f}{unit} ({trend})")
            else:
                items.append(f"{label}: no recent data")

        return items
    except Exception:
        logger.exception("daily_brief.checkin_trends_error")
        return []


async def _query_loom_tasks() -> list[str]:
    """Ready/blocked tasks from Loom via CLI."""
    try:
        result = subprocess.run(
            ["uv", "run", "python", "-m", "loom", "ready", "--json"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            logger.warning("daily_brief.loom_error: exit %d", result.returncode)
            return []

        tasks = json.loads(result.stdout) if result.stdout.strip() else []
        items = []
        for t in tasks[:BRIEF_MAX_ITEMS_PER_SECTION]:
            title = t.get("title", "untitled")
            priority = t.get("priority", "")
            items.append(f"[{priority}] {title}")
        return items
    except FileNotFoundError:
        logger.info("daily_brief.loom_not_found — skipping task section")
        return []
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        logger.warning("daily_brief.loom_error: %s", e)
        return []
    except Exception:
        logger.exception("daily_brief.loom_error")
        return []


async def _query_alerts(pool: asyncpg.Pool, as_of: datetime) -> list[str]:
    """Pending alerts due today."""
    try:
        from weft.alerts import list_alerts
        from weft.models import AlertStatus

        alerts = await list_alerts(pool, status=AlertStatus.pending, limit=BRIEF_MAX_ITEMS_PER_SECTION)
        end_of_day = as_of.replace(hour=23, minute=59, second=59)
        items = []
        for a in alerts:
            if a.trigger_at <= end_of_day:
                time_str = a.trigger_at.strftime("%H:%M")
                items.append(f"[{a.alert_type.value}] {a.title} (due {time_str})")
        return items
    except Exception:
        logger.exception("daily_brief.alerts_error")
        return []


# --- Formatters ---


def format_markdown(sections: dict[str, list[str]], generated_at: datetime) -> str:
    """Format sections dict as readable markdown."""
    lines = [
        f"## 🌅 Daily Brief",
        f"_{generated_at.strftime('%A, %B %d %Y at %H:%M %Z')}_",
        "",
    ]

    for key, (emoji, title) in SECTION_META.items():
        items = sections.get(key, [])
        lines.append(f"### {emoji} {title}")
        if items:
            for item in items:
                lines.append(f"- {item}")
        else:
            lines.append("_Nothing here — all clear!_")
        lines.append("")

    lines.append("---")
    return "\n".join(lines)


def format_slack_blocks(sections: dict[str, list[str]], generated_at: datetime) -> list[dict]:
    """Format sections dict as Slack Block Kit blocks."""
    blocks: list[dict] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"🌅 Daily Brief — {generated_at.strftime('%A, %B %d')}",
                "emoji": True,
            },
        }
    ]

    any_items = False
    for key, (emoji, title) in SECTION_META.items():
        items = sections.get(key, [])
        if not items:
            continue

        any_items = True
        # Build bullet text, respecting Slack's 3000-char limit
        bullet_lines = [f"• {item}" for item in items]
        text = f"*{emoji} {title}*\n" + "\n".join(bullet_lines)

        if len(text) > _SLACK_TEXT_LIMIT:
            # Truncate and indicate overflow
            shown = 0
            truncated = f"*{emoji} {title}*\n"
            for line in bullet_lines:
                if len(truncated) + len(line) + 30 > _SLACK_TEXT_LIMIT:
                    break
                truncated += line + "\n"
                shown += 1
            remaining = len(items) - shown
            text = truncated + f"_… and {remaining} more_"

        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": text}})
        blocks.append({"type": "divider"})

    if not any_items:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "All clear — nothing needs attention today 🎉"},
        })

    # Cap at 50 blocks (Slack limit)
    return blocks[:50]


# --- Main assembly ---


async def assemble_daily_brief(
    pool: asyncpg.Pool,
    brief_config: DailyBriefConfig | None = None,
    *,
    target_date: datetime | None = None,
) -> BriefResult:
    """Assemble the daily brief from all data sources.

    Args:
        pool: database connection pool.
        brief_config: daily brief configuration (timezone, etc.).
        target_date: reference datetime. Defaults to now in configured timezone.
    """
    if brief_config is None:
        brief_config = DailyBriefConfig()

    tz = ZoneInfo(brief_config.timezone)

    if target_date is None:
        target_date = datetime.now(timezone.utc)

    local_now = target_date.astimezone(tz)

    # Query all sources — each wrapped so one failure can't block others
    async def _safe(coro, label: str) -> list[str]:
        try:
            return await coro
        except Exception:
            logger.exception(f"daily_brief.{label}_error")
            return []

    review_queue = await _safe(_query_review_queue(pool, target_date), "review_queue")
    handoffs = await _safe(_query_handoffs(pool, target_date), "handoffs")
    checkin_trends = await _safe(_query_checkin_trends(pool), "checkin_trends")
    ready_tasks = await _safe(_query_loom_tasks(), "ready_tasks")
    alerts = await _safe(_query_alerts(pool, target_date), "alerts")

    sections: dict[str, list[str]] = {
        "review_queue": review_queue,
        "handoffs": handoffs,
        "checkin_trends": checkin_trends,
        "ready_tasks": ready_tasks,
        "alerts": alerts,
    }

    generated_at = local_now
    markdown = format_markdown(sections, generated_at)
    slack_blocks = format_slack_blocks(sections, generated_at)

    return BriefResult(
        markdown=markdown,
        slack_blocks=slack_blocks,
        generated_at=generated_at,
    )
