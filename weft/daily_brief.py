"""Daily brief assembly — morning digest, grouped Personal vs Code.

Assembles data from multiple sources into a structured brief with both
markdown (for CLI/MCP) and Slack Block Kit (for scheduled delivery) formats.
Each data source is queried independently — a failure in one never blocks others.

Section layout (see SECTION_GROUPS):

    👤 Personal
        📅 Today's Calendar
        💤 Check-in Trends
        🎂 Birthdays Today (stub until entities have birth_date)

    💻 Code
        🔥 Top Active Projects   (commits + handoffs in window, with next-step)
        💰 Daily Spend           (today + 7d-vs-prior-7d trend)
        📋 Review Queue
        🔄 Recent Handoffs
        🔔 Alerts Due Today
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import asyncpg

from weft.config import DailyBriefConfig
from weft.cost_tracking import get_spend_trend
from weft.git_activity import RepoActivity, scan_repos
from weft.loom_query import LoomQueryError, get_ready_tasks, loom_tables_exist
from weft.retrieval_modes import sources_for_mode

logger = logging.getLogger(__name__)

BRIEF_MAX_ITEMS_PER_SECTION = 10
_SLACK_TEXT_LIMIT = 2800  # leave headroom under Slack's 3000-char limit


@dataclass
class BriefResult:
    markdown: str = ""
    slack_blocks: list[dict] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# --- Section names (insertion order = display order within group) ---
SECTION_META = {
    "calendar": ("📅", "Today's Calendar"),
    "checkin_trends": ("💤", "Check-in Trends"),
    "birthdays": ("🎂", "Birthdays Today"),
    "active_projects": ("🔥", "Top Active Projects"),
    "daily_spend": ("💰", "Daily Spend"),
    "review_queue": ("📋", "Review Queue"),
    "handoffs": ("🔄", "Recent Handoffs"),
    "alerts": ("🔔", "Alerts Due Today"),
}

# Group ordering renders top-to-bottom.
SECTION_GROUPS: list[tuple[str, str, list[str]]] = [
    ("👤", "Personal", ["calendar", "checkin_trends", "birthdays"]),
    ("💻", "Code", ["active_projects", "daily_spend", "review_queue", "handoffs", "alerts"]),
]


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


# --- Handoff next-step extractor ---

_NEXT_STEPS_PATTERNS = [
    re.compile(r"\*\*Next Steps?:?\*\*\s*(.*?)(?=\n\s*\*\*[A-Z][^*]*?:\*\*|\Z)", re.DOTALL | re.IGNORECASE),
    re.compile(r"^##+\s*Next Steps?\s*\n(.*?)(?=\n##+\s|\Z)", re.DOTALL | re.IGNORECASE | re.MULTILINE),
]


def extract_next_step(handoff_content: str, max_chars: int = 220) -> str | None:
    """Pull the 'Next Steps' section out of a handoff memory body.

    Returns a truncated single-line summary, or None if no section is found.
    """
    if not handoff_content:
        return None
    for pat in _NEXT_STEPS_PATTERNS:
        m = pat.search(handoff_content)
        if not m:
            continue
        body = m.group(1).strip()
        if not body:
            continue
        # Collapse whitespace and grab the first sentence-ish chunk.
        single = re.sub(r"\s+", " ", body)
        if len(single) > max_chars:
            single = single[:max_chars].rsplit(" ", 1)[0] + "…"
        return single
    return None


# --- Data source queries ---


async def _query_review_queue(pool: asyncpg.Pool, as_of: datetime) -> list[str]:
    """Memories with review_after <= today. Face-mode filter: excludes codebase ingest noise."""
    try:
        from weft.db.connection import get_db

        face_sources = sources_for_mode("face") or []
        rows = await get_db(pool).fetch(
            """
            SELECT id, type, content, review_after
            FROM memories
            WHERE status = 'active'
              AND review_after IS NOT NULL
              AND review_after <= $1
              AND source = ANY($3::text[])
            ORDER BY review_after ASC
            LIMIT $2
            """,
            as_of,
            BRIEF_MAX_ITEMS_PER_SECTION,
            face_sources,
        )
        items = []
        for r in rows:
            content = r["content"].replace("\n", " ")
            truncated = len(content) > 140
            preview = content[:140].rsplit(" ", 1)[0] if truncated else content
            ellipsis = "…" if truncated else ""
            items.append(f"[{r['type']}] {preview}{ellipsis} (review due {r['review_after'].strftime('%Y-%m-%d')})")
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


async def _query_birthdays(pool: asyncpg.Pool, as_of: datetime) -> list[str]:
    """Birthdays today.

    Stub. Will read entity metadata once entities carry a `birth_date` field
    (planned: add `birth_date DATE` column to entities and a query that
    matches month+day to today). Returns [] until then so the section renders
    cleanly without lying about coverage.
    """
    return []


async def _query_active_projects(
    pool: asyncpg.Pool,
    *,
    project_repos: dict[str, str],
    window_hours: int,
    top_n: int,
    as_of: datetime,
) -> list[str]:
    """Top-N projects by activity (commits + handoffs in window) with next-step.

    Activity score = commit_count + 2 * handoff_count. Handoffs weight more
    because they're explicit work-session boundaries; commits are noisy
    (one session can produce 10).

    For each surfaced project, attempts to extract "Next Steps" from the
    most recent handoff. Falls back to the latest commit subject if no
    handoff exists in the window.
    """
    try:
        from weft.db.connection import get_db

        as_of_utc = as_of.astimezone(timezone.utc)
        cutoff = as_of_utc - timedelta(hours=window_hours)

        # Handoff counts and most recent body per project_id
        handoff_rows = await get_db(pool).fetch(
            """
            SELECT project_id,
                   COUNT(*) AS handoff_count,
                   (
                     SELECT content FROM memories m2
                     WHERE m2.project_id = m1.project_id
                       AND m2.type = 'handoff' AND m2.status = 'active'
                       AND m2.created_at >= $1
                     ORDER BY m2.created_at DESC LIMIT 1
                   ) AS latest_handoff
            FROM memories m1
            WHERE status = 'active'
              AND type = 'handoff'
              AND project_id IS NOT NULL
              AND created_at >= $1
            GROUP BY project_id
            """,
            cutoff,
        )

        handoff_by_project: dict[str, dict] = {
            r["project_id"]: {
                "count": r["handoff_count"],
                "content": r["latest_handoff"] or "",
            }
            for r in handoff_rows
        }

        # Git commits per configured repo
        repo_activities: list[RepoActivity] = await scan_repos(
            project_repos, window_hours=window_hours, as_of=as_of_utc,
        )
        commits_by_project = {a.project_id: a for a in repo_activities}

        # Union of project_ids with any signal
        all_pids = set(handoff_by_project.keys()) | {
            a.project_id for a in repo_activities if a.commit_count > 0
        }

        scored: list[tuple[int, str, dict | None, RepoActivity | None]] = []
        for pid in all_pids:
            handoff_info = handoff_by_project.get(pid)
            commit_info = commits_by_project.get(pid)
            handoff_count = handoff_info["count"] if handoff_info else 0
            commit_count = commit_info.commit_count if commit_info else 0
            score = commit_count + 2 * handoff_count
            if score == 0:
                continue
            scored.append((score, pid, handoff_info, commit_info))

        scored.sort(key=lambda t: (-t[0], t[1]))
        top = scored[:top_n]

        items: list[str] = []
        for score, pid, handoff_info, commit_info in top:
            commit_count = commit_info.commit_count if commit_info else 0
            handoff_count = handoff_info["count"] if handoff_info else 0
            parts = []
            if commit_count:
                parts.append(f"{commit_count} commit{'s' if commit_count != 1 else ''}")
            if handoff_count:
                parts.append(f"{handoff_count} handoff{'s' if handoff_count != 1 else ''}")
            activity_summary = ", ".join(parts) or "no activity"

            next_step: str | None = None
            if handoff_info and handoff_info.get("content"):
                next_step = extract_next_step(handoff_info["content"])
            if not next_step and commit_info and commit_info.last_commit_subject:
                next_step = f"latest commit — {commit_info.last_commit_subject}"
            if not next_step:
                next_step = "no recent handoff"

            items.append(f"**{pid}** ({activity_summary}) — next: {next_step}")
        return items
    except Exception:
        logger.exception("daily_brief.active_projects_error")
        return []


async def _query_daily_spend(pool: asyncpg.Pool, as_of: datetime) -> list[str]:
    """Today's spend + 7d-vs-prior-7d trend.

    Returns 1-3 lines depending on data availability. Uses USD with
    4-decimal precision because routine days run sub-dollar.
    """
    try:
        trend = await get_spend_trend(pool, as_of=as_of)
        if trend.trend == "insufficient data":
            return [f"Today: ${trend.today_usd:.2f} (no prior data)"]
        return [
            f"Today: ${trend.today_usd:.2f}",
            (
                f"7d total: ${trend.recent_total_usd:.2f} "
                f"(daily avg ${trend.daily_avg_recent:.2f}) "
                f"vs prior 7d ${trend.prior_total_usd:.2f} ({trend.trend})"
            ),
        ]
    except Exception:
        logger.exception("daily_brief.daily_spend_error")
        return []


async def _query_loom_tasks(pool: asyncpg.Pool) -> list[str]:
    """DEPRECATED — kept as a fallback; no longer rendered by default.

    Replaced by _query_active_projects, which surfaces per-project next-steps
    instead of a flat ready queue. Retained for callers that import the
    symbol and for any future debugging override.
    """
    try:
        if not await loom_tables_exist(pool):
            logger.info("daily_brief.loom_not_found — skipping task section")
            return []

        tasks = await get_ready_tasks(pool, limit=BRIEF_MAX_ITEMS_PER_SECTION)
        items = []
        for t in tasks:
            title = t.get("title", "untitled")
            priority = t.get("priority", "")
            items.append(f"[{priority}] {title}")
        return items
    except LoomQueryError as e:
        logger.warning("daily_brief.loom_error: %s", e)
        return []
    except Exception:
        logger.exception("daily_brief.loom_error")
        return []


async def _query_calendar_events(
    calendar_id: str, tz: ZoneInfo, as_of: datetime,
) -> list[str]:
    """Today's Google Calendar events — all-day first, then timed chronologically."""
    from weft.google_calendar import build_service

    local_now = as_of.astimezone(tz)
    day_start = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)

    def _fetch() -> list[dict]:
        service = build_service()
        result = service.events().list(
            calendarId=calendar_id,
            timeMin=day_start.isoformat(),
            timeMax=day_end.isoformat(),
            timeZone=str(tz),
            singleEvents=True,
            orderBy="startTime",
            maxResults=BRIEF_MAX_ITEMS_PER_SECTION,
        ).execute()
        return result.get("items", [])

    events = await asyncio.to_thread(_fetch)

    all_day: list[str] = []
    timed: list[str] = []
    for event in events:
        if event.get("status") == "cancelled":
            continue
        summary = event.get("summary", "(No title)")
        start = event.get("start", {})
        if "date" in start:
            all_day.append(f"All day: {summary}")
        elif "dateTime" in start:
            dt = datetime.fromisoformat(start["dateTime"]).astimezone(tz)
            timed.append(f"{dt.strftime('%H:%M')} {summary}")

    return all_day + timed


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
    """Format sections dict as readable markdown, grouped Personal vs Code."""
    lines = [
        f"## 🌅 Daily Brief",
        f"_{generated_at.strftime('%A, %B %d %Y at %H:%M %Z')}_",
        "",
    ]

    for group_emoji, group_title, section_keys in SECTION_GROUPS:
        lines.append(f"### {group_emoji} {group_title}")
        lines.append("")
        for key in section_keys:
            emoji, title = SECTION_META[key]
            items = sections.get(key, [])
            lines.append(f"#### {emoji} {title}")
            if items:
                for item in items:
                    lines.append(f"- {item}")
            else:
                lines.append("_Nothing here — all clear!_")
            lines.append("")

    lines.append("---")
    return "\n".join(lines)


def format_slack_blocks(sections: dict[str, list[str]], generated_at: datetime) -> list[dict]:
    """Format sections dict as Slack Block Kit blocks, grouped Personal vs Code."""
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
    for group_emoji, group_title, section_keys in SECTION_GROUPS:
        group_has_items = any(sections.get(k) for k in section_keys)
        if not group_has_items:
            continue

        any_items = True
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*{group_emoji} {group_title}*"},
        })

        for key in section_keys:
            emoji, title = SECTION_META[key]
            items = sections.get(key, [])
            if not items:
                continue

            bullet_lines = [f"• {item}" for item in items]
            text = f"*{emoji} {title}*\n" + "\n".join(bullet_lines)

            if len(text) > _SLACK_TEXT_LIMIT:
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
        brief_config: daily brief configuration (timezone, project repos, etc.).
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

    calendar = await _safe(
        _query_calendar_events(brief_config.calendar_id, tz, target_date), "calendar"
    )
    review_queue = await _safe(_query_review_queue(pool, target_date), "review_queue")
    handoffs = await _safe(_query_handoffs(pool, target_date), "handoffs")
    checkin_trends = await _safe(_query_checkin_trends(pool), "checkin_trends")
    birthdays = await _safe(_query_birthdays(pool, target_date), "birthdays")
    active_projects = await _safe(
        _query_active_projects(
            pool,
            project_repos=brief_config.project_repos,
            window_hours=brief_config.active_projects_window_hours,
            top_n=brief_config.active_projects_top_n,
            as_of=target_date,
        ),
        "active_projects",
    )
    daily_spend = await _safe(_query_daily_spend(pool, target_date), "daily_spend")
    alerts = await _safe(_query_alerts(pool, target_date), "alerts")

    sections: dict[str, list[str]] = {
        "calendar": calendar,
        "checkin_trends": checkin_trends,
        "birthdays": birthdays,
        "active_projects": active_projects,
        "daily_spend": daily_spend,
        "review_queue": review_queue,
        "handoffs": handoffs,
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
