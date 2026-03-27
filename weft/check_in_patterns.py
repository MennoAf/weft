"""Check-in pattern detection — pure Python analysis over CheckIn lists.

No new DB tables. Functions accept lists of CheckIn objects (fetched via
list_check_ins) and return structured dicts. analyze_all() orchestrates
everything into a combined report.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import asyncpg

from weft.models import CheckIn

logger = logging.getLogger(__name__)


@dataclass
class CheckInAlertConfig:
    """Configuration for check-in pattern alert thresholds."""

    low_mood_streak: int = 3  # consecutive days at mood <= 2
    low_sleep_hours: float = 6.0  # average sleep below this
    low_sleep_days: int = 5  # over this many days
    dedup_hours: int = 24  # suppress duplicate alerts within this window


# Day names for readability
_DAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

# Minimum data points for meaningful analysis
_MIN_CORRELATION_POINTS = 5
_MIN_TREND_POINTS = 5
_MIN_STREAK_LENGTH = 3


def day_of_week_stats(check_ins: list[CheckIn]) -> dict[str, Any]:
    """Average mood, energy, and sleep grouped by day of week.

    Returns a dict with day names as keys, each mapping to
    {avg_mood, avg_energy, avg_sleep, count}. Also includes
    best_day and worst_day (by mood, if available).
    """
    buckets: dict[int, list[CheckIn]] = defaultdict(list)
    for ci in check_ins:
        buckets[ci.logged_at.weekday()].append(ci)

    days: dict[str, dict[str, Any]] = {}
    for dow in range(7):
        items = buckets.get(dow, [])
        name = _DAY_NAMES[dow]
        if not items:
            days[name] = {"avg_mood": None, "avg_energy": None, "avg_sleep": None, "count": 0}
            continue
        moods = [ci.mood for ci in items if ci.mood is not None]
        energies = [ci.energy for ci in items if ci.energy is not None]
        sleeps = [ci.sleep_hours for ci in items if ci.sleep_hours is not None]
        days[name] = {
            "avg_mood": round(sum(moods) / len(moods), 2) if moods else None,
            "avg_energy": round(sum(energies) / len(energies), 2) if energies else None,
            "avg_sleep": round(sum(sleeps) / len(sleeps), 2) if sleeps else None,
            "count": len(items),
        }

    # Best/worst by mood
    scored = [(name, d["avg_mood"]) for name, d in days.items() if d["avg_mood"] is not None]
    best_day = max(scored, key=lambda x: x[1])[0] if scored else None
    worst_day = min(scored, key=lambda x: x[1])[0] if scored else None

    return {"days": days, "best_day": best_day, "worst_day": worst_day}


def sleep_energy_correlation(check_ins: list[CheckIn]) -> dict[str, Any]:
    """Pearson correlation between sleep_hours and energy.

    Pure Python implementation — no scipy needed. Returns correlation
    coefficient r, sample size n, and a human-readable interpretation.
    """
    pairs = [
        (ci.sleep_hours, ci.energy)
        for ci in check_ins
        if ci.sleep_hours is not None and ci.energy is not None
    ]
    n = len(pairs)
    if n < _MIN_CORRELATION_POINTS:
        return {
            "r": None,
            "n": n,
            "interpretation": f"insufficient data (need {_MIN_CORRELATION_POINTS}+, have {n})",
        }

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    r = _pearson(xs, ys)

    if r is None:
        interpretation = "no variance in data"
    elif abs(r) < 0.3:
        interpretation = "weak"
    elif abs(r) < 0.7:
        interpretation = "moderate"
    else:
        interpretation = "strong"
    if r is not None and r < 0:
        interpretation += " negative"

    return {"r": round(r, 3) if r is not None else None, "n": n, "interpretation": interpretation}


def sleep_mood_correlation(check_ins: list[CheckIn]) -> dict[str, Any]:
    """Pearson correlation between sleep_hours and mood."""
    pairs = [
        (ci.sleep_hours, ci.mood)
        for ci in check_ins
        if ci.sleep_hours is not None and ci.mood is not None
    ]
    n = len(pairs)
    if n < _MIN_CORRELATION_POINTS:
        return {
            "r": None,
            "n": n,
            "interpretation": f"insufficient data (need {_MIN_CORRELATION_POINTS}+, have {n})",
        }

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    r = _pearson(xs, ys)

    if r is None:
        interpretation = "no variance in data"
    elif abs(r) < 0.3:
        interpretation = "weak"
    elif abs(r) < 0.7:
        interpretation = "moderate"
    else:
        interpretation = "strong"
    if r is not None and r < 0:
        interpretation += " negative"

    return {"r": round(r, 3) if r is not None else None, "n": n, "interpretation": interpretation}


def detect_streaks(check_ins: list[CheckIn]) -> dict[str, Any]:
    """Detect consecutive-day streaks in check-in logging.

    Also detects mood streaks (consecutive days at mood >= 4 = "good",
    consecutive days at mood <= 2 = "low"). Returns current logging
    streak and notable mood streaks.
    """
    if not check_ins:
        return {"logging_streak": 0, "good_mood_streaks": [], "low_mood_streaks": []}

    # Sort by logged_at ascending
    sorted_cis = sorted(check_ins, key=lambda ci: ci.logged_at)

    # Logging streak: consecutive calendar days ending at the most recent
    dates = sorted({ci.logged_at.date() for ci in sorted_cis})
    logging_streak = 1
    for i in range(len(dates) - 1, 0, -1):
        if (dates[i] - dates[i - 1]).days == 1:
            logging_streak += 1
        else:
            break

    # Mood streaks
    good_streaks = _find_mood_streaks(sorted_cis, lambda m: m >= 4)
    low_streaks = _find_mood_streaks(sorted_cis, lambda m: m <= 2)

    return {
        "logging_streak": logging_streak,
        "good_mood_streaks": good_streaks,
        "low_mood_streaks": low_streaks,
    }


def rolling_averages(
    check_ins: list[CheckIn],
    *,
    window_days: int = 7,
    days: int = 30,
) -> dict[str, Any]:
    """Rolling averages for mood, energy, and sleep over the last N days.

    Returns a list of {date, avg_mood, avg_energy, avg_sleep} dicts,
    one per day that has data within the window.
    """
    if not check_ins:
        return {"window_days": window_days, "period_days": days, "series": []}

    sorted_cis = sorted(check_ins, key=lambda ci: ci.logged_at)
    cutoff = sorted_cis[-1].logged_at - timedelta(days=days)
    recent = [ci for ci in sorted_cis if ci.logged_at >= cutoff]

    if not recent:
        return {"window_days": window_days, "period_days": days, "series": []}

    # Build daily buckets
    start_date = recent[0].logged_at.date()
    end_date = recent[-1].logged_at.date()
    series = []

    current = start_date
    while current <= end_date:
        window_start = current - timedelta(days=window_days - 1)
        window_cis = [
            ci
            for ci in recent
            if window_start <= ci.logged_at.date() <= current
        ]
        if window_cis:
            moods = [ci.mood for ci in window_cis if ci.mood is not None]
            energies = [ci.energy for ci in window_cis if ci.energy is not None]
            sleeps = [ci.sleep_hours for ci in window_cis if ci.sleep_hours is not None]
            series.append(
                {
                    "date": current.isoformat(),
                    "avg_mood": round(sum(moods) / len(moods), 2) if moods else None,
                    "avg_energy": round(sum(energies) / len(energies), 2) if energies else None,
                    "avg_sleep": round(sum(sleeps) / len(sleeps), 2) if sleeps else None,
                    "n": len(window_cis),
                }
            )
        current += timedelta(days=1)

    return {"window_days": window_days, "period_days": days, "series": series}


def trend_direction(
    check_ins: list[CheckIn],
    *,
    days: int = 90,
) -> dict[str, Any]:
    """Detect whether mood, energy, and sleep are trending up, down, or flat.

    Uses simple linear regression (least squares) over the last N days.
    Returns slope, direction ("up"/"down"/"flat"), and data point count
    for each metric.
    """
    if not check_ins:
        return {"period_days": days, "mood": None, "energy": None, "sleep": None}

    sorted_cis = sorted(check_ins, key=lambda ci: ci.logged_at)
    cutoff = sorted_cis[-1].logged_at - timedelta(days=days)
    recent = [ci for ci in sorted_cis if ci.logged_at >= cutoff]

    base_date = recent[0].logged_at.date() if recent else None

    def _trend(values: list[tuple[float, float]]) -> dict[str, Any] | None:
        if len(values) < _MIN_TREND_POINTS:
            return None
        xs = [v[0] for v in values]
        ys = [v[1] for v in values]
        slope = _linear_slope(xs, ys)
        if slope is None:
            return {"slope": 0, "direction": "flat", "n": len(values)}
        # Threshold: slope per day. For 1-5 scales, +-0.02/day ≈ +-0.6/month
        if abs(slope) < 0.02:
            direction = "flat"
        elif slope > 0:
            direction = "up"
        else:
            direction = "down"
        return {"slope": round(slope, 4), "direction": direction, "n": len(values)}

    mood_pts = [
        ((ci.logged_at.date() - base_date).days, ci.mood)
        for ci in recent
        if ci.mood is not None and base_date is not None
    ]
    energy_pts = [
        ((ci.logged_at.date() - base_date).days, ci.energy)
        for ci in recent
        if ci.energy is not None and base_date is not None
    ]
    sleep_pts = [
        ((ci.logged_at.date() - base_date).days, ci.sleep_hours)
        for ci in recent
        if ci.sleep_hours is not None and base_date is not None
    ]

    return {
        "period_days": days,
        "mood": _trend(mood_pts),
        "energy": _trend(energy_pts),
        "sleep": _trend(sleep_pts),
    }


def analyze_all(
    check_ins: list[CheckIn],
    *,
    trend_days: int = 90,
    rolling_days: int = 30,
    rolling_window: int = 7,
) -> dict[str, Any]:
    """Run all pattern analyses and return a combined report."""
    return {
        "day_of_week": day_of_week_stats(check_ins),
        "sleep_energy_correlation": sleep_energy_correlation(check_ins),
        "sleep_mood_correlation": sleep_mood_correlation(check_ins),
        "streaks": detect_streaks(check_ins),
        "rolling_averages": rolling_averages(
            check_ins, window_days=rolling_window, days=rolling_days
        ),
        "trends": trend_direction(check_ins, days=trend_days),
        "total_check_ins": len(check_ins),
    }


# --- Alert evaluation ---


async def evaluate_check_in_alerts(
    pool: asyncpg.Pool,
    check_ins: list[CheckIn],
    *,
    config: CheckInAlertConfig | None = None,
) -> list[dict[str, Any]]:
    """Evaluate check-in patterns against thresholds and create alerts.

    Checks:
    1. Low mood streak >= 3 consecutive days (mood <= 2)
    2. Average sleep below 6h over last 5+ days
    3. Declining mood trend (7-day window)

    Dedup: skips alert creation if same alert_type was fired in last 24h.
    Returns list of created alert dicts (empty if no thresholds crossed or all deduped).
    """
    from weft.alerts import create_alert, list_alerts
    from weft.models import AlertCreate, AlertStatus, AlertType

    if not check_ins:
        return []

    cfg = config or CheckInAlertConfig()
    report = analyze_all(check_ins, trend_days=90, rolling_days=30)
    created: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc)

    # Fetch recent alerts for dedup
    recent_alerts = await list_alerts(pool, status=AlertStatus.pending, limit=100)
    fired_alerts = await list_alerts(pool, status=AlertStatus.fired, limit=100)
    all_recent = recent_alerts + fired_alerts
    dedup_cutoff = now - timedelta(hours=cfg.dedup_hours)
    recent_types: set[str] = set()
    for a in all_recent:
        if a.created_at >= dedup_cutoff:
            recent_types.add(a.alert_type.value)

    # 1. Low mood streak
    low_streaks = report["streaks"]["low_mood_streaks"]
    if low_streaks and AlertType.check_in_low_mood.value not in recent_types:
        longest = max(low_streaks, key=lambda s: s["length"])
        if longest["length"] >= cfg.low_mood_streak:
            alert = await create_alert(
                pool,
                AlertCreate(
                    alert_type=AlertType.check_in_low_mood,
                    title=f"Low mood streak: {longest['length']} consecutive days",
                    body=f"Mood has been at 2 or below for {longest['length']} days starting {longest['start']}.",
                    trigger_at=now,
                ),
            )
            created.append(alert.to_dict())

    # 2. Low average sleep
    sorted_cis = sorted(check_ins, key=lambda ci: ci.logged_at, reverse=True)
    recent_sleep = [
        ci.sleep_hours
        for ci in sorted_cis[:cfg.low_sleep_days * 2]  # look at recent entries
        if ci.sleep_hours is not None
        and ci.logged_at >= now - timedelta(days=cfg.low_sleep_days + 1)
    ]
    if (
        len(recent_sleep) >= cfg.low_sleep_days
        and AlertType.check_in_low_sleep.value not in recent_types
    ):
        avg_sleep = sum(recent_sleep) / len(recent_sleep)
        if avg_sleep < cfg.low_sleep_hours:
            alert = await create_alert(
                pool,
                AlertCreate(
                    alert_type=AlertType.check_in_low_sleep,
                    title=f"Low sleep average: {avg_sleep:.1f}h over last {len(recent_sleep)} entries",
                    body=f"Average sleep has been {avg_sleep:.1f}h (below {cfg.low_sleep_hours}h threshold).",
                    trigger_at=now,
                ),
            )
            created.append(alert.to_dict())

    # 3. Declining mood trend
    mood_trend = report["trends"].get("mood")
    if (
        mood_trend
        and mood_trend["direction"] == "down"
        and AlertType.check_in_declining_trend.value not in recent_types
    ):
        alert = await create_alert(
            pool,
            AlertCreate(
                alert_type=AlertType.check_in_declining_trend,
                title="Declining mood trend detected",
                body=f"Mood is trending down (slope: {mood_trend['slope']}/day over {mood_trend['n']} data points).",
                trigger_at=now,
            ),
        )
        created.append(alert.to_dict())

    if created:
        logger.info("check_in_alerts: created %d alerts", len(created))

    return created


# --- Internal helpers ---


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation coefficient. Returns None if no variance."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return cov / math.sqrt(var_x * var_y)


def _linear_slope(xs: list[float], ys: list[float]) -> float | None:
    """Slope of least-squares linear fit. Returns None if no x variance."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x == 0:
        return None
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return cov / var_x


def _find_mood_streaks(
    sorted_cis: list[CheckIn],
    predicate: Any,
) -> list[dict[str, Any]]:
    """Find streaks of consecutive calendar days where mood matches predicate."""
    streaks: list[dict[str, Any]] = []
    current_streak_start: datetime | None = None
    current_streak_len = 0
    last_date = None

    for ci in sorted_cis:
        if ci.mood is None:
            continue
        ci_date = ci.logged_at.date()
        if ci_date == last_date:
            # Same day — skip duplicate
            continue

        if predicate(ci.mood):
            if last_date is not None and (ci_date - last_date).days == 1:
                current_streak_len += 1
            else:
                # Save previous streak if long enough
                if current_streak_len >= _MIN_STREAK_LENGTH and current_streak_start:
                    streaks.append(
                        {
                            "start": current_streak_start.isoformat(),
                            "length": current_streak_len,
                        }
                    )
                current_streak_start = ci.logged_at
                current_streak_len = 1
        else:
            if current_streak_len >= _MIN_STREAK_LENGTH and current_streak_start:
                streaks.append(
                    {
                        "start": current_streak_start.isoformat(),
                        "length": current_streak_len,
                    }
                )
            current_streak_start = None
            current_streak_len = 0

        last_date = ci_date

    # Don't forget the last streak
    if current_streak_len >= _MIN_STREAK_LENGTH and current_streak_start:
        streaks.append(
            {
                "start": current_streak_start.isoformat(),
                "length": current_streak_len,
            }
        )

    return streaks
