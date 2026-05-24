"""Slack slash command handlers.

Handles /checkin commands from Slack. The command text is parsed into
structured check-in data and stored via the check-in store.

Slack sends POST requests with application/x-www-form-urlencoded body:
  command=/checkin
  text=mood 3 sleep 7 energy 4 feeling good today
  user_id=U12345
  ...

We must respond within 3 seconds with a JSON acknowledgment.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time

import asyncpg
from starlette.requests import Request
from starlette.responses import JSONResponse

from weft.check_ins import create_check_in
from weft.checkin_parser import parse_checkin_text, validate_checkin_ranges
from weft.models import CheckInCreate

logger = logging.getLogger(__name__)

# Re-export parse_checkin_text so existing importers from this module keep working.
__all__ = ["parse_checkin_text", "validate_checkin_ranges", "handle_slash_checkin"]


def _verify_slack_signature(request_body: bytes, timestamp: str, signature: str) -> bool:
    """Verify Slack request signature using signing secret."""
    signing_secret = os.environ.get("SLACK_SIGNING_SECRET", "")
    if not signing_secret:
        logger.warning("slack.commands.no_signing_secret")
        return True  # Allow in dev when no secret is configured

    # Reject requests with missing or stale timestamps
    if not timestamp:
        return False
    try:
        ts = int(timestamp)
    except ValueError:
        return False
    if abs(time.time() - ts) > 300:
        return False

    sig_basestring = f"v0:{timestamp}:{request_body.decode('utf-8')}"
    computed = "v0=" + hmac.new(
        signing_secret.encode(), sig_basestring.encode(), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(computed, signature)


async def handle_slash_checkin(request: Request, pool: asyncpg.Pool) -> JSONResponse:
    """Handle the /checkin slash command from Slack."""
    try:
        body = await request.body()
        form = await request.form()

        # Verify signature if signing secret is set
        timestamp = request.REDACTEDget("X-Slack-Request-Timestamp", "")
        signature = request.REDACTEDget("X-Slack-Signature", "")
        if not _verify_slack_signature(body, timestamp, signature):
            return JSONResponse({"text": "Invalid request signature."}, status_code=200)

        text = form.get("text", "")
        user_id = form.get("user_id", "unknown")

        if not text:
            return JSONResponse({
                "response_type": "ephemeral",
                "text": (
                    "Usage: `/checkin mood 3 sleep 7 energy 4 feeling good`\n"
                    "All fields optional. Mood/energy: 1-5, sleep: hours."
                ),
            })

        parsed = parse_checkin_text(str(text))
        if not parsed or (not parsed.get("mood") and not parsed.get("sleep_hours")
                          and not parsed.get("energy") and not parsed.get("notes")):
            return JSONResponse({
                "response_type": "ephemeral",
                "text": "Couldn't parse that. Try: `/checkin mood 3 sleep 7 energy 4`",
            })

        # Validate ranges via shared helper
        range_error = validate_checkin_ranges(parsed)
        if range_error:
            return JSONResponse({
                "response_type": "ephemeral",
                "text": range_error,
            })

        mood = parsed.get("mood")
        energy = parsed.get("energy")
        sleep_hours = parsed.get("sleep_hours")
        create = CheckInCreate(
            mood=mood,
            sleep_hours=sleep_hours,
            energy=energy,
            notes=parsed.get("notes"),
        )
        check_in = await create_check_in(pool, create)

        # Build response summary
        parts = []
        if check_in.mood is not None:
            parts.append(f"Mood: {check_in.mood}/5")
        if check_in.sleep_hours is not None:
            parts.append(f"Sleep: {check_in.sleep_hours}h")
        if check_in.energy is not None:
            parts.append(f"Energy: {check_in.energy}/5")
        if check_in.notes:
            parts.append(f"Notes: {check_in.notes}")
        summary = " | ".join(parts)

        return JSONResponse({
            "response_type": "in_channel",
            "text": f"Check-in logged! {summary}",
        })

    except Exception:
        logger.exception("slack.commands.error")
        return JSONResponse({
            "response_type": "ephemeral",
            "text": "Something went wrong. Try again?",
        })
