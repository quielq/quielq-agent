"""Models have no reliable sense of "now" - this fills that gap.

Computed fresh per model call (see loop.py) rather than baked into the
static system prompt once, so a long-running session doesn't go stale.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

DEFAULT_TIMEZONE = "Asia/Manila"


def current_time_message(timezone: str = DEFAULT_TIMEZONE) -> dict:
    now = datetime.now(ZoneInfo(timezone))
    return {
        "role": "system",
        "content": f"Current date and time: {now.strftime('%A, %B %d, %Y, %H:%M')} ({timezone}).",
    }
