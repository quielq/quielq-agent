"""request_approval: the mechanism behind Phase 3's human-approval-for-high-stakes-actions rule.

No v1 agent currently has a tool that needs this gate - every tool built so
far is read-only or sandboxed-write (see the plan doc's non-goals). This
exists so the pattern is already proven the day one does, instead of being
designed under pressure later.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from quielq_agent.tools import ToolContext

REQUEST_APPROVAL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "request_approval",
        "description": (
            "Request human approval before taking a high-stakes action (sending "
            "something, posting somewhere, writing outside your own memory folder). "
            "This does NOT perform the action - it only records a pending request "
            "for Quiel to review. Treat the action as NOT done until approved."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "The action being requested, in plain language."},
                "reason": {"type": "string", "description": "Why this action is being proposed."},
            },
            "required": ["action", "reason"],
        },
    },
}


def request_approval(action: str, reason: str, context: ToolContext | None = None) -> str:
    if context is None or context.memory_dir is None:
        return "error: request_approval needs an agent with a memory_dir configured"

    pending_dir = context.memory_dir / "pending_actions"
    pending_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    record = {
        "requested_at": now.isoformat(),
        "agent": context.agent_name,
        "action": action,
        "reason": reason,
        "status": "pending",
    }
    filename = f"{now.strftime('%Y%m%dT%H%M%S%f')}.json"
    (pending_dir / filename).write_text(json.dumps(record, indent=2))

    return (
        f"Recorded as a pending action ({filename}) for Quiel to review. "
        "Not approved yet - do not proceed as if it already happened."
    )
