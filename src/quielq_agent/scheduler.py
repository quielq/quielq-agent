"""Runs an agent's `schedule` entries on their cron times.

Deliberately simple - no APScheduler/Celery, just croniter and a sleep loop.
Phase 1 has exactly one scheduled agent (ops-watch) and no need for
concurrency, persistence across restarts, or missed-run catch-up (see the
plan doc: missed scheduled runs are an accepted gap, not a bug, in Phase 1).

Usage: python -m quielq_agent.scheduler --config fleet/agents/ops-watch.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from croniter import croniter
from dotenv import load_dotenv

from quielq_agent.config import PROJECT_ROOT, AgentConfig, ConfigError, ScheduleEntry, load_agent_config
from quielq_agent.llm import OllamaAuthError, OllamaCallError
from quielq_agent.loop import MaxIterationsExceeded, run_turn
from quielq_agent.tools import ToolRegistry, default_registry

POLL_INTERVAL_SECONDS = 60


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="quielq-agent-scheduler")
    parser.add_argument("--config", required=True, help="Path to an agent YAML config.")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run every schedule entry immediately and exit, instead of waiting for its cron time.",
    )
    return parser.parse_args(argv)


def _experiences_path(memory_dir: Path) -> Path:
    experiences_dir = memory_dir / "experiences"
    experiences_dir.mkdir(parents=True, exist_ok=True)
    return experiences_dir / f"{datetime.now(timezone.utc):%Y-%m}.jsonl"


async def run_scheduled_entry(config: AgentConfig, entry: ScheduleEntry, registry: ToolRegistry) -> dict:
    system_prompt = config.load_system_prompt()
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": entry.message},
    ]

    record: dict = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "agent": config.name,
        "trigger": entry.message,
        "deliver_to": entry.deliver_to,
    }
    try:
        result = await run_turn(config, messages, registry)
        record["status"] = "ok"
        record["result"] = result[-1].get("content", "")
    except (OllamaAuthError, OllamaCallError, MaxIterationsExceeded) as exc:
        record["status"] = "error"
        record["result"] = str(exc)

    if entry.deliver_to and entry.deliver_to != "memory":
        record["note"] = (
            f"deliver_to={entry.deliver_to!r} isn't implemented yet (deferred, see the plan's "
            "Phase 1 backlog note) - logged to memory instead"
        )

    if config.memory_path is not None:
        path = _experiences_path(config.memory_path)
        with path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    print(f"[{record['ran_at']}] {config.name}: {record['status']}")
    return record


async def _scheduler_loop(config: AgentConfig, registry: ToolRegistry, run_once: bool) -> None:
    if not config.schedule:
        print(f"{config.name} has no schedule entries - nothing to do.", file=sys.stderr)
        return

    if run_once:
        for entry in config.schedule:
            await run_scheduled_entry(config, entry, registry)
        return

    now = datetime.now()
    iters = [croniter(entry.cron, now) for entry in config.schedule]
    next_fires = [it.get_next(datetime) for it in iters]

    plural = "y" if len(config.schedule) == 1 else "ies"
    print(f"{config.name}: scheduler started, {len(config.schedule)} entr{plural}")

    while True:
        soonest_index = min(range(len(next_fires)), key=lambda i: next_fires[i])
        soonest_time = next_fires[soonest_index]
        wait_seconds = max(0.0, (soonest_time - datetime.now()).total_seconds())
        await asyncio.sleep(min(wait_seconds, POLL_INTERVAL_SECONDS))

        if datetime.now() >= soonest_time:
            await run_scheduled_entry(config, config.schedule[soonest_index], registry)
            next_fires[soonest_index] = iters[soonest_index].get_next(datetime)


def main(argv: list[str] | None = None) -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args(argv)
    try:
        config = load_agent_config(args.config)
        registry = default_registry()
        registry.validate_allow(config.tools.allow)
    except (ConfigError, ValueError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1

    asyncio.run(_scheduler_loop(config, registry, args.once))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
