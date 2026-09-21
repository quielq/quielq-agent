"""Interactive terminal chat for one fleet agent.

Usage: python -m quielq_agent.cli --config fleet/agents/echo.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from quielq_agent.config import AgentConfig, ConfigError, load_agent_config
from quielq_agent.llm import OllamaAuthError, OllamaCallError
from quielq_agent.loop import MaxIterationsExceeded, run_turn
from quielq_agent.tools import ToolRegistry, default_registry


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="quielq-agent", description="Chat with one fleet agent.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to an agent YAML config, e.g. fleet/agents/echo.yaml",
    )
    return parser.parse_args(argv)


async def _chat_loop(config: AgentConfig, registry: ToolRegistry) -> None:
    system_prompt = config.load_system_prompt()
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    print(f"{config.display_name} ({config.name}) - model: {config.model}")
    print("Type a message, or 'exit'/Ctrl-D to quit.\n")

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            return

        snapshot = list(messages)
        messages.append({"role": "user", "content": user_input})
        try:
            messages = await run_turn(config, messages, registry)
        except OllamaAuthError as exc:
            print(f"\n[auth error] {exc}\n")
            messages = snapshot
            continue
        except (OllamaCallError, MaxIterationsExceeded) as exc:
            print(f"\n[error] {exc}\n")
            messages = snapshot
            continue

        print(f"\n{messages[-1].get('content', '')}\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_agent_config(args.config)
        registry = default_registry()
        registry.validate_allow(config.tools.allow)
    except (ConfigError, ValueError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1

    asyncio.run(_chat_loop(config, registry))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
