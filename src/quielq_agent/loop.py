"""Minimal agent loop: call model -> tool_calls? -> execute -> repeat.

No context compaction, no answer verification - just the loop and a hard
iteration cap, deliberately smaller than OnIt's loop (see the plan doc).
"""

from __future__ import annotations

from quielq_agent import llm
from quielq_agent.config import AgentConfig
from quielq_agent.tools import ToolRegistry

MAX_ITERATIONS = 8


class MaxIterationsExceeded(RuntimeError):
    """The agent looped MAX_ITERATIONS times without producing a final answer."""


async def run_turn(
    config: AgentConfig,
    messages: list[dict],
    registry: ToolRegistry,
) -> list[dict]:
    tool_schemas = registry.schemas_for(config.tools.allow)

    for _ in range(MAX_ITERATIONS):
        response = await llm.chat(config.model, messages, tool_schemas)
        messages.append(response.message.model_dump(exclude_none=True))

        if not response.message.tool_calls:
            return messages

        for call in response.message.tool_calls:
            result = await registry.execute(call.function.name, dict(call.function.arguments))
            messages.append(
                {
                    "role": "tool",
                    "tool_name": call.function.name,
                    "content": result,
                }
            )

    raise MaxIterationsExceeded(
        f"{config.name}: hit {MAX_ITERATIONS} iterations without a final answer"
    )
