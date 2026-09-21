"""Plain tool registry: name -> (callable, JSON schema)."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Union

ToolFunc = Callable[..., Union[str, Awaitable[str]]]


@dataclass
class RegisteredTool:
    name: str
    func: ToolFunc
    schema: dict[str, Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, name: str, func: ToolFunc, schema: dict[str, Any]) -> None:
        if name in self._tools:
            raise ValueError(f"tool already registered: {name}")
        self._tools[name] = RegisteredTool(name=name, func=func, schema=schema)

    def known_names(self) -> set[str]:
        return set(self._tools)

    def validate_allow(self, allow: list[str]) -> None:
        unknown = set(allow) - self.known_names()
        if unknown:
            raise ValueError(f"unknown tool(s) in tools.allow: {sorted(unknown)}")

    def schemas_for(self, names: list[str]) -> list[dict[str, Any]]:
        return [self._tools[name].schema for name in names]

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"error: unknown tool {name!r}"
        try:
            result = tool.func(**arguments)
            if inspect.isawaitable(result):
                result = await result
            return str(result)
        except Exception as exc:  # a bad tool call must not crash the loop
            return f"error executing tool {name!r}: {exc}"


def default_registry() -> ToolRegistry:
    from quielq_agent.tools.web_search import WEB_SEARCH_SCHEMA, web_search

    registry = ToolRegistry()
    registry.register("web_search", web_search, WEB_SEARCH_SCHEMA)
    return registry
