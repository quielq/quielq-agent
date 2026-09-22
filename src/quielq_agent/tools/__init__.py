"""Plain tool registry: name -> (callable, JSON schema)."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Union

ToolFunc = Callable[..., Union[str, Awaitable[str]]]


@dataclass
class ToolContext:
    """Per-call agent context, injected only into tools that ask for it.

    A tool function opts in by declaring a `context` parameter; the registry
    checks with `inspect.signature` before passing it, so tools that don't
    need agent scoping (like web_search) are unaffected.
    """

    agent_name: str
    memory_dir: Path | None


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

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolContext | None = None,
    ) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"error: unknown tool {name!r}"
        try:
            call_kwargs = dict(arguments)
            if context is not None and "context" in inspect.signature(tool.func).parameters:
                call_kwargs["context"] = context
            result = tool.func(**call_kwargs)
            if inspect.isawaitable(result):
                result = await result
            return str(result)
        except Exception as exc:  # a bad tool call must not crash the loop
            return f"error executing tool {name!r}: {exc}"


def default_registry() -> ToolRegistry:
    from quielq_agent.tools.approval import REQUEST_APPROVAL_SCHEMA, request_approval
    from quielq_agent.tools.docx import CREATE_DOCX_SCHEMA, create_docx
    from quielq_agent.tools.github_repo import GITHUB_REPO_SCHEMA, github_repo
    from quielq_agent.tools.local_search import LOCAL_SEARCH_SCHEMA, local_search
    from quielq_agent.tools.web_fetch import WEB_FETCH_SCHEMA, web_fetch
    from quielq_agent.tools.web_search import WEB_SEARCH_SCHEMA, web_search

    registry = ToolRegistry()
    registry.register("web_search", web_search, WEB_SEARCH_SCHEMA)
    registry.register("web_fetch", web_fetch, WEB_FETCH_SCHEMA)
    registry.register("local_search", local_search, LOCAL_SEARCH_SCHEMA)
    registry.register("github_repo", github_repo, GITHUB_REPO_SCHEMA)
    registry.register("request_approval", request_approval, REQUEST_APPROVAL_SCHEMA)
    registry.register("create_docx", create_docx, CREATE_DOCX_SCHEMA)
    return registry
