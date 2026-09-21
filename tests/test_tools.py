import pytest

from quielq_agent.llm import OllamaAuthError
from quielq_agent.tools import ToolRegistry, default_registry


def test_registry_validate_allow_rejects_unknown():
    registry = ToolRegistry()
    registry.register("dummy", lambda: "ok", {"type": "function", "function": {"name": "dummy"}})
    with pytest.raises(ValueError):
        registry.validate_allow(["not_a_tool"])
    registry.validate_allow(["dummy"])  # should not raise


async def test_registry_execute_catches_tool_errors():
    registry = ToolRegistry()

    def boom():
        raise RuntimeError("kaboom")

    registry.register("boom", boom, {"type": "function", "function": {"name": "boom"}})
    result = await registry.execute("boom", {})
    assert "kaboom" in result


async def test_registry_execute_unknown_tool():
    registry = ToolRegistry()
    result = await registry.execute("nope", {})
    assert "unknown tool" in result


def test_default_registry_has_web_search():
    registry = default_registry()
    assert "web_search" in registry.known_names()


def test_web_search_requires_api_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    from quielq_agent.tools.web_search import web_search

    with pytest.raises(OllamaAuthError):
        web_search("test query")
