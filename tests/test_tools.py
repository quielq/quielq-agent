import pytest

from quielq_agent.llm import OllamaAuthError
from quielq_agent.tools import ToolContext, ToolRegistry, default_registry


def test_registry_validate_allow_rejects_unknown():
    registry = ToolRegistry()
    registry.register("dummy", lambda: "ok", {"type": "function", "function": {"name": "dummy"}})
    with pytest.raises(ValueError):
        registry.validate_allow(["not_a_tool"])
    registry.validate_allow(["dummy"])  # should not raise


async def test_registry_injects_context_only_when_requested(tmp_path):
    registry = ToolRegistry()

    def wants_context(context=None):
        return context.agent_name

    def ignores_context():
        return "fine"

    registry.register("wants", wants_context, {"type": "function", "function": {"name": "wants"}})
    registry.register("ignores", ignores_context, {"type": "function", "function": {"name": "ignores"}})

    context = ToolContext(agent_name="research", memory_dir=tmp_path)

    assert await registry.execute("wants", {}, context=context) == "research"
    assert await registry.execute("ignores", {}, context=context) == "fine"


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


def test_default_registry_has_request_approval():
    registry = default_registry()
    assert "request_approval" in registry.known_names()


def test_request_approval_writes_pending_action(tmp_path):
    from quielq_agent.tools.approval import request_approval

    context = ToolContext(agent_name="daily-brief", memory_dir=tmp_path)
    result = request_approval("send an email to legal", "flagged in inbox triage", context=context)

    assert "Recorded as a pending action" in result
    pending_files = list((tmp_path / "pending_actions").glob("*.json"))
    assert len(pending_files) == 1

    import json

    record = json.loads(pending_files[0].read_text())
    assert record["agent"] == "daily-brief"
    assert record["status"] == "pending"
    assert record["action"] == "send an email to legal"


def test_request_approval_without_context_errors_cleanly():
    from quielq_agent.tools.approval import request_approval

    result = request_approval("do something", "because", context=None)
    assert result.startswith("error:")
