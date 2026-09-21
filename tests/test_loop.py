import pytest

from quielq_agent import loop
from quielq_agent.config import AgentConfig
from quielq_agent.tools import ToolRegistry


def _config(tools_allow):
    return AgentConfig(
        name="t",
        display_name="T",
        model="m",
        system_prompt_file="fleet/prompts/echo.md",
        tools={"allow": tools_allow},
    )


class FakeToolCallFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, name, arguments):
        self.function = FakeToolCallFunction(name, arguments)


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none=True):
        data = {"role": "assistant"}
        if self.content is not None:
            data["content"] = self.content
        if self.tool_calls:
            data["tool_calls"] = [
                {"function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in self.tool_calls
            ]
        return data


class FakeResponse:
    def __init__(self, message):
        self.message = message


async def test_run_turn_no_tools_returns_immediately(monkeypatch):
    responses = [FakeResponse(FakeMessage(content="hi there"))]

    async def fake_chat(model, messages, tools=None, num_ctx=8192):
        return responses.pop(0)

    monkeypatch.setattr(loop.llm, "chat", fake_chat)

    registry = ToolRegistry()
    config = _config([])
    messages = [{"role": "user", "content": "hello"}]

    result = await loop.run_turn(config, messages, registry)

    assert result[-1]["content"] == "hi there"


async def test_run_turn_executes_tool_then_answers(monkeypatch):
    responses = [
        FakeResponse(FakeMessage(tool_calls=[FakeToolCall("dummy", {"q": "x"})])),
        FakeResponse(FakeMessage(content="final answer")),
    ]

    async def fake_chat(model, messages, tools=None, num_ctx=8192):
        return responses.pop(0)

    monkeypatch.setattr(loop.llm, "chat", fake_chat)

    registry = ToolRegistry()
    registry.register(
        "dummy", lambda q: f"result for {q}", {"type": "function", "function": {"name": "dummy"}}
    )
    config = _config(["dummy"])
    messages = [{"role": "user", "content": "hello"}]

    result = await loop.run_turn(config, messages, registry)

    assert result[-1]["content"] == "final answer"
    tool_messages = [m for m in result if m.get("role") == "tool"]
    assert tool_messages[0]["content"] == "result for x"


async def test_run_turn_raises_after_max_iterations(monkeypatch):
    async def fake_chat(model, messages, tools=None, num_ctx=8192):
        return FakeResponse(FakeMessage(tool_calls=[FakeToolCall("dummy", {})]))

    monkeypatch.setattr(loop.llm, "chat", fake_chat)

    registry = ToolRegistry()
    registry.register("dummy", lambda: "x", {"type": "function", "function": {"name": "dummy"}})
    config = _config(["dummy"])
    messages = [{"role": "user", "content": "hello"}]

    with pytest.raises(loop.MaxIterationsExceeded):
        await loop.run_turn(config, messages, registry)


async def test_run_turn_injects_time_context_without_persisting_it(monkeypatch):
    seen_messages = []

    async def fake_chat(model, messages, tools=None, num_ctx=8192):
        seen_messages.append(messages)
        return FakeResponse(FakeMessage(content="hi there"))

    monkeypatch.setattr(loop.llm, "chat", fake_chat)

    registry = ToolRegistry()
    config = _config([])
    messages = [{"role": "system", "content": "you are echo"}, {"role": "user", "content": "hello"}]

    result = await loop.run_turn(config, messages, registry)

    # llm.chat saw a time-context message inserted right after the system prompt...
    call_messages = seen_messages[0]
    assert call_messages[0] == {"role": "system", "content": "you are echo"}
    assert call_messages[1]["role"] == "system"
    assert "Current date and time" in call_messages[1]["content"]
    assert call_messages[2] == {"role": "user", "content": "hello"}
    # ...but it never leaks into the persisted/returned history.
    assert all("Current date and time" not in m.get("content", "") for m in result)
    assert len(result) == 3  # system, user, assistant - no extra message appended


async def test_run_turn_passes_tool_context(monkeypatch):
    async def fake_chat(model, messages, tools=None, num_ctx=8192):
        return FakeResponse(FakeMessage(tool_calls=[FakeToolCall("whoami", {})]))

    monkeypatch.setattr(loop.llm, "chat", fake_chat)

    captured = {}

    def whoami(context=None):
        captured["context"] = context
        return "ok"

    registry = ToolRegistry()
    registry.register("whoami", whoami, {"type": "function", "function": {"name": "whoami"}})
    config = _config(["whoami"])
    messages = [{"role": "user", "content": "hello"}]

    with pytest.raises(loop.MaxIterationsExceeded):
        await loop.run_turn(config, messages, registry)

    assert captured["context"].agent_name == "t"
