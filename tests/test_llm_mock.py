import asyncio

import pytest

from quielq_agent import llm


def test_chat_raises_auth_error_without_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    with pytest.raises(llm.OllamaAuthError):
        asyncio.run(llm.chat("some-model", [{"role": "user", "content": "hi"}]))


async def test_chat_calls_async_client_with_bearer_auth(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "fake-key")
    captured = {}

    class FakeMessage:
        content = "hello"
        tool_calls = None

    class FakeResponse:
        message = FakeMessage()

    class FakeAsyncClient:
        def __init__(self, host, headers):
            captured["host"] = host
            captured["headers"] = headers

        async def chat(self, **kwargs):
            captured["kwargs"] = kwargs
            return FakeResponse()

    monkeypatch.setattr(llm.ollama, "AsyncClient", FakeAsyncClient)

    response = await llm.chat("some-model", [{"role": "user", "content": "hi"}])

    assert response.message.content == "hello"
    assert captured["host"] == llm.OLLAMA_CLOUD_HOST
    assert captured["headers"]["Authorization"] == "Bearer fake-key"
    assert captured["kwargs"]["model"] == "some-model"
    assert captured["kwargs"]["stream"] is False
