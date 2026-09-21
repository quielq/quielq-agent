"""Ollama cloud calling. Non-streaming for Phase 0 - see the plan's rationale."""

from __future__ import annotations

import os

import ollama

OLLAMA_CLOUD_HOST = "https://ollama.com"
DEFAULT_NUM_CTX = 8192


class OllamaAuthError(RuntimeError):
    """OLLAMA_API_KEY is missing, or Ollama cloud rejected it."""


class OllamaCallError(RuntimeError):
    """Any other failure calling Ollama cloud (network, bad request, etc.)."""


def _client() -> ollama.AsyncClient:
    api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        raise OllamaAuthError(
            "OLLAMA_API_KEY is not set. Run `ollama signin` (or generate a cloud API "
            "key at ollama.com) and put it in .env - see .env.example."
        )
    return ollama.AsyncClient(
        host=OLLAMA_CLOUD_HOST,
        headers={"Authorization": f"Bearer {api_key}"},
    )


async def chat(
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    num_ctx: int = DEFAULT_NUM_CTX,
) -> ollama.ChatResponse:
    client = _client()
    try:
        return await client.chat(
            model=model,
            messages=messages,
            tools=tools or None,
            options={"num_ctx": num_ctx},
            stream=False,
        )
    except ollama.ResponseError as exc:
        if exc.status_code in (401, 403):
            raise OllamaAuthError(f"Ollama cloud rejected the API key: {exc}") from exc
        raise OllamaCallError(str(exc)) from exc
    except Exception as exc:  # network errors etc.
        raise OllamaCallError(str(exc)) from exc
