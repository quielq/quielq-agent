"""web_fetch: fetch a specific URL's content via Ollama cloud's built-in API.

Same OLLAMA_API_KEY as web_search/chat - no separate provider needed.
"""

from __future__ import annotations

import os

import ollama

from quielq_agent.llm import OLLAMA_CLOUD_HOST, OllamaAuthError

WEB_FETCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_fetch",
        "description": "Fetch the content of a specific web page URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch."},
            },
            "required": ["url"],
        },
    },
}


def web_fetch(url: str) -> str:
    api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        raise OllamaAuthError("OLLAMA_API_KEY is not set - web_fetch needs it (same key as chat).")
    client = ollama.Client(host=OLLAMA_CLOUD_HOST, headers={"Authorization": f"Bearer {api_key}"})
    response = client.web_fetch(url)
    title = response.title or "(no title)"
    content = (response.content or "")[:2000]
    return f"# {title}\n\n{content}"
