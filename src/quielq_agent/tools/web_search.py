"""Real web search for the `research` agent, via Ollama cloud's built-in web_search API.

Uses the same OLLAMA_API_KEY as chat calls - no separate search provider/signup needed.
"""

from __future__ import annotations

import os

import ollama

from quielq_agent.llm import OLLAMA_CLOUD_HOST, OllamaAuthError

WEB_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web and return a short list of results (title, url, snippet).",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
            },
            "required": ["query"],
        },
    },
}


def web_search(query: str, max_results: int = 3) -> str:
    api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        raise OllamaAuthError(
            "OLLAMA_API_KEY is not set - web_search needs it (same key as chat)."
        )
    client = ollama.Client(host=OLLAMA_CLOUD_HOST, headers={"Authorization": f"Bearer {api_key}"})
    response = client.web_search(query, max_results=max_results)
    if not response.results:
        return "No results found."
    lines = []
    for r in response.results:
        snippet = (r.content or "")[:300]
        lines.append(f"- {r.title}\n  {r.url}\n  {snippet}")
    return "\n".join(lines)
