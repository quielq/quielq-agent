# quielq-agent

A standalone personal AI agent fleet.

## Phase 0 status

Proves the core single-agent loop: a per-agent YAML config drives a model call to Ollama
cloud, with a tool-call round trip for the `research` agent. No Docker, no web UI, no
LiteLLM yet - those are later phases.

## Setup

```bash
uv sync
cp .env.example .env
# then: `ollama signin`, or put a cloud API key from https://ollama.com in .env as OLLAMA_API_KEY
```

## Run

```bash
uv run python -m quielq_agent.cli --config fleet/agents/echo.yaml
uv run python -m quielq_agent.cli --config fleet/agents/research.yaml
```

Before running, verify the `model:` field in each `fleet/agents/*.yaml` against
`ollama list` (while signed in) or the Ollama cloud catalog - don't trust the checked-in
placeholder name blindly, cloud model names change.

## Test

```bash
uv run pytest
```

All tests run without `OLLAMA_API_KEY` set (they mock the Ollama client). Real end-to-end
verification needs a real key - see Setup above.
