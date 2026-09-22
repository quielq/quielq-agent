"""github_repo: read-only GitHub CI status (recent workflow runs) for a repo.

Works unauthenticated (GitHub's public rate limit, 60 req/hour) or with an
optional GITHUB_TOKEN in .env for the higher authenticated limit. No write
access anywhere - matches the plan's non-goals (no v1 agent writes external).
"""

from __future__ import annotations

import os

import httpx

GITHUB_REPO_SCHEMA = {
    "type": "function",
    "function": {
        "name": "github_repo",
        "description": "Get the status of the most recent GitHub Actions workflow runs for a repo.",
        "parameters": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "owner/repo, e.g. 'quielq/quielq-agent'."},
                "limit": {"type": "integer", "description": "Max runs to return (default 5)."},
            },
            "required": ["repo"],
        },
    },
}


def github_repo(repo: str, limit: int = 5) -> str:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = httpx.get(
        f"https://api.github.com/repos/{repo}/actions/runs",
        headers=headers,
        params={"per_page": limit},
        timeout=15,
    )
    if response.status_code == 404:
        return f"error: repo {repo!r} not found (or has no Actions runs, or is private without a token)"
    response.raise_for_status()

    runs = response.json().get("workflow_runs", [])
    if not runs:
        return f"No workflow runs found for {repo}."

    lines = []
    for run in runs[:limit]:
        lines.append(
            f"- {run['name']}: {run['status']}/{run['conclusion']} "
            f"(branch {run['head_branch']}, updated {run['updated_at']})"
        )
    return "\n".join(lines)
