"""local_search: keyword/BM25 search over an agent's own fleet/memory/<agent>/knowledge/.

No embeddings needed at Quiel's personal-fleet scale (see the plan doc's
memory taxonomy). Agent-scoped via ToolContext, same as request_approval.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from quielq_agent.tools import ToolContext

LOCAL_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "local_search",
        "description": "Search this agent's own knowledge/ folder (documents, notes) for a query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for."},
            },
            "required": ["query"],
        },
    },
}

_WORD_RE = re.compile(r"[a-z0-9]+")
_DOC_EXTENSIONS = {".md", ".txt"}


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _load_documents(knowledge_dir: Path) -> list[tuple[str, str]]:
    docs = []
    for path in sorted(knowledge_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in _DOC_EXTENSIONS:
            docs.append((str(path.relative_to(knowledge_dir)), path.read_text(errors="ignore")))
    return docs


def _bm25_scores(query_tokens: list[str], docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> list[float]:
    n = len(docs_tokens)
    if n == 0:
        return []
    avg_len = sum(len(tokens) for tokens in docs_tokens) / n

    doc_freq: Counter[str] = Counter()
    for tokens in docs_tokens:
        doc_freq.update(set(tokens))
    idf = {term: math.log((n - freq + 0.5) / (freq + 0.5) + 1) for term, freq in doc_freq.items()}

    scores = []
    for tokens in docs_tokens:
        term_freq = Counter(tokens)
        doc_len = len(tokens)
        score = 0.0
        for term in query_tokens:
            freq = term_freq.get(term, 0)
            if freq == 0:
                continue
            score += idf.get(term, 0.0) * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * doc_len / avg_len))
        scores.append(score)
    return scores


def local_search(query: str, max_results: int = 3, context: ToolContext | None = None) -> str:
    if context is None or context.memory_dir is None:
        return "error: local_search needs an agent with a memory_dir configured"

    knowledge_dir = context.memory_dir / "knowledge"
    if not knowledge_dir.is_dir():
        return "No knowledge/ folder yet for this agent."

    docs = _load_documents(knowledge_dir)
    if not docs:
        return "knowledge/ folder is empty - nothing to search."

    query_tokens = _tokenize(query)
    docs_tokens = [_tokenize(content) for _, content in docs]
    scores = _bm25_scores(query_tokens, docs_tokens)

    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
    results = [(name, content, score) for (name, content), score in ranked if score > 0][:max_results]
    if not results:
        return "No matching documents found."

    lines = []
    for name, content, score in results:
        snippet = content[:300].replace("\n", " ").strip()
        lines.append(f"- {name} (score {score:.2f}): {snippet}")
    return "\n".join(lines)
