from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from tinycontext.errors import SessionNotFoundError
from tinycontext.models import MemoryRow
from tinycontext.services.memory_store_service import fetch_candidates, session_exists
from tinycontext.services.token_counter_service import token_count


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[A-Za-z0-9_]+", text.lower()) if token]


def memory_recall_run(
    query: str,
    *,
    session_id: str | None,
    max_tokens: int,
    top_k: int,
    db_path: Path,
    encoding_name: str,
) -> dict[str, Any]:
    if session_id is not None and not session_exists(db_path, session_id):
        raise SessionNotFoundError(f"session not found: {session_id}")

    candidates = fetch_candidates(db_path, session_id=session_id)
    if not candidates:
        return {
            "query": query,
            "memories": [],
            "total_tokens": 0,
            "truncated": False,
        }

    query_tokens = _tokenize(query)
    corpus = [_tokenize(row.content) for row in candidates]
    if query_tokens and any(corpus):
        scores = BM25Okapi(corpus).get_scores(query_tokens)
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
    else:
        ranked = [(row, 0.0) for row in candidates[:top_k]]

    selected: list[dict[str, Any]] = []
    total_tokens = 0
    truncated = False
    for row, score in ranked:
        content_tokens = token_count(row.content, encoding_name)
        if selected and total_tokens + content_tokens > max_tokens:
            truncated = True
            break
        if not selected and content_tokens > max_tokens:
            truncated = True
            selected.append(_memory_payload(row, score, content_tokens))
            total_tokens = content_tokens
            break
        selected.append(_memory_payload(row, score, content_tokens))
        total_tokens += content_tokens

    return {
        "query": query,
        "memories": selected,
        "total_tokens": total_tokens,
        "truncated": truncated,
    }


def _memory_payload(row: MemoryRow, score: float, content_tokens: int) -> dict[str, Any]:
    return {
        "id": row.id,
        "content": row.content,
        "score": float(score),
        "content_tokens": content_tokens,
        "tags": row.tags,
        "metadata": row.metadata,
        "created_at": row.created_at,
    }
