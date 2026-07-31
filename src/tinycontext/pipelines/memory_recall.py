from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from tinycontext.errors import SessionNotFoundError
from tinycontext.models import MemoryRow
from tinycontext.services.embedding_service import embed_texts, embedding_model_key
from tinycontext.services.memory_store_service import (
    fetch_candidates,
    fetch_dense_scores,
    session_exists,
    update_memory_embeddings,
)
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
    models_dir: Path | None = None,
    embedding_model: str = "fast",
    embedding_batch_size: int = 32,
    dense_weight: float = 0.5,
    rrf_k: int = 60,
    query_prefix: str = "",
    document_prefix: str = "",
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

    model_key = embedding_model_key(
        embedding_model,
        models_dir=models_dir,
        document_prefix=document_prefix,
    )
    query_embedding = embed_texts(
        [query_prefix + query],
        embedding_model=embedding_model,
        models_dir=models_dir,
        batch_size=embedding_batch_size,
    )[0]
    stale = [
        row
        for row in candidates
        if row.embedding_model != model_key
        or row.embedding_dimensions != len(query_embedding)
    ]
    if stale:
        document_embeddings = embed_texts(
            [document_prefix + row.content for row in stale],
            embedding_model=embedding_model,
            models_dir=models_dir,
            batch_size=embedding_batch_size,
        )
        update_memory_embeddings(
            db_path,
            [
                (row.id, vector)
                for row, vector in zip(stale, document_embeddings, strict=True)
            ],
            embedding_model=model_key,
        )

    dense_scores = fetch_dense_scores(
        db_path,
        query_embedding,
        embedding_model=model_key,
        session_id=session_id,
    )
    query_tokens = _tokenize(query)
    corpus = [_tokenize(row.content) for row in candidates]
    if query_tokens and any(corpus):
        bm25_values = BM25Okapi(corpus).get_scores(query_tokens)
        lexical = sorted(
            (
                (row, score)
                for row, tokens, score in zip(
                    candidates,
                    corpus,
                    bm25_values,
                    strict=True,
                )
                if set(tokens).intersection(query_tokens)
            ),
            key=lambda item: item[1],
            reverse=True,
        )
    else:
        lexical = []

    lexical_ranks = {
        row.id: rank
        for rank, (row, _score) in enumerate(lexical, start=1)
    }
    dense_ranks = {
        memory_id: rank
        for rank, (memory_id, _score) in enumerate(
            sorted(
                dense_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            ),
            start=1,
        )
    }
    lexical_weight = 1.0 - dense_weight
    fused: list[tuple[MemoryRow, float]] = []
    for row in candidates:
        score = 0.0
        lexical_rank = lexical_ranks.get(row.id)
        if lexical_rank is not None:
            score += lexical_weight / (rrf_k + lexical_rank)
        dense_rank = dense_ranks.get(row.id)
        if dense_rank is not None:
            score += dense_weight / (rrf_k + dense_rank)
        fused.append((row, score))
    ranked = sorted(
        fused,
        key=lambda item: item[1],
        reverse=True,
    )[:top_k]

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
