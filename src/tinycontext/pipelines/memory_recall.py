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


_HIGH_RELEVANCE_THRESHOLD = 0.90
_MEDIUM_RELEVANCE_THRESHOLD = 0.75


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[A-Za-z0-9_]+", text.lower()) if token]


def _rank_by_score(scores: list[float]) -> dict[int, int]:
    return {
        index: rank
        for rank, index in enumerate(
            sorted(range(len(scores)), key=lambda item: scores[item], reverse=True),
            start=1,
        )
    }


def _relevance_label(rrf_similarity: float) -> str:
    if rrf_similarity >= _HIGH_RELEVANCE_THRESHOLD:
        return "high"
    if rrf_similarity >= _MEDIUM_RELEVANCE_THRESHOLD:
        return "medium"
    return "low"


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
    rrf_similarity_cutoff: float | None = None,
    dense_weight: float = 0.5,
    rrf_k: int = 60,
    query_prefix: str = "",
    document_prefix: str = "",
    skip_stale_backfill: bool = False,
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
    if stale and not skip_stale_backfill:
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
        bm25_scores = [
            float(score) for score in BM25Okapi(corpus).get_scores(query_tokens)
        ]
    else:
        bm25_scores = [0.0 for _row in candidates]

    dense_values = [float(dense_scores.get(row.id, 0.0)) for row in candidates]
    bm25_ranks = _rank_by_score(bm25_scores)
    dense_ranks = _rank_by_score(dense_values)
    sparse_weight = 1.0 - dense_weight
    max_rrf_score = (sparse_weight + dense_weight) / (rrf_k + 1)
    fused: list[tuple[MemoryRow, dict[str, float | int | str]]] = []
    for index, row in enumerate(candidates):
        bm25_rank = bm25_ranks[index]
        dense_rank = dense_ranks[index]
        rrf_score = (
            sparse_weight / (rrf_k + bm25_rank)
            + dense_weight / (rrf_k + dense_rank)
        )
        rrf_similarity = rrf_score / max_rrf_score if max_rrf_score else 0.0
        if (
            rrf_similarity_cutoff is not None
            and rrf_similarity < rrf_similarity_cutoff
        ):
            continue
        fused.append(
            (
                row,
                {
                    "bm25_score": bm25_scores[index],
                    "bm25_rank": bm25_rank,
                    "dense_score": dense_values[index],
                    "dense_rank": dense_rank,
                    "rrf_score": rrf_score,
                    "rrf_similarity": rrf_similarity,
                    "relevance": _relevance_label(rrf_similarity),
                },
            )
        )
    ranked = sorted(
        fused,
        key=lambda item: (
            item[1]["rrf_score"],
            item[1]["dense_score"],
            item[1]["bm25_score"],
        ),
        reverse=True,
    )[:top_k]

    selected: list[dict[str, Any]] = []
    total_tokens = 0
    truncated = False
    for rank, (row, scores) in enumerate(ranked, start=1):
        content_tokens = token_count(row.content, encoding_name)
        if selected and total_tokens + content_tokens > max_tokens:
            truncated = True
            break
        if not selected and content_tokens > max_tokens:
            truncated = True
            selected.append(_memory_payload(row, rank, scores, content_tokens))
            total_tokens = content_tokens
            break
        selected.append(_memory_payload(row, rank, scores, content_tokens))
        total_tokens += content_tokens

    return {
        "query": query,
        "memories": selected,
        "total_tokens": total_tokens,
        "truncated": truncated,
    }


def _memory_payload(
    row: MemoryRow,
    rank: int,
    retrieval: dict[str, float | int | str],
    content_tokens: int,
) -> dict[str, Any]:
    return {
        "id": row.id,
        "content": row.content,
        "rank": rank,
        "relevance": str(retrieval["relevance"]),
        "scores": {
            "rrf": float(retrieval["rrf_similarity"]),
            "dense": float(retrieval["dense_score"]),
            "bm25": float(retrieval["bm25_score"]),
        },
        "content_tokens": content_tokens,
        "created_at": row.created_at,
    }
