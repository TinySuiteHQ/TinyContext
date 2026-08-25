from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tinycontext.errors import SessionNotFoundError
from tinycontext.models import MemoryRow
from tinycontext.services.embedding_service import embed_texts, embedding_model_key
from tinycontext.services.memory_store_service import (
    count_memories,
    fetch_dense_scores,
    fetch_memories_by_ids,
    fetch_sparse_scores,
    fetch_stale_memories,
    record_recall_hits,
    session_exists,
    short_memory_ref,
    update_memory_embeddings,
)
from tinycontext.services.token_counter_service import token_count


_HIGH_RELEVANCE_THRESHOLD = 0.90
_MEDIUM_RELEVANCE_THRESHOLD = 0.75

# Bound on how many top-scoring rows each retriever (dense, sparse) hands to
# fusion, instead of ranking every stored memory in the session/kind scope on
# every call. RRF only needs relative rank among plausible candidates, so a
# generous top-N from each side (unioned) reproduces the same fused ordering
# for anything that would realistically surface, at a fraction of the cost --
# this is the same pattern hybrid search engines (Elasticsearch, Weaviate)
# use: rank within each retriever's top-N, not the full corpus.
_CANDIDATE_POOL_MULTIPLIER = 20
_CANDIDATE_POOL_FLOOR = 100
_CANDIDATE_POOL_CAP = 1000

# RRF similarity is rank-relative: it measures how a candidate ranks against
# the *other* candidates in this pool, not how semantically close it actually
# is to the query. With a small or uniformly-weak pool (e.g. one memory in a
# session), a candidate can trivially rank #1 in both signals -- and score
# rrf_similarity == 1.0 -- while being unrelated to the query. These floors on
# the raw dense cosine similarity (an absolute, pool-independent measure) stop
# a rank-only "best of a bad pool" match from being labeled high/medium. The
# values are conservative defaults, not calibrated per embedding model.
_HIGH_DENSE_FLOOR = 0.55
_MEDIUM_DENSE_FLOOR = 0.35


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _iso_utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _rank_by_score(scores: list[float]) -> dict[int, int]:
    return {
        index: rank
        for rank, index in enumerate(
            sorted(range(len(scores)), key=lambda item: scores[item], reverse=True),
            start=1,
        )
    }


def _relevance_label(rrf_similarity: float, dense_score: float) -> str:
    if rrf_similarity >= _HIGH_RELEVANCE_THRESHOLD and dense_score >= _HIGH_DENSE_FLOOR:
        return "high"
    if (
        rrf_similarity >= _MEDIUM_RELEVANCE_THRESHOLD
        and dense_score >= _MEDIUM_DENSE_FLOOR
    ):
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
    embedding_backend: str = "onnx",
    embedding_openai_env_file: str | None = None,
    embedding_batch_size: int = 32,
    rrf_similarity_cutoff: float | None = None,
    dense_weight: float = 0.5,
    access_weight: float = 0.0,
    rrf_k: int = 60,
    query_prefix: str = "",
    document_prefix: str = "",
    skip_stale_backfill: bool = False,
) -> dict[str, Any]:
    if session_id is not None and not session_exists(db_path, session_id):
        raise SessionNotFoundError(f"session not found: {session_id}")

    empty_result = {
        "query": query,
        "current_time": _iso_utc(_utc_now()),
        "memories": [],
        "total_tokens": 0,
        "truncated": False,
        "matched_count": 0,
    }

    total_candidates = count_memories(db_path, session_id=session_id, kind="episodic")
    if total_candidates == 0:
        return empty_result

    model_key = embedding_model_key(
        embedding_model,
        backend=embedding_backend,
        models_dir=models_dir,
        openai_env_file=embedding_openai_env_file,
        document_prefix=document_prefix,
    )
    query_embedding = embed_texts(
        [query_prefix + query],
        embedding_model=embedding_model,
        backend=embedding_backend,
        models_dir=models_dir,
        openai_env_file=embedding_openai_env_file,
        batch_size=embedding_batch_size,
    )[0]
    if not skip_stale_backfill:
        stale = fetch_stale_memories(
            db_path,
            session_id=session_id,
            kind="episodic",
            embedding_model=model_key,
            embedding_dimensions=len(query_embedding),
        )
        if stale:
            document_embeddings = embed_texts(
                [document_prefix + row.content for row in stale],
                embedding_model=embedding_model,
                backend=embedding_backend,
                models_dir=models_dir,
                openai_env_file=embedding_openai_env_file,
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

    pool_size = min(
        max(top_k * _CANDIDATE_POOL_MULTIPLIER, _CANDIDATE_POOL_FLOOR),
        _CANDIDATE_POOL_CAP,
    )
    dense_scores = fetch_dense_scores(
        db_path,
        query_embedding,
        embedding_model=model_key,
        session_id=session_id,
        kind="episodic",
        limit=pool_size,
    )
    sparse_scores = fetch_sparse_scores(
        db_path, query, session_id=session_id, kind="episodic", limit=pool_size
    )
    candidate_ids = set(dense_scores) | set(sparse_scores)
    if not candidate_ids:
        return empty_result
    candidates = fetch_memories_by_ids(db_path, candidate_ids)

    bm25_scores = [float(sparse_scores.get(row.id, 0.0)) for row in candidates]

    dense_values = [float(dense_scores.get(row.id, 0.0)) for row in candidates]
    bm25_ranks = _rank_by_score(bm25_scores)
    dense_ranks = _rank_by_score(dense_values)
    access_ranks = _rank_by_score([float(row.recall_count) for row in candidates])
    sparse_weight = 1.0 - dense_weight
    max_rrf_score = (sparse_weight + dense_weight + access_weight) / (rrf_k + 1)
    fused: list[tuple[MemoryRow, dict[str, float | int | str]]] = []
    for index, row in enumerate(candidates):
        bm25_rank = bm25_ranks[index]
        dense_rank = dense_ranks[index]
        access_rank = access_ranks[index]
        rrf_score = (
            sparse_weight / (rrf_k + bm25_rank)
            + dense_weight / (rrf_k + dense_rank)
            + access_weight / (rrf_k + access_rank)
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
                    "relevance": _relevance_label(rrf_similarity, dense_values[index]),
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

    current_time = _utc_now()
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
            selected.append(
                _memory_payload(row, rank, scores, content_tokens)
            )
            total_tokens = content_tokens
            break
        selected.append(
            _memory_payload(row, rank, scores, content_tokens)
        )
        total_tokens += content_tokens

    if selected:
        record_recall_hits(
            db_path,
            [str(memory["id"]) for memory in selected],
            _iso_utc(current_time),
        )

    return {
        "query": query,
        "current_time": _iso_utc(current_time),
        "memories": selected,
        "total_tokens": total_tokens,
        "truncated": truncated,
        "matched_count": len(ranked),
    }


def _memory_payload(
    row: MemoryRow,
    rank: int,
    retrieval: dict[str, float | int | str],
    content_tokens: int,
) -> dict[str, Any]:
    return {
        "id": row.id,
        "ref": short_memory_ref(row.id),
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
