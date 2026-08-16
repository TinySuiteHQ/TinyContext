"""TinyContext's save and recall operations, independent of transport."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from tinycontext.config import ConfigInput, resolve_config
from tinycontext.errors import (
    AmbiguousMemoryReferenceError,
    EmptyMemoryError,
    MemoryAlreadySupersededError,
    MemoryNotFoundError,
    RecallBudgetError,
    SessionNotFoundError,
)
from tinycontext.models import MemoryInput, MemoryRow
from tinycontext.pipelines.memory_recall import memory_recall_run
from tinycontext.services.embedding_reindex_service import (
    ensure_background_reindex,
    reindex_notice,
)
from tinycontext.services.embedding_service import embed_texts, embedding_model_key
from tinycontext.services.memory_store_service import (
    clear_superseded_by,
    delete_memory as _delete_memory_row,
    embedding_model_mismatch_count,
    embedding_storage_stats,
    fetch_dense_scores,
    fetch_memory_by_id,
    fetch_recent_memories,
    find_memory_ids_by_ref,
    insert_memories,
    looks_like_short_ref,
    short_memory_ref,
    supersede_memory,
)
from tinycontext.services.token_counter_service import token_count


def _kick_off_background_reindex(resolved: dict[str, Any], db_path: Path) -> str | None:
    """Start a background re-embed if the store has drifted; return a notice if so."""
    if not db_path.exists():
        return None
    ensure_background_reindex(
        db_path,
        embedding_model=str(resolved["embedding_model"]),
        embedding_backend=str(resolved["embedding_backend"]),
        models_dir=Path(str(resolved["models_dir"])),
        openai_env_file=str(resolved["embedding_openai_env_file"]),
        embedding_batch_size=int(resolved["embedding_batch_size"]),
        document_prefix=str(resolved["dense_document_prefix"]),
    )
    return reindex_notice(db_path)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _resolved_values(config: ConfigInput | None) -> dict[str, Any]:
    return resolve_config(config).to_dict()


def save_memories(
    memories: Sequence[MemoryInput],
    *,
    session_id: str | None = None,
    config: ConfigInput | None = None,
) -> dict[str, Any]:
    if not memories:
        raise EmptyMemoryError("memories must not be empty")

    resolved = _resolved_values(config)
    db_path = Path(str(resolved["memory_db_path"]))
    notice = _kick_off_background_reindex(resolved, db_path)
    encoding_name = str(resolved["encoding_name"])
    contents: list[str] = []
    normalized_items: list[tuple[MemoryInput, str]] = []
    for item in memories:
        content = item.content.strip()
        if not content:
            raise EmptyMemoryError("memory content must not be empty")
        contents.append(str(resolved["dense_document_prefix"]) + content)
        normalized_items.append((item, content))

    vectors = embed_texts(
        contents,
        embedding_model=str(resolved["embedding_model"]),
        backend=str(resolved["embedding_backend"]),
        models_dir=Path(str(resolved["models_dir"])),
        openai_env_file=str(resolved["embedding_openai_env_file"]),
        batch_size=int(resolved["embedding_batch_size"]),
    )
    model_key = embedding_model_key(
        str(resolved["embedding_model"]),
        backend=str(resolved["embedding_backend"]),
        models_dir=Path(str(resolved["models_dir"])),
        openai_env_file=str(resolved["embedding_openai_env_file"]),
        document_prefix=str(resolved["dense_document_prefix"]),
    )
    dedup_threshold = float(resolved["dedup_similarity_threshold"])
    saved: list[dict[str, Any]] = []
    skipped_duplicates: list[dict[str, Any]] = []
    for (item, content), vector in zip(normalized_items, vectors, strict=True):
        scores = fetch_dense_scores(
            db_path,
            vector,
            embedding_model=model_key,
            session_id=session_id,
        )
        if scores:
            duplicate_id, similarity = max(scores.items(), key=lambda kv: kv[1])
            if similarity >= dedup_threshold:
                skipped_duplicates.append(
                    {
                        "duplicate_of": short_memory_ref(duplicate_id),
                        "similarity": round(similarity, 4),
                    }
                )
                continue
        memory_id = str(uuid.uuid4())
        created_at = _utc_now_iso()
        row = MemoryRow(
            id=memory_id,
            session_id=session_id,
            content=content,
            created_at=created_at,
            embedding=vector,
            embedding_model=model_key,
            embedding_dimensions=len(vector),
        )
        insert_memories(db_path, [row])
        saved.append(
            {
                "id": memory_id,
                "ref": short_memory_ref(memory_id),
                "session_id": session_id,
                "content_tokens": token_count(content, encoding_name),
                "created_at": created_at,
            }
        )
    result: dict[str, Any] = {"saved": saved}
    if skipped_duplicates:
        result["skipped_duplicates"] = skipped_duplicates
    if notice:
        result["notice"] = notice
    return result


_RECENT_MODE_DEFAULT_TOP_K = 5


def recall_memories(
    query: str | None = None,
    *,
    session_id: str | None = None,
    max_tokens: int | None = None,
    top_k: int | None = None,
    config: ConfigInput | None = None,
) -> dict[str, Any]:
    """Recall memories relevant to ``query``, or the newest ones if omitted.

    Passing a non-empty ``query`` runs hybrid semantic search. Omitting it
    (or passing an empty/whitespace-only string) returns the newest stored
    memories in chronological order instead -- a bounded, non-semantic view
    useful for resuming a recent thread.
    """
    query = (query or "").strip()
    if max_tokens is not None and max_tokens < 1:
        raise RecallBudgetError("max_tokens must be at least 1")
    if top_k is not None and top_k < 1:
        raise RecallBudgetError("top_k must be at least 1")

    resolved = _resolved_values(config)
    db_path = Path(str(resolved["memory_db_path"]))

    if not query:
        return _recall_recent_memories(
            session_id=session_id,
            max_tokens=max_tokens,
            top_k=top_k or _RECENT_MODE_DEFAULT_TOP_K,
            resolved=resolved,
            db_path=db_path,
        )

    notice = _kick_off_background_reindex(resolved, db_path)
    result = memory_recall_run(
        query=query,
        session_id=session_id,
        max_tokens=max_tokens or int(resolved["recall_max_tokens"]),
        top_k=top_k or int(resolved["recall_top_k"]),
        db_path=db_path,
        encoding_name=str(resolved["encoding_name"]),
        models_dir=Path(str(resolved["models_dir"])),
        embedding_model=str(resolved["embedding_model"]),
        embedding_backend=str(resolved["embedding_backend"]),
        embedding_openai_env_file=str(resolved["embedding_openai_env_file"]),
        embedding_batch_size=int(resolved["embedding_batch_size"]),
        rrf_similarity_cutoff=float(resolved["recall_rrf_cutoff"]),
        dense_weight=float(resolved["recall_dense_weight"]),
        access_weight=float(resolved["recall_access_weight"]),
        rrf_k=int(resolved["recall_rrf_k"]),
        query_prefix=str(resolved["dense_query_prefix"]),
        document_prefix=str(resolved["dense_document_prefix"]),
        skip_stale_backfill=notice is not None,
    )
    if notice:
        result["notice"] = notice
    return result


def _recall_recent_memories(
    *,
    session_id: str | None,
    max_tokens: int | None,
    top_k: int,
    resolved: dict[str, Any],
    db_path: Path,
) -> dict[str, Any]:
    """Return the newest durable memories within the configured token budget."""
    rows = fetch_recent_memories(db_path, session_id=session_id, limit=top_k)
    if session_id is not None and not rows:
        raise SessionNotFoundError(f"session not found: {session_id}")

    current_time = _utc_now_iso()
    selected: list[dict[str, Any]] = []
    total_tokens = 0
    truncated = False
    resolved_max_tokens = max_tokens or int(resolved["recall_max_tokens"])
    encoding_name = str(resolved["encoding_name"])
    for rank, row in enumerate(rows, start=1):
        content_tokens = token_count(row.content, encoding_name)
        if selected and total_tokens + content_tokens > resolved_max_tokens:
            truncated = True
            break
        if not selected and content_tokens > resolved_max_tokens:
            truncated = True
            selected.append(_recent_memory_payload(row, rank, content_tokens))
            total_tokens = content_tokens
            break
        selected.append(_recent_memory_payload(row, rank, content_tokens))
        total_tokens += content_tokens

    return {
        "mode": "recent",
        "current_time": current_time,
        "memories": selected,
        "total_tokens": total_tokens,
        "truncated": truncated,
    }


def _recent_memory_payload(
    row: MemoryRow,
    rank: int,
    content_tokens: int,
) -> dict[str, Any]:
    return {
        "id": row.id,
        "ref": short_memory_ref(row.id),
        "content": row.content,
        "rank": rank,
        "content_tokens": content_tokens,
        "created_at": row.created_at,
    }


def _resolve_memory_id(db_path: Path, memory_id: str) -> str:
    if not looks_like_short_ref(memory_id):
        return memory_id
    matches = find_memory_ids_by_ref(db_path, memory_id.lower())
    if not matches:
        raise MemoryNotFoundError(f"no memory found with ref {memory_id!r}")
    if len(matches) > 1:
        raise AmbiguousMemoryReferenceError(
            f"ref {memory_id!r} matches {len(matches)} memories; "
            "use the full id instead"
        )
    return matches[0]


def update_memory(
    memory_id: str,
    content: str,
    *,
    config: ConfigInput | None = None,
) -> dict[str, Any]:
    """Supersede a stored memory with corrected content, preserving history."""
    memory_id = memory_id.strip()
    if not memory_id:
        raise EmptyMemoryError("memory_id must not be empty")
    content = content.strip()
    if not content:
        raise EmptyMemoryError("memory content must not be empty")

    resolved = _resolved_values(config)
    db_path = Path(str(resolved["memory_db_path"]))
    encoding_name = str(resolved["encoding_name"])

    resolved_id = _resolve_memory_id(db_path, memory_id)
    old_row = fetch_memory_by_id(db_path, resolved_id)
    if old_row is None:
        raise MemoryNotFoundError(f"no memory found with id {memory_id!r}")
    if old_row.superseded_by is not None:
        raise MemoryAlreadySupersededError(
            f"memory {memory_id!r} was already superseded by "
            f"{short_memory_ref(old_row.superseded_by)!r}; update that memory instead"
        )

    vector = embed_texts(
        [str(resolved["dense_document_prefix"]) + content],
        embedding_model=str(resolved["embedding_model"]),
        backend=str(resolved["embedding_backend"]),
        models_dir=Path(str(resolved["models_dir"])),
        openai_env_file=str(resolved["embedding_openai_env_file"]),
        batch_size=int(resolved["embedding_batch_size"]),
    )[0]
    model_key = embedding_model_key(
        str(resolved["embedding_model"]),
        backend=str(resolved["embedding_backend"]),
        models_dir=Path(str(resolved["models_dir"])),
        openai_env_file=str(resolved["embedding_openai_env_file"]),
        document_prefix=str(resolved["dense_document_prefix"]),
    )
    new_id = str(uuid.uuid4())
    created_at = _utc_now_iso()
    new_row = MemoryRow(
        id=new_id,
        session_id=old_row.session_id,
        content=content,
        created_at=created_at,
        embedding=vector,
        embedding_model=model_key,
        embedding_dimensions=len(vector),
    )
    insert_memories(db_path, [new_row])
    supersede_memory(db_path, resolved_id, new_id, superseded_at=created_at)

    return {
        "id": new_id,
        "ref": short_memory_ref(new_id),
        "session_id": old_row.session_id,
        "content_tokens": token_count(content, encoding_name),
        "created_at": created_at,
        "supersedes": {"id": resolved_id, "ref": short_memory_ref(resolved_id)},
    }


def delete_memory(
    memory_id: str,
    *,
    config: ConfigInput | None = None,
) -> dict[str, Any]:
    memory_id = memory_id.strip()
    if not memory_id:
        raise EmptyMemoryError("memory_id must not be empty")

    resolved = _resolved_values(config)
    db_path = Path(str(resolved["memory_db_path"]))

    resolved_id = _resolve_memory_id(db_path, memory_id)

    deleted = _delete_memory_row(db_path, resolved_id)
    if not deleted:
        raise MemoryNotFoundError(f"no memory found with id {memory_id!r}")
    clear_superseded_by(db_path, resolved_id)
    return {"id": resolved_id, "ref": short_memory_ref(resolved_id), "deleted": True}


def start_background_reembed_if_needed(config: ConfigInput | None = None) -> str | None:
    """Kick off a background re-embed if the store has drifted. For server startup."""
    resolved = _resolved_values(config)
    db_path = Path(str(resolved["memory_db_path"]))
    return _kick_off_background_reindex(resolved, db_path)


def describe_embedding_drift(config: ConfigInput | None = None) -> str | None:
    """Report whether stored memories don't match the currently configured embedding model.

    This is a read-only diagnostic for callers (like ``doctor``) that don't
    otherwise touch the store. ``save_memories``/``recall_memories`` detect
    and fix this themselves by starting a background re-embed job (see
    ``embedding_reindex_service``) and surfacing a ``notice`` in their
    response while it runs. Returns None when there's nothing to warn about,
    including when the database doesn't exist yet (nothing has been saved).
    """
    resolved = _resolved_values(config)
    db_path = Path(str(resolved["memory_db_path"]))
    if not db_path.exists():
        return None

    model_key = embedding_model_key(
        str(resolved["embedding_model"]),
        backend=str(resolved["embedding_backend"]),
        models_dir=Path(str(resolved["models_dir"])),
        openai_env_file=str(resolved["embedding_openai_env_file"]),
        document_prefix=str(resolved["dense_document_prefix"]),
    )
    mismatched = embedding_model_mismatch_count(db_path, model_key)
    if mismatched <= 0:
        return None

    total = embedding_storage_stats(db_path)["total"]
    return (
        f"{mismatched} of {total} stored memories were embedded with a different "
        f"model/config than is currently set (embedding_model={resolved['embedding_model']!r}). "
        "They will be re-embedded synchronously, inline, the next time recall_memories "
        "fetches them -- this can be slow for a large memory store."
    )
