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
    InvalidMemoryKindError,
    InvalidSortError,
    MemoryAlreadySupersededError,
    MemoryNotFoundError,
    RecallBudgetError,
    SessionNotFoundError,
)
from tinycontext.models import MEMORY_KINDS, MemoryInput, MemoryRow
from tinycontext.pipelines.memory_recall import memory_recall_run
from tinycontext.services.embedding_reindex_service import (
    ensure_background_reindex,
    reindex_notice,
)
from tinycontext.services.embedding_service import embed_texts, embedding_model_key
from tinycontext.services.memory_store_service import (
    clear_superseded_by,
    count_memories,
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
    normalized_items: list[tuple[str, str]] = []
    for item in memories:
        content = item.content.strip()
        if not content:
            raise EmptyMemoryError("memory content must not be empty")
        kind = (item.kind or "episodic").strip()
        if kind not in MEMORY_KINDS:
            raise InvalidMemoryKindError(
                f"kind must be one of {sorted(MEMORY_KINDS)}, got {kind!r}"
            )
        contents.append(str(resolved["dense_document_prefix"]) + content)
        normalized_items.append((content, kind))

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
    dedup_review_threshold = float(resolved["dedup_review_similarity_threshold"])
    length_notice_tokens = int(resolved["save_length_notice_tokens"])
    saved: list[dict[str, Any]] = []
    skipped_duplicates: list[dict[str, Any]] = []
    for (content, kind), vector in zip(normalized_items, vectors, strict=True):
        # Profile facts are global to the store (not scoped to a session),
        # since identity/preference facts apply regardless of which
        # project/session is active.
        item_session_id = None if kind == "profile" else session_id
        scores = fetch_dense_scores(
            db_path,
            vector,
            embedding_model=model_key,
            session_id=item_session_id,
            kind=kind,
        )
        similar_to: dict[str, Any] | None = None
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
            if similarity >= dedup_review_threshold:
                # Not similar enough to auto-skip, but close enough that it's
                # probably a near-duplicate wording of an existing memory --
                # surface it so the caller can consolidate with update_memory
                # instead of the store quietly accumulating restated copies.
                similar_to = {
                    "ref": short_memory_ref(duplicate_id),
                    "similarity": round(similarity, 4),
                }
        memory_id = str(uuid.uuid4())
        created_at = _utc_now_iso()
        row = MemoryRow(
            id=memory_id,
            session_id=item_session_id,
            content=content,
            created_at=created_at,
            embedding=vector,
            embedding_model=model_key,
            embedding_dimensions=len(vector),
            kind=kind,
        )
        insert_memories(db_path, [row])
        content_tokens = token_count(content, encoding_name)
        saved_item: dict[str, Any] = {
            "id": memory_id,
            "ref": short_memory_ref(memory_id),
            "session_id": item_session_id,
            "kind": kind,
            "content_tokens": content_tokens,
            "created_at": created_at,
        }
        if similar_to is not None:
            saved_item["similar_to"] = similar_to
        if content_tokens > length_notice_tokens:
            saved_item["notice"] = (
                f"This memory is {content_tokens} tokens, above the recommended "
                f"{length_notice_tokens} -- consider splitting it into smaller, "
                "atomic facts for cleaner recall."
            )
        saved.append(saved_item)
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
        result = _recall_recent_memories(
            session_id=session_id,
            max_tokens=max_tokens,
            top_k=top_k or _RECENT_MODE_DEFAULT_TOP_K,
            resolved=resolved,
            db_path=db_path,
        )
        result["profile"] = _fetch_profile_block(resolved, db_path)
        return result

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
    result["profile"] = _fetch_profile_block(resolved, db_path)
    return result


def _fetch_profile_block(resolved: dict[str, Any], db_path: Path) -> list[dict[str, Any]]:
    """Fetch the durable, always-surfaced profile pool, trimmed to its own budget.

    Profile memories are global to the store (session_id=None), newest-first,
    and not semantically ranked -- the pool is small by design, so every
    recall attaches it unconditionally instead of requiring a separate call.
    """
    rows = fetch_recent_memories(db_path, session_id=None, kind="profile")
    if not rows:
        return []

    encoding_name = str(resolved["encoding_name"])
    profile_max_tokens = int(resolved["profile_max_tokens"])
    selected: list[dict[str, Any]] = []
    total_tokens = 0
    for rank, row in enumerate(rows, start=1):
        content_tokens = token_count(row.content, encoding_name)
        if not selected and content_tokens > profile_max_tokens:
            selected.append(_recent_memory_payload(row, rank, content_tokens))
            break
        if selected and total_tokens + content_tokens > profile_max_tokens:
            break
        selected.append(_recent_memory_payload(row, rank, content_tokens))
        total_tokens += content_tokens
    return selected


def _recall_recent_memories(
    *,
    session_id: str | None,
    max_tokens: int | None,
    top_k: int,
    resolved: dict[str, Any],
    db_path: Path,
) -> dict[str, Any]:
    """Return the newest durable memories within the configured token budget."""
    rows = fetch_recent_memories(
        db_path, session_id=session_id, kind="episodic", limit=top_k
    )
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
        "matched_count": len(rows),
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
        kind=old_row.kind,
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


_LIST_DEFAULT_LIMIT = 20
_LIST_MAX_LIMIT = 200
_LIST_PREVIEW_CHARS = 200
_LIST_SORT_VALUES = frozenset({"recent", "stale"})


def get_memory(memory_id: str, *, config: ConfigInput | None = None) -> dict[str, Any]:
    """Fetch a single memory's full content by ref or id, bypassing recall/ranking."""
    memory_id = memory_id.strip()
    if not memory_id:
        raise EmptyMemoryError("memory_id must not be empty")

    resolved = _resolved_values(config)
    db_path = Path(str(resolved["memory_db_path"]))
    encoding_name = str(resolved["encoding_name"])

    resolved_id = _resolve_memory_id(db_path, memory_id)
    row = fetch_memory_by_id(db_path, resolved_id)
    if row is None:
        raise MemoryNotFoundError(f"no memory found with id {memory_id!r}")

    return {
        "id": row.id,
        "ref": short_memory_ref(row.id),
        "session_id": row.session_id,
        "kind": row.kind,
        "content": row.content,
        "content_tokens": token_count(row.content, encoding_name),
        "created_at": row.created_at,
        "superseded_by": short_memory_ref(row.superseded_by)
        if row.superseded_by
        else None,
        "superseded_at": row.superseded_at,
        "last_recalled_at": row.last_recalled_at,
        "recall_count": row.recall_count,
    }


def list_memories(
    *,
    session_id: str | None = None,
    kind: str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    sort: str = "recent",
    config: ConfigInput | None = None,
) -> dict[str, Any]:
    """Browse stored memories: a cheap, deterministic catalog view.

    Unlike ``recall_memories`` this does no embedding calls, no ranking, and
    no token-budget cutoff -- it's meant for paging through what's stored
    (optionally scoped to a date range) rather than finding what's relevant
    to a query. Content is truncated to a short preview per entry; use
    ``get_memory`` to read one entry in full.

    ``sort="recent"`` (default) is newest-first. ``sort="stale"`` instead
    surfaces the least-recalled, oldest memories first -- candidates to
    review, consolidate with ``update_memory``, or remove with
    ``delete_memory``.
    """
    if limit is not None and limit < 1:
        raise RecallBudgetError("limit must be at least 1")
    if offset < 0:
        raise RecallBudgetError("offset must be at least 0")
    if kind is not None and kind not in MEMORY_KINDS:
        raise InvalidMemoryKindError(
            f"kind must be one of {sorted(MEMORY_KINDS)}, got {kind!r}"
        )
    if sort not in _LIST_SORT_VALUES:
        raise InvalidSortError(
            f"sort must be one of {sorted(_LIST_SORT_VALUES)}, got {sort!r}"
        )

    resolved = _resolved_values(config)
    db_path = Path(str(resolved["memory_db_path"]))
    encoding_name = str(resolved["encoding_name"])
    resolved_limit = min(limit or _LIST_DEFAULT_LIMIT, _LIST_MAX_LIMIT)

    rows = fetch_recent_memories(
        db_path,
        session_id=session_id,
        kind=kind,
        limit=resolved_limit,
        offset=offset,
        since=since,
        until=until,
        order_by=sort,
    )
    total_count = count_memories(
        db_path, session_id=session_id, kind=kind, since=since, until=until
    )

    memories: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        content_tokens = token_count(row.content, encoding_name)
        preview_truncated = len(row.content) > _LIST_PREVIEW_CHARS
        content = row.content[:_LIST_PREVIEW_CHARS] + "…" if preview_truncated else row.content
        memories.append(
            {
                "id": row.id,
                "ref": short_memory_ref(row.id),
                "session_id": row.session_id,
                "kind": row.kind,
                "content": content,
                "content_tokens": content_tokens,
                "preview_truncated": preview_truncated,
                "created_at": row.created_at,
                "recall_count": row.recall_count,
                "last_recalled_at": row.last_recalled_at,
                "rank": offset + index,
            }
        )

    return {
        "current_time": _utc_now_iso(),
        "memories": memories,
        "returned_count": len(memories),
        "total_count": total_count,
        "limit": resolved_limit,
        "offset": offset,
        "has_more": offset + len(memories) < total_count,
        "sort": sort,
    }


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
