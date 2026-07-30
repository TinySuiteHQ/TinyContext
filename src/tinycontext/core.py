"""TinyContext's save and recall operations, independent of transport."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from tinycontext.config import ConfigInput, resolve_config
from tinycontext.errors import EmptyMemoryError, RecallBudgetError
from tinycontext.models import MemoryInput, MemoryRow
from tinycontext.pipelines.memory_recall import memory_recall_run
from tinycontext.services.embedding_service import embed_texts, embedding_model_key
from tinycontext.services.memory_store_service import insert_memories
from tinycontext.services.token_counter_service import token_count


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
        models_dir=Path(str(resolved["models_dir"])),
        batch_size=int(resolved["embedding_batch_size"]),
    )
    model_key = embedding_model_key(
        str(resolved["embedding_model"]),
        models_dir=Path(str(resolved["models_dir"])),
        document_prefix=str(resolved["dense_document_prefix"]),
    )
    rows: list[MemoryRow] = []
    saved: list[dict[str, Any]] = []
    for (item, content), vector in zip(normalized_items, vectors, strict=True):
        memory_id = str(uuid.uuid4())
        created_at = _utc_now_iso()
        tags = list(item.tags or [])
        metadata = dict(item.metadata or {})
        rows.append(
            MemoryRow(
                id=memory_id,
                session_id=session_id,
                content=content,
                tags=tags,
                metadata=metadata,
                created_at=created_at,
                embedding=vector,
                embedding_model=model_key,
                embedding_dimensions=len(vector),
            )
        )
        saved.append(
            {
                "id": memory_id,
                "session_id": session_id,
                "content_tokens": token_count(content, encoding_name),
                "created_at": created_at,
            }
        )
    insert_memories(db_path, rows)
    return {"saved": saved}


def recall_memories(
    query: str,
    *,
    session_id: str | None = None,
    max_tokens: int | None = None,
    top_k: int | None = None,
    config: ConfigInput | None = None,
) -> dict[str, Any]:
    query = query.strip()
    if not query:
        raise EmptyMemoryError("query must not be empty")
    if max_tokens is not None and max_tokens < 1:
        raise RecallBudgetError("max_tokens must be at least 1")
    if top_k is not None and top_k < 1:
        raise RecallBudgetError("top_k must be at least 1")

    resolved = _resolved_values(config)
    return memory_recall_run(
        query=query,
        session_id=session_id,
        max_tokens=max_tokens or int(resolved["recall_max_tokens"]),
        top_k=top_k or int(resolved["recall_top_k"]),
        db_path=Path(str(resolved["memory_db_path"])),
        encoding_name=str(resolved["encoding_name"]),
        models_dir=Path(str(resolved["models_dir"])),
        embedding_model=str(resolved["embedding_model"]),
        embedding_batch_size=int(resolved["embedding_batch_size"]),
        dense_weight=float(resolved["recall_dense_weight"]),
        rrf_k=int(resolved["recall_rrf_k"]),
        query_prefix=str(resolved["dense_query_prefix"]),
        document_prefix=str(resolved["dense_document_prefix"]),
    )
