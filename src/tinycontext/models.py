from __future__ import annotations

from dataclasses import dataclass

MEMORY_KINDS = frozenset({"episodic", "profile"})


@dataclass(frozen=True)
class MemoryInput:
    content: str
    kind: str = "episodic"


@dataclass(frozen=True)
class MemoryRow:
    id: str
    session_id: str | None
    content: str
    created_at: str
    embedding: list[float] | None = None
    embedding_model: str | None = None
    embedding_dimensions: int | None = None
    superseded_by: str | None = None
    superseded_at: str | None = None
    last_recalled_at: str | None = None
    recall_count: int = 0
    kind: str = "episodic"
