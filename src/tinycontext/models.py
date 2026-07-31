from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryInput:
    content: str


@dataclass(frozen=True)
class MemoryRow:
    id: str
    session_id: str | None
    content: str
    created_at: str
    embedding: list[float] | None = None
    embedding_model: str | None = None
    embedding_dimensions: int | None = None
