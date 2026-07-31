from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MemoryInput:
    content: str
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class MemoryRow:
    id: str
    session_id: str | None
    content: str
    tags: list[str]
    metadata: dict[str, Any]
    created_at: str
    embedding: list[float] | None = None
    embedding_model: str | None = None
    embedding_dimensions: int | None = None
