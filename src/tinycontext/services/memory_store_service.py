from __future__ import annotations

import math
import os
import sqlite3
import struct
import threading
from importlib.metadata import version
from pathlib import Path
from typing import Any, Sequence

import sqlite_vec

from tinycontext.models import MemoryRow


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
  id TEXT PRIMARY KEY,
  session_id TEXT,
  content TEXT NOT NULL,
  created_at TEXT NOT NULL,
  embedding BLOB,
  embedding_model TEXT,
  embedding_dimensions INTEGER
);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
"""
_EMBEDDING_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_memories_embedding_model
ON memories(embedding_model, embedding_dimensions)
WHERE embedding IS NOT NULL;
"""

_BUSY_TIMEOUT_MS = 5000
_EMBEDDING_COLUMNS: dict[str, str] = {
    "embedding": "BLOB",
    "embedding_model": "TEXT",
    "embedding_dimensions": "INTEGER",
}


def _cosine_distance(left: bytes, right: bytes) -> float:
    """Compute cosine distance for sqlite-vec float32 BLOBs."""
    if len(left) != len(right) or len(left) % 4:
        raise ValueError("vectors must have matching float32 dimensions")
    dimensions = len(left) // 4
    if dimensions == 0:
        raise ValueError("vectors must not be empty")
    left_values = struct.unpack(f"{dimensions}f", left)
    right_values = struct.unpack(f"{dimensions}f", right)
    dot_product = sum(a * b for a, b in zip(left_values, right_values))
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    similarity = dot_product / (left_norm * right_norm)
    return 1.0 - max(-1.0, min(1.0, similarity))


def _register_python_vector_functions(conn: sqlite3.Connection) -> None:
    conn.create_function(
        "vec_distance_cosine",
        2,
        _cosine_distance,
        deterministic=True,
    )


def _load_sqlite_vec(conn: sqlite3.Connection) -> bool:
    """Load sqlite-vec, falling back when SQLite extensions are unavailable."""
    enable_load_extension = getattr(conn, "enable_load_extension", None)
    if enable_load_extension is None:
        _register_python_vector_functions(conn)
        return False

    try:
        enable_load_extension(True)
        try:
            sqlite_vec.load(conn)
        finally:
            enable_load_extension(False)
    except (AttributeError, OSError, sqlite3.Error):
        _register_python_vector_functions(conn)
        return False
    return True


def _migrate_schema(conn: sqlite3.Connection) -> None:
    columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(memories)").fetchall()
    }
    for name, column_type in _EMBEDDING_COLUMNS.items():
        if name not in columns:
            conn.execute(f"ALTER TABLE memories ADD COLUMN {name} {column_type}")
    conn.executescript(_EMBEDDING_INDEX_SQL)
    conn.commit()


class _ConnectionPool:
    """Thread-safe, single-connection-per-path pool with WAL mode."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._connections: dict[Path, sqlite3.Connection] = {}
        self._locks: dict[Path, threading.Lock] = {}

    def _get_or_create(self, db_path: Path) -> tuple[sqlite3.Connection, threading.Lock]:
        db_path = Path(db_path)
        db_path = Path(os.path.normcase(os.path.normpath(str(db_path.expanduser()))))
        with self._guard:
            if db_path not in self._connections:
                db_path.parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(
                    db_path,
                    check_same_thread=False,
                    timeout=_BUSY_TIMEOUT_MS / 1000,
                )
                try:
                    conn.row_factory = sqlite3.Row
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
                    _load_sqlite_vec(conn)
                    conn.executescript(_SCHEMA_SQL)
                    _migrate_schema(conn)
                except Exception:
                    conn.close()
                    raise
                self._connections[db_path] = conn
                self._locks[db_path] = threading.Lock()
            return self._connections[db_path], self._locks[db_path]

    def execute(self, db_path: Path, fn: Any) -> Any:
        conn, lock = self._get_or_create(db_path)
        with lock:
            return fn(conn)

    def close(self, db_path: Path) -> None:
        db_path = Path(db_path)
        db_path = Path(os.path.normcase(os.path.normpath(str(db_path.expanduser()))))
        with self._guard:
            conn = self._connections.pop(db_path, None)
            self._locks.pop(db_path, None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


_pool = _ConnectionPool()


def close_connection(db_path: Path) -> None:
    """Release a pooled database connection, primarily for tests and shutdown."""
    _pool.close(db_path)


def init_db(db_path: Path) -> None:
    """Ensure the existing TinyContext schema is initialized."""
    _pool._get_or_create(db_path)


def insert_memories(db_path: Path, rows: list[MemoryRow]) -> None:
    def _insert(conn: sqlite3.Connection) -> None:
        conn.executemany(
            """
            INSERT INTO memories (
              id,
              session_id,
              content,
              created_at,
              embedding,
              embedding_model,
              embedding_dimensions
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row.id,
                    row.session_id,
                    row.content,
                    row.created_at,
                    (
                        sqlite_vec.serialize_float32(row.embedding)
                        if row.embedding is not None
                        else None
                    ),
                    row.embedding_model,
                    (
                        row.embedding_dimensions
                        if row.embedding_dimensions is not None
                        else len(row.embedding or [])
                    )
                    or None,
                )
                for row in rows
            ],
        )
        conn.commit()

    _pool.execute(db_path, _insert)


def _row_to_memory(row: sqlite3.Row) -> MemoryRow:
    return MemoryRow(
        id=str(row["id"]),
        session_id=row["session_id"],
        content=str(row["content"]),
        created_at=str(row["created_at"]),
        embedding_model=row["embedding_model"],
        embedding_dimensions=row["embedding_dimensions"],
    )


def fetch_candidates(
    db_path: Path,
    *,
    session_id: str | None = None,
    limit: int | None = None,
) -> list[MemoryRow]:
    def _fetch(conn: sqlite3.Connection) -> list[MemoryRow]:
        query = """
        SELECT
          id,
          session_id,
          content,
          created_at,
          embedding_model,
          embedding_dimensions
        FROM memories
        """
        params: list[Any] = []
        if session_id is not None:
            query += " WHERE session_id = ?"
            params.append(session_id)
        query += " ORDER BY created_at DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(int(limit))
        rows = conn.execute(query, params).fetchall()
        return [_row_to_memory(row) for row in rows]

    return _pool.execute(db_path, _fetch)


def update_memory_embeddings(
    db_path: Path,
    rows: Sequence[tuple[str, Sequence[float]]],
    *,
    embedding_model: str,
) -> None:
    """Store float32 embeddings for existing rows without changing their payloads."""
    updates = [
        (
            sqlite_vec.serialize_float32(vector),
            embedding_model,
            len(vector),
            memory_id,
        )
        for memory_id, vector in rows
    ]
    if not updates:
        return

    def _update(conn: sqlite3.Connection) -> None:
        conn.executemany(
            """
            UPDATE memories
            SET embedding = ?, embedding_model = ?, embedding_dimensions = ?
            WHERE id = ?
            """,
            updates,
        )
        conn.commit()

    _pool.execute(db_path, _update)


def fetch_dense_scores(
    db_path: Path,
    query_embedding: Sequence[float],
    *,
    embedding_model: str,
    session_id: str | None = None,
) -> dict[str, float]:
    """Run cosine similarity inside SQLite through sqlite-vec."""
    if not query_embedding:
        return {}

    def _fetch(conn: sqlite3.Connection) -> dict[str, float]:
        query = """
        SELECT
          id,
          1.0 - vec_distance_cosine(embedding, ?) AS dense_score
        FROM memories
        WHERE embedding IS NOT NULL
          AND embedding_model = ?
          AND embedding_dimensions = ?
        """
        params: list[Any] = [
            sqlite_vec.serialize_float32(query_embedding),
            embedding_model,
            len(query_embedding),
        ]
        if session_id is not None:
            query += " AND session_id = ?"
            params.append(session_id)
        query += " ORDER BY dense_score DESC, created_at DESC"
        rows = conn.execute(query, params).fetchall()
        return {str(row["id"]): float(row["dense_score"]) for row in rows}

    return _pool.execute(db_path, _fetch)


def embedding_storage_stats(db_path: Path) -> dict[str, int]:
    """Return storage counts used by doctor and tests."""
    def _stats(conn: sqlite3.Connection) -> dict[str, int]:
        row = conn.execute(
            """
            SELECT
              COUNT(*) AS total,
              COUNT(embedding) AS embedded
            FROM memories
            """
        ).fetchone()
        return {
            "total": int(row["total"]),
            "embedded": int(row["embedded"]),
        }

    return _pool.execute(db_path, _stats)


def embedding_model_mismatch_count(db_path: Path, current_embedding_model: str) -> int:
    """Count rows embedded under a different model key than ``current_embedding_model``.

    These rows get re-embedded synchronously, inline, the next time a recall
    touches them (see ``memory_recall_run``'s stale-row backfill) -- which can
    stall a recall call for a large memory store.
    """
    def _count(conn: sqlite3.Connection) -> int:
        row = conn.execute(
            """
            SELECT COUNT(*) AS mismatched
            FROM memories
            WHERE embedding_model IS NULL OR embedding_model != ?
            """,
            (current_embedding_model,),
        ).fetchone()
        return int(row["mismatched"])

    return _pool.execute(db_path, _count)


def sqlite_vec_version() -> str:
    """Return the loaded extension version using an isolated SQLite connection."""
    conn = sqlite3.connect(":memory:")
    try:
        if _load_sqlite_vec(conn):
            row = conn.execute("SELECT vec_version()").fetchone()
            return str(row[0])
        return f"{version('sqlite-vec')} (Python cosine fallback)"
    finally:
        conn.close()


def session_exists(db_path: Path, session_id: str) -> bool:
    def _exists(conn: sqlite3.Connection) -> bool:
        row = conn.execute(
            "SELECT 1 FROM memories WHERE session_id = ? LIMIT 1",
            (session_id,),
        ).fetchone()
        return row is not None

    return _pool.execute(db_path, _exists)
